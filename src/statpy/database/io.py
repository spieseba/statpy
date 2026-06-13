import os
import base64
import json
import re
import h5py
import numpy as np
from statpy.log import message
from statpy.database.core import DB

_CFG_ID_RE = re.compile(r"n(\d+)$")


def _parse_cfg_id(name):
    """Parse the trailing ``n<digits>`` block of a CLS configlist entry."""
    m = _CFG_ID_RE.search(name)
    if m is None:
        raise ValueError(f"Cannot parse cfg id from name: {name!r}")
    return int(m.group(1))


def load_CLS(fn, rwf_fn, correlator_patterns, stream_tag, run_tag, cfgs_to_be_removed=None, meas_group="messpec", reverse=False, silent=False):
    """Load CLS hdf5 measurements + reweighting factors into a fresh ``DB``.

    Each hdf5 dataset under ``meas_group/data`` whose key contains a
    pattern in ``correlator_patterns`` becomes an atomic data entry at
    ``{stream_tag}/{run_tag}/{key}`` with cfg labels
    ``{stream_tag}-{cfg_id}`` sorted by ascending cfg id (descending if
    ``reverse=True``). The raw rwf is embedded as ``weights`` — no
    separate ``/rwf`` / ``/nrwf`` entries.

    ``cfgs_to_be_removed`` filters both streams before insertion; their
    remaining cfg sets must agree exactly.
    """
    if not os.path.isfile(fn):
        raise FileNotFoundError(f"hdf5 file {fn!r} not found!")
    if not os.path.isfile(rwf_fn):
        raise FileNotFoundError(f"rwf file {rwf_fn!r} not found!")
    if cfgs_to_be_removed is not None and not isinstance(cfgs_to_be_removed, (list, np.ndarray)):
        raise TypeError("'cfgs_to_be_removed' must be list | np.ndarray | None")

    message("---------------------------------", silent)
    message(f"Load CLS data from {fn}", silent)
    message(f"Load rw factors from: {rwf_fn}", silent)
    message(f" -- correlator patterns: {correlator_patterns}", silent)
    message(f" -- ensemble tag = {stream_tag}", silent)
    message(f" -- run tag: {run_tag}", silent)
    message(f" -- cfgs to be removed: {cfgs_to_be_removed}", silent)
    with h5py.File(fn, "r") as h5:
        f = h5[meas_group]
        h5_cfgs = np.array([_parse_cfg_id(cfg.decode("utf-8")) for cfg in f["configlist"]])
        h5_cfgs_filtered = h5_cfgs[~np.isin(h5_cfgs, cfgs_to_be_removed)] if cfgs_to_be_removed is not None else h5_cfgs
        _log_h5_git(f, silent)
        message(f"Number of cfgs in hdf5 file: {len(h5_cfgs)} | Number of filtered configs in hdf5 file: {len(h5_cfgs_filtered)}", silent)

        _log_rwf_git(rwf_fn, silent)
        rwf_cfgs, rwf_values = _load_rwf_dispatch(rwf_fn)
        rwf_cfgs_filtered = rwf_cfgs[~np.isin(rwf_cfgs, cfgs_to_be_removed)] if cfgs_to_be_removed is not None else rwf_cfgs
        message(f"Number of cfgs in rwf file: {rwf_cfgs.shape[0]} | Number of filtered configs in rwf file : {rwf_cfgs_filtered.shape[0]}", silent)

        common_cfgs = _resolve_common_cfgs(h5_cfgs_filtered, rwf_cfgs_filtered, stream_tag)
        if reverse:
            common_cfgs = common_cfgs[::-1]
        message(f"Number of filtered configs in hdf5 file and rwf file: {common_cfgs.shape[0]}", silent)

        rwf_idx = _argsort_to(rwf_cfgs, common_cfgs)
        h5_idx = _argsort_to(h5_cfgs, common_cfgs)

        # Raw rwf stored un-normalised; callers normalise after combining
        # streams (per-stream or globally) so FP-rounding matches their intent.
        weights = rwf_values[rwf_idx]
        cfg_labels = np.array([f"{stream_tag}-{int(c)}" for c in common_cfgs])

        db = DB()
        _populate_data(db, f, correlator_patterns, h5_idx, cfg_labels, weights, stream_tag, run_tag)
    message("---------------------------------", silent)
    return db


def _argsort_to(src_cfgs, target_cfgs):
    """Return ``idx`` such that ``src_cfgs[idx] == target_cfgs``."""
    pos = {int(c): i for i, c in enumerate(src_cfgs)}
    return np.array([pos[int(c)] for c in target_cfgs])


def _log_h5_git(f, silent):
    git_dict = f["description"].get("git")
    if git_dict is None:
        message("Git info not found for hdf5 file!", silent)
        return
    message("hdf5 git info:", silent)
    for key, val in git_dict.items():
        message(f"--- {key}: {val[()].decode()}", silent)


def _log_rwf_git(rwf_fn, silent):
    rwf_fn_git = rwf_fn + ".git"
    if not os.path.isfile(rwf_fn_git):
        message("Git info not found for rwf file!", silent)
        return
    message("rwf git info:", silent)
    with open(rwf_fn_git) as rwf_f:
        rwf_info_dict = json.load(rwf_f)
    for key, val in rwf_info_dict.items():
        message(f"--- {key}: {val}", silent)


def _load_rwf_dispatch(rwf_fn):
    if rwf_fn.endswith(".rwf"):
        return _load_rwf(rwf_fn)
    if rwf_fn.endswith(".rwms.txt"):
        return _load_rwms(rwf_fn)
    raise ValueError(f"Unknown rwf file format: {rwf_fn}")


def _load_rwf(fn):
    """Two-column ``.rwf`` → ``(cfg_ids, rwf_values)``."""
    data = np.loadtxt(fn)
    return data[:, 0].astype(int), data[:, 1]


def _load_rwms(fn):
    """Multi-column ``.rwms.txt`` → ``(cfg_ids, prod_of_rwf_columns)``."""
    data = np.loadtxt(fn)
    return data[:, 0].astype(int), np.prod(data[:, 1:], axis=1)


def parse_bootstrap_file(fn):
    """Parse a CLS ``.boot.txt`` → ``(bootstraps, configlist)``."""
    bootstraps = np.loadtxt(fn, dtype=int)
    with open(fn) as f:
        configlist = f.readlines()[3][:-1].replace("n", "-").split(" ")[1:]
    return bootstraps, configlist


def _resolve_common_cfgs(h5_cfgs_filtered, rwf_cfgs_filtered, stream_tag):
    """Assert hdf5 and rwf cfg sets match; return cfg ids sorted ascending."""
    if set(h5_cfgs_filtered.tolist()) != set(rwf_cfgs_filtered.tolist()):
        non_common = np.setxor1d(h5_cfgs_filtered, rwf_cfgs_filtered)
        raise ValueError(
            f"hdf5/rwf cfg mismatch for {stream_tag}: configs in only one file: {non_common}"
        )
    return np.sort(rwf_cfgs_filtered)


def _populate_data(db, f, correlator_patterns, h5_idx, cfg_labels, weights, stream_tag, run_tag):
    data_keys = list(f["data"].keys())
    for pattern in correlator_patterns:
        for key in data_keys:
            if pattern in key:
                f_vals = f["data"].get(key)[:]
                sample = f_vals[h5_idx]
                f_tag = f"{stream_tag}/{run_tag}/{key}"
                db.add_entry(
                    tag=f_tag,
                    sample=sample, weights=weights, cfgs=cfg_labels,
                )


def decode_v1_ndarray(blob):
    """Decode a v1 ``{"__ndarray__": <base64>, "dtype":..., "shape":...}`` blob."""
    buf = base64.b64decode(blob["__ndarray__"])
    return np.frombuffer(buf, dtype=np.dtype(blob["dtype"])).reshape(blob["shape"])


def load_v1_json(fn, silent=False):
    """Migrate a retired v1 (custom-JSON) statpy database into a fresh v2 ``DB``.

    The v1 format serialised each leaf as
    ``{tag: {"__leaf__": {mean, jks, sample, misc, checksum}}}`` with ndarrays
    base64-encoded (``{"__ndarray__": <b64>, "dtype": ..., "shape": ...}``) and
    ``sample``/``jks`` stored as cfg-keyed dicts. v2 dropped this format (it
    saves pickle + CRC32); this is a one-way importer for legacy data, not a
    revival of the format.

    Each leaf's per-cfg ``sample`` dict -- rectangular within a leaf -- is
    stacked into an array ``(n_cfgs, *value_shape)`` in file order, with the cfg
    labels as ``cfgs`` and uniform ``weights`` (v1 data carried none). ``mean``
    and ``jks`` are re-derived by :meth:`DB.add_entry`; for uniform weights this
    reproduces the v1 leave-one-out jackknife exactly. The v1 ``misc`` is carried
    through unchanged; the per-leaf ``checksum`` is dropped (v2 checksums the
    whole file via the CRC32 header).

    Raises ``ValueError`` if a leaf's per-cfg samples are not rectangular (they
    cannot be stacked into a v2 array entry).
    """
    if not os.path.isfile(fn):
        raise FileNotFoundError(f"{fn} not found")
    message(f"Migrate v1 JSON database from {fn}", silent)
    with open(fn) as f:
        raw = json.load(f)

    db = DB()
    for tag, wrapped in raw.items():
        leaf = wrapped["__leaf__"]
        sample_dict = leaf["sample"]
        cfgs = np.array(list(sample_dict.keys()))
        rows = [decode_v1_ndarray(sample_dict[c]) for c in cfgs]
        shapes = {r.shape for r in rows}
        if len(shapes) > 1:
            raise ValueError(
                f"load_v1_json({fn!r}): leaf {tag!r} has ragged per-cfg sample "
                f"shapes {sorted(shapes)}; cannot stack into a v2 array entry"
            )
        sample = np.array(rows)
        weights = np.ones(len(cfgs))
        db.add_entry(tag, sample=sample, weights=weights, cfgs=cfgs, misc=leaf.get("misc"))
    message(f" -- migrated {len(raw)} entries", silent)
    return db
