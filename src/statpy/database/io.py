import base64
import json
import os
import re
import textwrap

import h5py
import numpy as np

from statpy.database.core import DB
from statpy.log import message

_CFG_ID_RE = re.compile(r"n(\d+)$")


def _parse_cfg_id(name):
    """Parse the trailing ``n<digits>`` block of a CLS configlist entry."""
    m = _CFG_ID_RE.search(name)
    if m is None:
        raise ValueError(f"Cannot parse cfg id from name: {name!r}")
    return int(m.group(1))


def load_CLS(fn, rwf_fn, correlator_patterns, stream_tag, run_tag, configurations_to_be_removed=None, meas_group="messpec", reverse=False, silent=False):
    """Load CLS hdf5 measurements + reweighting factors into a fresh ``DB``.

    Each hdf5 dataset under ``meas_group/data`` whose key matches any regex
    in ``correlator_patterns`` via ``re.search`` becomes an atomic data entry at
    ``{stream_tag}/{run_tag}/{key}`` with cfg labels
    ``{stream_tag}-{cfg_id}`` sorted by ascending cfg id (descending if
    ``reverse=True``). The raw rwf is embedded as ``weights`` — no
    separate ``/rwf`` / ``/nrwf`` entries.

    ``configurations_to_be_removed`` filters both streams before insertion; their
    remaining cfg sets must agree exactly.

    Patterns are processed in order; overlapping matches load each dataset
    once. Escape literal regex metacharacters with ``re.escape``.
    """
    if not os.path.isfile(fn):
        raise FileNotFoundError(f"hdf5 file {fn!r} not found!")
    if not os.path.isfile(rwf_fn):
        raise FileNotFoundError(f"rwf file {rwf_fn!r} not found!")
    if configurations_to_be_removed is not None and not isinstance(configurations_to_be_removed, (list, np.ndarray)):
        raise TypeError("'configurations_to_be_removed' must be list | np.ndarray | None")

    compiled_patterns = [re.compile(pattern) for pattern in correlator_patterns]

    if not silent:
        print()
    message(
        f"Load CLS data — {stream_tag} / {run_tag}\n"
        f"  {'HDF5':<14} {fn}\n"
        f"  {'Reweighting':<14} {rwf_fn}\n"
        f"  {'Stream':<14} {stream_tag}\n"
        f"  {'Run':<14} {run_tag}\n"
        + textwrap.fill(
            str(correlator_patterns), width=72,
            initial_indent=f"  {'Regex patterns':<14} ", subsequent_indent=" " * 17,
            break_long_words=False, break_on_hyphens=False,
        )
        + "\n"
        + textwrap.fill(
            str(configurations_to_be_removed), width=72,
            initial_indent=f"  {'Excluded cfgs':<14} ", subsequent_indent=" " * 17,
            break_long_words=False, break_on_hyphens=False,
        ),
        silent,
    )
    with h5py.File(fn, "r") as h5:
        f = h5[meas_group]
        h5_cfgs = np.array([_parse_cfg_id(cfg.decode("utf-8")) for cfg in f["configlist"]])
        h5_cfgs_filtered = h5_cfgs[~np.isin(h5_cfgs, configurations_to_be_removed)] if configurations_to_be_removed is not None else h5_cfgs
        _log_h5_git(f, silent)
        message(
            "  HDF5 configurations\n"
            f"    {'Total':<16}  {len(h5_cfgs)}\n"
            f"    {'After filter':<16}  {len(h5_cfgs_filtered)}",
            silent,
            continuation=True,
        )

        _log_rwf_git(rwf_fn, silent)
        rwf_cfgs, rwf_values = _load_rwf_dispatch(rwf_fn)
        rwf_cfgs_filtered = rwf_cfgs[~np.isin(rwf_cfgs, configurations_to_be_removed)] if configurations_to_be_removed is not None else rwf_cfgs
        message(
            "  Reweighting configurations\n"
            f"    {'Total':<16}  {rwf_cfgs.shape[0]}\n"
            f"    {'After filter':<16}  {rwf_cfgs_filtered.shape[0]}",
            silent,
            continuation=True,
        )

        common_cfgs = _resolve_common_cfgs(h5_cfgs_filtered, rwf_cfgs_filtered, stream_tag)
        if reverse:
            common_cfgs = common_cfgs[::-1]
        message(f"  Matched configurations: {common_cfgs.shape[0]}", silent, continuation=True)

        rwf_idx = _argsort_to(rwf_cfgs, common_cfgs)
        h5_idx = _argsort_to(h5_cfgs, common_cfgs)

        # Raw rwf stored un-normalised; callers normalise after combining
        # streams (per-stream or globally) so FP-rounding matches their intent.
        weights = rwf_values[rwf_idx]
        cfg_labels = np.array([f"{stream_tag}-{int(c)}" for c in common_cfgs])

        db = DB()
        _populate_data(db, f, compiled_patterns, h5_idx, cfg_labels, weights, stream_tag, run_tag)
    return db


def _argsort_to(src_cfgs, target_cfgs):
    """Return ``idx`` such that ``src_cfgs[idx] == target_cfgs``."""
    pos = {int(c): i for i, c in enumerate(src_cfgs)}
    return np.array([pos[int(c)] for c in target_cfgs])


def _log_h5_git(f, silent):
    git_dict = f["description"].get("git")
    if git_dict is None:
        message("\n  HDF5 Git metadata: unavailable", silent, continuation=True)
        return
    message("\n  HDF5 Git metadata\n" + "\n".join(
        f"    {key:<16}  {val[()].decode()}" for key, val in git_dict.items()
    ), silent, continuation=True)


def _log_rwf_git(rwf_fn, silent):
    rwf_fn_git = rwf_fn + ".git"
    if not os.path.isfile(rwf_fn_git):
        message("\n  Reweighting Git metadata: unavailable", silent, continuation=True)
        return
    with open(rwf_fn_git) as rwf_f:
        rwf_info_dict = json.load(rwf_f)
    message("\n  Reweighting Git metadata\n" + "\n".join(
        f"    {key:<16}  {val}" for key, val in rwf_info_dict.items()
    ), silent, continuation=True)


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


def _populate_data(db, f, compiled_patterns, h5_idx, cfg_labels, weights, stream_tag, run_tag):
    data_keys = list(f["data"].keys())
    loaded_keys = set()
    for pattern in compiled_patterns:
        for key in data_keys:
            if key not in loaded_keys and pattern.search(key):
                f_vals = f["data"].get(key)[:]
                sample = f_vals[h5_idx]
                f_tag = f"{stream_tag}/{run_tag}/{key}"
                db.add_entry(
                    tag=f_tag,
                    samples=sample, weights=weights, configurations=cfg_labels,
                )
                loaded_keys.add(key)


def decode_v1_ndarray(blob):
    """Decode a v1 ``{"__ndarray__": <base64>, "dtype":..., "shape":...}`` blob."""
    buf = base64.b64decode(blob["__ndarray__"])
    return np.frombuffer(buf, dtype=np.dtype(blob["dtype"])).reshape(blob["shape"])


def load_v1_json(fn, silent=False):
    """Convert a retired v1 (custom-JSON) statpy database into a new ``DB``.

    Samples get uniform weights; ``mean`` and ``jackknife_samples`` are recomputed
    and ``misc`` is kept as ``metadata``. Raises ``ValueError`` if a leaf's samples cannot be stacked.
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
        db.add_entry(tag, samples=sample, weights=weights, configurations=cfgs, metadata=leaf.get("misc"))
    message(f" -- migrated {len(raw)} entries", silent)
    return db
