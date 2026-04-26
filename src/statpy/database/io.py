import os
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

def load_CLS(fn, rwf_fn, correlator_patterns, stream_tag, run_tag, cfgs_to_be_removed=None, meas_group="messpec", silent=False):
    """Load CLS hdf5 measurements + reweighting factors into a fresh ``DB``.

    For each pattern in ``correlator_patterns``, every dataset under
    ``f["data"]`` whose key contains the pattern becomes a leaf at
    ``"{stream_tag}/{run_tag}/{key}"``. Sample keys within each leaf
    are ``f"{stream_tag}-{cfg_id}"``. The rwf is added at
    ``"{stream_tag}/rwf"`` together with its normalized form via
    ``add_nrwf``.

    Configs whose ids are listed in ``cfgs_to_be_removed`` are filtered
    out of both the hdf5 and rwf streams before any leaves are added.
    The remaining hdf5 and rwf cfg sets must agree exactly; otherwise a
    ``ValueError`` is raised.

    Args:
        fn: Path to the hdf5 measurement file.
        rwf_fn: Path to the rwf file (``.rwf`` or ``.rwms.txt``); required.
        correlator_patterns: Substring patterns; any hdf5 dataset key
            under ``meas_group/data`` containing one is loaded.
        stream_tag: Ensemble/stream label, used as the top-level DB prefix.
        run_tag: Sub-prefix between stream_tag and the hdf5 key.
        cfgs_to_be_removed: Iterable of integer cfg ids to drop, or None.
        meas_group: Top-level hdf5 group, e.g. ``"messpec"`` (mesons) or
            ``"barspec"`` (baryons). Default ``"messpec"``.
        silent: Forwarded to the underlying ``DB``/``add_leaf`` calls;
            ``True`` suppresses all log output.

    Returns:
        A ``DB`` populated with the rwf leaf, nrwf leaf, and one leaf per
        matched correlator key.

    Raises:
        FileNotFoundError: If ``fn`` or ``rwf_fn`` does not point to an
            existing file.
        TypeError: If ``cfgs_to_be_removed`` is not a ``list``,
            ``np.ndarray``, or ``None``.
        ValueError: If the rwf file extension is unknown, a configlist
            entry's cfg id cannot be parsed, or the hdf5 and rwf cfg
            sets disagree after filtering.
    """
    if not os.path.isfile(fn):
        raise FileNotFoundError(f"hdf5 file {fn!r} not found!")
    if not os.path.isfile(rwf_fn):
        raise FileNotFoundError(f"rwf file {rwf_fn!r} not found!")
    if cfgs_to_be_removed is not None and not isinstance(cfgs_to_be_removed, (list, np.ndarray)):
        raise TypeError("'cfgs_to_be_removed' must be list | np.ndarray | None")

    message("---------------------------------")
    message(f"Load CLS data from {fn}")
    message(f"Load rw factors from: {rwf_fn}")
    message(f" -- correlator patterns: {correlator_patterns}")
    message(f" -- ensemble tag = {stream_tag}")
    message(f" -- run tag: {run_tag}")
    message(f" -- cfgs to be removed: {cfgs_to_be_removed}")
    with h5py.File(fn, "r") as h5:
        # hdf5
        f = h5[meas_group]
        f_cfgs = np.array([_parse_cfg_id(cfg.decode("utf-8")) for cfg in f["configlist"]])
        f_cfgs_filtered = f_cfgs[~np.isin(f_cfgs, cfgs_to_be_removed)] if cfgs_to_be_removed is not None else f_cfgs
        _log_h5_git(f)
        message(f"Number of cfgs in hdf5 file: {len(f_cfgs)} | Number of filtered configs in hdf5 file: {len(f_cfgs_filtered)}")
        # rwf
        _log_rwf_git(rwf_fn)
        rwf_cfgs, rwf = _load_rwf_dispatch(rwf_fn)
        rwf_cfgs_filtered = rwf_cfgs[~np.isin(rwf_cfgs, cfgs_to_be_removed)] if cfgs_to_be_removed is not None else rwf_cfgs
        message(f"Number of cfgs in rwf file: {rwf_cfgs.shape[0]} | Number of filtered configs in rwf file : {rwf_cfgs_filtered.shape[0]}")
        common_cfgs = _resolve_common_cfgs(f_cfgs_filtered, rwf_cfgs_filtered, stream_tag)
        message(f"Number of filtered configs in hdf5 file and rwf file: {common_cfgs.shape[0]}")
        rwf_mask = np.isin(rwf_cfgs, common_cfgs)
        rwf = dict(zip(
            (f"{stream_tag}-{int(c)}" for c in rwf_cfgs[rwf_mask]),
            rwf[rwf_mask],
        ))
        # database
        db = DB(silent=silent)
        db.add_leaf(tag=f"{stream_tag}/rwf", mean=None, jks=None, sample=rwf, misc=None)
        db.add_nrwf(rwf_tag=f"{stream_tag}/rwf")
        _populate_data(db, f, f_cfgs, common_cfgs, correlator_patterns, stream_tag, run_tag, silent)
    message("---------------------------------")
    return db


def _log_h5_git(f):
    """Log git info from ``f["description"]["git"]``, or warn if absent."""
    git_dict = f["description"].get("git")
    if git_dict is None:
        message("Git info not found for hdf5 file!")
        return
    message("hdf5 git info:")
    for key, val in git_dict.items():
        message(f"--- {key}: {val[()].decode()}")


def _log_rwf_git(rwf_fn):
    """Log git info from the ``<rwf_fn>.git`` JSON sidecar, or warn if absent."""
    rwf_fn_git = rwf_fn + ".git"
    if not os.path.isfile(rwf_fn_git):
        message("Git info not found for rwf file!")
        return
    message("rwf git info:")
    with open(rwf_fn_git) as rwf_f:
        rwf_info_dict = json.load(rwf_f)
    for key, val in rwf_info_dict.items():
        message(f"--- {key}: {val}")


def _load_rwf_dispatch(rwf_fn):
    """Dispatch to the loader matching ``rwf_fn``'s extension (``.rwf`` or ``.rwms.txt``)."""
    if rwf_fn.endswith(".rwf"):
        return _load_rwf(rwf_fn)
    if rwf_fn.endswith(".rwms.txt"):
        return _load_rwms(rwf_fn)
    raise ValueError(f"Unknown rwf file format: {rwf_fn}")


def _load_rwf(fn):
    """Load a two-column ``.rwf`` file; returns ``(cfg_ids, rwf_values)``."""
    rwf_cfgs = np.array(np.loadtxt(fn)[:,0], dtype=int)
    rwf = np.loadtxt(fn)[:,1]
    return rwf_cfgs, rwf


def _load_rwms(fn):
    """Load a multi-column ``.rwms.txt`` file; returns ``(cfg_ids, prod_of_rwf_columns)``."""
    rwf_cfgs = np.array(np.loadtxt(fn)[:,0], dtype=int)
    rwf = np.prod(np.loadtxt(fn)[:,1:], axis=1)
    return rwf_cfgs, rwf


def parse_bootstrap_file(fn):
    """Parse a CLS-style ``.boot.txt`` file.

    Returns ``(bootstraps, configlist)``: the integer bootstrap-index
    matrix plus the configuration label list parsed from line 4 of the
    header.
    """
    bootstraps = np.loadtxt(fn, dtype=int)
    with open(fn) as f:
        configlist = f.readlines()[3][:-1].replace("n", "-").split(" ")[1:]
    return bootstraps, configlist


def _resolve_common_cfgs(h5_cfgs_filtered, rwf_cfgs_filtered, stream_tag):
    """Verify hdf5 and rwf cfg sets match; return cfg ids in rwf order.

    Raises ValueError if the two sets disagree, listing the symmetric
    difference.
    """
    if set(h5_cfgs_filtered.tolist()) != set(rwf_cfgs_filtered.tolist()):
        non_common = np.setxor1d(h5_cfgs_filtered, rwf_cfgs_filtered)
        raise ValueError(
            f"hdf5/rwf cfg mismatch for {stream_tag}: configs in only one file: {non_common}"
        )
    return np.array(rwf_cfgs_filtered)


def _populate_data(db, f, h5_cfgs, common_cfgs, correlator_patterns, stream_tag, run_tag, silent):
    """Add a leaf for every ``f["data"]`` key matching a correlator pattern.

    Each leaf's sample dict is keyed by ``f"{stream_tag}-{cfg_id}"`` and
    restricted to ``common_cfgs``. The mask and key list are precomputed
    once and reused across all matched keys.
    """
    h5_mask = np.isin(h5_cfgs, common_cfgs)
    sample_keys = [f"{stream_tag}-{int(c)}" for c in h5_cfgs[h5_mask]]
    data_keys = list(f["data"].keys())
    for pattern in correlator_patterns:
        for key in data_keys:
            if pattern in key:
                f_vals = f["data"].get(key)[:]
                sample = dict(zip(sample_keys, f_vals[h5_mask]))
                f_tag = f"{stream_tag}/{run_tag}/{key}"
                db.add_leaf(tag=f_tag, mean=None, jks=None, sample=sample, misc=None, weights_tag=f"{stream_tag}/nrwf", silent=silent)



