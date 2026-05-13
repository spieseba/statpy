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


def load_CLS(fn, rwf_fn, correlator_patterns, stream_tag, run_tag, cfgs_to_be_removed=None, meas_group="messpec", reverse=False, silent=False):
    """Load CLS hdf5 measurements + reweighting factors into a fresh ``DB``.

    Each hdf5 dataset under ``meas_group/data`` whose key contains a
    pattern in ``correlator_patterns`` becomes an atomic data leaf at
    ``{stream_tag}/{run_tag}/{key}`` with cfg labels
    ``{stream_tag}-{cfg_id}`` sorted by ascending cfg id (descending if
    ``reverse=True``). The raw rwf is embedded as ``weights`` — no
    separate ``/rwf`` / ``/nrwf`` leaves.

    ``cfgs_to_be_removed`` filters both streams before insertion; their
    remaining cfg sets must agree exactly.
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
        f = h5[meas_group]
        h5_cfgs = np.array([_parse_cfg_id(cfg.decode("utf-8")) for cfg in f["configlist"]])
        h5_cfgs_filtered = h5_cfgs[~np.isin(h5_cfgs, cfgs_to_be_removed)] if cfgs_to_be_removed is not None else h5_cfgs
        _log_h5_git(f)
        message(f"Number of cfgs in hdf5 file: {len(h5_cfgs)} | Number of filtered configs in hdf5 file: {len(h5_cfgs_filtered)}")

        _log_rwf_git(rwf_fn)
        rwf_cfgs, rwf_values = _load_rwf_dispatch(rwf_fn)
        rwf_cfgs_filtered = rwf_cfgs[~np.isin(rwf_cfgs, cfgs_to_be_removed)] if cfgs_to_be_removed is not None else rwf_cfgs
        message(f"Number of cfgs in rwf file: {rwf_cfgs.shape[0]} | Number of filtered configs in rwf file : {rwf_cfgs_filtered.shape[0]}")

        common_cfgs = _resolve_common_cfgs(h5_cfgs_filtered, rwf_cfgs_filtered, stream_tag)
        if reverse:
            common_cfgs = common_cfgs[::-1]
        message(f"Number of filtered configs in hdf5 file and rwf file: {common_cfgs.shape[0]}")

        rwf_idx = _argsort_to(rwf_cfgs, common_cfgs)
        h5_idx = _argsort_to(h5_cfgs, common_cfgs)

        # Raw rwf stored un-normalised; jackknife/bootstrap are scale-invariant in
        # weights, which makes per-stream and post-concat results agree bit-for-bit
        # with the legacy global-normalisation pipeline.
        weights = rwf_values[rwf_idx]
        cfg_labels = np.array([f"{stream_tag}-{int(c)}" for c in common_cfgs])

        db = DB(silent=silent)
        _populate_data(db, f, correlator_patterns, h5_idx, cfg_labels, weights, stream_tag, run_tag, silent)
    message("---------------------------------")
    return db


def _argsort_to(src_cfgs, target_cfgs):
    """Return ``idx`` such that ``src_cfgs[idx] == target_cfgs``."""
    pos = {int(c): i for i, c in enumerate(src_cfgs)}
    return np.array([pos[int(c)] for c in target_cfgs])


def _log_h5_git(f):
    git_dict = f["description"].get("git")
    if git_dict is None:
        message("Git info not found for hdf5 file!")
        return
    message("hdf5 git info:")
    for key, val in git_dict.items():
        message(f"--- {key}: {val[()].decode()}")


def _log_rwf_git(rwf_fn):
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
    if rwf_fn.endswith(".rwf"):
        return _load_rwf(rwf_fn)
    if rwf_fn.endswith(".rwms.txt"):
        return _load_rwms(rwf_fn)
    raise ValueError(f"Unknown rwf file format: {rwf_fn}")


def _load_rwf(fn):
    """Two-column ``.rwf`` → ``(cfg_ids, rwf_values)``."""
    rwf_cfgs = np.array(np.loadtxt(fn)[:,0], dtype=int)
    rwf = np.loadtxt(fn)[:,1]
    return rwf_cfgs, rwf


def _load_rwms(fn):
    """Multi-column ``.rwms.txt`` → ``(cfg_ids, prod_of_rwf_columns)``."""
    rwf_cfgs = np.array(np.loadtxt(fn)[:,0], dtype=int)
    rwf = np.prod(np.loadtxt(fn)[:,1:], axis=1)
    return rwf_cfgs, rwf


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


def _populate_data(db, f, correlator_patterns, h5_idx, cfg_labels, weights, stream_tag, run_tag, silent):
    data_keys = list(f["data"].keys())
    for pattern in correlator_patterns:
        for key in data_keys:
            if pattern in key:
                f_vals = f["data"].get(key)[:]
                sample = f_vals[h5_idx]
                f_tag = f"{stream_tag}/{run_tag}/{key}"
                db.add_leaf(
                    tag=f_tag,
                    sample=sample, weights=weights, cfgs=cfg_labels,
                    silent=silent,
                )
