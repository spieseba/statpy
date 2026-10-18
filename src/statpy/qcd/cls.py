import json
import os
import re
import textwrap

import h5py
import numpy as np

from statpy.database import Database
from statpy.log import message
from statpy.variable import Layout, Variable

# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def load_CLS(fn, rwf_fn, correlator_patterns, stream_tag, run_tag, configurations_to_be_removed=None, meas_group="messpec", reverse=False, silent=False):
    """Load CLS hdf5 measurements and reweighting factors into a fresh Database.

    ``stream_tag`` is ``<ensemble>r<stream number>`` (e.g. ``U101r001``); the stream is one replica,
    numbered by its stream number. Monte Carlo times are the cfg ids, negated for ``reverse=True`` so
    that they increase in array order; missing ids are gaps.
    Returns ``(db, cfg_ids)`` with the cfg ids in array order.
    """
    stream = re.match(r"^(.+)r(\d+)$", stream_tag)
    if stream is None:
        raise ValueError(f"stream_tag {stream_tag!r} is not <ensemble>r<stream number>")
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
        layout = Layout(stream[1], -common_cfgs if reverse else common_cfgs,
                        replicas=np.full(len(common_cfgs), int(stream[2])), weights=weights)

        db = Database()
        _populate_data(db, f, compiled_patterns, h5_idx, layout, stream_tag, run_tag)
    return db, common_cfgs


def parse_bootstrap_file(fn):
    """Parse a CLS ``.boot.txt`` → ``(bootstrap_indices, configlist)``, with CLS cfg names (``<stream>n<cfg id>``)."""
    bootstrap_indices = np.loadtxt(fn, dtype=int)
    with open(fn) as f:
        configlist = f.readlines()[3][:-1].split(" ")[1:]
    return bootstrap_indices, configlist


# ---------------------------------------------------------------------------
# Configuration ids
# ---------------------------------------------------------------------------

def _parse_cfg_id(name):
    """Parse the trailing ``n<digits>`` block of a CLS configlist entry."""
    m = re.search(r"n(\d+)$", name)
    if m is None:
        raise ValueError(f"Cannot parse cfg id from name: {name!r}")
    return int(m.group(1))


def _resolve_common_cfgs(h5_cfgs_filtered, rwf_cfgs_filtered, stream_tag):
    """Assert hdf5 and rwf cfg sets match; return cfg ids sorted ascending."""
    if set(h5_cfgs_filtered.tolist()) != set(rwf_cfgs_filtered.tolist()):
        non_common = np.setxor1d(h5_cfgs_filtered, rwf_cfgs_filtered)
        raise ValueError(
            f"hdf5/rwf cfg mismatch for {stream_tag}: configs in only one file: {non_common}"
        )
    return np.sort(rwf_cfgs_filtered)


def _argsort_to(src_cfgs, target_cfgs):
    """Return ``idx`` such that ``src_cfgs[idx] == target_cfgs``."""
    pos = {int(c): i for i, c in enumerate(src_cfgs)}
    return np.array([pos[int(c)] for c in target_cfgs])


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------

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


def _populate_data(db, f, compiled_patterns, h5_idx, layout, stream_tag, run_tag):
    data_keys = list(f["data"].keys())
    loaded_keys = set()
    for pattern in compiled_patterns:
        for key in data_keys:
            if key not in loaded_keys and pattern.search(key):
                f_vals = f["data"].get(key)[:]
                observations = f_vals[h5_idx]
                f_tag = f"{stream_tag}/{run_tag}/{key}"
                db[f_tag] = Variable.from_data(observations, layout)
                loaded_keys.add(key)


# ---------------------------------------------------------------------------
# Git metadata
# ---------------------------------------------------------------------------

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
