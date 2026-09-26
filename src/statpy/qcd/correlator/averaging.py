"""Source averaging and folding of correlators (preprocessing, before fitting).

All functions take a database handle as first argument and write their result
back into it. The OBC/fold routines are valid for mesonic channels only.
"""
import re

import numpy as np

from statpy.log import message
from statpy.qcd.correlator._masking import (
    _flip_sign_boundary,
    _fold_meson_boundary,
    _get_masked_corrs_boundary,
    _get_masked_meson_sample,
    _get_tmax_fw_bw,
)
from statpy.qcd.correlator.primitives import (
    _validate_time_parity,
    binned_tag,
    effective_mass,
    meson_fold_correlator,
)
from statpy.statistics import core as statistics
from statpy.statistics import jackknife


def pbc_correlator_average(db, corr_tag, store_as):
    """PBC source-axis average; write to ``store_as``. Any channel (no folding)."""
    assert isinstance(corr_tag, str)
    entry = db.database[corr_tag]
    new_sample = entry.sample.mean(axis=1)
    db.add_entry(store_as, sample=new_sample, weights=entry.weights, cfgs=entry.cfgs, misc=entry.misc)


def obc_meson_correlator_average(db, corr_tags, bulk_range, store_as, tmax_from_tsrc=None, time_parity=1):
    """OBC source average over tsrcs in bulk_range: per-src fw/bw mask, then fold-and-mean.

    Mesons only: time_parity is +1 (even) or -1 (odd); booleans are rejected.
    """
    time_parity = _validate_time_parity(time_parity)
    message(f"Perform obc tsrc average over all srcs in bulk_range = [[{bulk_range[0]},{bulk_range[-1]}]] with correlator tags: {corr_tags}")
    message(f"tmax_from_tsrc: {tmax_from_tsrc}")
    # Get src positions in bulk
    tsrcs = [int(re.search(r'tsrc(\d+)', t)[1]) for t in corr_tags]
    assert len(corr_tags) == len(tsrcs)
    corr_tags_in_bulk = []
    tsrcs_in_bulk = []
    for corr_tag, tsrc in zip(corr_tags, tsrcs):
        if (tsrc >= bulk_range[0]) and (tsrc <= bulk_range[-1]):
            corr_tags_in_bulk.append(corr_tag)
            tsrcs_in_bulk.append(tsrc)
    message(f"tsrcs in bulk: {tsrcs_in_bulk}")
    tmax_fw, tmax_bw = _get_tmax_fw_bw(tsrcs_in_bulk, bulk_range)
    if tmax_from_tsrc is not None:
        tmax_fw = np.minimum(tmax_fw, tmax_from_tsrc+1)
        tmax_bw = np.minimum(tmax_bw, tmax_from_tsrc+1)
    masked_samples = []
    for src_idx, corr_tag in enumerate(corr_tags_in_bulk):
        entry = db.database[corr_tag]
        masked_sample = _get_masked_meson_sample(entry.sample, tmax_fw[src_idx], tmax_bw[src_idx], time_parity)
        masked_samples.append(masked_sample)
    ref_lf = db.database[corr_tags_in_bulk[0]]
    # mask pattern is config-independent, so mean+compress over the whole stack at once
    combined = np.ma.concatenate(masked_samples, axis=1).mean(axis=1)  # (N_cfg, T)
    combined_sample = np.ma.compress_cols(combined)
    db.add_entry(
        store_as, sample=combined_sample, weights=ref_lf.weights, cfgs=ref_lf.cfgs,
        misc={"tsrcs": tsrcs_in_bulk, "bulk_range": bulk_range, "time_parity": time_parity},
    )


def meson_fold_correlator_entry(db, corr_tag, store_as, time_parity=1):
    """Fold a DB correlator entry around T/2 into ``store_as``. Mesons only.

    Unrelated to the pipeline-level ``fold_correlators`` config toggle.
    time_parity is +1 (even) or -1 (odd); booleans are rejected.
    """
    time_parity = _validate_time_parity(time_parity)
    message(f"Fold correlator {corr_tag}.")
    entry = db.database[corr_tag]
    folded = np.array([meson_fold_correlator(corr, time_parity) for corr in entry.sample])
    db.add_entry(store_as, sample=folded, weights=entry.weights, cfgs=entry.cfgs, misc=entry.misc)


def _boundary_eff_mass(corr, tsrc):
    """Boundary effective mass of a (masked) correlator, sign-corrected per tsrc.

    Masked / non-finite time slices collapse to 0 -- the marker the source
    average reads as "no contribution here".
    """
    return np.nan_to_num(
        _flip_sign_boundary(effective_mass(corr, estimator="log_symmetric"), tsrc), nan=0.0, posinf=0.0, neginf=0.0
    )


def obc_meson_boundary_average(db, corr_tags, tmin_excited, binsize, tmax_from_tsrc=None, time_parity=1):
    """Source-averaged, folded boundary effective mass (excited region masked). Mesons only.

    Returns the source-averaged tag (``tsrc<None>/am_t``); ``<tag>/folded`` and
    ``misc["nsrc_hist"]`` (source positions contributing per time slice) are also written.
    time_parity is +1 (even) or -1 (odd); booleans are rejected.
    """
    time_parity = _validate_time_parity(time_parity)
    message(f"Perform boundary average over all tsrcs with correlator tags: {corr_tags}")
    message(f"Excited state contributions expected to be removed at t = {tmin_excited}")
    message(f"tmax_from_tsrc = {tmax_from_tsrc}")
    tsrcs = [int(re.search(r'tsrc(\d+)', t)[1]) for t in corr_tags]
    assert len(corr_tags) == len(tsrcs)

    am_t_means, am_t_jks, cfgs = [], [], None
    for corr_tag, tsrc in zip(corr_tags, tsrcs):
        entry = db.database[corr_tag]
        masked_excited_state_sample = np.array([
            _get_masked_corrs_boundary(corrs, tsrc, tmin_excited, tmax_from_tsrc).mean(axis=0)
            for corrs in entry.sample
        ])
        b_sample = statistics.bin(masked_excited_state_sample, binsize, weights=entry.weights)
        b_weights = statistics.bin(entry.weights, binsize)
        jks = jackknife.sample(b_sample, weights=b_weights)
        am_t_means.append(_boundary_eff_mass(np.average(b_sample, axis=0, weights=b_weights), tsrc))
        am_t_jks.append(np.array([_boundary_eff_mass(jk, tsrc) for jk in jks]))
        if cfgs is None:
            # every tsrc shares the same (binned) cfg set, so the cross-tsrc
            # average is a plain stack -- no cfg alignment needed. Binned labels
            # mirror DB.bin_entry; the tag still routes through binned_tag().
            cfgs = entry.cfgs if binsize == 1 else np.array(
                [f"{corr_tag.split('/')[0]}-bin{i}" for i in range(len(b_sample))]
            )

    def source_average(stack):
        # masked mean over tsrcs (0 marks a masked time slice)
        return np.ma.filled(np.ma.masked_equal(stack, 0).mean(axis=0), 0)
    avg_mean = source_average(np.array(am_t_means))
    avg_jks = source_average(np.array(am_t_jks))
    nsrc_hist = np.sum(np.array(am_t_means) != 0, axis=0)

    masked_tag = f"{corr_tags[0]}/maskedES"
    avg_mt_tag = re.sub(r'(tsrc)\d+', r'\1None', f"{binned_tag(masked_tag, binsize)}/am_t")
    db.add_entry(avg_mt_tag, central_value=avg_mean, jks=avg_jks, cfgs=cfgs, misc={"nsrc_hist": nsrc_hist})
    db.transform(avg_mt_tag, f=lambda mt: _fold_meson_boundary(mt, time_parity), store_as=f"{avg_mt_tag}/folded")
    return avg_mt_tag
