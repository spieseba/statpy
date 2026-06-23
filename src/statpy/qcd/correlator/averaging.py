"""Source averaging and folding of correlators (preprocessing, before fitting).

All functions take a database handle as first argument and write their result
back into it. The OBC/fold routines are valid for mesonic channels only.
"""
import re

import numpy as np

from statpy.log import message
from statpy.qcd.correlator.primitives import (
    meson_fold_correlator, meff_exp_symmetric, binned_tag,
)
from statpy.qcd.correlator._masking import (
    _get_masked_meson_sample, _get_tmax_fw_bw, _get_masked_Cts_boundary,
    _fold_meson_boundary, _flip_sign_boundary,
)


def pbc_correlator_average(db, Ct_tag, store_as):
    """PBC source-axis average; write to ``store_as``. Any channel (no folding)."""
    assert isinstance(Ct_tag, str)
    entry = db.database[Ct_tag]
    new_sample = entry.sample.mean(axis=1)
    db.add_entry(store_as, sample=new_sample, weights=entry.weights, cfgs=entry.cfgs, misc=entry.misc)


def obc_meson_correlator_average(db, Ct_tags, tbulk, store_as, tmax_from_tsrc=None, antisymmetric=False):
    """OBC source average over tsrcs in tbulk: per-src fw/bw mask, then fold-and-mean.

    Mesons only: backward half folded as the same state (antisymmetric -> sinh, else cosh).
    """
    message(f"Perform obc tsrc average over all srcs in tbulk = [[{tbulk[0]},{tbulk[-1]}]] with correlator tags: {Ct_tags}")
    message(f"tmax_from_tsrc: {tmax_from_tsrc}")
    # Get src positions in bulk
    tsrcs = [int(re.search(r'tsrc(\d+)', t)[1]) for t in Ct_tags]
    assert len(Ct_tags) == len(tsrcs)
    Ct_tags_in_bulk = []
    tsrcs_in_bulk = []
    for Ct_tag, tsrc in zip(Ct_tags, tsrcs):
        if (tsrc >= tbulk[0]) and (tsrc <= tbulk[-1]):
            Ct_tags_in_bulk.append(Ct_tag)
            tsrcs_in_bulk.append(tsrc)
    message(f"tsrcs in bulk: {tsrcs_in_bulk}")
    tmax_fw, tmax_bw = _get_tmax_fw_bw(tsrcs_in_bulk, tbulk)
    if tmax_from_tsrc is not None:
        tmax_fw = np.minimum(tmax_fw, tmax_from_tsrc+1)
        tmax_bw = np.minimum(tmax_bw, tmax_from_tsrc+1)
    masked_samples = []
    for src_idx, Ct_tag in enumerate(Ct_tags_in_bulk):
        entry = db.database[Ct_tag]
        masked_sample = _get_masked_meson_sample(entry.sample, tmax_fw[src_idx], tmax_bw[src_idx], antisymmetric)
        masked_samples.append(masked_sample)
    ref_lf = db.database[Ct_tags_in_bulk[0]]
    # mask pattern is config-independent, so mean+compress over the whole stack at once
    combined = np.ma.concatenate(masked_samples, axis=1).mean(axis=1)  # (N_cfg, T)
    combined_sample = np.ma.compress_cols(combined)
    db.add_entry(
        store_as, sample=combined_sample, weights=ref_lf.weights, cfgs=ref_lf.cfgs,
        misc={"tsrcs": tsrcs_in_bulk, "tbulk": tbulk, "antisymmetric": antisymmetric},
    )


def meson_fold_correlator_entry(db, Ct_tag, store_as, antisymmetric=False):
    """Fold a DB correlator entry around T/2 into ``store_as``. Mesons only.

    Unrelated to the pipeline-level ``fold_correlators`` config toggle.
    """
    message(f"Fold correlator {Ct_tag}.")
    entry = db.database[Ct_tag]
    folded = np.array([meson_fold_correlator(Ct, antisymmetric) for Ct in entry.sample])
    db.add_entry(store_as, sample=folded, weights=entry.weights, cfgs=entry.cfgs, misc=entry.misc)


def obc_meson_boundary_average(db, Ct_tags, tmin_excited, binsize, tmax_from_tsrc=None, antisymmetric=False, cleanup=False):
    """Per-tsrc boundary effective mass (excited region masked), source-averaged then folded. Mesons only.

    Returns the source-averaged tag (``tsrc<None>/am_t``); ``<tag>/folded`` is also written.
    """
    message(f"Perform boundary average over all tsrcs with correlator tags: {Ct_tags}")
    message(f"Excited state contributions expected to be removed at t = {tmin_excited}")
    message(f"tmax_from_tsrc = {tmax_from_tsrc}")
    tsrcs = [int(re.search(r'tsrc(\d+)', t)[1]) for t in Ct_tags]
    assert len(Ct_tags) == len(tsrcs)
    mt_tags = []
    for Ct_tag, tsrc in zip(Ct_tags, tsrcs):
        entry = db.database[Ct_tag]
        maskedES_sample = np.array([
            _get_masked_Cts_boundary(Ct, tsrc, tmin_excited, tmax_from_tsrc).mean(axis=0)
            for Ct in entry.sample
        ])
        masked_tag = f"{Ct_tag}/maskedES"
        db.add_entry(masked_tag, sample=maskedES_sample, weights=entry.weights, cfgs=entry.cfgs)
        binned_Ct_tag = binned_tag(masked_tag, binsize)
        if binned_Ct_tag != masked_tag and binned_Ct_tag not in db.database:
            db.add_entry(binned_Ct_tag, **db.bin_entry(masked_tag, binsize))
        mt_tag = f"{binned_Ct_tag}/am_t"
        mt_tags.append(mt_tag)
        db.transform(binned_Ct_tag, f=lambda Ct: np.nan_to_num(_flip_sign_boundary(meff_exp_symmetric(Ct), tsrc), nan=0.0, posinf=0.0, neginf=0.0), store_as=mt_tag)
        if cleanup:
            db.remove_entry(masked_tag)
            db.remove_entry(binned_Ct_tag)
    avg_mt_tag = re.sub(r'(tsrc)\d+', r'\1None', mt_tags[0])
    db.combine(*mt_tags, f=lambda *eff_mass: np.ma.filled(np.ma.masked_equal(eff_mass, 0).mean(axis=0), 0), store_as=avg_mt_tag)
    db.transform(avg_mt_tag, f=lambda mt: _fold_meson_boundary(mt, antisymmetric), store_as=f"{avg_mt_tag}/folded")
    if cleanup:
        for mt_tag in mt_tags:
            db.remove_entry(mt_tag)
    return avg_mt_tag
