"""Source averaging and folding of correlators (preprocessing, before fitting).

All functions take a database handle as first argument and write their result
back into it. The OBC/fold routines are valid for mesonic channels only.
"""
import re
import textwrap

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
from statpy.variable import Layout, Variable


def pbc_correlator_average(db, corr_tag, store_as):
    """PBC source-axis average; write to ``store_as``. Any channel (no folding)."""
    assert isinstance(corr_tag, str)
    variable = db[corr_tag]
    new_sample = variable.observations.mean(axis=1)
    db[store_as] = Variable.from_data(new_sample, variable.layout, metadata=variable.metadata)


def obc_meson_correlator_average(db, corr_tags, bulk_range, store_as, tmax_from_tsrc=None, time_parity=1):
    """OBC source average over tsrcs in bulk_range: per-src fw/bw mask, then fold-and-mean.

    Mesons only: time_parity is +1 (even) or -1 (odd); booleans are rejected.
    """
    time_parity = _validate_time_parity(time_parity)
    # Get src positions in bulk
    tsrcs = [int(re.search(r'tsrc(\d+)', t)[1]) for t in corr_tags]
    assert len(corr_tags) == len(tsrcs)
    corr_tags_in_bulk = []
    tsrcs_in_bulk = []
    for corr_tag, tsrc in zip(corr_tags, tsrcs):
        if (tsrc >= bulk_range[0]) and (tsrc <= bulk_range[-1]):
            corr_tags_in_bulk.append(corr_tag)
            tsrcs_in_bulk.append(tsrc)
    message(
        "OBC source averaging\n"
        f"  {'Bulk range':<14} [{bulk_range[0]}, {bulk_range[-1]}]\n"
        f"  {'tmax_from_tsrc':<14} {tmax_from_tsrc}\n"
        + textwrap.fill(
            ", ".join(str(t) for t in tsrcs_in_bulk), width=72,
            initial_indent=f"  {'tsrcs in bulk':<14} ", subsequent_indent=" " * 17,
        )
        + "\n  Input tags\n"
        + "\n".join(f"    {tag}" for tag in corr_tags)
    )
    tmax_fw, tmax_bw = _get_tmax_fw_bw(tsrcs_in_bulk, bulk_range)
    if tmax_from_tsrc is not None:
        tmax_fw = np.minimum(tmax_fw, tmax_from_tsrc+1)
        tmax_bw = np.minimum(tmax_bw, tmax_from_tsrc+1)
    masked_samples = []
    for src_idx, corr_tag in enumerate(corr_tags_in_bulk):
        variable = db[corr_tag]
        masked_sample = _get_masked_meson_sample(variable.observations, tmax_fw[src_idx], tmax_bw[src_idx], time_parity)
        masked_samples.append(masked_sample)
    ref_lf = db[corr_tags_in_bulk[0]]
    # mask pattern is config-independent, so mean+compress over the whole stack at once
    combined = np.ma.concatenate(masked_samples, axis=1).mean(axis=1)  # (N_cfg, T)
    combined_sample = np.ma.compress_cols(combined)
    db[store_as] = Variable.from_data(
        combined_sample,
        ref_lf.layout,
        metadata={"tsrcs": tsrcs_in_bulk, "bulk_range": bulk_range, "time_parity": time_parity},
    )


def meson_fold_correlator_entry(db, corr_tag, store_as, time_parity=1):
    """Fold a meson correlator variable around T/2 into ``store_as``."""
    time_parity = _validate_time_parity(time_parity)
    message(f"Fold correlator {corr_tag}.")
    variable = db[corr_tag]
    folded = np.array([meson_fold_correlator(corr, time_parity) for corr in variable.observations])
    db[store_as] = Variable.from_data(folded, variable.layout, metadata=variable.metadata)


def _boundary_eff_mass(corr, tsrc):
    """Boundary effective mass of a (masked) correlator, sign-corrected per tsrc.

    Masked / non-finite time slices collapse to 0 -- the marker the source
    average reads as "no contribution here".
    """
    return np.nan_to_num(
        _flip_sign_boundary(effective_mass(corr, estimator="log_symmetric"), tsrc), nan=0.0, posinf=0.0, neginf=0.0
    )


def obc_meson_boundary_average(db, corr_tags, tmin_excited, bin_size, tmax_from_tsrc=None, time_parity=1):
    """Source-averaged, folded boundary effective mass (excited region masked). Mesons only.

    Returns the source-averaged tag (``tsrc<None>/am_t``); ``<tag>/folded`` and
    ``metadata["nsrc_hist"]`` (source positions contributing per time slice) are also written.
    time_parity is +1 (even) or -1 (odd); booleans are rejected.
    """
    time_parity = _validate_time_parity(time_parity)
    message(
        "OBC boundary averaging\n"
        f"  {'tmin_excited':<14} {tmin_excited} (excited-state cutoff)\n"
        f"  {'tmax_from_tsrc':<14} {tmax_from_tsrc}\n"
        "  Input tags\n"
        + "\n".join(f"    {tag}" for tag in corr_tags)
    )
    tsrcs = [int(re.search(r'tsrc(\d+)', t)[1]) for t in corr_tags]
    assert len(corr_tags) == len(tsrcs)

    am_t_means, am_t_jks, cfgs = [], [], None
    for corr_tag, tsrc in zip(corr_tags, tsrcs):
        variable = db[corr_tag]
        masked_excited_state_sample = np.array([
            _get_masked_corrs_boundary(corrs, tsrc, tmin_excited, tmax_from_tsrc).mean(axis=0)
            for corrs in variable.observations
        ])
        b_sample = statistics.bin(masked_excited_state_sample, bin_size, weights=variable.layout.weights)
        b_weights = statistics.bin(variable.layout.weights, bin_size, weights=np.ones(len(variable.layout.weights)))
        jks = jackknife.mean_replicates(b_sample, weights=b_weights)
        am_t_means.append(_boundary_eff_mass(np.average(b_sample, axis=0, weights=b_weights), tsrc))
        am_t_jks.append(np.array([_boundary_eff_mass(jk, tsrc) for jk in jks]))
        if cfgs is None:
            # binned labels as in Variable.bin
            cfgs = variable.layout.configurations if bin_size == 1 else np.array(
                [f"{corr_tag.split('/')[0]}/bin_size{bin_size}-{i}" for i in range(len(b_sample))]
            )

    def source_average(stack):
        # masked mean over tsrcs (0 marks a masked time slice)
        return np.ma.filled(np.ma.masked_equal(stack, 0).mean(axis=0), 0)
    avg_mean = source_average(np.array(am_t_means))
    avg_jks = source_average(np.array(am_t_jks))
    nsrc_hist = np.sum(np.array(am_t_means) != 0, axis=0)

    masked_tag = f"{corr_tags[0]}/maskedES"
    avg_mt_tag = re.sub(r'(tsrc)\d+', r'\1None', f"{binned_tag(masked_tag, bin_size)}/am_t")
    db[avg_mt_tag] = Variable.from_results(
        central_value=avg_mean,
        jackknife_replicates=avg_jks,
        metadata={"nsrc_hist": nsrc_hist},
        layout=Layout(cfgs),
    )
    db[f"{avg_mt_tag}/folded"] = Variable.derive(lambda mt: _fold_meson_boundary(mt, time_parity), db[avg_mt_tag])
    return avg_mt_tag
