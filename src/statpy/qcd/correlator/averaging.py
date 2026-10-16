"""Source averaging of meson correlator Variables."""
from dataclasses import dataclass

import numpy as np

from statpy.qcd.correlator._masking import (
    _flip_sign_boundary,
    _fold_meson_boundary,
    _get_masked_corrs_boundary,
    _get_masked_meson_sample,
    _get_tmax_fw_bw,
)
from statpy.qcd.correlator.primitives import (
    _validate_time_parity,
    effective_mass,
)
from statpy.statistics import core as statistics
from statpy.statistics import jackknife
from statpy.variable import Layout, Variable


def obc_meson_correlator_average(correlators: dict[int, Variable], bulk_range, *, tmax_from_tsrc=None, time_parity=1) -> Variable:
    """Return the OBC meson source average over sources in bulk_range."""
    time_parity = _validate_time_parity(time_parity)
    tsrcs_in_bulk = [tsrc for tsrc in correlators if (tsrc >= bulk_range[0]) and (tsrc <= bulk_range[-1])]
    tmax_fw, tmax_bw = _get_tmax_fw_bw(tsrcs_in_bulk, bulk_range)
    if tmax_from_tsrc is not None:
        tmax_fw = np.minimum(tmax_fw, tmax_from_tsrc+1)
        tmax_bw = np.minimum(tmax_bw, tmax_from_tsrc+1)
    masked_samples = []
    for src_idx, tsrc in enumerate(tsrcs_in_bulk):
        variable = correlators[tsrc]
        masked_sample = _get_masked_meson_sample(variable.observations, tmax_fw[src_idx], tmax_bw[src_idx], time_parity)
        masked_samples.append(masked_sample)
    ref_lf = correlators[tsrcs_in_bulk[0]]
    # mask pattern is config-independent, so mean+compress over the whole stack at once
    combined = np.ma.concatenate(masked_samples, axis=1).mean(axis=1)  # (N_cfg, T)
    combined_sample = np.ma.compress_cols(combined)
    return Variable.from_data(
        combined_sample,
        ref_lf.layout,
        metadata={"tsrcs": tsrcs_in_bulk, "bulk_range": bulk_range, "time_parity": time_parity},
    )


def _boundary_eff_mass(corr, tsrc):
    """Boundary effective mass of a (masked) correlator, sign-corrected per tsrc.

    Masked / non-finite time slices collapse to 0 -- the marker the source
    average reads as "no contribution here".
    """
    return np.nan_to_num(
        _flip_sign_boundary(effective_mass(corr, estimator="log_symmetric"), tsrc), nan=0.0, posinf=0.0, neginf=0.0
    )


@dataclass(frozen=True)
class BoundaryAverage:
    """Boundary effective mass, folded effective mass, and source counts."""

    effective_mass: Variable
    folded: Variable
    nsrc_hist: np.ndarray


def obc_meson_boundary_average(correlators: dict[int, Variable], tmin_excited, bin_size, prefix, *, tmax_from_tsrc=None, time_parity=1) -> BoundaryAverage:
    """Return source-averaged boundary effective masses and source counts."""
    time_parity = _validate_time_parity(time_parity)

    am_t_means, am_t_jks, cfgs = [], [], None
    for tsrc, variable in correlators.items():
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
                [f"{prefix}/bin_size{bin_size}-{i}" for i in range(len(b_sample))]
            )

    def source_average(stack):
        # masked mean over tsrcs (0 marks a masked time slice)
        return np.ma.filled(np.ma.masked_equal(stack, 0).mean(axis=0), 0)
    avg_mean = source_average(np.array(am_t_means))
    avg_jks = source_average(np.array(am_t_jks))
    nsrc_hist = np.sum(np.array(am_t_means) != 0, axis=0)

    effective_mass = Variable.from_results(
        central_value=avg_mean,
        jackknife_replicates=avg_jks,
        layout=Layout(cfgs),
    )
    folded = Variable.derive(lambda mt: _fold_meson_boundary(mt, time_parity), effective_mass)
    return BoundaryAverage(effective_mass, folded, nsrc_hist)
