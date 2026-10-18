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
from statpy.variable import Variable


def obc_meson_correlator_average(correlators: dict[int, Variable], bulk_range, *, tmax_from_tsrc=None, time_parity=1) -> Variable:
    """Return the OBC meson source average over sources in bulk_range."""
    time_parity = _validate_time_parity(time_parity)
    tsrcs_in_bulk = [tsrc for tsrc in correlators if (tsrc >= bulk_range[0]) and (tsrc <= bulk_range[-1])]
    tmax_fw, tmax_bw = _get_tmax_fw_bw(tsrcs_in_bulk, bulk_range)
    if tmax_from_tsrc is not None:
        tmax_fw = np.minimum(tmax_fw, tmax_from_tsrc+1)
        tmax_bw = np.minimum(tmax_bw, tmax_from_tsrc+1)

    def source_average(*observations):
        masked = [_get_masked_meson_sample(sample, fw, bw, time_parity)
                  for sample, fw, bw in zip(observations, tmax_fw, tmax_bw)]
        # mask pattern is config-independent, so mean+compress over the whole stack at once
        return np.ma.compress_cols(np.ma.concatenate(masked, axis=1).mean(axis=1))  # (N_cfg, T)

    return Variable.map_observations(source_average, *[correlators[tsrc] for tsrc in tsrcs_in_bulk])


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


def obc_meson_boundary_average(correlators: dict[int, Variable], tmin_excited, *, tmax_from_tsrc=None, time_parity=1) -> BoundaryAverage:
    """Return source-averaged boundary effective masses and source counts."""
    time_parity = _validate_time_parity(time_parity)

    effective_masses = []
    for tsrc, variable in correlators.items():
        masked = Variable.map_observations(lambda observations, tsrc=tsrc: np.array([
            _get_masked_corrs_boundary(corrs, tsrc, tmin_excited, tmax_from_tsrc).mean(axis=0)
            for corrs in observations
        ]), variable)
        effective_masses.append(Variable.derive(lambda corr, tsrc=tsrc: _boundary_eff_mass(corr, tsrc), masked))

    def source_average(*masses):
        # masked mean over tsrcs (0 marks a masked time slice)
        return np.ma.filled(np.ma.masked_equal(np.array(masses), 0).mean(axis=0), 0)
    effective_mass = Variable.derive(source_average, *effective_masses)
    nsrc_hist = np.sum(np.array([mass.central_value for mass in effective_masses]) != 0, axis=0)
    folded = Variable.derive(lambda mt: _fold_meson_boundary(mt, time_parity), effective_mass)
    return BoundaryAverage(effective_mass, folded, nsrc_hist)
