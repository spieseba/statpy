"""Source averaging and folding of meson correlators."""
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from statpy.qcd.correlator.effective_observables import effective_mass
from statpy.variable import Variable

# ---------------------------------------------------------------------------
# Folding
# ---------------------------------------------------------------------------

def meson_fold_correlator(correlator: np.ndarray, time_parity: int = 1) -> np.ndarray:
    """Fold a meson correlator around T/2; time_parity is +1 (even) or -1 (odd)."""
    time_parity = _validate_time_parity(time_parity)
    half = len(correlator) // 2
    first_half = correlator[:half]
    second_half = np.roll(np.flip(correlator[half:]), 1) * time_parity
    second_half[0] = first_half[0]
    return np.mean([first_half, second_half], axis=0)


def _validate_time_parity(time_parity: int) -> int:
    """Return the sign for even (+1) or odd (-1) time parity; reject booleans."""
    if (
        isinstance(time_parity, (bool, np.bool_))
        or not isinstance(time_parity, Real)
        or time_parity not in (-1, 1)
    ):
        raise ValueError("time_parity must be +1 (even) or -1 (odd), not a boolean")
    return int(time_parity)


# ---------------------------------------------------------------------------
# Correlator average
# ---------------------------------------------------------------------------

def obc_meson_correlator_average(correlators: dict[int, Variable], bulk_range: Sequence[int] | NDArray[np.integer], *,
                                 max_source_distance: int | None = None, time_parity: int = 1) -> Variable:
    """Return the OBC meson source average over sources in bulk_range."""
    time_parity = _validate_time_parity(time_parity)
    tsrcs_in_bulk = [tsrc for tsrc in correlators if (tsrc >= bulk_range[0]) and (tsrc <= bulk_range[-1])]
    tmax_forward, tmax_backward = _get_tmax_forward_backward(tsrcs_in_bulk, bulk_range)
    # max_source_distance is inclusive: keep |t - tsrc| <= max_source_distance
    if max_source_distance is not None:
        tmax_forward = np.minimum(tmax_forward, max_source_distance+1)
        tmax_backward = np.minimum(tmax_backward, max_source_distance+1)

    def source_average(*observations_per_source: np.ndarray) -> np.ndarray:
        masked = [_get_masked_meson_observations(observations, forward, backward, time_parity)
                  for observations, forward, backward in zip(observations_per_source, tmax_forward, tmax_backward)]
        # mask pattern is config-independent, so mean+compress over the whole stack at once
        return np.ma.compress_cols(np.ma.concatenate(masked, axis=1).mean(axis=1))  # (N_cfg, T)
    # map_observations does not trace, numpy is fine
    return Variable.map_observations(source_average, *[correlators[tsrc] for tsrc in tsrcs_in_bulk])


def _get_tmax_forward_backward(tsrcs: list[int], bulk_range: Sequence[int] | NDArray[np.integer]
                               ) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
    """Return per source the number of time slices in bulk_range forward and backward of it, source included."""
    tmin = bulk_range[0]
    tmax = bulk_range[-1]
    tmax_forward = tmax + 1 - np.array(tsrcs)
    tmax_backward = np.array(tsrcs) - tmin + 1
    return tmax_forward, tmax_backward


def _get_masked_meson_observations(observations: np.ndarray, tmax_forward: int, tmax_backward: int,
                                   time_parity: int) -> np.ma.MaskedArray:
    """Return forward and time-reversed backward correlators, masked beyond tmax_forward and tmax_backward.

    Shape (N, 2 * num_correlators, T); the backward half is multiplied by time_parity except at t = 0.
    """
    time_parity = _validate_time_parity(time_parity)
    N, num_correlators, T = observations.shape
    # backward = time-reverse each correlator (flip then roll by 1)
    backward = np.roll(np.flip(observations, axis=2), 1, axis=2)
    backward[:, :, 1:] *= time_parity
    masked = np.ma.empty((N, 2*num_correlators, T))
    masked.mask = True
    masked[:, :num_correlators, :tmax_forward] = observations[:, :, :tmax_forward]
    masked[:, num_correlators:, :tmax_backward] = backward[:, :, :tmax_backward]
    return masked


# ---------------------------------------------------------------------------
# Boundary average
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoundaryAverage:
    """Boundary effective mass, folded effective mass, and source counts."""

    effective_mass: Variable
    folded: Variable
    nsrc_hist: np.ndarray


def obc_meson_boundary_average(correlators: dict[int, Variable], tmin_excited: int, *, max_source_distance: int | None = None,
                               time_parity: int = 1) -> BoundaryAverage:
    """Return source-averaged boundary effective masses and source counts."""
    time_parity = _validate_time_parity(time_parity)

    effective_masses = []
    for tsrc, variable in correlators.items():
        masked = Variable.map_observations(lambda observations, tsrc=tsrc: np.array([
            _get_masked_correlators_boundary(observation, tsrc, tmin_excited, max_source_distance).mean(axis=0)
            for observation in observations
        ]), variable)
        effective_masses.append(Variable.derive(
            lambda correlator, tsrc=tsrc: _boundary_effective_mass(correlator, tsrc), masked))

    def source_average(*masses: np.ndarray) -> np.ndarray:
        # masked mean over tsrcs (0 marks a masked time slice)
        return np.ma.filled(np.ma.masked_equal(np.array(masses), 0).mean(axis=0), 0)
    effective_mass = Variable.derive(source_average, *effective_masses)
    nsrc_hist = np.sum(np.array([mass.central_value for mass in effective_masses]) != 0, axis=0)
    folded = Variable.derive(lambda mass: _fold_meson_boundary(mass, time_parity), effective_mass)
    return BoundaryAverage(effective_mass, folded, nsrc_hist)


def _boundary_effective_mass(correlator: np.ndarray, tsrc: int) -> np.ndarray:
    """Boundary effective mass of a (masked) correlator, sign-corrected per tsrc.

    Masked / non-finite time slices collapse to 0 -- the marker the source
    average reads as "no contribution here".
    """
    return np.nan_to_num(
        _flip_sign_boundary(effective_mass(correlator, estimator="log_symmetric"), tsrc), nan=0.0, posinf=0.0, neginf=0.0
    )


def _get_masked_correlators_boundary(correlators: np.ndarray, tsrc: int, tmin_excited: int,
                                     max_source_distance: int | None = None) -> np.ma.MaskedArray:
    """Return correlators in lattice time, masked where |t - tsrc| < tmin_excited or |t - tsrc| > max_source_distance."""
    # get correlators in terms of original lattice
    aligned = np.roll(correlators, tsrc, axis=1)
    num_correlators = correlators.shape[0]
    # 1: create masked array
    masked = np.ma.empty((num_correlators, correlators.shape[1]) )
    masked.mask = True
    # 2: fill in all elements that are not excited states
    masked[:,:tsrc-(tmin_excited-1)] = aligned[:,:tsrc-(tmin_excited-1)]
    masked[:,tsrc+tmin_excited:] = aligned[:,tsrc+tmin_excited:]
    # 3: mask time slices with |t - tsrc| > max_source_distance (inclusive distance)
    if max_source_distance is not None:
        times = np.arange(correlators.shape[1])
        tmax_mask = np.abs(times - tsrc) > max_source_distance
        masked.mask = np.logical_or(masked.mask, tmax_mask[np.newaxis,:])
    return masked


def _flip_sign_boundary(mass: np.ndarray | jax.Array, tsrc: int) -> jax.Array:
    """Negate effective masses before tsrc."""
    return jnp.where(jnp.arange(len(mass)) < tsrc, -1, 1) * mass


def _fold_meson_boundary(mass: np.ndarray, time_parity: int) -> np.ndarray:
    """Average mass with its mirror image around (T - 1)/2, times time_parity."""
    time_parity = _validate_time_parity(time_parity)
    half = len(mass) // 2
    first_half = mass[:half]
    second_half = np.flip(mass[half:]) * time_parity
    return np.mean([first_half, second_half], axis=0)
