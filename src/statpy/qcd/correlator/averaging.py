"""Source averaging and folding of meson correlators."""
from collections.abc import Sequence
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


def obc_fold_effective_mass(mass: np.ndarray | jax.Array) -> jax.Array:
    """Average an OBC effective mass with its mirror image around (T - 1)/2, using the available half.

    Where only one of m(t) and m(T - 1 - t) is finite, that value is returned; NaN where neither is.
    """
    half = len(mass) // 2
    stacked = jnp.stack([mass[:half], jnp.flip(mass[half:])], axis=0)
    finite = jnp.isfinite(stacked)
    return jnp.sum(jnp.where(finite, stacked, 0), axis=0) / jnp.sum(finite, axis=0)


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

def obc_meson_boundary_average(correlators: dict[int, Variable], tmin_excited: int, *,
                               max_source_distance: int | None = None,
                               estimator: str = "log_symmetric") -> tuple[Variable, np.ndarray]:
    """Return the source-averaged boundary effective mass and the number of sources per time slice.

    The average is NaN where no source has a value.
    """
    # source count from the central values; raises here, before derive, for an unsupported estimator
    effective_masses = np.array([
        _masked_source_mass(correlators[tsrc].central_value, tsrc, tmin_excited, max_source_distance, estimator)
        for tsrc in correlators
    ])
    num_sources = np.sum(np.isfinite(effective_masses), axis=0)

    tsrcs = list(correlators)

    def source_average(*arrays: np.ndarray) -> jax.Array:
        """Return the mean over sources of the finite masked masses; arrays in the order of tsrcs."""
        stacked = jnp.stack([
            _masked_source_mass(array, tsrc, tmin_excited, max_source_distance, estimator)
            for array, tsrc in zip(arrays, tsrcs, strict=True)
        ])
        finite = jnp.isfinite(stacked)
        # sum and count only finite masses; 0/0 = NaN where no source is finite
        return jnp.sum(jnp.where(finite, stacked, 0), axis=0) / jnp.sum(finite, axis=0)

    # one derive over all sources
    average = Variable.derive(source_average, *correlators.values())
    return average, num_sources


def _masked_source_mass(correlators: np.ndarray, tsrc: int, tmin_excited: int, max_source_distance: int | None,
                        estimator: str) -> jax.Array:
    """Return one source's effective mass in lattice time.

    NaN where the mass uses a correlator slice with |t - tsrc| < tmin_excited or
    |t - tsrc| > max_source_distance, or where it is undefined.
    """
    # average over runs and get correlators in terms of original lattice
    aligned = jnp.roll(jnp.mean(correlators, axis=0), tsrc)
    mass = _flip_sign_boundary(effective_mass(aligned, estimator=estimator), tsrc)
    # correlator mask: keep tmin_excited <= |t - tsrc| <= max_source_distance
    times = np.arange(len(mass))
    distance = np.abs(times - tsrc)
    keep = distance >= tmin_excited
    if max_source_distance is not None:
        keep &= distance <= max_source_distance
    # mass mask: keep the mass at t only if all correlator slices it uses are kept
    if estimator == "log":
        keep_mass = keep & np.roll(keep, -1)  # C(t), C(t+1)
    elif estimator == "log_symmetric":
        keep_mass = np.roll(keep, 1) & np.roll(keep, -1)  # C(t-1), C(t+1)
    else:
        raise ValueError(f"Boundary effective mass needs estimator 'log' or 'log_symmetric', not {estimator!r}")
    return jnp.where(keep_mass, mass, jnp.nan)


def _flip_sign_boundary(mass: np.ndarray | jax.Array, tsrc: int) -> jax.Array:
    """Negate effective masses before tsrc."""
    return jnp.where(jnp.arange(len(mass)) < tsrc, -1, 1) * mass
