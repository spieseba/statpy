"""Gamma-method error analysis (Wolff, hep-lat/0306017).

Arrays have configurations (or lags, windows) on axis 0; every element of the
remaining axes, e.g. a time slice, is a component and is analysed independently.
"""

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.fft import irfft, next_fast_len, rfft

# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def mean_and_fluctuations(observations: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean of primary ``observations`` over axis 0 and their fluctuations per configuration.

    ``delta_i = weights_i (observations_i - mean) / mean(weights)``: the fluctuations
    of ``weights * observations`` and ``weights`` projected with the gradient of
    their ratio of means. Uniform weights give ``observations_i - mean``.
    """
    mean = np.average(observations, axis=0, weights=weights)
    w_bcast = weights.reshape((-1,) + (1,) * (observations.ndim - 1))
    delta = w_bcast * (observations - mean) / np.mean(weights)
    return mean, delta


@dataclass(frozen=True)
class GammaErrorAnalysis:
    """Gamma-method error analysis of one quantity, per component."""
    error: np.ndarray
    error_of_error: np.ndarray
    tau_int: np.ndarray          # bias-corrected, at chosen W
    error_tau_int: np.ndarray    # at chosen W
    window: np.ndarray           # chosen W
    autocovariance: np.ndarray   # all lags, for autocorrelation plot
    tau_ints: np.ndarray         # all W, for plateau plot
    error_tau_ints: np.ndarray   # all W


def error_analysis(fluctuations: np.ndarray, monte_carlo_times: NDArray[np.integer], replicas: NDArray[np.integer], S: float) -> GammaErrorAnalysis:
    """Gamma-method error analysis of ``fluctuations`` per component (Wolff)."""
    num_configurations = len(fluctuations)
    gamma = autocovariance(fluctuations, monte_carlo_times, replicas)
    tau_ints = integrated_autocorrelation_times(gamma)
    error_tau_ints = error_integrated_autocorrelation_times(tau_ints, num_configurations)
    W = window(tau_ints, num_configurations, S)
    constant = gamma[0] == 0    # constant: error = 0, tau_int = 1/2, error_tau_int = 0
    error = np.where(constant, 0.0, np.sqrt(variance(gamma, tau_ints, W, num_configurations)))
    return GammaErrorAnalysis(
        error=error,
        error_of_error=error_of_error(error, W, num_configurations),
        tau_int=np.where(constant, 0.5, _at_window(tau_ints, W) * _tau_int_bias_correction(W, num_configurations)),
        error_tau_int=np.where(constant, 0.0, _at_window(error_tau_ints, W)),
        window=W,
        autocovariance=gamma,
        tau_ints=tau_ints,
        error_tau_ints=error_tau_ints,
    )

# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------

def value_and_jacobians(func: Callable[..., np.ndarray | jax.Array], central_values: Sequence[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Value of ``func`` and its Jacobian with respect to each input, at ``central_values``.

    Jacobian k has the output's shape followed by input k's shape: entry
    ``[o..., i...]`` is ``d func[o...] / d input_k[i...]``. Reverse mode if the
    inputs have more components than the output, else forward mode.
    """
    input_size = sum(np.size(value) for value in central_values)
    output_size = jax.eval_shape(func, *central_values).size

    def value_twice(*args):
        value = func(*args)
        return value, value

    jacobian_function = jax.jacrev if input_size > output_size else jax.jacfwd
    jacobians, value = jacobian_function(value_twice, argnums=tuple(range(len(central_values))), has_aux=True)(*central_values)
    return np.asarray(value), [np.asarray(jacobian) for jacobian in jacobians]


def project_fluctuations(jacobians: Sequence[np.ndarray], fluctuations: Sequence[np.ndarray]) -> np.ndarray:
    """Fluctuations of the derived quantity: sum over inputs k of ``jacobians[k]`` applied to ``fluctuations[k]``.

    ``jacobians[k]`` has shape ``output_shape + input_shape_k`` (as from
    ``value_and_jacobians``), ``fluctuations[k]`` has shape
    ``(num_configurations,) + input_shape_k``. Returns shape
    ``(num_configurations,) + output_shape``.
    """
    projections = []
    for jacobian, fluctuation in zip(jacobians, fluctuations, strict=True):
        fluctuation_input_axes = range(1, fluctuation.ndim)
        jacobian_input_axes = range(jacobian.ndim - len(fluctuation_input_axes), jacobian.ndim)
        projections.append(np.tensordot(fluctuation, jacobian, axes=(fluctuation_input_axes, jacobian_input_axes)))
    return np.sum(projections, axis=0)

# ---------------------------------------------------------------------------
# Analysis steps
# ---------------------------------------------------------------------------

def autocovariance(fluctuations: np.ndarray, monte_carlo_times: NDArray[np.integer], replicas: NDArray[np.integer], max_lag: int | None = None, fft: bool = True) -> np.ndarray:
    """Autocovariance of ``fluctuations`` for lags 0 to ``max_lag``, using only
    pairs within the same replica. Lags are in units of the greatest common divisor of the Monte Carlo
    time differences. Default ``max_lag``: half the longest replica span minus 1.
    ``fft=False`` sums the products directly instead of via FFT."""
    differences = [np.diff(monte_carlo_times[replicas == replica]) for replica in np.unique(replicas)]
    monte_carlo_times = monte_carlo_times // (np.gcd.reduce(np.concatenate(differences)) or 1)
    if max_lag is None:
        spans = [np.ptp(monte_carlo_times[replicas == replica]) + 1 for replica in np.unique(replicas)]
        max_lag = max(spans) // 2 - 1
    product_sums = np.zeros((max_lag + 1,) + fluctuations.shape[1:])
    num_pairs = np.zeros(max_lag + 1)
    for replica in np.unique(replicas):
        in_replica = (replicas == replica)
        replica_fluctuations, measured = _insert_gaps(fluctuations[in_replica], monte_carlo_times[in_replica])
        product_sums += _lag_sum(replica_fluctuations, max_lag, fft)
        num_pairs += np.rint(_lag_sum(measured, max_lag, fft))
    lags_without_pairs = np.flatnonzero(num_pairs == 0)
    if lags_without_pairs.size:
        raise ValueError(f"No pairs at lag {lags_without_pairs[0]}; check monte_carlo_times or max_lag.")
    return product_sums / num_pairs.reshape((-1,) + (1,) * (product_sums.ndim - 1))


def _lag_sum(values: np.ndarray, max_lag: int, fft: bool = True) -> np.ndarray:
    """Sums of ``values[i] * values[i + t]`` over i for lags t = 0, ..., ``max_lag``."""
    length = len(values)
    if not fft:
        return np.array([np.sum(values[:max(length - lag, 0)] * values[lag:], axis=0) for lag in range(max_lag + 1)])
    padded_length = next_fast_len(length + max_lag)
    transform = rfft(values, padded_length, axis=0)
    return irfft(np.abs(transform)**2, padded_length, axis=0)[:max_lag+1]


def _insert_gaps(values: np.ndarray, monte_carlo_times: NDArray[np.integer]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``values`` with zeros inserted at missing Monte Carlo times, and an array
    that is 1 at measured times and 0 at missing ones."""
    indices = monte_carlo_times - monte_carlo_times.min()
    values_with_gaps = np.zeros((indices.max() + 1,) + values.shape[1:])
    values_with_gaps[indices] = values
    measured = np.zeros(indices.max() + 1)
    measured[indices] = 1
    return values_with_gaps, measured


def autocorrelation(gamma: np.ndarray) -> np.ndarray:
    """Autocorrelation ``gamma / gamma[0]``, with ``gamma`` from ``autocovariance``; NaN for a constant component."""
    constant = gamma[0] == 0
    return np.divide(gamma, gamma[0], out=np.full(gamma.shape, np.nan), where=~constant)


def integrated_autocorrelation_times(gamma: np.ndarray) -> np.ndarray:
    """Integrated autocorrelation times ``0.5 + sum(gamma[1:W+1]) / gamma[0]`` for
    windows W = 0, ..., len(gamma) - 1, with ``gamma`` from ``autocovariance``."""
    return np.cumsum(autocorrelation(gamma), axis=0) - 0.5


def error_integrated_autocorrelation_times(tau_ints: np.ndarray, num_configurations: int) -> np.ndarray:
    """Errors of ``tau_ints`` for every W per component (Wolff eq. 42); NaN with a warning where
    tau_int(W) > W + 1/2."""
    Ws = np.arange(len(tau_ints)).reshape((-1,) + (1,) * (tau_ints.ndim - 1))
    above = Ws + 0.5 - tau_ints < 0    # an estimated autocorrelation above 1, e.g. from few pairs at a lag
    if np.any(above):
        warnings.warn(f"tau_int(W) > W + 1/2 for {np.count_nonzero(above)} windows, first at W = "
                      f"{np.argwhere(above)[0, 0]}; their errors of tau_int are NaN.")
    # abs: far beyond the window, noise can make tau_int < 0
    return 2 * np.abs(tau_ints) * (np.where(above, np.nan, Ws + 0.5 - tau_ints) / num_configurations)**.5


def window(tau_ints: np.ndarray, num_configurations: int, S: float) -> NDArray[np.integer]:
    """Summation window W per component, with ``tau_ints`` from ``integrated_autocorrelation_times``."""
    components = tau_ints.reshape(len(tau_ints), -1).T
    windows = np.array([_window_1d(component, num_configurations, S) for component in components])
    return windows.reshape(tau_ints.shape[1:])


def _window_1d(tau_ints: np.ndarray, num_configurations: int, S: float) -> int:
    """First W >= 1 with g(W) < 0 or ``tau_int <= 1/2`` (Wolff eq. 52); the largest W with a warning if none; 0 for a constant component."""
    if np.isnan(tau_ints[0]):   # constant component
        return 0
    tau_ints = tau_ints[1:]
    Ws = np.arange(1, len(tau_ints) + 1)
    accepted = tau_ints <= 0.5
    taus = S / np.log( (2 * tau_ints[~accepted] + 1) / (2 * tau_ints[~accepted] - 1) )
    accepted[~accepted] = _g(Ws[~accepted], taus, num_configurations) < 0
    if not accepted.any():
        warnings.warn(f"No window found; using the largest W = {Ws[-1]}.")
        return Ws[-1]
    return Ws[np.argmax(accepted)]


def _g(Ws: NDArray[np.integer], taus: np.ndarray, num_configurations: int) -> np.ndarray:
    """Wolff's window criterion g(W) (eq. 52)."""
    return np.exp(- Ws / taus) - taus / np.sqrt(Ws * num_configurations)


def variance(gamma: np.ndarray, tau_ints: np.ndarray, W: NDArray[np.integer], num_configurations: int) -> np.ndarray:
    """Variance of the estimate, ``2 tau_int(W) gamma[0] / N`` per component, bias-corrected (Wolff eqs. 44, 49)."""
    tau_int = _at_window(tau_ints, W) * _tau_int_bias_correction(W, num_configurations)
    return 2 * tau_int * gamma[0] / num_configurations


def _at_window(values: np.ndarray, W: NDArray[np.integer]) -> np.ndarray:
    """``values`` at the chosen W per component."""
    return np.take_along_axis(values, W[np.newaxis], axis=0)[0]


def _tau_int_bias_correction(W: NDArray[np.integer], num_configurations: int) -> np.ndarray:
    """Factor ``1 + (2W+1)/N`` correcting the bias of ``tau_int(W)``, i.e. of ``C_F(W)`` (Wolff eq. 49)."""
    return 1 + (2 * W + 1) / num_configurations


def error_of_error(error: np.ndarray, W: NDArray[np.integer], num_configurations: int) -> np.ndarray:
    """Error of ``error = sqrt(variance)`` per component (Wolff eq. 40)."""
    return error * ((W + 0.5) / num_configurations)**.5


def bias_corrected_value(central_value: np.ndarray, replica_central_values: np.ndarray,
                         sizes: NDArray[np.integer]) -> np.ndarray:
    """Wolff eq. 20: ``(R F - Fbar) / (R - 1)`` with ``Fbar`` from ``_replica_mean``; cancels the leading
    O(1/N) bias of a nonlinear derived quantity. ``sizes``: configurations per replica; needs R >= 2."""
    num_replicas = len(sizes)
    return (num_replicas * central_value - _replica_mean(replica_central_values, sizes)) / (num_replicas - 1)


def replica_q_value(replica_central_values: np.ndarray, sizes: NDArray[np.integer], error: np.ndarray) -> np.ndarray:
    """Wolff eqs. 27-28 per component: probability of a chi^2 of the replica estimates around their mean
    ``Fbar`` at least as large as observed, ``chi^2 = sum_r N_r (F_r - Fbar)^2 / (N error^2)``. NaN for a
    constant component (error 0)."""
    deviations = replica_central_values - _replica_mean(replica_central_values, sizes)
    constant = error == 0    # constant: no fluctuations, Q undefined
    chi2 = np.tensordot(sizes, deviations**2, axes=1) / (np.sum(sizes) * np.where(constant, 1.0, error)**2)
    return np.where(constant, np.nan, stats.chi2.sf(chi2, len(sizes) - 1))


def _replica_mean(replica_central_values: np.ndarray, sizes: NDArray[np.integer]) -> np.ndarray:
    """Wolff eq. 15: ``Fbar = sum_r N_r F_r / N``."""
    return np.tensordot(sizes, replica_central_values, axes=1) / np.sum(sizes)

# ---------------------------------------------------------------------------
# Errors for autocorrelation plots
# ---------------------------------------------------------------------------

def error_autocovariance(gamma: np.ndarray, W: NDArray[np.integer], num_configurations: int, max_lag: int) -> np.ndarray:
    """Errors of ``gamma`` from ``autocovariance`` for lags 0 to ``max_lag`` per component
    (Lüscher hep-lat/0409106 eq. E.9 with s = t); the sum over k stops at t + W, NaN where
    that needs lags beyond ``len(gamma) - 1``."""
    components = gamma.reshape(len(gamma), -1).T
    errors = [_error_autocovariance_1d(component, int(w), num_configurations, max_lag) for component, w in zip(components, W.ravel(), strict=True)]
    return np.stack(errors, axis=1).reshape((max_lag + 1,) + gamma.shape[1:])


def _error_autocovariance_1d(gamma: np.ndarray, W: int, num_configurations: int, max_lag: int) -> np.ndarray:
    """``error_autocovariance`` for one component."""
    errors = np.full(max_lag + 1, np.nan)
    for t in range(min(max_lag, (len(gamma) - 1 - W) // 2) + 1):
        k = np.arange(1, t + W + 1)
        variance = gamma[0]**2 + gamma[t]**2 + 2 * np.sum(gamma[k]**2 + gamma[k + t] * gamma[np.abs(k - t)])
        errors[t] = np.sqrt(variance / num_configurations)
    return errors


def error_autocorrelation(rho: np.ndarray, W: NDArray[np.integer], num_configurations: int, max_lag: int) -> np.ndarray:
    """Errors of ``rho`` from ``autocorrelation`` for lags 0 to ``max_lag`` per component
    (Lüscher hep-lat/0409106 eq. E.11); the sum over k stops at t + W, NaN where that
    needs lags beyond ``len(rho) - 1``."""
    components = rho.reshape(len(rho), -1).T
    errors = [_error_autocorrelation_1d(component, int(w), num_configurations, max_lag) for component, w in zip(components, W.ravel(), strict=True)]
    return np.stack(errors, axis=1).reshape((max_lag + 1,) + rho.shape[1:])


def _error_autocorrelation_1d(rho: np.ndarray, W: int, num_configurations: int, max_lag: int) -> np.ndarray:
    """``error_autocorrelation`` for one component."""
    errors = np.full(max_lag + 1, np.nan)
    for t in range(min(max_lag, (len(rho) - 1 - W) // 2) + 1):
        k = np.arange(1, t + W + 1)
        errors[t] = np.sqrt(np.sum((rho[k + t] + rho[np.abs(k - t)] - 2 * rho[k] * rho[t])**2) / num_configurations)
    return errors
