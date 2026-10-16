"""Gamma-method error analysis (Wolff, hep-lat/0306017).

Arrays have configurations (or lags, windows) on axis 0; every element of the
remaining axes, e.g. a time slice, is a component and is analysed independently.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def mean_and_fluctuations(values, weights):
    """Weighted mean of primary data ``values`` over axis 0 and its fluctuations per configuration.

    ``delta_i = weights_i (values_i - mean) / mean(weights)``: the fluctuations
    of ``weights * values`` and ``weights`` projected with the gradient of
    their ratio of means. Uniform weights give ``values_i - mean``.
    """
    mean = np.average(values, axis=0, weights=weights)
    w_bcast = weights.reshape((-1,) + (1,) * (values.ndim - 1))
    delta = w_bcast * (values - mean) / np.mean(weights)
    return mean, delta


@dataclass(frozen=True)
class ErrorAnalysis:
    """Gamma-method error analysis of one quantity, per component."""
    error: np.ndarray
    error_of_error: np.ndarray
    tau_int: np.ndarray          # bias-corrected, at chosen W
    error_tau_int: np.ndarray    # at chosen W
    window: np.ndarray           # chosen W
    gamma: np.ndarray            # all lags, for autocorrelation plot
    tau_ints: np.ndarray         # all W, for plateau plot
    error_tau_ints: np.ndarray   # all W


def error_analysis(fluctuations, replicas, positions, S=1.5):
    """Gamma-method error analysis of ``fluctuations`` per component (Wolff, S default 1.5)."""
    num_configurations = len(fluctuations)
    gamma = autocovariance(fluctuations, replicas, positions)
    tau_ints = integrated_autocorrelation_times(gamma)
    error_tau_ints = error_integrated_autocorrelation_times(tau_ints, num_configurations)
    W = window(tau_ints, num_configurations, S)
    constant = gamma[0] == 0    # constant: error = 0, tau_int = 1/2, error_tau_int = 0
    error = np.where(constant, 0.0, np.sqrt(variance(gamma, tau_ints, W, num_configurations)))
    return ErrorAnalysis(
        error=error,
        error_of_error=error_of_error(error, W, num_configurations),
        tau_int=np.where(constant, 0.5, _at_window(tau_ints, W) * _tau_int_bias_correction(W, num_configurations)),
        error_tau_int=np.where(constant, 0.0, _at_window(error_tau_ints, W)),
        window=W,
        gamma=gamma,
        tau_ints=tau_ints,
        error_tau_ints=error_tau_ints,
    )


# ---------------------------------------------------------------------------
# Steps of the analysis, in order
# ---------------------------------------------------------------------------

def autocovariance(fluctuations, replicas, positions, max_lag=None, fft=True):
    """Autocovariance of ``fluctuations`` for lags 0 to ``max_lag``, using only
    pairs within the same replica. Default ``max_lag``: half the longest replica span minus 1.
    ``fft=False`` sums the products directly instead of via FFT."""
    if max_lag is None:
        spans = [np.ptp(positions[replicas == replica]) + 1 for replica in np.unique(replicas)]
        max_lag = max(spans) // 2 - 1
    product_sums = np.zeros((max_lag + 1,) + fluctuations.shape[1:])
    num_pairs = 0
    for replica in np.unique(replicas):
        in_replica = (replicas == replica)
        replica_fluctuations, measured = _insert_gaps(fluctuations[in_replica], positions[in_replica])
        product_sums += _lag_sum(replica_fluctuations, max_lag, fft)
        num_pairs += np.rint(_lag_sum(measured, max_lag, fft))
    lags_without_pairs = np.flatnonzero(num_pairs == 0)
    if lags_without_pairs.size:
        raise ValueError(f"No pairs at lag {lags_without_pairs[0]}; check positions or max_lag.")
    return product_sums / num_pairs.reshape((-1,) + (1,) * (product_sums.ndim - 1))


def _lag_sum(values, max_lag, fft=True):
    """Sums of ``values[i] * values[i + t]`` over i for lags t = 0, ..., ``max_lag``."""
    length = len(values)
    if not fft:
        return np.array([np.sum(values[:max(length - lag, 0)] * values[lag:], axis=0) for lag in range(max_lag + 1)])
    padded_length = next_fast_len(length + max_lag)
    transform = rfft(values, padded_length, axis=0)
    return irfft(np.abs(transform)**2, padded_length, axis=0)[:max_lag+1]


def _insert_gaps(values, positions):
    """Return ``values`` with zeros inserted at missing positions, and an array
    that is 1 at measured positions and 0 at missing ones."""
    indices = positions - positions.min()
    values_with_gaps = np.zeros((indices.max() + 1,) + values.shape[1:])
    values_with_gaps[indices] = values
    measured = np.zeros(indices.max() + 1)
    measured[indices] = 1
    return values_with_gaps, measured


def autocorrelation(gamma):
    """Autocorrelation ``gamma / gamma[0]``, with ``gamma`` from ``autocovariance``; NaN for a constant component."""
    constant = gamma[0] == 0
    return np.divide(gamma, gamma[0], out=np.full(gamma.shape, np.nan), where=~constant)


def integrated_autocorrelation_times(gamma):
    """Integrated autocorrelation times ``0.5 + sum(gamma[1:W+1]) / gamma[0]`` for
    windows W = 0, ..., len(gamma) - 1, with ``gamma`` from ``autocovariance``."""
    return np.cumsum(autocorrelation(gamma), axis=0) - 0.5


def error_integrated_autocorrelation_times(tau_ints, num_configurations):
    """Errors of ``tau_ints`` for every W per component (Wolff eq. 42)."""
    Ws = np.arange(len(tau_ints)).reshape((-1,) + (1,) * (tau_ints.ndim - 1))
    return 2 * tau_ints * (np.abs(Ws + 0.5 - tau_ints) / num_configurations)**.5  # abs: noise can make tau_int > W + 1/2 (as pyerrors)


def window(tau_ints, num_configurations, S=1.5):
    """Summation window W per component, with ``tau_ints`` from ``integrated_autocorrelation_times``."""
    components = tau_ints.reshape(len(tau_ints), -1).T
    windows = np.array([_window_1d(component, num_configurations, S) for component in components])
    return windows.reshape(tau_ints.shape[1:])


def _window_1d(tau_ints, num_configurations, S):
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


def _g(W, tau, num_configurations):
    """Wolff's window criterion g(W) (eq. 52)."""
    return np.exp(- W / tau) - tau / np.sqrt(W * num_configurations)


def variance(gamma, tau_ints, W, num_configurations):
    """Variance of the estimate, ``2 tau_int(W) gamma[0] / N`` per component, bias-corrected (Wolff eqs. 44, 49)."""
    tau_int = _at_window(tau_ints, W) * _tau_int_bias_correction(W, num_configurations)
    return 2 * tau_int * gamma[0] / num_configurations


def _at_window(values, W):
    """``values`` at the chosen W per component."""
    return np.take_along_axis(values, W[np.newaxis], axis=0)[0]


def _tau_int_bias_correction(W, num_configurations):
    """Factor ``1 + (2W+1)/N`` correcting the bias of ``tau_int(W)``, i.e. of ``C_F(W)`` (Wolff eq. 49)."""
    return 1 + (2 * W + 1) / num_configurations


def error_of_error(error, W, num_configurations):
    """Error of ``error = sqrt(variance)`` per component (Wolff eq. 40)."""
    return error * ((W + 0.5) / num_configurations)**.5
