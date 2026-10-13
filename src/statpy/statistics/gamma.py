import numpy as np
from scipy.fft import irfft, next_fast_len, rfft


def mean_and_fluctuations(values, weights):
    """Weighted mean of ``values`` over axis 0 and its fluctuations per configuration.

    ``delta_i = weights_i (values_i - mean) / mean(weights)``: the fluctuations
    of ``weights * values`` and ``weights`` projected with the gradient of
    their ratio of means. Uniform weights give ``values_i - mean``.
    """
    mean = np.average(values, axis=0, weights=weights)
    w_bcast = weights.reshape((-1,) + (1,) * (values.ndim - 1))
    delta = w_bcast * (values - mean) / np.mean(weights)
    return mean, delta


def autocovariance(fluctuations, chains, positions, max_lag=None, fft=True):
    """Autocovariance of ``fluctuations`` for lags 0 to ``max_lag``, using only
    pairs within the same chain. Default ``max_lag``: half the longest chain span minus 1.
    ``fft=False`` sums the products directly instead of via FFT."""
    if max_lag is None:
        spans = [np.ptp(positions[chains == chain]) + 1 for chain in np.unique(chains)]
        max_lag = max(spans) // 2 - 1
    product_sums = np.zeros((max_lag + 1,) + fluctuations.shape[1:])
    num_pairs = 0
    for chain in np.unique(chains):
        in_chain = (chains == chain)
        chain_fluctuations, measured = _insert_gaps(fluctuations[in_chain], positions[in_chain])
        product_sums += _lag_sum(chain_fluctuations, max_lag, fft)
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
