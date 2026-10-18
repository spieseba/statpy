import numpy as np


def mean_replicates(observations, weights):
    """Weighted mean of ``observations`` over axis 0 with each configuration left out in turn."""
    mean = np.average(observations, axis=0, weights=weights)
    W = np.sum(weights)
    w_bcast = weights.reshape((-1,) + (1,) * (observations.ndim - 1))
    return mean + w_bcast * (mean - observations) / (W - w_bcast)

def variance(jackknife_replicates):
    mean = np.mean(jackknife_replicates, axis=0)
    N = len(jackknife_replicates)
    return np.sum((jackknife_replicates - mean)**2, axis=0) * (N-1) / N

def covariance(jackknife_replicates):
    mean = np.mean(jackknife_replicates, axis=0)
    N = len(jackknife_replicates)
    d = (jackknife_replicates - mean).reshape(N, -1)
    return d.T @ d * (N-1) / N


def delayed_binning(jackknife_replicates, bin_size, weights):
    """Return delayed-binned jackknife replicates, truncating the trailing incomplete bin."""
    if bin_size <= 1:
        raise ValueError(f"bin_size must be > 1, got {bin_size}")
    N = len(jackknife_replicates)
    num_bins = N // bin_size          # trailing incomplete bin is truncated
    keep = num_bins * bin_size
    bcast = (-1,) + (1,) * (jackknife_replicates.ndim - 1)

    W = np.sum(weights)
    W_minus_w = (W - weights).reshape(bcast)                     # W - w_j
    mean = np.sum(W_minus_w * jackknife_replicates, axis=0) / (W * (N - 1))

    # jk_(bin i) = mean + sum_{j in bin i} (W - w_j)(jk_j - mean) / (W - W_bins[i])
    terms = W_minus_w[:keep] * (jackknife_replicates[:keep] - mean)
    bin_sums = terms.reshape(num_bins, bin_size, *jackknife_replicates.shape[1:]).sum(axis=1)
    W_bins = weights[:keep].reshape(num_bins, bin_size).sum(axis=1).reshape(bcast)   # weight sum per bin
    return mean + bin_sums / (W - W_bins)
