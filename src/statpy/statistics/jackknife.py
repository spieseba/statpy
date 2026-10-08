import numpy as np


def mean_sample(x, weights):
    """Weighted mean of ``x`` over axis 0 with each configuration left out in turn."""
    mean = np.average(x, axis=0, weights=weights)
    W = np.sum(weights)
    w_bcast = weights.reshape((-1,) + (1,) * (x.ndim - 1))
    return mean + w_bcast * (mean - x) / (W - w_bcast)

def variance(jackknife_samples):
    mean = np.mean(jackknife_samples, axis=0)
    N = len(jackknife_samples)
    return np.sum((jackknife_samples - mean)**2, axis=0) * (N-1) / N

def covariance(jackknife_samples):
    mean = np.mean(jackknife_samples, axis=0)
    N = len(jackknife_samples)
    d = (jackknife_samples - mean).reshape(N, -1)   # np.outer flattens its inputs
    return np.sum(d[:, :, None] * d[:, None, :], axis=0) * (N-1) / N


def delayed_binning(jackknife_samples, bin_size, weights):
    """Binned jackknife sample from an unbinned one (delayed binning, arxiv:2410.17053).

    Bins over axis 0 and truncates the trailing incomplete bin. ``weights`` are
    those the jackknife sample was built with.
    """
    if bin_size <= 1:
        raise ValueError(f"bin_size must be > 1, got {bin_size}")
    N = len(jackknife_samples)
    num_bins = N // bin_size          # trailing incomplete bin is truncated
    keep = num_bins * bin_size
    bcast = (-1,) + (1,) * (jackknife_samples.ndim - 1)

    W = np.sum(weights)
    W_minus_w = (W - weights).reshape(bcast)                     # W - w_j
    mean = np.sum(W_minus_w * jackknife_samples, axis=0) / (W * (N - 1))

    # jk_(bin i) = mean + sum_{j in bin i} (W - w_j)(jk_j - mean) / (W - W_bins[i])
    terms = W_minus_w[:keep] * (jackknife_samples[:keep] - mean)
    bin_sums = terms.reshape(num_bins, bin_size, *jackknife_samples.shape[1:]).sum(axis=1)
    W_bins = weights[:keep].reshape(num_bins, bin_size).sum(axis=1).reshape(bcast)   # weight sum per bin
    return mean + bin_sums / (W - W_bins)
