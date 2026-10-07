import warnings

import numpy as np


def sample(x, weights=None):
    N = len(x)
    w = np.ones(N) if weights is None else weights
    if len(w) != N:
        raise ValueError(f"jackknife.sample: weights length {len(w)} != sample length {N}")
    mean = np.average(x, axis=0, weights=w)
    N_w = np.sum(w)
    w_col = w.reshape((-1,) + (1,) * (x.ndim - 1))
    return mean + w_col * (mean - x) / (N_w - w_col)

def variance(jackknife_samples, mean=None):
    if mean is None: mean = np.mean(jackknife_samples, axis=0)
    N = len(jackknife_samples)
    return np.sum((jackknife_samples - mean)**2, axis=0) * (N-1) / N

def covariance(jackknife_samples, mean=None):
    if mean is None: mean = np.mean(jackknife_samples, axis=0)
    N = len(jackknife_samples)
    d = (jackknife_samples - mean).reshape(N, -1)   # np.outer flattens its inputs
    return np.sum(d[:, :, None] * d[:, None, :], axis=0) * (N-1) / N


def delayed_binning(jackknife_samples, bin_size, weights=None, mean=None):
    """Construct the binned jackknife sample from an unbinned one (delayed
    binning, arxiv:2410.17053).

    The idea is to defer binning to the end of an analysis: run the whole
    pipeline once on the unbinned (bin-size-1) jackknife sample -- applying
    functions to the samples as usual -- and only then call this to obtain the
    binned jackknife sample at any ``bin_size`` for error estimation, instead of
    re-running the analysis per bin_size. The result agrees statistically with
    having binned the raw data up front (identical when ``bin_size`` divides
    ``len(jackknife_samples)``). The trailing incomplete bin is truncated.

    Binning is over axis 0. For a weighted jackknife sample (built by ``sample``
    with non-uniform ``weights``), pass the same ``weights``: the reconstruction
    then uses the per-sample factor (W - w_j)/(W - omega_i) about the weighted
    mean. With ``weights`` None or uniform this reduces to the scalar
    (N-1)/(N-bin_size) form.

    Parameters
    ----------
    ``jackknife_samples``: ndarray
        The unbinned (bin-size-1) jackknife sample.
    ``bin_size``: int
        Desired bin_size for the binned jackknife sample.
    ``weights``: ndarray, optional
        Per-sample weights the jackknife sample was built with. Required only
        for non-uniform weights.
    ``mean``: ndarray or float, optional
        Mean of the data. Recovered from the jackknife sample (and ``weights``)
        if not specified.

    Returns:
    --------
    ndarray
        Binned jackknife sample with bin_size ``bin_size``.
    """
    if bin_size <= 1:
        raise ValueError(f"bin_size must be > 1, got {bin_size}")
    N = len(jackknife_samples)
    num_bins = N // bin_size          # trailing incomplete bin is truncated
    keep = num_bins * bin_size

    # uniform weights (None or all equal): the scalar factor is exact, as the
    # constant cancels in both the mean reconstruction and the per-sample factor
    if weights is None or np.all(weights == weights[0]):
        if mean is None: 
            mean = np.mean(jackknife_samples, axis=0)
        bin_sums = jackknife_samples[:keep].reshape(num_bins, bin_size, *jackknife_samples.shape[1:]).sum(axis=1)
        return mean + (bin_sums - bin_size * mean) * (N-1)/(N-bin_size)

    # weighted case (! This needs to be checked at some point !)
    warnings.warn("delayed_binning: I have not yet validated the weighted delayed binning; "
                  "results should be cross-checked before use.", stacklevel=2)
    bcast = (-1,) + (1,) * (jackknife_samples.ndim - 1)
    W = weights.sum()
    Ww = (W - weights).reshape(bcast)                                       # (W - w_j), per sample
    if mean is None: 
        mean = (jackknife_samples * Ww).sum(axis=0) / (W * (N-1))            # weighted mean from replicates
    omega = weights[:keep].reshape(num_bins, bin_size).sum(axis=1).reshape(bcast)   # omega_i, per bin
    bin_terms = ((jackknife_samples[:keep] - mean) * Ww[:keep]).reshape(num_bins, bin_size, *jackknife_samples.shape[1:]).sum(axis=1)
    return mean + bin_terms / (W - omega)
