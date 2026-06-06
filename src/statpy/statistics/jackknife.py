import warnings

import numpy as np

def sample(x, weights=None, f=lambda x: x):
    N = len(x)
    w = np.ones(N) if weights is None else weights
    if len(w) != N:
        raise ValueError(f"jackknife.sample: weights length {len(w)} != sample length {N}")
    with np.errstate(invalid='ignore'):
        mean = np.average(x, axis=0, weights=w)
        N_w = np.sum(w)
        return np.array([ f( mean + w[j] * (mean - x[j]) / (N_w - w[j]) ) for j in range(N)])

def binned_sample(jks, binsize, weights=None, mean=None):
    """Construct the binned jackknife sample from an unbinned one (delayed
    binning, arxiv:2410.17053).

    The idea is to defer binning to the end of an analysis: run the whole
    pipeline once on the unbinned (binsize-1) jackknife sample -- applying
    functions to the samples as usual -- and only then call this to obtain the
    binned jackknife sample at any ``binsize`` for error estimation, instead of
    re-running the analysis per binsize. The result agrees statistically with
    having binned the raw data up front (identical when ``binsize`` divides
    ``len(jks)``). The trailing incomplete bin is truncated.

    Binning is over axis 0. For a weighted jackknife sample (built by ``sample``
    with non-uniform ``weights``), pass the same ``weights``: the reconstruction
    then uses the per-sample factor (W - w_j)/(W - omega_i) about the weighted
    mean. With ``weights`` None or uniform this reduces to the scalar
    (N-1)/(N-binsize) form.

    Parameters
    ----------
    ``jks``: ndarray
        The unbinned (binsize-1) jackknife sample.
    ``binsize``: int
        Desired binsize for the binned jackknife sample.
    ``weights``: ndarray, optional
        Per-sample weights the jackknife sample was built with. Required only
        for non-uniform weights.
    ``mean``: ndarray or float, optional
        Mean of the data. Recovered from the jackknife sample (and ``weights``)
        if not specified.

    Returns:
    --------
    ndarray
        Binned jackknife sample with binsize ``binsize``.
    """
    if binsize <= 1:
        raise ValueError(f"binsize must be > 1, got {binsize}")
    N = len(jks)
    n_bins = N // binsize          # trailing incomplete bin is truncated
    keep = n_bins * binsize

    # uniform weights case 
    if weights is None:            # equal weights: scalar factor about the replicate mean
        if mean is None: 
            mean = np.mean(jks, axis=0)
        bin_sums = jks[:keep].reshape(n_bins, binsize, *jks.shape[1:]).sum(axis=1)
        return mean + (bin_sums - binsize * mean) * (N-1)/(N-binsize)

    # weighted case (! This needs to be checked at some point !)
    warnings.warn("binned_sample: I have not yet validated the weighted delayed binning; "
                  "results should be cross-checked before use.", stacklevel=2)
    bcast = (-1,) + (1,) * (jks.ndim - 1)
    W = weights.sum()
    Ww = (W - weights).reshape(bcast)                                       # (W - w_j), per sample
    if mean is None: 
        mean = (jks * Ww).sum(axis=0) / (W * (N-1))            # weighted mean from replicates
    omega = weights[:keep].reshape(n_bins, binsize).sum(axis=1).reshape(bcast)   # omega_i, per bin
    bin_terms = ((jks[:keep] - mean) * Ww[:keep]).reshape(n_bins, binsize, *jks.shape[1:]).sum(axis=1)
    return mean + bin_terms / (W - omega)


def variance(jks, mean=None):
    if mean is None: mean = np.mean(jks, axis=0)
    N = len(jks)
    with np.errstate(invalid='ignore'):
        return np.sum(np.array([(jks[j] - mean)**2 for j in range(N)]), axis=0) * (N-1) / N

def covariance(jks, mean=None):
    if mean is None: mean = np.mean(jks, axis=0)
    N = len(jks)
    def outer_sqr(a):
        return np.outer(a,a)
    return np.sum(np.array([outer_sqr(jks[j] - mean) for j in range(N)]), axis=0) * (N-1) / N  
