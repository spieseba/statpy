import numpy as np

# binning
def bin(data, binsize, weights=None):
    assert binsize is not None
    if binsize == 1:
        return data
    N = len(data)
    w = np.ones(N) if weights is None else weights
    Nb = N // binsize # cut off data of last incomplete bin
    keep = Nb * binsize
    tail = (1,) * (np.ndim(data) - 1)
    d = np.asarray(data)[:keep].reshape((Nb, binsize) + np.shape(data)[1:])
    wb = np.asarray(w)[:keep].reshape(Nb, binsize)
    return (d * wb.reshape((Nb, binsize) + tail)).sum(axis=1) / wb.sum(axis=1).reshape((-1,) + tail)


def normalize_covariance(cov):
    """Divide a covariance matrix by the outer product of its diagonal stds,
    yielding a correlation matrix with ones on the diagonal."""
    stds = np.sqrt(np.diag(cov))
    return cov / np.outer(stds, stds)


def inflated_covariance(jk_cov, bs_std):
    """Re-scale a (binned) jackknife covariance to match a bootstrap-std scale. 

    Parameters
    ----------
    jk_cov : (N, N) array
        Jackknife covariance.
    bs_std : (N,) array
        Per-component bootstrap std dev (e.g. from
        ``np.sqrt(bootstrap.variance(bss))``).
    """
    cov_n = normalize_covariance(jk_cov)
    return np.outer(bs_std, bs_std) * cov_n 