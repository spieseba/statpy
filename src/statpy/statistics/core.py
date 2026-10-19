import numpy as np


def bin(data: np.ndarray, bin_size: int, weights: np.ndarray) -> np.ndarray:
    assert bin_size is not None
    if bin_size == 1:
        return data
    N = len(data)
    Nb = N // bin_size  # cut off data of last incomplete bin
    keep = Nb * bin_size
    bcast = (1,) * (np.ndim(data) - 1)
    d = np.asarray(data)[:keep].reshape((Nb, bin_size) + np.shape(data)[1:])
    wb = np.asarray(weights)[:keep].reshape(Nb, bin_size)
    return (d * wb.reshape((Nb, bin_size) + bcast)).sum(axis=1) / wb.sum(axis=1).reshape((-1,) + bcast)


def normalize_covariance(cov: np.ndarray) -> np.ndarray:
    """Divide a covariance matrix by the outer product of its diagonal stds,
    yielding a correlation matrix with ones on the diagonal."""
    stds = np.sqrt(np.diag(cov))
    return cov / np.outer(stds, stds)


def inflated_covariance(jk_cov: np.ndarray, bs_std: np.ndarray) -> np.ndarray:
    """Re-scale a (binned) jackknife covariance to match a bootstrap-std scale."""
    cov_n = normalize_covariance(jk_cov)
    return np.outer(bs_std, bs_std) * cov_n
