import numpy as np


def bin(data: np.ndarray, bin_size: int, weights: np.ndarray | None = None) -> np.ndarray:
    """Means of consecutive blocks of ``bin_size`` rows of ``data``, weighted by ``weights`` (one per row) if
    given: sum(w x) / sum(w) per block. A trailing incomplete block is dropped."""
    num_bins = len(data) // bin_size
    blocks = data[:num_bins * bin_size].reshape((num_bins, bin_size) + data.shape[1:])
    if weights is None:
        return blocks.mean(axis=1)
    w = weights[:num_bins * bin_size].reshape((num_bins, bin_size) + (1,) * (data.ndim - 1))
    return (w * blocks).sum(axis=1) / w.sum(axis=1)


def normalize_covariance(cov: np.ndarray) -> np.ndarray:
    """Divide a covariance matrix by the outer product of its diagonal stds,
    yielding a correlation matrix with ones on the diagonal."""
    stds = np.sqrt(np.diag(cov))
    return cov / np.outer(stds, stds)


def inflated_covariance(jk_cov: np.ndarray, bs_std: np.ndarray) -> np.ndarray:
    """Re-scale a (binned) jackknife covariance to match a bootstrap-std scale."""
    cov_n = normalize_covariance(jk_cov)
    return np.outer(bs_std, bs_std) * cov_n
