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


def correlation_matrix(covariance: np.ndarray) -> np.ndarray:
    """Correlation matrix of ``covariance``: ``covariance_ij / (sigma_i sigma_j)``, ones on the diagonal."""
    sigmas = np.sqrt(np.diag(covariance))
    return covariance / np.outer(sigmas, sigmas)


def rescale_covariance(covariance: np.ndarray, errors: np.ndarray) -> np.ndarray:
    """Covariance with the correlations of ``covariance`` and the standard deviations ``errors``:
    ``corr_ij errors_i errors_j``. If ``errors`` include autocorrelations and ``covariance`` does not, this
    assumes tau_int,ij = sqrt(tau_int,i tau_int,j)."""
    return np.outer(errors, errors) * correlation_matrix(covariance)
