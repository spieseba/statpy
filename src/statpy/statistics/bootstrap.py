import numpy as np
from numpy.typing import NDArray


def generate_bootstrap_indices(num_bootstraps: int, num_configurations: int, seed: int) -> NDArray[np.integer]:
    """Return a (num_bootstraps, num_configurations) bootstrap index matrix."""
    rng = np.random.RandomState(seed)
    return rng.randint(low=0, high=num_configurations, size=(num_bootstraps, num_configurations))


def mean_replicates(observations: np.ndarray, bootstrap_indices: NDArray[np.integer], weights: np.ndarray) -> np.ndarray:
    """Weighted mean of ``observations`` over axis 0 for each bootstrap resampling."""
    return np.array([np.average(observations[idx], axis=0, weights=weights[idx]) for idx in bootstrap_indices])


def variance(bootstrap_replicates: np.ndarray) -> np.ndarray:
    return np.var(bootstrap_replicates, ddof=1, axis=0)


def covariance(bootstrap_replicates: np.ndarray) -> np.ndarray:
    mean = np.mean(bootstrap_replicates, axis=0)
    B = len(bootstrap_replicates)
    d = (bootstrap_replicates - mean).reshape(B, -1)
    return d.T @ d / (B-1)


def rescale(bootstrap_replicates: np.ndarray, s: float) -> np.ndarray:
    mean = np.mean(bootstrap_replicates, axis=0)
    return mean + s * (bootstrap_replicates - mean)
