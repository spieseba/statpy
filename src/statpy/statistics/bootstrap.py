import numpy as np


def generate_bootstrap_indices(num_bootstraps, num_configurations, seed):
    """Return a (num_bootstraps, num_configurations) bootstrap index matrix."""
    rng = np.random.RandomState(seed)
    return rng.randint(low=0, high=num_configurations, size=(num_bootstraps, num_configurations))


def mean_replicates(observations, bootstrap_indices, weights):
    """Weighted mean of ``observations`` over axis 0 for each bootstrap resampling."""
    return np.array([np.average(observations[idx], axis=0, weights=weights[idx]) for idx in bootstrap_indices])


def variance(bootstrap_replicates):
    return np.var(bootstrap_replicates, ddof=1, axis=0)


def covariance(bootstrap_replicates):
    mean = np.mean(bootstrap_replicates, axis=0)
    B = len(bootstrap_replicates)
    d = (bootstrap_replicates - mean).reshape(B, -1)
    return d.T @ d / (B-1)


def rescale(bootstrap_replicates, s):
    mean = np.mean(bootstrap_replicates, axis=0)
    return mean + s * (bootstrap_replicates - mean)
