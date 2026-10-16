import numpy as np


# generate B bootstraps for N samples
def generate_bootstraps(B, N, seed):
    rng = np.random.RandomState(seed)
    return rng.randint(low=0, high=N, size=(B, N))

def replicates(x, bootstraps, weights, estimator):
    """``estimator(x[idx], weights[idx])`` for each row ``idx`` of ``bootstraps``."""
    return np.array([estimator(x[idx], weights[idx]) for idx in bootstraps])

def mean_replicates(x, bootstraps, weights):
    """Weighted mean of ``x`` over axis 0 for each bootstrap resampling."""
    return replicates(x, bootstraps, weights, lambda x, w: np.average(x, axis=0, weights=w))

def variance(bootstrap_replicates):
    return np.var(bootstrap_replicates, ddof=1, axis=0)

def covariance(bootstrap_replicates):
    mean = np.mean(bootstrap_replicates, axis=0)
    B = len(bootstrap_replicates)
    d = (bootstrap_replicates - mean).reshape(B, -1)   # np.outer flattens its inputs
    return np.sum(d[:, :, None] * d[:, None, :], axis=0) / (B-1)

def rescale(bootstrap_replicates, s):
    mean = np.mean(bootstrap_replicates, axis=0)
    return mean + s * (bootstrap_replicates - mean)
