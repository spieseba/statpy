import numpy as np


# generate B bootstraps for N samples
def generate_bootstraps(B, N, seed):
    rng = np.random.RandomState(seed)
    return rng.randint(low=0, high=N, size=(B, N))

def sample(x, bootstraps, weights, estimator):
    """``estimator(x[idx], weights[idx])`` for each row ``idx`` of ``bootstraps``."""
    return np.array([estimator(x[idx], weights[idx]) for idx in bootstraps])

def mean_sample(x, bootstraps, weights):
    """Weighted mean of ``x`` over axis 0 for each bootstrap resampling."""
    return sample(x, bootstraps, weights, lambda x, w: np.average(x, axis=0, weights=w))

def variance(bootstrap_samples):
    return np.var(bootstrap_samples, ddof=1, axis=0)

def covariance(bootstrap_samples):
    mean = np.mean(bootstrap_samples, axis=0)
    B = len(bootstrap_samples)
    d = (bootstrap_samples - mean).reshape(B, -1)   # np.outer flattens its inputs
    return np.sum(d[:, :, None] * d[:, None, :], axis=0) / (B-1)

def rescale(bootstrap_samples, s):
    mean = np.mean(bootstrap_samples, axis=0)
    return mean + s * (bootstrap_samples - mean)
