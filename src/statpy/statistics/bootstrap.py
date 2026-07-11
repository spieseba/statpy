import numpy as np

# generate B bootstraps for N samples
def generate_bootstraps(B, N, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(low=0, high=N, size=(B, N))

# compute bootstrap sample from sample x with bootstraps
def sample(x, bootstraps, weights=None):
    N = len(x)
    B = bootstraps.shape[0]
    w = np.ones(N) if weights is None else weights
    out = np.empty((B,) + x.shape[1:])
    tail = (1,) * (x.ndim - 1)
    chunk = max(1, 4_000_000 // max(1, x.size))   # ~32 MB gather buffer
    for i in range(0, B, chunk):
        bs = bootstraps[i:i+chunk]
        wb = w[bs]
        out[i:i+chunk] = (x[bs] * wb.reshape(wb.shape + tail)).sum(axis=1) \
                         / wb.sum(axis=1).reshape((-1,) + tail)
    return out

def variance(bss, mean=None):
    return np.var(bss, mean=mean, ddof=1, axis=0)

def covariance(bss, mean=None):
    if mean is None: mean = np.mean(bss, axis=0)
    B = len(bss)
    d = (bss - mean).reshape(B, -1)   # np.outer flattens its inputs
    return np.sum(d[:, :, None] * d[:, None, :], axis=0) / (B-1)

def rescale(bss, s):
    mean = np.mean(bss, axis=0)
    return mean + s * (bss - mean)
