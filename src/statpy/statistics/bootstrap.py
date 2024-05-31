import numpy as np

# generate B bootstraps for N samples
def generate_bootstraps(B, N, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(low=0, high=N, size=(B, N))

# compute bootstrap sample from sample x with bootstraps and function f
def sample(x, bootstraps, weights=None, f=lambda x: x):
    N = len(x); D = x.shape[1]
    B = bootstraps.shape[0]
    w = np.ones(N) if weights is None else weights
    bss = np.zeros(shape=(B,D))
    for k,bs in enumerate(bootstraps):
        bss[k] = f(np.average(x[bs], axis=0, weights=w[bs]))
    return bss 

def variance(bss, mean=None):
    if mean is None: mean = np.mean(bss, axis=0)
    B = len(bss)
    return np.sum(np.array([(bss[b] - mean)**2 for b in range(B)]), axis=0) / (B-1) 

def covariance(bss, mean=None):
    if mean is None: mean = np.mean(bss, axis=0)
    B = len(bss)
    def outer_sqr(a):
        return np.outer(a,a)
    return np.sum(np.array([outer_sqr(bss[b] - mean) for b in range(B)]), axis=0) / B 

def rescale(bss, s):
    mean = np.mean(bss, axis=0)
    if isinstance(np.mean(bss, axis=0), np.float64):
        return mean + s * (bss - mean) 
    return mean[None,:] + s * (bss - mean[None,:])
