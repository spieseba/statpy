import numpy as np

def sample(x, weights=None, f=lambda x: x): 
    N = len(x)
    w = np.ones(N) if weights is None else weights; assert len(w) == N
    mean = np.average(x, axis=0, weights=w)
    N_w = np.sum(w)
    return np.array([ f( mean + w[j] * (mean - x[j]) / (N_w - w[j]) ) for j in range(N)])

def variance(jks, mean=None):
    if mean is None: mean = np.mean(jks, axis=0)
    N = len(jks)
    return np.sum(np.array([(jks[j] - mean)**2 for j in range(N)]), axis=0) * (N-1) / N  

def covariance(jks, mean=None):
    if mean is None: mean = np.mean(jks, axis=0)
    N = len(jks)
    def outer_sqr(a):
        return np.outer(a,a)
    return np.sum(np.array([outer_sqr(jks[j] - mean) for j in range(N)]), axis=0) * (N-1) / N  
