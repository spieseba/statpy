import numpy as np

def sample(x, *argv, f=lambda x: x): 
    N = len(x)
    if len(argv) != 0:
        weights = argv[0]
        s = []
        for j in range(N):
            xj = np.delete(x, j, axis=0)
            wj = np.delete(weights, j, axis=0); wj = wj / np.mean(wj) # renormalize weights. This is a choice.
            s.append( f(np.mean(wj * xj, axis=0)) )
        return np.array(s)
    mean = np.mean(x, axis=0)
    return np.array([ f(mean + (mean - x[j]) / (N-1)) for j in range(N)])

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
