import numpy as np

def covariance(sample, tmax):
    N = len(sample)
    mean = np.mean(sample, axis=0)
    d = sample - mean
    Cy = np.zeros(tmax)
    for t in range(tmax):
        Cy[t] = np.dot(d[:N-t], d[t:]) / (N-t)
    return Cy

def correlation_function(sample, tmax):
    cov = covariance(sample, tmax)
    return cov/cov[0]

def integrated_time(gamma, tmax):
    return 0.5 + np.sum(gamma[1:tmax])
