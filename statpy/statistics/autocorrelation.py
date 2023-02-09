import numpy as np

# autocovariance
def acov(sample, tmax):
    n = len(sample)
    y_bar = np.mean(sample, axis=0)
    Cy = np.zeros(tmax)
    for t in range(tmax):
        Cy_t = 0.0
        for i in range(n-t):
            Cy_t += (sample[i] - y_bar) * (sample[i+t] - y_bar)
        Cy_t /= n-t
        Cy[t] = Cy_t
    return Cy

# autocorrelation function
def acfunc(sample, tmax):
    cov = acov(sample, tmax)
    return cov/cov[0]

# integrated autocorrelation time 
def actime(gamma, tmax):
    return 0.5 + np.sum(gamma[1:tmax])
