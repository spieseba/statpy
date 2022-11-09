import numpy as np

# autocovariance
def autocovariance(sample, tmax):
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
def gamma(sample, tmax):
    cov = autocovariance(sample, tmax)
    return cov/cov[0]

# integrated autocorrelation time 
def tau_int(gamma, tmax):
    assert isinstance(gamma,np.ndarray)
    if gamma.ndim == 1: 
        gamma_arr = np.array([gamma])
    else:
        gamma_arr = gamma
    K = len(gamma_arr)
    tau_int = np.array([0.5 for i in range(K)])
    for k in range(K):
        for t in range(1,tmax):
            tau_int[k] += gamma_arr[k][t]

    if K==1:
        return tau_int[0]
    else:
        return tau_int


def stdErr(sample, tau_int=0.0):
    return np.sqrt(2.0*tau_int / len(sample) ) * np.std(sample, ddof=1) 

    
   
    

