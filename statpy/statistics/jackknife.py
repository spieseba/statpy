import numpy as np

################################################ functions of means ################################################

def samples(f, x, weights=None):
    N = len(x)
    if weights == None:
        mean = np.mean(x, axis=0)
        return np.array([ f(mean + (mean - x[j]) / (N-1)) for j in range(N)])
    else:
        s = []
        for j in range(N):
            xj = np.delete(x, j, axis=0)
            wj = np.delete(weights, j, axis=0); wj = wj / np.mean(wj) # renormalize weights. This is a choice.
            s.append( f(np.mean(wj * xj, axis=0)) )
        return np.array(s)

def variance(f, x, f_samples=None):
    N = len(x)
    mean = np.mean(x, axis=0)
    f_mean = f(mean)
    if f_samples == None:
        return np.mean([ ( f( (N * mean - x[k]) / (N - 1) ) - f_mean )**2 for k in range(N) ], axis=0) * (N-1)
    return np.sum(np.array([(f_samples[j] - f_mean) for j in range(N)]), axis=0) * (N-1) / N   #np.mean((f_samples - f_mean)**2, axis=0) * (N-1)

def covariance(f, x, f_samples=None):
    N = len(x)
    mean = np.mean(x, axis=0)
    f_mean = f(mean)
    def outer_sqr(a):
        return np.outer(a,a)
    if f_samples == None:
        return np.mean( [outer_sqr( f( (N * mean - x[k]) / (N - 1) ) - f_mean  ) for k in range(N)], axis=0) * (N-1) 
    return np.sum(np.array([outer_sqr(f_samples[j] - f_mean) for j in range(N)]), axis=0) * (N-1) / N   

def covariance_samples2(f_mean, f_samples):
    N = len(f_samples)
    def outer_sqr(a):
        return np.outer(a,a)
    return np.sum(np.array([outer_sqr(f_samples[j] - f_mean) for j in range(N)]), axis=0) * (N-1) / N  

################################################ arbitrary functions ################################################

def variance_general(f, x, f_samples=None):
    if f_samples == None:
        N = len(x)
        f_x = f(x)
        return np.mean([ (f(np.delete(x, k, axis=0)) - f_x)**2 for k in range(N)], axis=0) * (N-1)
        
def covariance_general(f, x):
    N = len(x)
    f_x = f(x)
    def outer_prod(a):
        return np.outer(a,a)
    return np.mean([outer_prod( (f(np.delete(x, k, axis=0)) - f_x) ) for k in range(N)], axis=0) * (N - 1) 