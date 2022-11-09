import numpy as np

################################################ functions of means ################################################

def samples(f, x):
    N = len(x)
    mean = np.mean(x, axis=0)
    return np.array([ f(mean + (mean - x[j]) / (N-1)) for j in range(N)])

def variance(f, x):
    N = len(x)
    mean = np.mean(x, axis=0)
    f_mean = f(mean)
    return np.mean([ ( f( (N * mean - x[k]) / (N - 1) ) - f_mean )**2 for k in range(N) ], axis=0) * (N - 1)

def covariance(f, x):
    N = len(x)
    mean = np.mean(x, axis=0)
    f_mean = f(mean)
    def outer_prod(a):
        return np.outer(a,a)
    return np.mean( [outer_prod( f( (N * mean - x[k]) / (N - 1) ) - f_mean  ) for k in range(N)], axis=0) * (N - 1) 

def covariance_samples(f, x, f_samples):
    N = len(x)
    f_mean = f(np.mean(x, axis=0))
    def outer_sqr(a):
        return np.outer(a,a)
    return np.sum(np.array([outer_sqr(f_samples[j] - f_mean) for j in range(N)]), axis=0) * (N-1) / N   

################################################ arbitrary functions ################################################

def variance_general(f, x):
    N = len(x)
    f_x = f(x)
    return np.mean([ (f(np.delete(x, k, axis=0)) - f_x)**2 for k in range(N)], axis=0) * (N - 1)
        
def covariance_general(f, x):
    N = len(x)
    f_x = f(x)
    def outer_prod(a):
        return np.outer(a,a)
    return np.mean([outer_prod( (f(np.delete(x, k, axis=0)) - f_x) ) for k in range(N)], axis=0) * (N - 1) 