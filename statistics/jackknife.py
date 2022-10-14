import numpy as np


class jackknife():
    def __init__(self, x):
        self.x = x
        self.N = len(x)
    
    def __call__(self):
        pass

    def sample(self, idx):
        return np.delete(self.x, idx)
    
    def variance(self, f):
        f_x = f(self.x)
        return np.mean([ (f(np.delete(self.x, k, axis=0)) - f_x)**2 for k in range(self.N)], axis=0) * (self.N - 1)

    # fast version for functions of means
    def variance_mean(self, f):
        x_mean = np.mean(self.x, axis=0)
        f_mean = f(x_mean)
        return np.mean([ ( f( (self.N * x_mean - self.x[k]) / (self.N - 1) ) - f_mean )**2 for k in range(self.N) ], axis=0) * (self.N - 1)

    # fast version for functions of means
    def covariance_mean(self, f):
        x_mean = np.mean(self.x)
        f_mean = f(x_mean)
        def outer_prod(a):
            return np.outer(a,a)
        return np.mean( outer_prod( f( (self.N * x_mean - self.x[k]) / (self.N - 1) ) - f_mean  ) for k in range(self.N) )  * (self.N - 1)        
