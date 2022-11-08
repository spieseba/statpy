import numpy as np


class jackknife():
    def __init__(self, x):
        self.x = x
        self.N = len(x)
    
    def __call__(self):
        pass

    def sample(self, idx):
        return np.delete(self.x, idx, axis=0)

    # function of means
    def variance(self, f):
        x_mean = np.mean(self.x, axis=0)
        f_mean = f(x_mean)
        return np.mean([ ( f( (self.N * x_mean - self.x[k]) / (self.N - 1) ) - f_mean )**2 for k in range(self.N) ], axis=0) * (self.N - 1)

    # function of means
    def covariance(self, f):
        x_mean = np.mean(self.x, axis=0)
        f_mean = f(x_mean)
        def outer_prod(a):
            return np.outer(a,a)
        return np.mean( [outer_prod( f( (self.N * x_mean - self.x[k]) / (self.N - 1) ) - f_mean  ) for k in range(self.N)], axis=0)  * (self.N - 1)        

    def variance2(self, f):
        f_x = f(self.x)
        return np.mean([ (f(np.delete(self.x, k, axis=0)) - f_x)**2 for k in range(self.N)], axis=0) * (self.N - 1)

    def covariance2(self, f, return_samples=False):
        f_x = f(self.x)
        def outer_prod(a):
            return np.outer(a,a)
        if return_samples:
            return np.array([outer_prod( (f(np.delete(self.x, k, axis=0)) - f_x) ) for k in range(self.N)]) * (self.N - 1)
        return np.mean([outer_prod( (f(np.delete(self.x, k, axis=0)) - f_x) ) for k in range(self.N)], axis=0) * (self.N - 1) 