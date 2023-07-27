#!/usr/bin/env python3

import numpy as np

class bootstrap():
    def __init__(self, x=None, seed=None):
        self.x = x
        self.N = len(x)
        self.seed = seed
        np.random.seed(seed)

    def __call__(self):
        pass

    def sample(self, og_sample):
        assert self.x == None
        n = len(og_sample)
        return og_sample[ np.random.randint(n, size=n) ]

    def variance(self, f, K):
        f_arr = np.array([ f(self.x[np.random.randint(self.N, size=self.N)]) for k in range(K) ])
        f_arr_mean = np.mean(f_arr, axis=0)
        return np.mean( [ (f_arr[k] - f_arr_mean)**2 for k in range(K)])

    def variance_input(self, samples, f=lambda y: y): 
        assert self.x == None
        K = len(samples)
        f_arr = np.array([f(sample) for sample in samples])
        f_arr_mean = np.mean(f_arr, axis=0)
        return np.mean( [ (f_arr[k] - f_arr_mean)**2  for k in range(K) ] )