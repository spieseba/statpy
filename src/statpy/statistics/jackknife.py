#!/usr/bin/env python3

import numpy as np

################################################ functions of means ################################################

def sample(f, x, *argv): 
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

def variance_jks(mean, jks):
    N = len(jks)
    return np.sum(np.array([(jks[j] - mean)**2 for j in range(N)]), axis=0) * (N-1) / N  

def covariance_jks(mean, jks):
    N = len(jks)
    def outer_sqr(a):
        return np.outer(a,a)
    return np.sum(np.array([outer_sqr(jks[j] - mean) for j in range(N)]), axis=0) * (N-1) / N  

def variance(f, x, *argv):
    N = len(x)
    mean = np.mean(x, axis=0)
    f_mean = f(mean)
    if len(argv) != 0:
        f_samples = argv[0]
        N = len(f_samples)
        return np.sum(np.array([(f_samples[j] - f_mean)**2 for j in range(N)]), axis=0) * (N-1) / N  
    return np.mean([ ( f( (N * mean - x[k]) / (N - 1) ) - f_mean )**2 for k in range(N) ], axis=0) * (N-1)

def covariance(f, x, *argv):
    N = len(x)
    mean = np.mean(x, axis=0)
    f_mean = f(mean)
    def outer_sqr(a):
        return np.outer(a,a)
    if len(argv) != 0:
        f_samples = argv[0]
        return np.sum(np.array([outer_sqr(f_samples[j] - f_mean) for j in range(N)]), axis=0) * (N-1) / N 
    return np.mean( [outer_sqr( f( (N * mean - x[k]) / (N - 1) ) - f_mean  ) for k in range(N)], axis=0) * (N-1) 
      
################################################ arbitrary functions ################################################

def variance_general(f, x, data_axis=0):
    N = len(x)
    f_x = f(x)
    return np.mean([ (f(np.delete(x, k, axis=data_axis)) - f_x)**2 for k in range(N)], axis=0) * (N-1)
        
def covariance_general(f, x, data_axis=0):
    N = len(x)
    f_x = f(x)
    def outer_sqr(a):
        return np.outer(a,a)
    return np.mean([outer_sqr( (f(np.delete(x, k, axis=data_axis)) - f_x) ) for k in range(N)], axis=0) * (N-1) 
