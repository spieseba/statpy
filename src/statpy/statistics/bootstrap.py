#!/usr/bin/env python3

import numpy as np
from ..log import message
from ..database.leafs import Leaf

# compute bootstrap sample from sample x with bootstraps and function f
def sample(f, x, bootstraps, *argv):
    B = bootstraps.shape[0]; I = x.shape[1]
    nrwf = None
    if len(argv) != 0:
        nrwf = argv[0]
    bss = np.zeros((B,I))
    for b in range(B):
        bs = bootstraps[b]
        nrwf_bs = None
        if nrwf is not None:
            nrwf_bs = nrwf[bs] / np.mean(nrwf[bs])
        bss[b] = f(np.average(x[bs], axis=0, weights=nrwf_bs))
    return bss

def variance_bss(bss, mean=None):
    if mean is None: mean = np.mean(bss, axis=0)
    B = len(bss)
    return np.sum(np.array([(bss[b] - mean)**2 for b in range(B)]), axis=0) / (B-1) 

def covariance_bss(bss, mean=None):
    if mean is None: mean = np.mean(bss, axis=0)
    B = len(bss)
    def outer_sqr(a):
        return np.outer(a,a)
    return np.sum(np.array([outer_sqr(bss[b] - mean) for b in range(B)]), axis=0) / B 

def rescale_bss(bss, s):
    mean = np.mean(bss, axis=0)
    if isinstance(np.mean(bss, axis=0), np.float64):
        return mean + s * (bss - mean) 
    return mean[None,:] + s * (bss - mean[None,:])


class LatticeCharmBootstrap():
    def __init__(self, bs_fn, db, bootstrap_tag):
        self.db = db
        self.bs_fn = bs_fn
        if self.bs_fn is not None:
            self.bootstraps, self.configlist = self.get_bootstraps(self.bs_fn)
            self.bootstrap_tag = bootstrap_tag
            self.db.database[self.bootstrap_tag] = Leaf(self.bootstraps, None, None, misc={"configlist": self.configlist})
        else:
            message(f"NO BOOTSTRAP FILE AVAILABLE - SAMPLES ARE NOT DETERMINED")

    def __call__(self, tag, nrwf_tag, check_nrwf=True):
        if self.bs_fn is None:
            pass
        else:
            lf = self.db.database[tag]; x = np.array([lf.sample[cfg] for cfg in self.configlist])
            if nrwf_tag is not None:
                nrwf_lf = self.db.database[nrwf_tag]; nrwf = np.array([nrwf_lf.sample[cfg] for cfg in self.configlist])
                bss = sample(lambda y: y, x, self.bootstraps, nrwf)
            else:
                if check_nrwf: message(f"!NRWF FOR BOOTSTRAP OF {tag} NOT FOUND!")
                bss = sample(lambda y: y, x, self.bootstraps)
            if lf.misc is None:
                lf.misc = {"bss": bss}
            else:
                lf.misc["bss"] = bss

    def get_bootstraps(self, bs_fn):
        def get_line(bs_fn, n):
            with open(bs_fn) as f:
                for i, line in enumerate(f):
                    if i==n:
                        return line
            return None
        bootstraps = np.loadtxt(bs_fn, dtype=int)
        configlist = get_line(bs_fn, 3).replace("n", "-").split(" ")[1:]
        configlist[-1] = configlist[-1].replace("\n", "")
        return bootstraps, configlist