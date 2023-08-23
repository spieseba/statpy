#!/usr/bin/env python3

import numpy as np
from ..database.leafs import Leaf


class LatticeCharmBootstrap():
    def __init__(self, path):
        self.path = path
        self.bootstraps, self.configlist = self.get_bootstraps(self.path)

    def __call__(self, db, tag, nrwf_tag, bootstraps_tag, s=1.0):
        db.database[bootstraps_tag] = Leaf(self.bootstraps, None, None, misc={"configlist": self.configlist})
        lf = self.database[tag]; sample = np.array([lf.sample[cfg] for cfg in self.configlist])
        nrwf_lf = self.database[nrwf_tag]; nrwf = np.array([nrwf_lf.sample[cfg] for cfg in self.configlist])
        self.bss = self.compute_bss(sample, self.bootstraps, nrwf)
        self.rescaled_bss = self.rescale_bss(self.bss, s)
        self.avg = np.mean(self.rescaled_bss, axis=0)
        return self.avg, self.rescaled_bss

    def get_bootstraps(self, path):
        def get_line(path, n):
            with open(path) as f:
                for i, line in enumerate(f):
                    if i==n:
                        return line
            return None
        bootstraps = np.loadtxt(path, dtype=int)
        configlist = get_line(path, 3).replace("n", "-").split(" ")[1:]
        configlist[-1] = configlist[-1].replace("\n", "")
        return bootstraps, configlist

    def compute_bss(self, sample, bootstraps, *argv):
        B = bootstraps.shape[0]; I = sample.shape[1]
        nrwf = None
        if len(argv) != 0:
            nrwf = argv[0]
        bss = np.zeros((B,I))
        for b in range(B):
            bs = bootstraps[b]
            nrwf_bs = None
            if nrwf is not None:
                nrwf_bs = nrwf[bs] / np.mean(nrwf[bs])
            bss[b] = np.average(sample[bs], axis=0, weights=nrwf_bs)
        return bss

    def rescale_bss(self, bss, s):
        mean = np.mean(bss, axis=0)
        return mean[None,:] + s * (bss - mean[None,:])