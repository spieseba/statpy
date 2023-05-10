#!/usr/bin/env python3

import os, h5py
from typing import Any
import numpy as np
from statpy.dbpy import custom_json as json

try:
    import gpt as g
    def rank_func():
        return g.rank()
    def barrier():
        return g.barrier()
except ImportError:
    def rank_func():
        return 0
    def barrier():
        pass

class IO:
    def __init__(self, path, filename):
        self.src = path
        self.dst = path + filename
        if rank_func() == 0:
            if os.path.isfile(self.dst):
                with open(self.dst) as f:
                    self.data = json.load(f)
            else:
                self.data = {}
    def store(self, key, value):
        if rank_func() == 0:
            if self.data is not None:
                    self.data[key] = value
    def save(self):
        if rank_func() == 0:
            with open(self.dst, "w") as f:
                json.dump(self.data, f)
        barrier()


class CLS_IO:
    def __init__(self, data_path, rwf_path, obs_tags, ensemble_label):
        assert os.path.isfile(data_path)
        assert os.path.isfile(rwf_path)
        self.lfs = []
        for obs_tag in obs_tags:
            tag = f"{ensemble_label}/{obs_tag}" 
            sample = np.array(h5py.File(data_path, "r").get(obs_tag)[:])
            sample = {f"{ensemble_label}-{cfg}":val for cfg, val in enumerate(sample)}
            rwf = np.loadtxt(rwf_path)[:,1]; nrwf = rwf / np.mean(rwf)
            nrwf = {f"{ensemble_label}-{cfg}":val for cfg, val in enumerate(nrwf)} 
            self.lfs.append((tag, None, None, sample, nrwf, None))

    def __call__(self):
        return self.lfs