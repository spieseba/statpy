#!/usr/bin/env python3

import binascii, h5py, os, re, struct, sys
import numpy as np
from statpy.database import custom_json as json
from .leafs import Leaf

def load_CLS(fn, rwf_fn, tags, branch_tag):
    assert os.path.isfile(fn), f"{fn} not found!"
    # data
    f = h5py.File(fn, "r")
    f_cfgs = np.array([int(cfg.decode("utf-8").split("n")[1]) for cfg in f.get("configlist")])
    measurements = {}
    # rwfs
    if rwf_fn is None:
        cfgs = f_cfgs 
        rwf = {f"{branch_tag}-{cfg}":1.0 for cfg in f_cfgs}
    else:
        assert os.path.isfile(rwf_fn) 
        rwf_cfgs = np.array(np.loadtxt(rwf_fn)[:,0], dtype=int)
        rwf = np.loadtxt(rwf_fn)[:,1] 
        rwf = {f"{branch_tag}-{cfg}":val for cfg, val in zip(rwf_cfgs, rwf)} 
        cfgs = np.array([cfg for cfg in rwf_cfgs if cfg in f_cfgs])
    measurements[f"{branch_tag}/rwf"] = Leaf(mean=None, jks=None, sample=rwf)
    # data
    for t in tags:
        for key in f.keys():
            if t in key:
                sample = {f"{branch_tag}-{cfg}":val for cfg, val in zip(cfgs, f.get(key)[cfgs-1])}
                measurements[f"{branch_tag}/{key}"] = Leaf(mean=None, jks=None, sample=sample)
    return measurements