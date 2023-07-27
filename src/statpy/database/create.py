#!/usr/bin/env python3

import os, h5py
import numpy as np
from . import custom_json as json
from .leafs import Leaf

def load(src_file, tag):
    with open(src_file) as f:
        src_data = json.load(f)
    return src_data[tag]

def sample_db(src_dir, src_tags, dst, branch_tag, leaf_prefix, meas_postfix="", dst_tags=None):
    filenames = sorted([os.path.join(src_dir, f) for f in os.listdir(src_dir) if (os.path.isfile(os.path.join(src_dir, f)) and meas_postfix in f.split("."))], key=lambda x: int(x.split("ckpoint_lat.")[-1].split(".")[0]))
    cfgs = [branch_tag + "-" + x.split("ckpoint_lat.")[-1].split(".")[0] for x in filenames]
    database = {}
    if dst_tags == None: dst_tags = src_tags
    for src_tag, dst_tag in zip(src_tags, dst_tags):
        print(f"reading src_tag={src_tag}")
        sample = {}
        for cfg, f in zip(cfgs, filenames):
            print(f"\tcfg={cfg}")
            sample[cfg] = load(f, src_tag)
        database[leaf_prefix + "/" + dst_tag] = Leaf(mean=None, jks=None, sample=sample)
    with open(dst, "w") as f:
        json.dump(database, f)

def cls_sample_db(data_path, rwf_path, src_tags, leaf_prefix, dst=None):
    assert os.path.isfile(data_path)
    assert os.path.isfile(rwf_path)
    database = {}
    for tag in src_tags:
        for key in h5py.File(data_path, "r").keys():
            if tag in key:
                sample = {f"{leaf_prefix}-{cfg}":val for cfg, val in enumerate(np.array(h5py.File(data_path, "r").get(key)[:]))}
                rwf = np.loadtxt(rwf_path)[:,1]; nrwf = rwf / np.mean(rwf) 
                nrwf = {f"{leaf_prefix}-{cfg}":val for cfg, val in enumerate(nrwf)}   
                database[f"{leaf_prefix}/{key}"] = Leaf(mean=None, jks=None, sample=sample, nrwf=nrwf)
    if dst == None:
        return database
    with open(dst, "w") as f:
        json.dump(database, f)