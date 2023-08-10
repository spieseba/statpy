#!/usr/bin/env python3

import os, h5py
import numpy as np
from statpy.database import custom_json as json
from .leafs import Leaf

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

class MeasureIO:
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

class DatabaseIO:
    def __init__(self):
        pass

    def load_measurement(self, src, tag):
        with open(src) as f:
            data = json.load(f)
        return data[tag]
    
    def create_SAMPLE_DB(self, src_dir, src_tags, branch_tag, leaf_prefix, filter_str=None, dst_tags=None, dst=None):
        filenames = sorted([os.path.join(src_dir, f) for f in os.listdir(src_dir) if (os.path.isfile(os.path.join(src_dir, f)))], key=lambda x: int(x.split("ckpoint_lat.")[-1].split(".")[0]))
        if filter_str is not None:
            filenames = [x for x in filenames if filter_str in x]
        cfgs = [branch_tag + "-" + x.split("ckpoint_lat.")[-1].split(".")[0] for x in filenames]
        if dst_tags is None:
            dst_tags = src_tags
        database = {}
        for stag, dtag in zip(src_tags, dst_tags):
            print(f"store {stag} as {dtag} in database {dst}")
            sample = {}
            for cfg, f in zip(cfgs, filenames):
                sample[cfg] = self.load_measurement(f, stag)
            database[f"{leaf_prefix}/{dtag}"] = Leaf(mean=None, jks=None, sample=sample)
        if dst is None:
            return database
        with open(dst, "w") as f:
            json.dump(database, f)

    def create_SAMPLE_DB_from_CLS(self, data_path, rwf_path, src_tags, leaf_prefix, dst=None):
        assert os.path.isfile(data_path)
        assert os.path.isfile(rwf_path)
        rwf_cfgs = np.array(np.loadtxt(rwf_path)[:,0] - 1, dtype=int)
        rwf = np.loadtxt(rwf_path)[:,1]; nrwf = rwf / np.mean(rwf)
        nrwf = {f"{leaf_prefix}-{cfg}":val for cfg, val in zip(rwf_cfgs, nrwf)}    
        f = h5py.File(data_path, "r")
        f_cfgs = np.array([int(cfg.decode("ascii").split("n")[1]) for cfg in f['configlist']]) - 1; f_cfgs = f_cfgs[rwf_cfgs]
        database = {}
        for stag in src_tags:
            for key in f.keys():
                if stag in key:
                    sample = {f"{leaf_prefix}-{cfg}":val for cfg,val in zip(f_cfgs, np.array(f.get(key))[f_cfgs])}
                    database[f"{leaf_prefix}/{key}"] = Leaf(mean=None, jks=None, sample=sample, nrwf=nrwf)
        if dst is None:
            return database
        with open(dst, "w") as file:
            json.dump(database, file)