#!/usr/bin/env python3

import os, re, ast
import dbpy.np_json as json
import numpy as np

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

def convert(src, dst, src_tags, dst_tags, cfg_prefix="", verbose=False):
    if os.path.isfile(dst):
        print(f"file {dst} already exists. Delete or rename it")
        exit()
    database = {}
    for dst_tag in dst_tags:
        database[dst_tag] = {}
    ensemble = [os.path.join(src, f) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
    ensemble.sort(key=lambda f: int(re.sub('\D', '', f)))
    for cfg in ensemble:
        if verbose:
            print(cfg)
        with open(cfg) as f:
            data = json.load(f)
        for src_tag, dst_tag in zip(src_tags, dst_tags):
            database[dst_tag][cfg_prefix + cfg.split(".")[-1]] = data[src_tag]
    with open(dst, "w") as f:
        json.dump(database, f)

def convert_from_old_format(src, dst, src_tags, dst_tags, cfg_prefix="", verbose=False):
    if os.path.isfile(dst):
        print(f"file {dst} already exists. Delete or rename it")
        exit()
    database = {}
    for dst_tag in dst_tags:
        database[dst_tag] = {}
    ensemble = [os.path.join(src, f) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
    ensemble.sort(key=lambda f: int(re.sub('\D', '', f)))
    for cfg in ensemble:
        if verbose:
            print(cfg)
        with open(cfg) as f:
            data = json.load(f)
        for src_tag, dst_tag in zip(src_tags, dst_tags):
            database[dst_tag][cfg_prefix + cfg.split(".")[-1]] = np.array(ast.literal_eval(data[src_tag]))
    with open(dst, "w") as f:
        json.dump(database, f)