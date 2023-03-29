#!/usr/bin/env python3

import os, re, ast
import numpy as np
from statpy.dbpy import np_json as json
from statpy.dbpy.core import query_yes_no

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

def load(src_file, tag):
    with open(src_file) as f:
        src_data = json.load(f)
    return src_data[tag]

def create_db(src_dir, meas_tag, src_tags, dst, dst_tags, ensemble_tag):
    if os.path.isfile(dst):
        if query_yes_no(f"file {dst} already exists. Overwrite?"):
            os.remove(dst)
        else:        
            exit()
    # get filenames and cfgs
    filenames = sorted([os.path.join(src_dir, f) for f in os.listdir(src_dir) if (os.path.isfile(os.path.join(src_dir, f)) and meas_tag in f.split("."))], key=lambda x: int(x.split("ckpoint_lat.")[-1].split(".")[0]))
    cfgs = [int(x.split("ckpoint_lat.")[-1].split(".")[0]) for x in filenames]
    #ensemble = {cfg:f for cfg,f in zip(cfgs,filenames)}
    assert len(cfgs) == len(set(cfgs)), f"meas_tag='{meas_tag}' does not yield a unique set of measurements"
    # create db
    db = {dst_tag: {ensemble_tag: {"sample": {}}} for dst_tag in dst_tags}
    for src_tag, dst_tag in zip(src_tags, dst_tags):
        print(f"reading src_tag={src_tag}")
        for cfg, f in zip(cfgs, filenames):
            print(f"\tcfg={cfg}")
            db[dst_tag][ensemble_tag]["sample"][cfg] = load(f, src_tag)
    db["cfgs"] = {ensemble_tag: cfgs}
    with open(dst, "w") as f:
        json.dump(db, f)

#def convert(src, dst, src_tags, dst_tags, cfg_prefix="", verbose=False):
#    if os.path.isfile(dst):
#        if query_yes_no(f"file {dst} already exists. Overwrite?"):
#            pass
#        else:        
#            exit()
#    database = {}
#    for dst_tag in dst_tags:
#        database[dst_tag] = {}
#    ensemble = [os.path.join(src, f) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
#    ensemble.sort(key=lambda f: int(re.sub('\D', '', f)))
#    for cfg in ensemble:
#        if verbose:
#            print(cfg)
#        with open(cfg) as f:
#            data = json.load(f)
#        for src_tag, dst_tag in zip(src_tags, dst_tags):
#            database[dst_tag][cfg_prefix + cfg.split(".")[-1]] = data[src_tag]
#    with open(dst, "w") as f:
#        json.dump(database, f)

#def onvert_old_to_new(src, src_tags, dst_tags):
#    ensemble = [os.path.join(src, f) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
#    ensemble.sort(key=lambda f: int(re.sub('\D', '', f)))
#    for cfg in ensemble:
#        print(cfg)
#        database = {}
#        with open(cfg) as f:
#            data = json.load(f)
#        for src_tag, dst_tag in zip(src_tags, dst_tags):
#            database[dst_tag] = np.array(ast.literal_eval(data[src_tag]))
#        with open(cfg, "w") as f:
#            json.dump(database, f)
#
#def convert_from_old_format(src, dst, src_tags, dst_tags, cfg_prefix="", verbose=False):
#    if os.path.isfile(dst):
#        if query_yes_no(f"file {dst} already exists. Overwrite?"):
#            pass
#        else:        
#            exit()
#    else:
#        database = {}
#    for dst_tag in dst_tags:
#        database[dst_tag] = {}
#    ensemble = [os.path.join(src, f) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
#    # remove ,json
#    ensemble = [ x for x in ensemble if ".json" not in x ]
#    ensemble.sort(key=lambda f: int(re.sub('\D', '', f)))
#    for cfg in ensemble:
#        if verbose:
#            print(cfg)
#        with open(cfg) as f:
#            data = json.load(f)
#        for src_tag, dst_tag in zip(src_tags, dst_tags):
#            database[dst_tag][cfg_prefix + cfg.split(".")[-1]] = np.array(ast.literal_eval(data[src_tag]))
#    with open(dst, "w") as f:
#        json.dump(database, f)
