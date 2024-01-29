#!/usr/bin/env python3

import os, copy
import numpy as np
from time import time
from functools import reduce
from operator import ior
from ..log import message
from . import custom_json as json
from .leafs import Leaf 
from ..statistics import core as statistics
from ..statistics import jackknife
# import multiprocessing module and overwrite its Pickle class using dill
import dill, multiprocessing
dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads
multiprocessing.reduction.ForkingPickler = dill.Pickler
multiprocessing.reduction.dump = dill.dump

##############################################################################################################################################################
##############################################################################################################################################################
###################################### DATABASE SYSTEM USING LEAFS CONTAINING MEAN AND JKS (SECONDARY OBSERVABLES) ###########################################
##############################################################################################################################################################
##############################################################################################################################################################

class DB:
    def __init__(self, *args, num_proc=None, verbosity=0):
        self.t0 = time()
        self.num_proc = num_proc
        self.verbosity = verbosity
        self.database = {}
        for src in args:
            # init db using src files
            if isinstance(src, str):
                self.add_src(src)
            # init db using src dict
            if isinstance(src, dict):
                for t, lf in src.items():
                    self.database[t] = lf

    def add_src(self, *srcs):
        for src in srcs:
            assert os.path.isfile(src)
            message(f"LOAD {src}")
            with open(src) as f:
                src_db = json.load(f)
            for t, lf in src_db.items():
                self.database[t] = Leaf(lf.mean, lf.jks, lf.sample, lf.misc)

    def add_Leaf(self, tag, mean, jks, sample, misc):
        assert (isinstance(sample, dict) or sample==None)
        assert (isinstance(jks, dict) or jks==None)
        assert (isinstance(misc, dict) or misc==None)
        self.database[tag] = Leaf(mean, jks, sample, misc)

    def rename_Leaf(self, old, new):
        if old in self.database:
            self.database[new] = self.database.pop(old)
 
    def save(self, dst):
        with open(dst, "w") as f:
            json.dump(self.database, f)

    def print(self, key="", verbosity=0):
        message(self.__str__(key, verbosity))    

    def __str__(self, key, verbosity):
        s = '\n\n\tDATABASE CONSISTS OF\n\n'
        for tag, lf in self.database.items():
            if key in tag:
                s += f'\t{tag:20s}\n'
                if verbosity >= 1:
                    if np.array(lf.mean).any() != None:
                        s += f'\t└── mean\n'
                    if np.array(lf.jks).any() != None:
                        s += f'\t└── jks\n'
                    if np.array(lf.sample).any() != None:
                        s += f'\t└── sample\n' 
                    if lf.misc != None:
                        s += f'\t└── misc\n'
        return s

    def print_misc(self, tag):
        message(self.__misc_str__(tag))

    def __misc_str__(self, tag):
        s = f'\n\n\tMISC DICT OF {tag}\n\n'
        for k, i in self.database[tag].misc.items():
            s += f'\t{k:20s}: {i}\n'
        return s
    
    def remove(self, *tags, verbosity=-1):
        for tag in tags:
            try:
                del self.database[tag]
            except KeyError:
                message(f"{tag} not in database", verbosity)

    def get_tags(self, key="", verbosity=0):
        return [tag for tag in self.database.keys() if key in tag]

    # helper function
    def as_array(self, obj, sorting_key=lambda x: int(x[0].split("-")[-1])):
        if isinstance(obj, np.ndarray):
            return obj
        sorted_obj = dict(sorted(obj.items(), key=sorting_key))
        return np.array(list(sorted_obj.values()))
    
    ################################## FUNCTIONS #######################################

    def combine_sample(self, *tags, f=lambda x: x, dst_tag=None, sorting_key=None):
        lfs = [self.database[tag] for tag in tags]
        f_sample = {}
        cfgs = np.unique([list(lf.sample.keys()) for lf in lfs]) 
        for cfg in cfgs:
            x = [lf.sample[cfg] if cfg in lf.sample else lf.mean for lf in lfs]
            f_sample[cfg] = f(*x) 
        f_sample = dict(sorted(f_sample.items(), key=sorting_key)) 
        if dst_tag is None:
            return Leaf(None, None, f_sample)
        self.database[dst_tag] = Leaf(None, None, f_sample)

    def concatenate_samples(self, *tags, dst_tag=None, sorting_key=None, dst_cfgs=None):
        lfs = [self.database[tag] for tag in tags]
        if dst_cfgs is None:
            sample = dict(sorted(reduce(ior, [lf.sample for lf in lfs], {}).items(), key=sorting_key))
        else:
            sample = {cfg:val for cfg,val in zip(dst_cfgs, np.concatenate([self.as_array(lf.sample) for lf in lfs], axis=0))}
        if dst_tag is None:
            return Leaf(None, None, sample)
        else:
            self.database[dst_tag] = Leaf(None, None, sample)
    
    def compute_nrwf(self, tag):
        rwf = self.database[tag].sample; n = np.mean(self.as_array(rwf, None))
        self.add_Leaf(tag.replace("rwf","nrwf"), None, None, {cfg:rwf/n for cfg,rwf in rwf.items()}, None)
    
    def get_nrwf(self, tag):
        lf = self.database.get(f"{tag.split('/')[0]}/nrwf") 
        if (lf == None) or ("rwf" in tag):
            return None
        return lf.sample 
    
    def init_sample_means(self, *tags):
        if len(tags) == 0:
            tags = self.database.keys()
        for tag in tags:
            lf = self.database[tag]
            nrwf = self.get_nrwf(tag)
            lf.mean = np.mean(self.as_array(nrwf)[:,None] * self.as_array(lf.sample), axis=0)

    def init_sample_jks(self, *tags, binsizes=[1]):
        if len(tags) == 0:
            tags = self.database.keys()
        for tag in tags:
            lf = self.database[tag]
            if lf.sample is None: continue
            lf.jks = {}
            for b in binsizes:
                lf.jks = self.jks(tag, b)
    
    def cfgs(self, tag):
        return sorted([int(x.split("-")[-1]) for x in self.database[tag].sample.keys()])
 
    def remove_cfgs(self, tag, cfgs, dst_tag=None):
        sample = copy.deepcopy(self.database[tag].sample)
        for cfg in cfgs:
            sample.pop(str(cfg), None)
        if dst_tag is None:
            self.database[tag].sample = sample
        else:
            self.add_Leaf(dst_tag, None, None, sample, None)

    ################################## STATISTICS ######################################

    def jks(self, tag, binsize, sorting_key=lambda x: int(x[0].split("-")[-1])):
        lf = self.database[tag]
        nrwf = self.get_nrwf(tag)
        bsample = statistics.bin(self.as_array(lf.sample, sorting_key=sorting_key), binsize, self.as_array(nrwf, sorting_key=sorting_key))
        bnrwf = statistics.bin(self.as_array(nrwf, sorting_key=sorting_key), binsize)
        jks = jackknife.sample(bsample, bnrwf[:, None])
        return jks
    
    def jackknife_variance(self, tag, binsize):
        jks = self.jks(tag, binsize)
        return jackknife.variance(jks)

    def jackknife_covariance(self, tag, binsize):
        jks = self.jks(tag, binsize)
        return jackknife.covariance(jks)
    
    def binning_study(self, tag, binsizes):
        message(f"Unbinned sample size: {len(self.database[tag].sample)}")
        var = {}
        for b in binsizes:
            var[b] = self.jackknife_variance(tag, b)
        return var
