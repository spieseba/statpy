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

    ################################ JKS ######################################

    def combine(self, *tags, f=lambda x: x, dst_tag=None, num_proc=None):
        if num_proc is None: num_proc = self.num_proc
        mean = self.combine_mean(*tags, f=f)
        jks = self.combine_jks(*tags, f=f, num_proc=num_proc)
        if dst_tag is None:
            return Leaf(mean, jks, None)
        self.database[dst_tag] = Leaf(mean, jks, None)

    def combine_mean(self, *tags, f=lambda x: x, dst_tag=None):
        lfs = [self.database[tag] for tag in tags]
        mean = f(*[lf.mean for lf in lfs])
        if dst_tag is None:
            return mean
        try:
            self.database[dst_tag].mean = mean
        except KeyError:
            self.database[dst_tag] = Leaf(mean, None, None)

    def combine_jks(self, *tags, f=lambda x: x, dst_tag=None, num_proc=None):
        if num_proc is None: num_proc = self.num_proc
        lfs = [self.database[tag] for tag in tags]
        cfgs = np.unique(np.concatenate([list(lf.jks.keys()) for lf in lfs]))
        xs = {cfg:[lf.jks[cfg] if cfg in lf.jks else lf.mean for lf in lfs] for cfg in cfgs}
        if num_proc is None:
            jks = {cfg:f(*x) for cfg,x in xs.items()}
        else:
            def wrapped_f(cfg, *x):
                return cfg, f(*x)
            message(f"Spawn {num_proc} processes to compute jackknife sample", verbosity=self.verbosity-1)
            with multiprocessing.Pool(num_proc) as pool:
                jks = dict(pool.starmap(wrapped_f, [(cfg, *x) for cfg,x in xs.items()]))
        if dst_tag is None:
            return jks
        try:
            self.database[dst_tag].jks = jks
        except KeyError:
            self.database[dst_tag] = Leaf(None, jks, None)

    ############################### SAMPLE ####################################
            
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

    def merge_sample(self, *tags, dst_tag=None, sorting_key=None, dst_cfgs=None):
        lfs = [self.database[tag] for tag in tags]
        if dst_cfgs is None:
            sample = dict(sorted(reduce(ior, [lf.sample for lf in lfs], {}).items(), key=sorting_key))
        else:
            sample = {cfg:val for cfg,val in zip(dst_cfgs, np.concatenate([self.as_array(lf.sample) for lf in lfs], axis=0))}
        if dst_tag is None:
            return Leaf(None, None, sample)
        else:
            self.database[dst_tag] = Leaf(None, None, sample)
    
    def init_sample_means(self, *tags):
        if len(tags) == 0:
            tags = self.database.keys()
        for tag in tags:
            lf = self.database[tag]
            if lf.sample is not None:
                lf.mean = np.mean(self.as_array(lf.sample), axis=0)

    def init_sample_jks(self, *tags):
        if len(tags) == 0:
            tags = self.database.keys()
        for tag in tags:
            lf = self.database[tag]
            if lf.sample is None: continue
            jks = {}
            for cfg in lf.sample:
                jks[cfg] = lf.mean + (lf.mean - lf.sample[cfg]) / (len(lf.sample) - 1)
            lf.jks = jks
 
    def remove_cfgs(self, tag, cfgs, dst_tag=None):
        sample = copy.deepcopy(self.database[tag].sample)
        for cfg in cfgs:
            sample.pop(str(cfg), None)
        if dst_tag is None:
            self.database[tag].sample = sample
        else:
            self.add_Leaf(dst_tag, None, None, sample, None)

    def cfgs(self, tag):
        return sorted([int(x.split("-")[-1]) for x in self.database[tag].sample.keys()])
     
    ################################## STATISTICS ######################################
    
    def delayed_binning(self, lf, binsize, branch_tag="", shift=0):
        jks_tags = sorted([t for t in list(lf.jks.keys()) if branch_tag in t], key=lambda x: int(x.split("-")[-1]))
        N = len(jks_tags)
        Nb = N // binsize 
        jks_tags = jks_tags[:Nb*binsize] # cut off excess data
        jks_bin = {}
        for i in range(Nb):
            s = sum([lf.jks[np.roll(jks_tags, shift)[idx]] for idx in np.arange(i*binsize, (i+1)*binsize)]) - binsize * lf.mean
            jks_bin[f"{branch_tag}/binsize{binsize}-{i}"] = lf.mean + s * (N-1) / (N-binsize)
        return jks_bin
    
    def jks(self, tag, binsize, shift=0):
        lf = self.database[tag]
        if binsize == 1 or not lf.jks:
            return lf.jks
        branch_tags = np.unique([t.split("-")[0] for t in list(lf.jks.keys())])
        def get_arr(v):
            if isinstance(v, (int, np.int64)):
                vs = len(branch_tags) * [v]
            elif isinstance(v, list):
                vs = v
            assert len(branch_tags) == len(vs)
            return vs
        binsizes = get_arr(binsize)
        shifts = get_arr(shift)
        jks_bin = {}
        for b,branch_tag,s in zip(binsizes, branch_tags, shifts):
            jks_bin.update(self.delayed_binning(lf, b, branch_tag, s))
        return jks_bin
 
    def jackknife_variance(self, tag, binsize, pavg=False):
        permutations = np.arange(1)
        if pavg:
            permutations = np.arange(np.min(binsize))
        var = []
        for p in permutations:
            jks = self.as_array(self.jks(tag, binsize, p))
            if len(jks) == 0: return 0.0
            var.append(jackknife.variance(jks))
        return np.mean(var, axis=0)
    
    def jackknife_covariance(self, tag, binsize, pavg=False):
        permutations = np.arange(1)
        if pavg:
            permutations = np.arange(np.min(binsize))
        cov = []
        for p in permutations:
            jks = self.as_array(self.jks(tag, binsize, p))
            cov.append(jackknife.covariance(jks))
        return np.mean(cov, axis=0)

    def binning_study(self, tag, binsizes, pavg=False):
        message(f"Unbinned sample size: {len(self.database[tag].jks)}")
        var = {}
        for b in binsizes:
            var[b] = self.jackknife_variance(tag, b, pavg)
        return var
    
    def AMA(self, exact_exact_tag, exact_sloppy_tag, sloppy_sloppy_tag, dst_tag):
        self.combine(exact_exact_tag, exact_sloppy_tag, f=lambda x,y: x-y, dst_tag=dst_tag+"_bias")
        self.combine(sloppy_sloppy_tag, dst_tag+"_bias", f=lambda x,y: x+y, dst_tag=dst_tag)
    
    ############################### SAMPLE ####################################

    def sample_jks(self, tag, binsize, sorting_key=lambda x: int(x[0].split("-")[-1])):
        lf = self.database[tag]
        if binsize == 1:
            if lf.jks is None:
                self.init_sample_jks(tag)
            jks = self.as_array(lf.jks)
        else:
            bsample = statistics.bin(self.as_array(lf.sample, sorting_key=sorting_key), binsize)
            jks = jackknife.sample(bsample)
        return jks
    
    def sample_jackknife_variance(self, tag, binsize):
        jks = self.sample_jks(tag, binsize)
        return jackknife.variance(jks)

    def sample_jackknife_covariance(self, tag, binsize):
        jks = self.sample_jks(tag, binsize)
        return jackknife.covariance(jks)
    
    def sample_binning_study(self, tag, binsizes):
        message(f"Unbinned sample size: {len(self.database[tag].sample)}")
        var = {}
        for b in binsizes:
            var[b] = self.sample_jackknife_variance(tag, b)
        return var
 
    ############################# SYSTEMATICS #################################

    def get_mean_shifted(self, *tags, f=lambda x: x , sys_tag=None):
        assert sys_tag != None
        x = []
        for tag in tags:
            try: 
                x.append(self.database[tag].misc[f"MEAN_SHIFTED_{sys_tag}"])
            except (TypeError, KeyError):
                x.append(self.database[tag].mean)
        return f(*x)

    def propagate_sys_var(self, mean_shifted, dst_tag, sys_tag=None):
        assert sys_tag != None
        sys_var = (self.database[dst_tag].mean - mean_shifted)**2.
        if self.database[dst_tag].misc is None:
            self.database[dst_tag].misc = {}
        self.database[dst_tag].misc[f"MEAN_SHIFTED_{sys_tag}"] = mean_shifted
        self.database[dst_tag].misc[f"SYS_VAR_{sys_tag}"] = sys_var

    def get_sys_tags(self, *tags):
        sys_tags = []
        for tag in tags:
            if self.database[tag].misc is not None:
                for k in self.database[tag].misc:
                    if "MEAN_SHIFTED" in k:
                        sys_tag = k.split("MEAN_SHIFTED_")[1] 
                        if sys_tag not in sys_tags:
                            sys_tags.append(k.split("MEAN_SHIFTED_")[1])
        return sys_tags

    def get_sys_var(self, tag):
        sys_var = np.zeros_like(self.database[tag].mean)
        if self.database[tag].misc != None:
            for k,v in self.database[tag].misc.items():
                if "SYS_VAR" in k:
                    sys_var += v
        return sys_var
    
    def get_tot_var(self, tag, binsize, pavg=False):
        return self.jackknife_variance(tag, binsize, pavg) + self.get_sys_var(tag)
    
    def print_estimate(self, tag, binsize, pavg=False, verbosity=0):
        s = f"\n ESTIMATE of {tag}:\n"
        s += f"   {self.database[tag].mean} +- {self.get_tot_var(tag, binsize, pavg)**.5} (STAT + SYS)\n"
        if verbosity > 0:
            s += " ERRORS:\n"
            s += f"   {self.jackknife_variance(tag, binsize, pavg)**.5} (STAT)\n"
            for sys_tag in self.get_sys_tags(tag):
                s += f"   {self.database[tag].misc[f'SYS_VAR_{sys_tag}']**.5} (SYS {sys_tag})\n"
        message(s, verbosity)
        
    def get_estimate(self, tag, binsize, pavg=False):
        return self.database[tag].mean, self.get_tot_var(tag, binsize, pavg)
