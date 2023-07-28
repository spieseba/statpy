#!/usr/bin/env python3

import os
import numpy as np
from time import time
from multiprocessing import Process, Manager
from . import custom_json as json
from .leafs import Leaf 
from ..statistics import core as statistics
from ..statistics import jackknife, auto

##############################################################################################################################################################
##############################################################################################################################################################
###################################### DATABASE SYSTEM USING LEAFS CONTAINING MEAN AND JKS (SECONDARY OBSERVABLES) ###########################################
##############################################################################################################################################################
##############################################################################################################################################################

class JKS_DB:
    db_type = "JKS-DB"
    def __init__(self, *args, verbosity=0):
        self.t0 = time()
        self.verbosity = verbosity
        self.database = {}
        for src in args:
            # init db using src files
            if isinstance(src, str):
                self.add_src(src)
            # init db using src dict
            if isinstance(src, dict):
                for t, lf in src.items():
                    if self.db_type == "JKS-DB":
                        self.database[t] = Leaf(lf.mean, lf.jks, None, None, lf.info)
                    elif self.db_type == "SAMPLE-DB":
                        self.database[t] = lf

    def add_src(self, src):
        assert os.path.isfile(src)
        self.message(f"load {src}")
        with open(src) as f:
            src_db = json.load(f)
        for t, lf in src_db.items():
            self.database[t] = Leaf(lf.mean, lf.jks, lf.sample, lf.nrwf, lf.info)

    def add_Leaf(self, tag, mean, jks, sample, nrwf, info):
        assert (isinstance(sample, dict) or sample==None)
        assert (isinstance(jks, dict) or jks==None)
        assert (isinstance(nrwf, dict) or nrwf==None)
        assert (isinstance(info, dict) or info==None)
        self.database[tag] = Leaf(mean, jks, sample, nrwf, info)

    def message(self, s, verbosity=None):
        if verbosity == None: verbosity = self.verbosity
        if verbosity >= 0: print(f"{self.db_type}:\t\t{time()-self.t0:.6f}s: " + s)
    
    def save(self, dst):
        with open(dst, "w") as f:
            json.dump(self.database, f)

    def print(self, key="", verbosity=0):
        self.message(self.__str__(key, verbosity))    

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
                    if np.array(lf.nrwf).any() != None:
                        s += f'\t└── nrwf\n' 
                    if lf.info != None:
                        s += f'\t└── info\n'
        return s

    def print_info(self, tag):
        self.message(self.__info_str__(tag))

    def __info_str__(self, tag):
        s = f'\n\n\tINFO DICT OF {tag}\n\n'
        for k, i in self.database[tag].info.items():
            s += f'\t{k:20s}: {i}\n'
        return s
    
    def remove(self, *tags):
        for tag in tags:
            try:
                del self.database[tag]
            except KeyError:
                print(f"{tag} not in database")

    def get_tags(self, key="", verbosity=0):
        return [tag for tag in self.database.keys() if key in tag]

    # helper function
    def as_array(self, obj):
        if isinstance(obj, np.ndarray):
            return obj
        else:
            return np.array(list(obj.values()))
    
    ################################## FUNCTIONS #######################################

    def combine_mean(self, *tags, f=lambda x: x, dst_tag=None):
        lfs = [self.database[tag] for tag in tags]
        mean = f(*[lf.mean for lf in lfs])
        if dst_tag is None:
            return mean
        try:
            self.database[dst_tag].mean = mean
        except KeyError:
            self.database[dst_tag] = Leaf(mean, None, None)

    def combine_jks(self, *tags, f=lambda x: x, dst_tag=None, processes=1):
        lfs = [self.database[tag] for tag in tags]
        cfgs = np.unique(np.concatenate([list(lf.jks.keys()) for lf in lfs]))
        xs = {cfg:[lf.jks[cfg] if cfg in lf.jks else lf.mean for lf in lfs] for cfg in cfgs}
        if processes > 1:
            def g(d, cfg):
                d[cfg] = f(*xs[cfg])
            m = Manager()
            jks = m.dict()
            job = [Process(target=g, args=(jks, cfg)) for cfg in cfgs]
            _ = [p.start() for p in job]
            _ = [p.join() for p in job]
        else:
            jks = {}
            for cfg, x in xs.items():
                jks[cfg] = f(*x)
        
        if dst_tag is None:
            return jks
        try:
            self.database[dst_tag].jks = jks
        except KeyError:
            self.database[dst_tag] = Leaf(None, jks, None)

    def combine(self, *tags, f=lambda x: x, dst_tag=None, processes=1):
        mean = self.combine_mean(*tags, f=f)
        jks = self.combine_jks(*tags, f=f, processes=processes)
        if dst_tag == None:
            return Leaf(mean, jks, None)
        self.database[dst_tag] = Leaf(mean, jks, None)

    ################################## STATISTICS ######################################

    def jks(self, tag, binsize, shift=0, verbose=False):
        lf = self.database[tag]
        if binsize == 1:
            return lf.jks
        jks_tags = sorted(list(lf.jks.keys()), key=lambda x: (x.split("-")[0], int(x.split("-")[-1])))
        if verbose:
            print(jks_tags)
        N = len(jks_tags)
        Nb = N // binsize 
        jks_tags = jks_tags[:Nb*binsize] # cut off excess data
        jks_bin = {}
        for i in range(Nb):
            s = sum([lf.jks[np.roll(jks_tags, shift)[idx]] for idx in np.arange(i*binsize, (i+1)*binsize)]) - binsize * lf.mean
            jks_bin[i] = lf.mean + s * (N-1) / (N-binsize)
        return jks_bin

    def jackknife_variance(self, tag, binsize, pavg=False):
        permutations = np.arange(1)
        if pavg:
            permutations = np.arange(binsize)
        var = []
        for p in permutations:
            jks = self.as_array(self.jks(tag, binsize, p))
            if len(jks) == 0: return 0.0
            var.append(jackknife.variance_jks(np.mean(jks, axis=0), jks))
        return np.mean(var, axis=0)
    
    def jackknife_covariance(self, tag, binsize, pavg=False):
        permutations = np.arange(1)
        if pavg:
            permutations = np.arange(binsize)
        cov = []
        for p in permutations:
            jks = self.as_array(self.jks(tag, binsize, p))
            cov.append(jackknife.covariance_jks(np.mean(jks, axis=0), jks))
        return np.mean(cov, axis=0)

    def binning_study(self, tag, binsizes, pavg=False):
        self.message(f"Unbinned sample size: {len(self.database[tag].jks)}")
        var = {}
        for b in binsizes:
            var[b] = self.jackknife_variance(tag, b, pavg)
        return var
    
    def AMA(self, exact_exact_tag, exact_sloppy_tag, sloppy_sloppy_tag, dst_tag):
        self.combine(exact_exact_tag, exact_sloppy_tag, f=lambda x,y: x-y, dst_tag=dst_tag+"_bias")
        self.combine(sloppy_sloppy_tag, dst_tag+"_bias", f=lambda x,y: x+y, dst_tag=dst_tag)
 
    ############################# SYSTEMATICS #################################

    def get_mean_shifted(self, *tags, f=lambda x: x , sys_tag=None):
        assert sys_tag != None
        x = []
        for tag in tags:
            try: 
                x.append(self.database[tag].info[f"MEAN_SHIFTED_{sys_tag}"])
            except (TypeError, KeyError):
                x.append(self.database[tag].mean)
        return f(*x)

    def propagate_sys_var(self, mean_shifted, dst_tag, sys_tag=None):
        assert sys_tag != None
        sys_var = (self.database[dst_tag].mean - mean_shifted)**2.
        if self.database[dst_tag].info == None:
            self.database[dst_tag].info = {}
        self.database[dst_tag].info[f"MEAN_SHIFTED_{sys_tag}"] = mean_shifted
        self.database[dst_tag].info[f"SYS_VAR_{sys_tag}"] = sys_var

    def get_sys_tags(self, *tags):
        sys_tags = []
        for tag in tags:
            if self.database[tag].info is not None:
                for k in self.database[tag].info:
                    if "MEAN_SHIFTED" in k:
                        sys_tag = k.split("MEAN_SHIFTED_")[1] 
                        if sys_tag not in sys_tags:
                            sys_tags.append(k.split("MEAN_SHIFTED_")[1])
        return sys_tags

    def get_sys_var(self, tag):
        sys_var = np.zeros_like(self.database[tag].mean)
        if self.database[tag].info != None:
            for k,v in self.database[tag].info.items():
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
                s += f"   {self.database[tag].info[f'SYS_VAR_{sys_tag}']**.5} (SYS {sys_tag})\n"
        self.message(s, verbosity)

###########################################################################################################################################################################
###########################################################################################################################################################################
################################# DATABASE SYSTEM USING LEAFS CONTAINING SAMPLE, RWF, MEAN AND JKS (PRIMARY and SECONDARY OBSERVABLES) #####################################
###########################################################################################################################################################################
###########################################################################################################################################################################

class Sample_DB(JKS_DB):
    db_type = "SAMPLE-DB" 

    ################################## FUNCTIONS MODIFYING THE DATABASE #######################################
    
    def merge_sample(self, *tags, dst_tag=None, dst_cfgs=None):
        assert dst_cfgs != None
        lfs = [self.database[tag] for tag in tags]
        sample = np.concatenate([self.as_array(lf.sample) for lf in lfs], axis=0)
        dst_sample = {}
        for cfg, val in zip(dst_cfgs, sample):
            dst_sample[cfg] = val
        if dst_tag == None:
            return Leaf(None, None, sample)
        else:
            self.database[dst_tag] = Leaf(None, None, dst_sample)
    
    def combine_sample(self, *tags, f=lambda x: x, dst_tag=None):
        lfs = [self.database[tag] for tag in tags]
        f_sample = {}
        cfgs = np.unique([list(lf.sample.keys()) for lf in lfs]) 
        for cfg in cfgs:
            x = [lf.sample[cfg] if cfg in lf.sample else lf.mean for lf in lfs]
            f_sample[cfg] = f(*x) 
        if dst_tag == None:
            return Leaf(None, None, f_sample)
        self.database[dst_tag] = Leaf(None, None, f_sample)

    def init_sample_means(self, *tags):
        if len(tags) == 0:
            tags = self.database.keys()
        for tag in tags:
            lf = self.database[tag]
            if lf.sample != None:
                if lf.nrwf == None:
                    lf.mean = np.mean(self.as_array(lf.sample), axis=0)
                else:
                    lf.mean = np.mean(self.as_array(lf.nrwf)[:,None] * self.as_array(lf.sample), axis=0)

    def init_sample_jks(self, *tags):
        if len(tags) == 0:
            tags = self.database.keys()
        for tag in tags:
            lf = self.database[tag]
            if lf.nrwf == None:
                jks = {}
                for cfg in lf.sample:
                    jks[cfg] = lf.mean + (lf.mean - lf.sample[cfg]) / (len(lf.sample) - 1)
            else:
                jks = {}
                for cfg in lf.sample:
                    jks[cfg] = lf.mean + (lf.mean - lf.sample[cfg]) * lf.nrwf[cfg] / (len(lf.sample) - lf.nrwf[cfg])
            lf.jks = jks
 
    def remove_cfgs(self, tag, cfgs):
        for cfg in cfgs:
            self.database[tag].sample.pop(str(cfg), None)

    ############################### FUNCTIONS NOT MODIFYING THE DATABASE #######################################

    def return_JKS_DB(self):
        return JKS_DB(self.database)

    def cfgs(self, tag):
        return sorted([int(x.split("-")[-1]) for x in self.database[tag].sample.keys()])

    ################################## STATISTICS ######################################

    def sample_jks(self, tag, binsize, f=lambda x: x):
        lf = self.database[tag]
        if binsize == 1:
            if lf.jks == None:
                self.init_sample_jks(tag)
            jks = self.as_array(lf.jks)
        else:
            if lf.nrwf == None:
                bsample = statistics.bin(self.as_array(lf.sample), binsize)
                jks = jackknife.samples(f, bsample)
            else:
                bsample = statistics.bin(self.as_array(lf.sample), binsize, self.as_array(lf.nrwf)); bnrwf = statistics.bin(self.as_array(lf.nrwf), binsize)
                jks = jackknife.samples(f, bsample, bnrwf[:, None])
        return jks
    
    def sample_jackknife_variance(self, tag, binsize, f=lambda x: x):
        jks = self.sample_jks(tag, binsize, f)
        return jackknife.variance_jks(np.mean(jks, axis=0), jks)

    def sample_jackknife_covariance(self, tag, binsize, f=lambda x: x):
        jks = self.sample_jks(tag, binsize, f)
        return jackknife.covariance_jks(np.mean(jks, axis=0), jks)
    
    def sample_binning_study(self, tag, binsizes):
        self.message(f"Unbinned sample size: {len(self.database[tag].sample)}")
        var = {}
        for b in binsizes:
            var[b] = self.sample_jackknife_variance(tag, b)
        return var
    
    ## autocorrelations ##
    def sample_autocovariance(self, tag, tmax, idx=None):        
        sorted_cfgs = sorted(self.database[tag].sample, key=lambda x: int(x.split("-")[-1]))
        if idx == None:
            sample = np.array([self.database[tag].sample[cfg] for cfg in sorted_cfgs]) 
        else: 
            sample = np.array([self.database[tag].sample[cfg][idx] for cfg in sorted_cfgs]) 
        return auto.covariance(sample, tmax)
    
    def sample_autocorrelation_function(self, tag, tmax, idx=None):
        cov = self.sample_autocovariance(tag, tmax, idx)
        return cov/cov[0]
    
    def sample_integrated_autocorrelation_time(self, tag, tmax, idx=None):
        gamma = self.sample_autocorrelation_function(tag, tmax, idx)
        return 0.5 + np.sum(gamma[1:])