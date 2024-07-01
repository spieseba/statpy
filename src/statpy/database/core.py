import os, subprocess, re
import numpy as np
from time import time
from functools import reduce
from operator import ior

# import multiprocessing module and overwrite its Pickle class using dill
import dill, multiprocessing
dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads
multiprocessing.reduction.ForkingPickler = dill.Pickler
multiprocessing.reduction.dump = dill.dump

from statpy.log import message 
from statpy.database import custom_json as json
from statpy.database.leafs import Leaf
from statpy.statistics import core as statistics
from statpy.statistics import jackknife 

##############################################################################################################################################################
##############################################################################################################################################################
###################################### DATABASE SYSTEM USING LEAFS CONTAINING MEAN AND JKS (SECONDARY OBSERVABLES) ###########################################
##############################################################################################################################################################
##############################################################################################################################################################


class DB:
    def __init__(self, *args, num_proc=None, verbosity=0, sorting_key=lambda x: int(x[0].split("-")[-1]), dev_mode=False, repo_path=None):
        self.t0 = time()
        self.num_proc = num_proc
        self.verbosity = verbosity
        self.sorting_key = sorting_key
        self.dev_mode = dev_mode
        self.database = {} 
        self.commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(repo_path)).decode('utf-8').strip() if repo_path is not None else None
        message(f"Initialized database with statpy commit hash {self.commit_hash} and {num_proc} processes.")
        if dev_mode: message(f"DEVELOPMENT MODE IS ACTIVATED - LEAFS CAN BE REPLACED")
        for src in args:
            # init db using src files
            if isinstance(src, str):
                self.load(src)
            # init db using src dict
            if isinstance(src, dict):
                for t, lf in src.items():
                    self.add_leaf(t, lf.mean, lf.jks, lf.sample, lf.misc)

    def load(self, *srcs):
        for src in srcs:
            message(f"LOAD {src}", self.verbosity)
            assert os.path.isfile(src)
            with open(src) as f:
                src_db = json.load(f)
            for t, lf in src_db.items():
                self.add_leaf(t, lf.mean, lf.jks, lf.sample, lf.misc, verbosity=self.verbosity)
    
    def save(self, dst, with_sample=False, verbosity=None):
        verbosity = self.verbosity if verbosity is None else verbosity
        db = {}
        for tag, lf in self.database.items():
            sample = lf.sample if with_sample else None
            misc = dict(lf.misc) if lf.misc is not None else dict(); misc["tag"] = tag
            self.add_leaf(tag, lf.mean, lf.jks, sample, lf.misc, database=db, verbosity=verbosity)
        message(f"write database to {dst} (with sample: {with_sample})", verbosity)
        with open(dst, "w") as f:
            json.dump(db, f)

    def add_leaf(self, tag, mean, jks, sample, misc, database=None, verbosity=None):
        verbosity = self.verbosity if verbosity is None else verbosity
        db = self.database if database is None else database
        if tag not in db or self.dev_mode:
            assert (isinstance(sample, dict) or sample==None)
            assert (isinstance(jks, dict) or jks==None)
            assert (isinstance(misc, dict) or misc==None)
            # compute mean and jks if necessary and possible
            if sample is not None: 
                if not isinstance(next(iter(sample.values())), dict):
                    if mean is None:
                        mean = np.mean(self.as_array(sample), axis=0)
                    if jks is None:
                        jks = {cfg:( mean + (mean - sample[cfg]) / (len(sample) - 1) ) for cfg in sample}
            db[tag] = Leaf(mean, jks, sample, misc)
        else:
            message(f"{tag} already in database. Leaf not added.", verbosity)

    def remove_leaf(self, *tags, verbosity=None):
        verbosity = self.verbosity if verbosity is None else verbosity
        for tag in tags:
            if tag in self.database:
                message(f"remove {tag}.", verbosity)
                del self.database[tag]
            else:
                message(f"{tag} not in database.")

    def rename_leaf(self, old, new):
        if old in self.database:
            if new not in self.database:
                old_lf = self.database[old]                
                self.add_leaf(new, old_lf.mean, old_lf.jks, old_lf.sample, old_lf.misc)
                self.remove_leaf(old)
            else:
                message(f"{new} already in database. Leaf not added.")
        else:
            message(f"{old} not in database.")
 
    ################################## VERBOSITY #######################################
   
    def print(self, pattern=".*", verbosity=None):
        verbosity = self.verbosity if verbosity is None else verbosity
        message(self.__str__(pattern, verbosity))    
    
    def __str__(self, pattern, verbosity):
        s = '\n\n\tDatabase consists of\n\n'
        for tag, lf in self.database.items():
            if re.search(pattern, tag):
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
        s = f'\n\n\tMisc dict for {tag} consists of\n\n'
        for k, i in self.database[tag].misc.items():
            s += f'\t{k:20s}: {i}\n'
        return s
    
    ################################## FUNCTIONS #######################################

    ################################ HELPER ###################################

    def get_tags(self, pattern=".*"):
        return [tag for tag in self.database.keys() if re.search(pattern, tag)]
 
    def as_array(self, dictionary):
        sorted_d = dict(sorted(dictionary.items(), key=self.sorting_key))
        return np.array(list(sorted_d.values()))

    ################################ JKS ######################################
    
    def combine(self, *tags, f=lambda x: x, dst_tag=None, propagate_sys=True):
        mean = self.combine_mean(*tags, f=f)
        jks = self.combine_jks(*tags, f=f)
        # Propagate systematics
        misc = self.propagate_systematics(*tags, f=f) if propagate_sys else None
        if dst_tag is None:
            return mean, jks, misc
        self.add_leaf(dst_tag, mean, jks, None, misc)

    def combine_mean(self, *tags, f=lambda x: x):
        lfs = [self.database[tag] for tag in tags]
        mean = f(*[lf.mean for lf in lfs])
        return mean

    def combine_jks(self, *tags, f=lambda x: x):
        lfs = [self.database[tag] for tag in tags]
        cfgs = np.unique(np.concatenate([list(lf.jks.keys()) for lf in lfs]))
        xs = {cfg:[lf.jks[cfg] if cfg in lf.jks else lf.mean for lf in lfs] for cfg in cfgs}
        if self.num_proc is None:
            jks = {cfg:f(*x) for cfg,x in xs.items()}
        else:
            def wrapped_f(cfg, *x):
                return cfg, f(*x)
            message(f"Spawn {self.num_proc} processes to compute jackknife sample.", verbosity=self.verbosity-1)
            with multiprocessing.Pool(self.num_proc) as pool:
                jks = dict(pool.starmap(wrapped_f, [(cfg, *x) for cfg,x in xs.items()]))
        return jks

    ############################### SAMPLE ####################################
            
    def combine_sample(self, *tags, f=lambda x: x, dst_tag=None):
        lfs = [self.database[tag] for tag in tags]
        f_sample = {}
        cfgs = np.unique([list(lf.sample.keys()) for lf in lfs]) 
        for cfg in cfgs:
            x = [lf.sample[cfg] if cfg in lf.sample else lf.mean for lf in lfs]
            f_sample[cfg] = f(*x) 
        f_sample = dict(sorted(f_sample.items(), key=self.sorting_key)) 
        if dst_tag is None: 
            return f_sample
        self.add_leaf(dst_tag, None, None, f_sample, None)

    def merge_sample(self, *tags, dst_tag=None, dst_cfgs=None):
        lfs = [self.database[tag] for tag in tags]
        if dst_cfgs is None:
            sample = dict(sorted(reduce(ior, [lf.sample for lf in lfs], {}).items(), key=self.sorting_key))
        else:
            sample = {cfg:val for cfg,val in zip(dst_cfgs, np.concatenate([self.as_array(lf.sample) for lf in lfs], axis=0))}
        if dst_tag is None:
            return sample
        self.add_leaf(dst_tag, None, None, sample, None)
 
    def remove_cfgs(self, tag, cfgs, dst_tag=None):
        sample = dict(self.database[tag].sample)
        for cfg in cfgs:
            sample.pop(str(cfg), None)
        if dst_tag is None:
            return sample
        self.add_leaf(dst_tag, None, None, sample, None)

    def get_cfgs(self, tag, numeric=False):
        lf = self.database[tag]
        obj = lf.jks if lf.jks is not None else lf.sample
        if numeric:
            return sorted([self.sorting_key(x) for x in obj.items()])
        return [x[0] for x in sorted(list(obj.items()), key=self.sorting_key)]
     
    ################################## STATISTICS ######################################
    
    def delayed_binning(self, lf, binsize, branch_tag, shift=0):
        jks_tags = [x[0] for x in sorted(list(lf.jks.items()), key=self.sorting_key) if branch_tag in x[0]]
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
        if binsize == 1:
            return lf.jks
        branch_tags = np.unique([t.split("-")[0] for t in list(lf.jks.keys())])
        binsizes = len(branch_tags) * [binsize] if isinstance(binsize, (int, np.int64)) else binsize; assert len(branch_tags) == len(binsizes), f"{len(branch_tags)} != {len(binsizes)}"
        shifts = len(branch_tags) * [shift] if isinstance(shift, (int, np.int64)) else shift; assert len(branch_tags) == len(shifts)
        jks_bin = {}
        for b,branch_tag,s in zip(binsizes, branch_tags, shifts):
            jks_bin.update(self.delayed_binning(lf, b, branch_tag, s))
        return jks_bin
 
    def jackknife_variance(self, tag, binsize, average_permutations=False):
        permutations = np.arange(1)
        if average_permutations:
            permutations = np.arange(np.min(binsize))
        var = []
        for p in permutations:
            jks = self.as_array(self.jks(tag, binsize, p))
            if len(jks) == 0: return 0.0
            var.append(jackknife.variance(jks))
        return np.mean(var, axis=0)
    
    def jackknife_covariance(self, tag, binsize, average_permutations=False):
        permutations = np.arange(1)
        if average_permutations:
            permutations = np.arange(np.min(binsize))
        cov = []
        for p in permutations:
            jks = self.as_array(self.jks(tag, binsize, p))
            cov.append(jackknife.covariance(jks))
        return np.mean(cov, axis=0)

    def binning_study(self, tag, binsizes, average_permutations=False):
        branch_tags = np.unique([t.split("-")[0] for t in list(self.database[tag].jks.keys())])
        unbinned_sample_sizes = {branch_tag:len([t for t in list(self.database[tag].jks.keys()) if branch_tag in t]) for branch_tag in branch_tags}
        message(f"Delayed binning study with unbinned sample size(s): {unbinned_sample_sizes} (number of unique cfgs: {len(self.database[tag].jks)})")
        var = {}
        for b in binsizes:
            var[b] = self.jackknife_variance(tag, b, average_permutations)
        return var
    
    def AMA(self, exact_exact_tag, exact_sloppy_tag, sloppy_sloppy_tag, dst_tag):
        self.combine(exact_exact_tag, exact_sloppy_tag, f=lambda x,y: x-y, dst_tag=dst_tag+"_bias")
        self.combine(sloppy_sloppy_tag, dst_tag+"_bias", f=lambda x,y: x+y, dst_tag=dst_tag)
    
    ############################### SAMPLE ####################################
         
    def sample_jks(self, tag, binsize):
        lf = self.database[tag]
        if binsize == 1:
            jks = self.as_array(lf.jks)
        else:
            bsample = statistics.bin(self.as_array(lf.sample), binsize)
            jks = jackknife.sample(bsample)
        return jks
    
    def sample_jackknife_variance(self, tag, binsize):
        jks = self.sample_jks(tag, binsize)
        return jackknife.variance(jks)

    def sample_jackknife_covariance(self, tag, binsize):
        jks = self.sample_jks(tag, binsize)
        return jackknife.covariance(jks)
    
    def sample_binning_study(self, tag, binsizes):
        message(f"Standard binning study with unbinned sample size: {len(self.database[tag].sample)}")
        var = {}
        for b in binsizes:
            var[b] = self.sample_jackknife_variance(tag, b)
        return var
 
    ############################# SYSTEMATICS #################################

    def propagate_systematics(self, *tags, f=lambda x: x, dst=None):
        dst = {} if dst is None else dst
        mean = self.combine_mean(*tags, f=f)
        sys_tags = self.get_sys_tags(*tags)
        for sys_tag in sys_tags:
            mean_shifted = self.compute_shifted_mean(*tags, f=f, sys_tag=sys_tag)
            dst = self.add_sys(mean, mean_shifted, sys_tag, dst)
        return dst

    def compute_shifted_mean(self, *tags, f=lambda x: x, sys_tag=None):
        assert sys_tag is not None
        x = []
        for tag in tags:
            lf = self.database[tag]
            if lf.misc is not None:
                try:
                    x.append(lf.misc[f"MEAN_SHIFTED_{sys_tag}"])
                except KeyError:
                    x.append(lf.mean)
            else:
                x.append(lf.mean)
        return f(*x)

    def add_sys(self, mean, mean_shifted, sys_tag, dst=None):
        dst = {} if dst is None else dst
        sys_var = (mean - mean_shifted)**2.
        dst[f"MEAN_SHIFTED_{sys_tag}"] = mean_shifted
        dst[f"SYS_VAR_{sys_tag}"] = sys_var
        return dst

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
        if self.database[tag].misc is not None:
            for sys_tag in self.get_sys_tags(tag):
                sys_var += self.database[tag].misc[f"SYS_VAR_{sys_tag}"]
        return sys_var
     
    ############################# ESTIMATES #################################
    
    def print_estimate(self, tag, binsize=1, significant_digits=None, exponent=None, average_permutations=False):
        s = f"\n ESTIMATE of {tag} (binsize = {binsize}, significant digits = {significant_digits}):\n"
        # get mean and std
        mean = self.database[tag].mean
        std = self.get_tot_var(tag, binsize, average_permutations)**.5
        # print full estimate and round it if desired
        if significant_digits is not None:
            mean, std, decimals = self._round_estimate(mean, std, significant_digits)
        s += f"   {self._get_rescaled_string(mean, std, exponent)} [STAT + SYS]\n"
        # print errors
        s += " ERRORS:\n"
        std_stat = self.jackknife_variance(tag, binsize, average_permutations)**.5
        std_sys = self.get_sys_var(tag)**.5
        if significant_digits is not None:
            _, std_stat, _ = self._round_estimate(0, self.jackknife_variance(tag, binsize, average_permutations)**.5, significant_digits)
            if self.get_sys_var(tag)**.5 == 0.:
                std_sys = 0.
            else:
                _, std_sys, _ = self._round_estimate(0, self.get_sys_var(tag)**.5, significant_digits)
        s += f"   {self._get_rescaled_string(None, std_stat, exponent)} [STAT]\n"
        s += f"   {self._get_rescaled_string(None, std_sys, exponent)} [SYS]\n"
        # print systematics
        sys_tags = self.get_sys_tags(tag)
        if len(sys_tags) > 0:
            s += "\t[\n"
            for sys_tag in sys_tags:
                std_sys = self.database[tag].misc[f'SYS_VAR_{sys_tag}']**.5
                if std_sys == 0.: 
                    s+= f"\t 0.0 [SYS {sys_tag}]\n"
                    continue
                if significant_digits is not None:
                    _, std_sys, _ = self._round_estimate(0, std_sys, significant_digits)
                s += f"\t {self._get_rescaled_string(None, std_sys, exponent)} [SYS {sys_tag}]\n"
            s += "\t]"
        message(s)

    def _get_rescaled_string(self, mean, std, exponent):
        rescaled_std_coeff = self._get_rescaled_coeff(std, exponent) if exponent is not None else None
        if mean is None:
            return f"{rescaled_std_coeff:.{abs(exponent)}}e{exponent:03d}" if exponent is not None else f"{std}"
        else:
            rescaled_mean_coeff = self._get_rescaled_coeff(mean, exponent) if exponent is not None else None
            return f"{rescaled_mean_coeff:.{abs(exponent)}}e{exponent:03d} +- {rescaled_std_coeff:.{abs(exponent)}}e{exponent:03d}" if exponent is not None else f"{mean} +- {std}"   

    def _get_rescaled_coeff(self, value, exponent):
         scientific_array = f"{value:e}".split("e")
         coeff = float(scientific_array[0])
         old_exp = int(scientific_array[1])
         rescaled_coeff = coeff * 10**(old_exp-exponent)
         return rescaled_coeff

    def _round_estimate(self, mean, std, significant_digits):
        decimals = -int(np.floor(np.log10(std))) + significant_digits-1
        scale = 10**(decimals)
        mean_rounded = np.round(mean, decimals)
        std_rounded = np.ceil(std * scale) / scale 
        return mean_rounded, std_rounded, decimals
     
    def get_estimate(self, tag, binsize, average_permutations=False):
        return self.database[tag].mean, self.get_tot_var(tag, binsize, average_permutations)

    def get_tot_var(self, tag, binsize, average_permutations=False):
        return self.jackknife_variance(tag, binsize, average_permutations) + self.get_sys_var(tag)
