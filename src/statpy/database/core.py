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
from statpy.statistics import jackknife, bootstrap


class DB:
    def __init__(self, *args, num_proc=None, verbosity=0, sorting_key=lambda x: (try_int(x[0].split("r")[-1].split("-")[0]),int(x[0].split("-")[-1])), dev_mode=False, repo_path=None):
        self.t0 = time()
        self.num_proc = num_proc
        self.verbosity = verbosity
        self.sorting_key = sorting_key
        self.dev_mode = dev_mode
        self.database = {} 
        self.commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(repo_path)).decode('utf-8').strip() if repo_path is not None else None
        message(f"Initialized database with statpy commit hash {self.commit_hash} and {num_proc} processes.", 0)
        if dev_mode: message(f"DEVELOPMENT MODE IS ACTIVATED - LEAFS CAN BE REPLACED", 0)
        for src in args:
            # init db using src files
            if isinstance(src, str):
                self.load(src)
            # init db using src database
            if isinstance(src, DB):
                self.merge(src)

    def load(self, *srcs):
        for src in srcs:
            message(f"Load {src}", self.verbosity)
            assert os.path.isfile(src), f"{src} not found."
            with open(src) as f:
                src_db = json.load(f)
            for t, lf in src_db.items():
                self.add_leaf(t, lf.mean, lf.jks, lf.sample, lf.misc, verbosity=self.verbosity)

    def merge(self, *srcs):
        for src in srcs:
            for t, lf in src.database.items():
                message(f"Merge {t} into database.", verbosity=self.verbosity)
                self.add_leaf(t, lf.mean, lf.jks, lf.sample, lf.misc, verbosity=self.verbosity)

    def save(self, dst, with_sample=False):
        db = {}
        for tag, lf in self.database.items():
            sample = lf.sample if with_sample else None
            misc = dict(lf.misc) if lf.misc is not None else dict(); misc["tag"] = tag
            self.add_leaf(tag, lf.mean, lf.jks, sample, lf.misc, database=db, verbosity=self.verbosity)
        with open(dst, "w") as f:
            json.dump(db, f)

    def add_leaf(self, tag, mean, jks, sample, misc, database=None, verbosity=None):
        verbosity = self.verbosity if verbosity is None else verbosity
        db = self.database if database is None else database
        if tag not in db or self.dev_mode:
            assert (isinstance(sample, dict) or sample is None)
            assert (isinstance(jks, dict) or jks is None)
            assert (isinstance(misc, dict) or misc is None)
            if "rwf" in tag:
                message(f"Add reweighting factors {tag} to database.", verbosity)
                db[tag] = Leaf(None, None, sample, None)
            else:
                if sample is not None:
                    if jks is None:
                        sample_arr = self.as_array(sample)
                        nrwf_arr = self.as_array(self.get_nrwf(tag))
                        jks_arr = jackknife.sample(sample_arr, weights=nrwf_arr); jks = {cfg:jk for cfg,jk in zip(sample,jks_arr)}
                        if mean is None:
                            mean = np.mean(jks_arr, axis=0)
                message(f"Add {tag} to database.", verbosity)                      
                db[tag] = Leaf(mean, jks, sample, misc) 
        else:
            message(f"{tag} already in database. Leaf not added.", verbosity)

    def remove_leaf(self, tag, verbosity=None):
        verbosity = self.verbosity if verbosity is None else verbosity
        if tag in self.database:
            message(f"remove {tag} from database.", verbosity)
            del self.database[tag]
        else:
            message(f"{tag} not in database.", verbosity)

    def rename_leaf(self, old, new, verbosity=None):
        verbosity = self.verbosity if verbosity is None else verbosity
        if old in self.database:
            if new not in self.database:
                old_lf = self.database[old]                
                self.add_leaf(new, old_lf.mean, old_lf.jks, old_lf.sample, old_lf.misc, verbosity=verbosity)
                self.remove_leaf(old, verbosity)
            else:
                message(f"{new} already in database. Leaf not added.", verbosity)
        else:
            message(f"{old} not in database.", verbosity)
 
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
    
    def combine(self, *tags, f=lambda x: x, dst_tag=None, combine_bss=False):
        mean = self.combine_mean(*tags, f=f)
        jks = self.combine_jks(*tags, f=f)
        # combine bootstrap
        misc = self.combine_bss() if combine_bss else None
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
    
    def combine_bss(self, src, f=lambda x: x):
        if isinstance(src, str):
            bss = self.bss(src) 
        elif isinstance(src, np.ndarray):
            bss = src
        else: 
            assert 0, "Invalid bss type."
        if self.num_proc is None:
            return np.array([f(bs) for bs in bss])
        else:
            with multiprocessing.Pool(self.num_proc) as pool:
                result = pool.map(f, bss)
            return np.array(result)

    ############################### SAMPLE ####################################
    
    def add_binned_leaf(self, tag, binsize):
        if binsize == 1:
            message(f"{tag} is already in database. Nothing to do.")
            return tag
        jks = self.jks(tag, binsize)
        mean = np.mean(jks, axis=0)
        binned_tag = f"{tag}/binsize{binsize}"; branch_tag = tag.split("/")[0]
        self.add_leaf(tag=binned_tag, mean=mean, jks={f"{branch_tag}-b{binsize}-{i}":jk for i,jk in enumerate(jks)}, sample=None, misc=None)
        return binned_tag

    def combine_sample(self, *tags, f=lambda x: x, dst_tag=None, parallel=False):
        lfs = [self.database[tag] for tag in tags]
        cfgs = np.unique(np.concatenate([list(lf.sample.keys()) for lf in lfs]))
        xs = {cfg:[lf.sample[cfg] if cfg in lf.sample else lf.mean for lf in lfs] for cfg in cfgs}
        if not parallel:
            sample = {cfg:f(*x) for cfg,x in xs.items()}
        else:
            def wrapped_f(cfg, *x):
                return cfg, f(*x)
            message(f"Spawn {self.num_proc} processes to compute sample.", self.verbosity-1)
            with multiprocessing.Pool(self.num_proc) as pool:
                sample = dict(pool.starmap(wrapped_f, [(cfg, *x) for cfg,x in xs.items()]))
        if dst_tag is None:
            return sample
        self.add_leaf(dst_tag, None, None, sample, None)

    def concatenate_samples(self, *tags, dst_tag=None, dst_cfgs=None):
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

    ################################ RWF ######################################
        
    def add_nrwf(self, rwf_tag, verbosity=None):
        verbosity = self.verbosity if verbosity is None else verbosity
        rwf = self.database[rwf_tag].sample; n = np.mean(self.as_array(rwf))
        self.add_leaf(tag=rwf_tag.replace("rwf","nrwf"), mean=None, jks=None, sample={cfg:rwf/n for cfg,rwf in rwf.items()}, misc=None, verbosity=verbosity)

    def get_nrwf(self, tag):
        return self.database[f"{tag.split('/')[0]}/nrwf"].sample
    
    ################################## STATISTICS ######################################

    def jks(self, tag, binsize):
        lf = self.database[tag]
        nrwf_arr = self.as_array(self.get_nrwf(tag))
        bsample = statistics.bin(self.as_array(lf.sample), binsize, weights=nrwf_arr)
        bnrwf = statistics.bin(nrwf_arr, binsize=binsize)
        jks = jackknife.sample(bsample, weights=bnrwf)
        return jks

    def jackknife_variance(self, tag, binsize=1):
        assert ("binsize" not in tag) or (binsize == 1)
        jks = self.as_array(self.database[tag].jks) if binsize == 1 else self.jks(tag, binsize)
        return jackknife.variance(jks)

    def jackknife_covariance(self, tag, binsize=1):
        assert ("binsize" not in tag) or (binsize == 1)
        jks = self.as_array(self.database[tag].jks) if binsize == 1 else self.jks(tag, binsize)
        return jackknife.covariance(jks)
    
    def sample_binning_study(self, tag, binsizes):
        message(f"Binning study with unbinned sample size: {len(self.database[tag].sample)}")
        var = {}
        for b in binsizes:
            var[b] = self.jackknife_variance(tag, b)
        return var
    
    def load_bootstrap(self, branch_tag, fn):
        bootstraps = np.loadtxt(fn, dtype=int)
        with open(fn, "r") as f:
            configlist = f.readlines()[3][:-1].replace("n", "-").split(" ")[1:]
        message(f"Add bootstraps for {branch_tag} from {fn} to database.")
        self.add_leaf(f"{branch_tag}/bootstraps", mean=bootstraps, jks=None, sample=None, misc={"configlist": configlist})
        self.database[f"{branch_tag}/bootstraps"].mean

    def bss(self, tag):
        assert "binsize" not in tag, "Can only compute bss for unbinned leafs"
        lf = self.database[tag]
        bootstraps = self.database[f"{tag.split('/')[0]}/bootstraps"].mean
        return bootstrap.sample(self.as_array(lf.sample), bootstraps, weights=self.as_array(self.get_nrwf(tag))) 
    
    def bootstrap_variance(self, tag):
        bss = self.database[tag].misc["bss"]
        return bootstrap.variance(bss)

    def bootstrap_covariance(self, tag):
        bss = self.database[tag].misc["bss"]
        return bootstrap.covariance(bss)


# helper function to allow sorting of concatenated branch_tags without r
def try_int(x):
    try:
        return int(x)
    except ValueError:
        return int(0)
