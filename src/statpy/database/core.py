import os, subprocess
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


class DB:
    def __init__(self, *args, num_proc=None, verbosity=0, sorting_key=lambda x: (int(x[0].split("r")[-1].split("-")[0]),int(x[0].split("-")[-1])), dev_mode=False, repo_path=None):
        self.t0 = time()
        self.num_proc = num_proc
        self.verbosity = verbosity
        self.sorting_key = sorting_key
        self.dev_mode = dev_mode
        self.database = {} 
        self.commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(repo_path)).decode('utf-8').strip() if repo_path is not None else None
        message(f"Initialized database with statpy commit hash {self.commit_hash} and {num_proc} processes.", self.verbosity)
        if dev_mode: message(f"DEVELOPMENT MODE IS ACTIVATED - LEAFS CAN BE REPLACED", self.verbosity)
        for src in args:
            # init db using src files
            if isinstance(src, str):
                self.load(src)
            # init db using src database
            if isinstance(src, DB):
                self.merge(src)

    def load(self, *srcs):
        for src in srcs:
            assert os.path.isfile(src)
            message(f"Load {src}")
            with open(src) as f:
                src_db = json.load(f)
            for t, lf in src_db.items():
                self.add_leaf(t, lf.mean, lf.jks, lf.sample, lf.misc)

    def merge(self, *srcs):
        for src in srcs:
            for t, lf in src.database.items():
                message(f"Merge {t} into database.")
                self.add_leaf(t, lf.mean, lf.jks, lf.sample, lf.misc)

    def save(self, dst, with_sample=False):
        db = {}
        for tag, lf in self.database.items():
            sample = lf.sample if with_sample else None
            misc = dict(lf.misc) if lf.misc is not None else dict(); misc["tag"] = tag
            self.add_leaf(tag, lf.mean, lf.jks, sample, lf.misc, database=db)
        with open(dst, "w") as f:
            json.dump(db, f)

    def add_leaf(self, tag, mean, jks, sample, misc, database=None):
        db = self.database if database is None else database
        if tag not in db or self.dev_mode:
            assert (isinstance(sample, dict) or sample==None)
            assert (isinstance(jks, dict) or jks==None)
            assert (isinstance(misc, dict) or misc==None)
            if "rwf" in tag:
                message(f"Add reweighting factors {tag} to database.", self.verbosity)
                db[tag] = Leaf(None, None, sample, None)
            else:
                if sample is not None:
                    if mean is None or jks is None:
                        nrwf = self.get_nrwf(tag)
                        mean = np.average(self.as_array(sample), axis=0, weights=self.as_array(nrwf))
                        jks = {cfg:( mean + (mean - sample[cfg]) * nrwf[cfg] / (len(sample) - nrwf[cfg]) ) for cfg in sample}                    
                db[tag] = Leaf(mean, jks, sample, misc) 
        else:
            message(f"{tag} already in database. Leaf not added.")

    def remove_leaf(self, tag, verbosity=None):
        verbosity = self.verbosity if verbosity is None else verbosity
        if tag in self.database:
            message(f"remove {tag} from database.", verbosity)
            del self.database[tag]
        else:
            message(f"{tag} not in database.")

    def rename_leaf(self, old, new):
        if old in self.database:
            if new not in self.database:
                old_lf = self.database[old]                
                self.add_leaf(new, old_lf.mean, old_lf.jks, old_lf.sample, old_lf.misc)
            else:
                message(f"{new} already in database. Leaf not added.")
        else:
            message(f"{old} not in database.")
 
    ################################## VERBOSITY #######################################
    
    def print(self, filter_key="", verbosity=None):
        verbosity = self.verbosity if verbosity is None else verbosity
        message(self.__str__(filter_key, verbosity))    

    def __str__(self, key, verbosity):
        s = '\n\n\tDatabase consists of\n\n'
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
        s = f'\n\n\tMisc dict for {tag} consists of\n\n'
        for k, i in self.database[tag].misc.items():
            s += f'\t{k:20s}: {i}\n'
        return s
    
    ################################## FUNCTIONS #######################################

    ################################ HELPER ###################################

    def get_tags(self, filter_key=""):
        return [tag for tag in self.database.keys() if filter_key in tag]
    
    def as_array(self, dictionary):
        sorted_d = dict(sorted(dictionary.items(), key=self.sorting_key))
        return np.array(list(sorted_d.values()))

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

    def add_binned_leaf(self, tag, binsize):
        jks = self.jks(tag, binsize)
        mean = np.mean(jks, axis=0)
        binned_tag = f"{tag}/binsize{binsize}"; branch_tag = tag.split("/")[0]
        self.add_leaf(tag=binned_tag, mean=mean, jks={f"{branch_tag}-{i}":jk for i,jk in enumerate(jks)}, sample=None, misc=None)
        return binned_tag

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
        
    def add_nrwf(self, rwf_tag):
        rwf = self.database[rwf_tag].sample; n = np.mean(self.as_array(rwf))
        self.add_leaf(tag=rwf_tag.replace("rwf","nrwf"), mean=None, jks=None, sample={cfg:rwf/n for cfg,rwf in rwf.items()}, misc=None)

    def get_nrwf(self, tag):
        return self.database[f"{tag.split('/')[0]}/nrwf"].sample
    
    ################################## STATISTICS ######################################

    def jks(self, tag, binsize):
        lf = self.database[tag]
        nrwf = self.get_nrwf(tag)
        bsample = statistics.bin(self.as_array(lf.sample), binsize, self.as_array(nrwf))
        bnrwf = statistics.bin(self.as_array(nrwf), binsize)
        jks = jackknife.sample(bsample, bnrwf[:, None])
        return jks

    def jackknife_variance(self, tag, binsize):
        jks = self.database[f"{tag}/binsize{binsize}"].jks if f"{tag}/binsize{binsize}" in self.database else self.jks(tag, binsize)
        return jackknife.variance(jks)

    def jackknife_covariance(self, tag, binsize):
        jks = self.database[f"{tag}/binsize{binsize}"].jks if f"{tag}/binsize{binsize}" in self.database else self.jks(tag, binsize)
        return jackknife.covariance(jks)
    
    def sample_binning_study(self, tag, binsizes):
        message(f"Binning study with unbinned sample size: {len(self.database[tag].sample)}")
        var = {}
        for b in binsizes:
            var[b] = self.jackknife_variance(tag, b)
        return var
