import os
import re 
import subprocess
import numpy as np
from time import time
from functools import reduce
from operator import ior

from statpy.log import message 
from statpy.database import custom_json as json
from statpy.database.leafs import Leaf
from statpy.statistics import core as statistics
from statpy.statistics import jackknife, bootstrap

# import multiprocessing module and overwrite its Pickle class using dill
import dill
import multiprocessing
dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads
multiprocessing.reduction.ForkingPickler = dill.Pickler
multiprocessing.reduction.dump = dill.dump



class DB:
    def __init__(self, *args, num_proc=None, silent=False, stream_order=None, reverse_order=None, repo_path=None):
        self.t0 = time()
        self.num_proc = num_proc
        self.silent = silent
        self._sort_key = lambda tag: _cls_sorting_key(tag, custom_major_order=stream_order, reverse_minor_order=reverse_order)
        self.database = {}
        self.commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(repo_path)).decode('utf-8').strip() if repo_path is not None else None
        message(f"Initialized database with statpy commit hash {self.commit_hash} and {num_proc} processes.")
        for src in args:
            if isinstance(src, str):
                self.load(src)
            elif isinstance(src, DB):
                for t, lf in src.database.items():
                    self.add_leaf(t, lf.mean, lf.jks, lf.sample, lf.misc, bss=lf.bss, weights_tag=lf.weights_tag, silent=self.silent)

    def load(self, src):
        """Load a JSON snapshot saved by :func:`save` and add every leaf."""
        message(f"Load {src}", self.silent)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"{src} not found")
        with open(src) as f:
            src_db = json.load(f)
        for t, lf in src_db.items():
            self.add_leaf(t, lf.mean, lf.jks, lf.sample, lf.misc, bss=lf.bss, weights_tag=lf.weights_tag, silent=self.silent)

    def save(self, dst):
        """Write the entire database (all fields, including samples) to ``dst``."""
        with open(dst, "w") as f:
            json.dump(self.database, f)

    def add_leaf(self, tag, mean, jks, sample, misc, bss=None, weights_tag=None, database=None, silent=None):
        silent = self.silent if silent is None else silent
        db = self.database if database is None else database
        if tag not in db:
            assert (isinstance(sample, dict) or sample is None)
            assert (isinstance(jks, dict) or jks is None)
            assert (isinstance(misc, dict) or misc is None)
            # Auto-compute jks from sample when the leaf carries a weights reference.
            # Leaves without weights_tag (rwf, nrwf, fit results, ...) skip this path.
            if sample is not None and jks is None and weights_tag is not None:
                sample_arr = self.as_array(sample)
                nrwf_arr = self.as_array(self.database[weights_tag].sample)
                jks_arr = jackknife.sample(sample_arr, weights=nrwf_arr)
                jks = {cfg: jk for cfg, jk in zip(sample, jks_arr)}
                if mean is None:
                    mean = np.mean(jks_arr, axis=0)
            message(f"Add {tag} to database.", silent)
            db[tag] = Leaf(mean, jks, sample, bss=bss, misc=misc, weights_tag=weights_tag)
        else:
            message(f"{tag} already in database. Leaf not added.", silent)

    def remove_leaf(self, tag, silent=None):
        silent = self.silent if silent is None else silent
        if tag in self.database:
            message(f"remove {tag} from database.", silent)
            del self.database[tag]
        else:
            message(f"{tag} not in database.", silent)

    def rename_leaf(self, old, new, silent=None):
        silent = self.silent if silent is None else silent
        if old in self.database:
            if new not in self.database:
                old_lf = self.database[old]
                self.add_leaf(new, old_lf.mean, old_lf.jks, old_lf.sample, old_lf.misc, bss=old_lf.bss, weights_tag=old_lf.weights_tag, silent=silent)
                self.remove_leaf(old, silent)
            else:
                message(f"{new} already in database. Leaf not added.", silent)
        else:
            message(f"{old} not in database.", silent)


    def print(self, pattern=".*"):
        message(self.__str__(pattern))

    def __str__(self, pattern):
        s = '\n\n\tDatabase consists of\n\n'
        for tag, lf in self.database.items():
            if re.search(pattern, tag):
                s += f'\t{tag:20s}\n'
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
        sorted_d = dict(sorted(dictionary.items(), key=lambda kv: self._sort_key(kv[0])))
        if isinstance(next(iter(sorted_d.values())), np.ma.MaskedArray):
            return np.ma.array(list(sorted_d.values()))
        return np.array(list(sorted_d.values()))

    ################################ JKS ######################################
    
    def combine(self, *tags, f=lambda x: x, dst_tag=None):
        """Combine ``f`` across mean, jackknife, and (when available) bootstrap
        samples of ``tags``.

        Bootstrap combination is performed only when every input leaf carries
        a stored ``bss`` (e.g. bootstrap-fit results). For raw-data leaves
        without stored ``bss``, call :func:`combine_bss` explicitly.

        Returns ``(mean, jks, bss)`` where ``bss`` is ``None`` when not
        combined. If ``dst_tag`` is given, the result is additionally added
        as a leaf at ``dst_tag``.
        """
        mean = self.combine_mean(*tags, f=f)
        jks = self.combine_jks(*tags, f=f)
        bss = (
            self.combine_bss(*tags, f=f)
            if all(self.database[tag].bss is not None for tag in tags)
            else None
        )
        if dst_tag is not None:
            self.add_leaf(dst_tag, mean, jks, None, None, bss=bss)
        return mean, jks, bss

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
            message(f"Spawn {self.num_proc} processes to compute jackknife sample.", silent=True)
            with multiprocessing.Pool(self.num_proc) as pool:
                jks = dict(pool.starmap(wrapped_f, [(cfg, *x) for cfg,x in xs.items()]))
        return jks
    
    def combine_bss(self, *tags, f=lambda x: x):
        """Apply ``f`` to bootstrap samples of one or more leaves.

        For each ``tag``, uses ``lf.bss`` if set (e.g. fit-result leaves)
        or computes ``db.bss(tag)`` from the sample (raw-data leaves).
        ``f`` receives the bootstrap value of each input leaf at the
        same bootstrap index, parallel to :func:`combine_jks`.
        """
        bsses = []
        for tag in tags:
            lf = self.database[tag]
            if lf.bss is not None:
                bsses.append(lf.bss)
            elif lf.sample is not None:
                bsses.append(self.bss(tag))
            else:
                raise ValueError(f"leaf {tag!r} has neither sample nor bss")
        n_bs = bsses[0].shape[0]
        if self.num_proc is None:
            return np.array([f(*[bs[i] for bs in bsses]) for i in range(n_bs)])
        with multiprocessing.Pool(self.num_proc) as pool:
            return np.array(pool.starmap(f, [tuple(bs[i] for bs in bsses) for i in range(n_bs)]))

    ############################### SAMPLE ####################################
    
    def add_binned_leaf(self, tag, binsize):
        if binsize == 1:
            message(f"{tag} is already in database. Nothing to do.")
            return tag
        if "binsize" in tag:
            message(f"{tag} is already binned. Can only bin unbinned leafs.")
            raise AssertionError
        jks = self.jks(tag, binsize)
        mean = np.mean(jks, axis=0)
        src_lf = self.database[tag]
        binned_tag = f"{tag}/binsize{binsize}"
        branch_tag = tag.split("/")[0]
        self.add_leaf(tag=binned_tag, mean=mean, jks={f"{branch_tag}-b{binsize}-{i}":jk for i,jk in enumerate(jks)}, sample=None, misc=src_lf.misc, weights_tag=src_lf.weights_tag)
        return binned_tag

    def combine_sample(self, *tags, f=lambda x: x, dst_tag=None, parallel=False, silent=None):
        silent = self.silent if silent is None else silent
        lfs = [self.database[tag] for tag in tags]
        cfgs = np.unique(np.concatenate([list(lf.sample.keys()) for lf in lfs]))
        xs = {cfg:[lf.sample[cfg] if cfg in lf.sample else lf.mean for lf in lfs] for cfg in cfgs}
        if not parallel:
            sample = {cfg:f(*x) for cfg,x in xs.items()}
        else:
            def wrapped_f(cfg, *x):
                return cfg, f(*x)
            message(f"Spawn {self.num_proc} processes to compute sample.", silent=True)
            with multiprocessing.Pool(self.num_proc) as pool:
                sample = dict(pool.starmap(wrapped_f, [(cfg, *x) for cfg,x in xs.items()]))
        if dst_tag is None:
            return sample
        self.add_leaf(dst_tag, None, None, sample, None, weights_tag=lfs[0].weights_tag, silent=silent)

    def concatenate_samples(self, *tags, dst_tag=None, dst_cfgs=None):
        lfs = [self.database[tag] for tag in tags]
        if dst_cfgs is None:
            sample = dict(sorted(reduce(ior, [lf.sample for lf in lfs], {}).items(), key=lambda kv: self._sort_key(kv[0])))
        else:
            sample = {cfg:val for cfg,val in zip(dst_cfgs, np.concatenate([self.as_array(lf.sample) for lf in lfs], axis=0))}
        if dst_tag is None:
            return sample
        self.add_leaf(dst_tag, None, None, sample, None, weights_tag=lfs[0].weights_tag)

    def remove_cfgs(self, *cfgs, tag=None, dst_tag=None):
        self.rename_leaf(tag, f"{tag}/tmp")
        lf = self.database[f"{tag}/tmp"]
        sample = dict(lf.sample)
        misc = dict(lf.misc) if lf.misc is not None else None
        for cfg in cfgs:
            sample.pop(str(cfg), None)
        if dst_tag is None:
            return sample, misc
        self.add_leaf(dst_tag, None, None, sample, misc, weights_tag=lf.weights_tag)

    def get_cfgs(self, tag):
        lf = self.database[tag]
        obj = lf.jks if lf.jks is not None else lf.sample
        return [str(k) for k, _ in sorted(obj.items(), key=lambda kv: self._sort_key(kv[0]))]

    ################################ RWF ######################################
        
    def add_nrwf(self, rwf_tag, silent=None):
        silent = self.silent if silent is None else silent
        rwf = self.database[rwf_tag].sample
        n = np.mean(self.as_array(rwf))
        self.add_leaf(tag=rwf_tag.replace("rwf","nrwf"), mean=None, jks=None, sample={cfg:rwf/n for cfg,rwf in rwf.items()}, misc=None, silent=silent)

    def get_nrwf(self, tag):
        weights_tag = self.database[tag].weights_tag
        if weights_tag is None:
            raise KeyError(f"leaf {tag!r} has no weights_tag set")
        return self.database[weights_tag].sample
    
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
    
    def add_bootstrap(self, branch_tag, bootstraps, configlist):
        message(f"Add bootstraps for {branch_tag} to database.")
        self.add_leaf(f"{branch_tag}/bootstraps", mean=bootstraps, jks=None, sample=None, misc={"configlist": configlist})

    def bss(self, tag):
        assert "binsize" not in tag, "Can only compute bss for unbinned leafs"
        lf = self.database[tag]
        bootstraps = self.database[f"{tag.split('/')[0]}/bootstraps"].mean
        return bootstrap.sample(self.as_array(lf.sample), bootstraps, weights=self.as_array(self.get_nrwf(tag)))
    
    def bootstrap_variance(self, tag):
        return bootstrap.variance(self.database[tag].bss)

    def bootstrap_covariance(self, tag):
        return bootstrap.covariance(self.database[tag].bss)


def _cls_sorting_key(tag, custom_major_order, reverse_minor_order):
    """Build a ``(major_index, minor_part)`` sort key for a CLS-style tag.

    Tags are expected to look like ``"<ensemble>r<replica>-<cfg>"``
    (e.g. ``"H101r000-123"``):

    - The **major part** ``<ensemble>r<replica>`` selects the stream;
      either parsed as ``int(replica)`` (when ``custom_major_order`` is
      ``None``) or looked up in ``custom_major_order`` (a list of major
      parts giving the desired stream order).
    - The **minor part** is ``int(cfg)`` (the substring after the last
      ``"-"``). If ``reverse_minor_order`` is provided (a per-major
      boolean list), the minor part is negated for streams flagged
      ``True`` so that the cfg sort runs backwards within those streams.

    Raises ``ValueError`` if ``tag`` does not match the expected format.
    """
    major_match = re.match(r"(.+?r\d+)", tag)
    if not major_match:
        raise ValueError(f"Invalid tag format: {tag}")
    major_part = major_match.group(1)
    major_index = int(major_part.split("r")[-1]) if custom_major_order is None else custom_major_order.index(major_part)
    minor_part = int(tag.split("-")[-1])
    if reverse_minor_order is not None and reverse_minor_order[major_index]:
        minor_part = -minor_part
    return (major_index, minor_part)
