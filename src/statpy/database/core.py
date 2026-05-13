import os
import pickle
import re
import struct
import subprocess
import zlib
import numpy as np
from time import time

from statpy.log import message
from statpy.database.leafs import Leaf

from statpy.statistics import core as statistics
from statpy.statistics import jackknife, bootstrap

import multiprocessing

_MAGIC = b"SPDB"  # statpy DB file marker; followed by 4-byte little-endian CRC32 of payload
_DILL_MP_PATCH_INSTALLED = False


def _install_dill_multiprocessing_patch():
    # Swap multiprocessing's pickler for dill so DB workers can ship lambdas/closures.
    # Done lazily on first DB(num_proc=...) so plain `import statpy` doesn't globally
    # mutate multiprocessing.reduction for processes that never spawn a Pool.
    global _DILL_MP_PATCH_INSTALLED
    if _DILL_MP_PATCH_INSTALLED:
        return
    import dill
    dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads
    multiprocessing.reduction.ForkingPickler = dill.Pickler
    multiprocessing.reduction.dump = dill.dump
    _DILL_MP_PATCH_INSTALLED = True


class DB:
    def __init__(self, *args, num_proc=None, silent=False, repo_path=None):
        if num_proc is not None:
            _install_dill_multiprocessing_patch()
        self.t0 = time()
        self.num_proc = num_proc
        self.silent = silent
        self.database = {}
        self.commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(repo_path)).decode('utf-8').strip() if repo_path is not None else None
        message(f"Initialized database with statpy commit hash {self.commit_hash} and {num_proc} processes.")
        for src in args:
            if isinstance(src, str):
                self.load(src)
            elif isinstance(src, DB):
                for t, lf in src.database.items():
                    self.database[t] = Leaf(
                        mean=lf.mean, jks=lf.jks, sample=lf.sample,
                        weights=lf.weights, cfgs=lf.cfgs, bss=lf.bss, misc=lf.misc,
                    )

    def load(self, src):
        """Load a snapshot saved by :func:`save` and add every leaf."""
        message(f"Load {src}", self.silent)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"{src} not found")
        with open(src, "rb") as f:
            header = f.read(8)
            if len(header) < 8 or header[:4] != _MAGIC:
                raise ValueError(f"{src}: not a statpy DB file (bad magic)")
            (crc_expected,) = struct.unpack("<I", header[4:8])
            payload = f.read()
        if (zlib.crc32(payload) & 0xFFFFFFFF) != crc_expected:
            raise ValueError(f"{src}: CRC mismatch -- file corrupted")
        src_db = pickle.loads(payload)
        for t, lf in src_db.items():
            self.database[t] = lf

    def save(self, dst):
        """Write the entire database (all fields, including samples) to ``dst``."""
        payload = pickle.dumps(self.database, protocol=pickle.HIGHEST_PROTOCOL)
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        with open(dst, "wb") as f:
            f.write(_MAGIC)
            f.write(struct.pack("<I", crc))
            f.write(payload)

    ################################ LEAF MANAGEMENT ###########################

    def add_leaf(self, tag, *, mean=None, jks=None, sample=None, weights=None,
                 cfgs=None, bss=None, misc=None, silent=None):
        """Add a new leaf at ``tag``.

        Invariants:
          - Data leaf: ``sample`` requires ``weights`` and ``cfgs``, all of
            matching length; ``jks`` must not be passed (it is computed
            from ``sample``+``weights`` via :func:`jackknife.sample`).
            ``mean`` is auto-computed from ``jks`` if not provided.
          - Combined / derived leaf: ``sample`` is ``None`` but ``jks`` is
            given; ``cfgs`` must match ``jks`` length.
          - Result leaf (fit/bootstrap): ``sample=jks=cfgs=weights=None``;
            carries ``mean`` and optionally ``bss``.
        """
        silent = self.silent if silent is None else silent
        if tag in self.database:
            message(f"{tag} already in database. Leaf not added.", silent)
            return

        if sample is not None:
            assert weights is not None, "sample requires weights"
            assert cfgs is not None,    "sample requires cfgs"
            assert isinstance(sample, np.ndarray), "sample must be np.ndarray"
            assert isinstance(weights, np.ndarray), "weights must be np.ndarray"
            assert isinstance(cfgs, np.ndarray), "cfgs must be np.ndarray"
            assert len(sample) == len(weights) == len(cfgs), \
                f"length mismatch: sample={len(sample)} weights={len(weights)} cfgs={len(cfgs)}"
            assert jks is None, "Don't pass jks when sample+weights are given; jks is derived."
            jks = jackknife.sample(sample, weights=weights)
            if mean is None:
                mean = np.mean(jks, axis=0)
        else:
            if cfgs is not None and jks is not None:
                assert len(jks) == len(cfgs), \
                    f"jks/cfgs length mismatch: jks={len(jks)} cfgs={len(cfgs)}"

        message(f"Add {tag} to database.", silent)
        self.database[tag] = Leaf(
            mean=mean, jks=jks, sample=sample, weights=weights,
            cfgs=cfgs, bss=bss, misc=misc,
        )

    def remove_leaf(self, tag, silent=None):
        silent = self.silent if silent is None else silent
        if tag in self.database:
            message(f"remove {tag} from database.", silent)
            del self.database[tag]
        else:
            message(f"{tag} not in database.", silent)

    def rename_leaf(self, old, new, silent=None):
        silent = self.silent if silent is None else silent
        if old not in self.database:
            message(f"{old} not in database.", silent)
            return
        if new in self.database:
            message(f"{new} already in database. Leaf not added.", silent)
            return
        self.database[new] = self.database[old]
        del self.database[old]
        message(f"renamed {old} -> {new}.", silent)

    def print(self, pattern=".*"):
        message(self.__str__(pattern))

    def __str__(self, pattern):
        s = '\n\n\tDatabase consists of\n\n'
        for tag, _ in self.database.items():
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

    ################################ QUERIES ###################################

    def get_tags(self, pattern=".*"):
        return [tag for tag in self.database.keys() if re.search(pattern, tag)]

    def get_cfgs(self, tag):
        return list(self.database[tag].cfgs)

    ################################ TRANSFORM #################################

    def transform(self, tag, f, dst_tag=None):
        """Apply ``f`` to ``mean``, each jackknife sample, and each bootstrap
        sample (when present) of the leaf at ``tag``.

        Returns ``(mean, jks, bss)`` where ``bss`` is ``None`` when the
        input leaf has no ``bss``. If ``dst_tag`` is given, the result is
        added as a leaf, inheriting ``cfgs`` from the source.
        """
        lf = self.database[tag]
        mean = f(lf.mean)
        jks = np.array([f(jk) for jk in lf.jks]) if lf.jks is not None else None
        bss = np.array([f(b) for b in lf.bss]) if lf.bss is not None else None
        if dst_tag is not None:
            self.add_leaf(dst_tag, mean=mean, jks=jks, cfgs=lf.cfgs, bss=bss)
        return mean, jks, bss

    def transform_jks(self, tag, f):
        """Return ``f``-mapped ``jks`` array of the leaf at ``tag``."""
        lf = self.database[tag]
        if self.num_proc is None:
            return np.array([f(jk) for jk in lf.jks])
        message(f"Spawn {self.num_proc} processes to transform jackknife sample.", silent=True)
        with multiprocessing.Pool(self.num_proc) as pool:
            return np.array(pool.map(f, lf.jks))

    def transform_bss(self, tag, f):
        """Return ``f``-mapped ``bss`` array of the leaf at ``tag``.

        If the leaf has no stored ``bss`` (raw-data leaf with sample+weights),
        bootstrap samples are computed on the fly via :meth:`bss`.
        """
        lf = self.database[tag]
        bss = lf.bss if lf.bss is not None else self.bss(tag)
        if self.num_proc is None:
            return np.array([f(b) for b in bss])
        message(f"Spawn {self.num_proc} processes to transform bootstrap sample.", silent=True)
        with multiprocessing.Pool(self.num_proc) as pool:
            return np.array(pool.map(f, bss))

    ################################ COMBINE ###################################

    def combine(self, *tags, f, dst_tag=None):
        """Combine leaves at ``tags`` cfg-wise by applying ``f`` across them.

        Cfg sets may differ: a union is taken in encounter order (first
        leaf's cfgs in their leaf order, then any new cfgs from subsequent
        leaves). For each cfg in the union, leaves missing that cfg
        contribute their ``mean`` (= "no fluctuation at this cfg").

        ``bss`` are aligned by bootstrap index (cfgs are irrelevant) and
        combined only if every input leaf has ``bss`` set.
        """
        lfs = [self.database[tag] for tag in tags]
        for lf in lfs:
            assert lf.cfgs is not None, "combine inputs must have cfgs"
            assert lf.jks is not None,  "combine inputs must have jks"

        # Union cfgs in encounter order — preserves tags[0]'s ordering.
        seen = set()
        union_cfgs = []
        for lf in lfs:
            for c in lf.cfgs:
                if c not in seen:
                    seen.add(c)
                    union_cfgs.append(c)
        union_cfgs = np.array(union_cfgs)

        idx_maps = [{c: i for i, c in enumerate(lf.cfgs)} for lf in lfs]

        mean = f(*[lf.mean for lf in lfs])
        jks = np.array([
            f(*[lf.jks[idx_maps[i][c]] if c in idx_maps[i] else lf.mean
                for i, lf in enumerate(lfs)])
            for c in union_cfgs
        ])
        bss = None
        if all(lf.bss is not None for lf in lfs):
            n_bs = lfs[0].bss.shape[0]
            bss = np.array([f(*[lf.bss[i] for lf in lfs]) for i in range(n_bs)])

        if dst_tag is not None:
            self.add_leaf(dst_tag, mean=mean, jks=jks, cfgs=union_cfgs, bss=bss)
        return mean, jks, bss

    ################################ BINNING / CONCAT ##########################

    def add_binned_leaf(self, tag, binsize):
        """Bin leaf ``tag`` into a new leaf at ``<tag>/binsize<binsize>``.

        Reshapes ``sample`` and ``weights`` into ``(n_bins, binsize, ...)``
        and averages along axis 1. Cfgs become synthetic ``f"{src}-bin{i}"``
        labels.
        """
        if binsize == 1:
            message(f"{tag} is already in database. Nothing to do.")
            return tag
        if "binsize" in tag:
            raise AssertionError(f"{tag} is already binned. Can only bin unbinned leafs.")
        src_lf = self.database[tag]
        assert src_lf.sample is not None and src_lf.weights is not None, \
            f"{tag} must be a data leaf (sample+weights) to bin"
        # statistics.bin truncates a trailing incomplete bin (matches legacy behaviour).
        n_bins = len(src_lf.sample) // binsize
        binned_sample = statistics.bin(src_lf.sample, binsize, weights=src_lf.weights)
        binned_weights = statistics.bin(src_lf.weights, binsize=binsize)
        # Synthetic cfg labels for bins.
        branch_tag = tag.split("/")[0]
        binned_cfgs = np.array([f"{branch_tag}-bin{i}" for i in range(n_bins)])
        binned_tag = f"{tag}/binsize{binsize}"
        self.add_leaf(
            binned_tag,
            sample=binned_sample, weights=binned_weights, cfgs=binned_cfgs,
            misc=src_lf.misc,
        )
        return binned_tag

    def concatenate_samples(self, *tags, dst_tag=None, dst_cfgs=None):
        """Concatenate the sample arrays of multiple data leaves.

        Cfgs are concatenated in order unless ``dst_cfgs`` is given
        (must match total length).
        """
        lfs = [self.database[tag] for tag in tags]
        for lf in lfs:
            assert lf.sample is not None and lf.weights is not None and lf.cfgs is not None, \
                "concatenate_samples inputs must be data leaves"
        sample = np.concatenate([lf.sample for lf in lfs], axis=0)
        weights = np.concatenate([lf.weights for lf in lfs], axis=0)
        if dst_cfgs is None:
            cfgs = np.concatenate([lf.cfgs for lf in lfs], axis=0)
        else:
            cfgs = np.asarray(dst_cfgs)
            assert len(cfgs) == len(sample), \
                f"dst_cfgs length {len(cfgs)} != total sample length {len(sample)}"
        if dst_tag is None:
            return cfgs, sample, weights
        self.add_leaf(dst_tag, sample=sample, weights=weights, cfgs=cfgs)

    def remove_cfgs(self, tag, *cfgs, dst_tag=None):
        """Drop cfgs from a data leaf by name; return or store a new leaf.

        If ``dst_tag`` is given, the result is added as a new leaf;
        otherwise returns the filtered ``(cfgs, sample, weights)`` tuple.
        """
        lf = self.database[tag]
        assert lf.sample is not None, f"remove_cfgs requires a data leaf at {tag}"
        cfgs_to_drop = set(str(c) for c in cfgs)
        mask = np.array([c not in cfgs_to_drop for c in lf.cfgs])
        new_cfgs = lf.cfgs[mask]
        new_sample = lf.sample[mask]
        new_weights = lf.weights[mask]
        if dst_tag is None:
            return new_cfgs, new_sample, new_weights
        self.add_leaf(
            dst_tag, sample=new_sample, weights=new_weights, cfgs=new_cfgs,
            misc=lf.misc,
        )

    ################################ STATISTICS ################################

    def jks(self, tag, binsize):
        """Compute jackknife resamples of ``tag``'s sample with ``binsize``."""
        lf = self.database[tag]
        bsample = statistics.bin(lf.sample, binsize, weights=lf.weights)
        bweights = statistics.bin(lf.weights, binsize=binsize)
        return jackknife.sample(bsample, weights=bweights)

    def jackknife_variance(self, tag, binsize=1):
        assert ("binsize" not in tag) or (binsize == 1)
        lf = self.database[tag]
        jks = lf.jks if binsize == 1 else self.jks(tag, binsize)
        return jackknife.variance(jks)

    def jackknife_covariance(self, tag, binsize=1):
        assert ("binsize" not in tag) or (binsize == 1)
        lf = self.database[tag]
        jks = lf.jks if binsize == 1 else self.jks(tag, binsize)
        return jackknife.covariance(jks)

    def sample_binning_study(self, tag, binsizes):
        message(f"Binning study with unbinned sample size: {len(self.database[tag].sample)}")
        var = {}
        for b in binsizes:
            var[b] = self.jackknife_variance(tag, b)
        return var

    def add_bootstrap(self, branch_tag, bootstraps, configlist):
        message(f"Add bootstraps for {branch_tag} to database.")
        self.add_leaf(
            f"{branch_tag}/bootstraps",
            mean=bootstraps, misc={"configlist": configlist},
        )

    def bss(self, tag):
        """Compute bootstrap samples of ``tag``'s sample using the matching
        ``/bootstraps`` leaf."""
        assert "binsize" not in tag, "Can only compute bss for unbinned leafs"
        lf = self.database[tag]
        bootstraps = self.database[f"{tag.split('/')[0]}/bootstraps"].mean
        return bootstrap.sample(lf.sample, bootstraps, weights=lf.weights)

    def bootstrap_variance(self, tag):
        return bootstrap.variance(self.database[tag].bss)

    def bootstrap_covariance(self, tag):
        return bootstrap.covariance(self.database[tag].bss)
