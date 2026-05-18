import os
import pickle
import re
import struct
import zlib
import numpy as np

import statpy
from statpy.log import message
from statpy.database.entries import Entry

from statpy.statistics import core as statistics
from statpy.statistics import jackknife, bootstrap

_MAGIC = b"SPDB"  # statpy DB file marker; followed by 4-byte little-endian CRC32 of payload
_commit_logged = False


class DB:
    """Entry store with statistics + I/O for lattice QCD analyses."""

    def __init__(self, *args):
        """Create an empty DB; ``*args`` of pickle paths or other ``DB``s are merged in."""
        global _commit_logged
        self.database = {}
        if not _commit_logged:
            message(f"Initialized database with statpy commit hash {statpy.__commit__}.")
            _commit_logged = True
        for src in args:
            if isinstance(src, str):
                self.load(src)
            elif isinstance(src, DB):
                for t, entry in src.database.items():
                    self.database[t] = Entry(
                        mean=entry.mean, jks=entry.jks, sample=entry.sample,
                        weights=entry.weights, cfgs=entry.cfgs, bss=entry.bss, misc=entry.misc,
                        binsize=entry.binsize,
                    )

    def load(self, src):
        """Load a snapshot saved by :func:`save` and add every entry."""
        message(f"Load {src}")
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
        for t, entry in src_db.items():
            self.database[t] = entry

    def save(self, dst):
        """Write the entire database (all fields, including samples) to ``dst``."""
        payload = pickle.dumps(self.database, protocol=pickle.HIGHEST_PROTOCOL)
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        with open(dst, "wb") as f:
            f.write(_MAGIC)
            f.write(struct.pack("<I", crc))
            f.write(payload)

    ################################ ENTRY MANAGEMENT ##########################

    def add_entry(self, tag, *, mean=None, jks=None, sample=None, weights=None,
                 cfgs=None, bss=None, misc=None, binsize=1):
        """Add a new entry at ``tag``.

        Three valid shapes:
          - Data entry: ``sample`` + ``weights`` + ``cfgs`` (matching length);
            ``jks`` and ``mean`` are auto-derived.
          - Derived entry: ``sample=None`` but ``jks`` and ``cfgs`` given
            (matching length).
          - Result entry: only ``mean`` and optionally ``bss``.
        ``binsize`` is 1 for raw, >1 for binned (see :meth:`bin_entry`).
        """
        if tag in self.database:
            message(f"{tag} already in database. Entry not added.")
            return

        if sample is not None:
            if weights is None:
                raise ValueError(f"add_entry({tag!r}): sample requires weights")
            if cfgs is None:
                raise ValueError(f"add_entry({tag!r}): sample requires cfgs")
            if not isinstance(sample, np.ndarray):
                raise TypeError(f"add_entry({tag!r}): sample must be np.ndarray, got {type(sample).__name__}")
            if not isinstance(weights, np.ndarray):
                raise TypeError(f"add_entry({tag!r}): weights must be np.ndarray, got {type(weights).__name__}")
            if not isinstance(cfgs, np.ndarray):
                raise TypeError(f"add_entry({tag!r}): cfgs must be np.ndarray, got {type(cfgs).__name__}")
            if not (len(sample) == len(weights) == len(cfgs)):
                raise ValueError(
                    f"add_entry({tag!r}): length mismatch sample={len(sample)} "
                    f"weights={len(weights)} cfgs={len(cfgs)}"
                )
            if jks is not None:
                raise ValueError(f"add_entry({tag!r}): do not pass jks when sample+weights are given; jks is derived")
            jks = jackknife.sample(sample, weights=weights)
            if mean is None:
                mean = np.mean(jks, axis=0)
        else:
            if cfgs is not None and jks is not None and len(jks) != len(cfgs):
                raise ValueError(
                    f"add_entry({tag!r}): jks/cfgs length mismatch jks={len(jks)} cfgs={len(cfgs)}"
                )

        self.database[tag] = Entry(
            mean=mean, jks=jks, sample=sample, weights=weights,
            cfgs=cfgs, bss=bss, misc=misc, binsize=binsize,
        )

    def remove_entry(self, tag):
        """Drop the entry at ``tag``."""
        if tag in self.database:
            del self.database[tag]
        else:
            message(f"remove_entry: {tag!r} not in database.")

    def rename_entry(self, old, new):
        """Move the entry from ``old`` to ``new``."""
        if old not in self.database:
            message(f"rename_entry: {old!r} not in database.")
            return
        if new in self.database:
            message(f"rename_entry: {new!r} already in database. Not renamed.")
            return
        self.database[new] = self.database[old]
        del self.database[old]

    def __repr__(self):
        return f"<DB n_entries={len(self.database)}>"

    def __str__(self):
        """Multi-line overview: entry counts per category."""
        counts = {"raw data": 0, "binned data": 0, "derived (jks)": 0, "result (mean/bss)": 0}
        for entry in self.database.values():
            if entry.sample is not None and entry.binsize == 1:
                counts["raw data"] += 1
            elif entry.sample is not None:
                counts["binned data"] += 1
            elif entry.jks is not None:
                counts["derived (jks)"] += 1
            else:
                counts["result (mean/bss)"] += 1
        lines = [f"DB with {len(self.database)} entries:"]
        for cat, n in counts.items():
            lines.append(f"  {cat:20s} {n}")
        return "\n".join(lines)

    ################################ QUERIES ###################################

    def get_tags(self, pattern=".*"):
        """Tags matching the regex ``pattern`` (via :func:`re.search`)."""
        return [tag for tag in self.database.keys() if re.search(pattern, tag)]

    def get_cfgs(self, tag):
        """Cfg labels of the entry at ``tag`` as a list."""
        return list(self.database[tag].cfgs)

    ################################ TRANSFORM #################################

    def transform(self, tag, f, dst_tag=None):
        """Apply ``f`` to ``mean``, every jackknife and (if present) every
        bootstrap sample of the entry at ``tag``; optionally store as ``dst_tag``."""
        entry = self.database[tag]
        mean = f(entry.mean)
        jks = self.transform_jks(tag, f) if entry.jks is not None else None
        bss = self.transform_bss(tag, f) if entry.bss is not None else None
        if dst_tag is not None:
            self.add_entry(dst_tag, mean=mean, jks=jks, cfgs=entry.cfgs, bss=bss)
        return mean, jks, bss

    def transform_jks(self, tag, f):
        """Return ``f``-mapped ``jks`` array of the entry at ``tag``."""
        return np.array([f(jk) for jk in self.database[tag].jks])

    def transform_bss(self, tag, f, bootstraps=None):
        """Return ``f``-mapped ``bss`` array of the entry at ``tag``.

        If the entry has no stored ``bss`` (raw-data entry with sample+weights),
        bootstrap samples are computed on the fly via :meth:`bss`; pass the
        ``(n_bs, n_cfgs)`` bootstrap-index matrix as ``bootstraps``.
        """
        entry = self.database[tag]
        if entry.bss is not None:
            bss = entry.bss
        else:
            if bootstraps is None:
                raise ValueError(f"transform_bss({tag!r}): entry has no stored bss; pass bootstraps=<index matrix>")
            bss = self.bss(tag, bootstraps)
        return np.array([f(b) for b in bss])

    ################################ COMBINE ###################################

    def combine(self, *tags, f, dst_tag=None):
        """Combine multiple entries cfg-wise by applying ``f`` across them.

        Cfg sets may differ — the union is taken in encounter order and
        entries missing a cfg contribute their ``mean`` (= "no fluctuation
        at this cfg"). ``bss`` are aligned by bootstrap index and combined
        only if every input has ``bss`` set.
        """
        entries = [self.database[tag] for tag in tags]
        for tag, entry in zip(tags, entries):
            if entry.cfgs is None:
                raise ValueError(f"combine({tag!r}): input entry has no cfgs")
            if entry.jks is None:
                raise ValueError(f"combine({tag!r}): input entry has no jks")

        # Union cfgs in encounter order — preserves tags[0]'s ordering.
        seen = set()
        union_cfgs = []
        for entry in entries:
            for c in entry.cfgs:
                if c not in seen:
                    seen.add(c)
                    union_cfgs.append(c)
        union_cfgs = np.array(union_cfgs)

        idx_maps = [{c: i for i, c in enumerate(entry.cfgs)} for entry in entries]

        mean = f(*[entry.mean for entry in entries])
        jks = np.array([
            f(*[entry.jks[idx_maps[i][c]] if c in idx_maps[i] else entry.mean
                for i, entry in enumerate(entries)])
            for c in union_cfgs
        ])
        bss = None
        if all(entry.bss is not None for entry in entries):
            n_bs = entries[0].bss.shape[0]
            bss = np.array([f(*[entry.bss[i] for entry in entries]) for i in range(n_bs)])

        if dst_tag is not None:
            self.add_entry(dst_tag, mean=mean, jks=jks, cfgs=union_cfgs, bss=bss)
        return mean, jks, bss

    ################################ BINNING ###################################

    def bin_entry(self, tag, binsize):
        """Bin the data entry at ``tag`` into bins of ``binsize`` configs.

        Returns a dict of the binned ``sample``, ``weights``, ``cfgs``,
        ``binsize`` and ``misc``, ready to splat into :meth:`add_entry` --
        the caller picks the destination tag. Nothing is stored here.

        Cfgs become synthetic ``f"{src}-bin{i}"`` labels; the trailing
        incomplete bin is truncated.
        """
        if binsize <= 1:
            raise ValueError(f"bin_entry({tag!r}): binsize must be > 1, got {binsize}")
        src_entry = self.database[tag]
        if src_entry.binsize != 1:
            raise ValueError(f"bin_entry({tag!r}): entry is already binned (binsize={src_entry.binsize})")
        if src_entry.sample is None or src_entry.weights is None:
            raise ValueError(f"bin_entry({tag!r}): entry must carry sample and weights")
        # statistics.bin truncates a trailing incomplete bin.
        n_bins = len(src_entry.sample) // binsize
        prefix = tag.split("/")[0]
        return {
            "sample": statistics.bin(src_entry.sample, binsize, weights=src_entry.weights),
            "weights": statistics.bin(src_entry.weights, binsize=binsize),
            "cfgs": np.array([f"{prefix}-bin{i}" for i in range(n_bins)]),
            "misc": src_entry.misc,
            "binsize": binsize,
        }

    ################################ STATISTICS ################################

    def jackknife_variance(self, tag):
        """Variance of the entry's stored jackknife samples."""
        return jackknife.variance(self.database[tag].jks)

    def jackknife_covariance(self, tag):
        """Covariance of the entry's stored jackknife samples."""
        return jackknife.covariance(self.database[tag].jks)

    def bss(self, tag, bootstraps):
        """Compute bootstrap samples of ``tag``'s sample.

        ``bootstraps`` is the ``(n_bs, n_cfgs)`` index matrix produced by
        :func:`statpy.statistics.bootstrap.generate_bootstraps` (or read
        from a ``.boot.txt`` file via :func:`parse_bootstrap_file`).
        ``tag`` must point at an unbinned entry — the indices reference
        cfg positions, not bin positions.
        """
        entry = self.database[tag]
        return bootstrap.sample(entry.sample, bootstraps, weights=entry.weights)

    def bootstrap_variance(self, tag):
        """Variance of the entry's stored bootstrap samples."""
        return bootstrap.variance(self.database[tag].bss)

    def bootstrap_covariance(self, tag):
        """Covariance of the entry's stored bootstrap samples."""
        return bootstrap.covariance(self.database[tag].bss)
