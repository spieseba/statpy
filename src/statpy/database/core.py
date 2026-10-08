import os
import pickle
import re
import struct
import zlib

import numpy as np

import statpy
from statpy.database.entries import Entry
from statpy.log import message
from statpy.statistics import bootstrap, jackknife
from statpy.statistics import core as statistics

# DB file: magic (format version) + u32 CRC32 + pickled {tag: entry state dict}.
# Bump the magic on any format change. Retired: v1 JSON, v2 SPDB, v3 SPD3.
_MAGIC = b"SPD4"
_commit_logged = False


class DuplicateTagError(ValueError):
    """An operation would write a tag that already exists in the DB."""


def _check_unique_cfgs(context, cfgs):
    """Raise ValueError if ``cfgs`` contains duplicate labels."""
    uniq, counts = np.unique(cfgs, return_counts=True)
    dup = uniq[counts > 1].tolist()
    if dup:
        raise ValueError(
            f"{context}: duplicate cfg label(s): {dup[:5]}"
            + ("..." if len(dup) > 5 else "")
        )


class DB:
    """Entry store with statistics + I/O for lattice QCD analyses."""

    def __init__(self, *args):
        """Create an empty DB; ``*args`` of pickle paths or other ``DB``s are merged in.

        Tag sets must be disjoint; overlaps raise :class:`DuplicateTagError`.
        """
        global _commit_logged
        self.database = {}
        if not _commit_logged:
            message(f"Initialized database with statpy commit hash {statpy.__commit__}.")
            _commit_logged = True
        for src in args:
            if isinstance(src, str):
                self.load(src)
            elif isinstance(src, DB):
                dup = [t for t in src.database if t in self.database]
                if dup:
                    raise DuplicateTagError(
                        f"DB merge: {len(dup)} overlapping tag(s): {dup[:5]}"
                        + ("..." if len(dup) > 5 else "")
                    )
                for t, entry in src.database.items():
                    if entry.configurations is not None:
                        _check_unique_cfgs(f"DB merge: entry {t!r}", entry.configurations)
                for t, entry in src.database.items():
                    self.database[t] = Entry(**vars(entry))

    def load(self, src):
        """Load a snapshot saved by :func:`save` and add every entry.

        Tags overlapping with existing entries raise :class:`DuplicateTagError`.
        """
        message(f"Load {src}")
        if not os.path.isfile(src):
            raise FileNotFoundError(f"{src} not found")
        with open(src, "rb") as f:
            header = f.read(8)
            if len(header) < 8 or header[:4] != _MAGIC:
                raise ValueError(
                    f"{src}: not a statpy DB file, or an outdated format "
                    f"(expected magic {_MAGIC.decode()}); regenerate if outdated"
                )
            (crc_expected,) = struct.unpack("<I", header[4:8])
            payload = f.read()
        if (zlib.crc32(payload) & 0xFFFFFFFF) != crc_expected:
            raise ValueError(f"{src}: CRC mismatch -- file corrupted")
        src_db = pickle.loads(payload)
        dup = [t for t in src_db if t in self.database]
        if dup:
            raise DuplicateTagError(
                f"load({src!r}): {len(dup)} overlapping tag(s): {dup[:5]}"
                + ("..." if len(dup) > 5 else "")
            )
        # Validate every entry before inserting any, so a bad file leaves
        # the DB untouched.
        for t, state in src_db.items():
            if state.get("configurations") is not None:
                _check_unique_cfgs(f"load({src!r}): entry {t!r}", state["configurations"])
        for t, state in src_db.items():
            self.database[t] = Entry(**state)

    def save(self, dst):
        """Write the entire database (all fields, including samples) to ``dst``."""
        payload = pickle.dumps(
            {t: vars(entry) for t, entry in self.database.items()},
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        with open(dst, "wb") as f:
            f.write(_MAGIC)
            f.write(struct.pack("<I", crc))
            f.write(payload)

    ################################ ENTRY MANAGEMENT ##########################

    def add_entry(self, tag, *, central_value=None, jackknife_samples=None, samples=None, weights=None,
                 configurations=None, bootstrap_samples=None, metadata=None, bin_size=1):
        """Add a new entry at ``tag``.

        With ``samples`` (data entry): requires ``weights`` and ``configurations``
        of the same length; ``jackknife_samples`` is derived for the weighted mean
        and ``central_value`` defaults to it.
        Without ``samples``: stores the given ``central_value``,
        ``jackknife_samples``, ``configurations`` and ``bootstrap_samples``
        unchanged; nothing is derived. ``jackknife_samples`` requires
        ``configurations`` of the same length.
        ``configurations`` labels must be unique; :meth:`combine` aligns entries
        by them. ``bin_size`` is 1 for raw, >1 for binned (see :meth:`bin_entry`).

        Entries are create-only: an existing ``tag`` raises
        :class:`DuplicateTagError`. To replace, :meth:`remove_entry` first.
        """
        if tag in self.database:
            raise DuplicateTagError(f"add_entry({tag!r}): tag already exists")

        if samples is not None:
            if weights is None:
                raise ValueError(f"add_entry({tag!r}): samples requires weights")
            if configurations is None:
                raise ValueError(f"add_entry({tag!r}): samples requires configurations")
            if not isinstance(samples, np.ndarray):
                raise TypeError(f"add_entry({tag!r}): samples must be np.ndarray, got {type(samples).__name__}")
            if not isinstance(weights, np.ndarray):
                raise TypeError(f"add_entry({tag!r}): weights must be np.ndarray, got {type(weights).__name__}")
            if not isinstance(configurations, np.ndarray):
                raise TypeError(f"add_entry({tag!r}): configurations must be np.ndarray, got {type(configurations).__name__}")
            if not (len(samples) == len(weights) == len(configurations)):
                raise ValueError(
                    f"add_entry({tag!r}): length mismatch samples={len(samples)} "
                    f"weights={len(weights)} configurations={len(configurations)}"
                )
            if jackknife_samples is not None:
                raise ValueError(f"add_entry({tag!r}): do not pass jackknife_samples when samples+weights are given; they are derived")
            jackknife_samples = jackknife.mean_sample(samples, weights=weights)
            if central_value is None:
                central_value = np.average(samples, axis=0, weights=weights)
        elif jackknife_samples is not None:
            if configurations is None:
                raise ValueError(f"add_entry({tag!r}): jackknife_samples requires configurations")
            if len(jackknife_samples) != len(configurations):
                raise ValueError(
                    f"add_entry({tag!r}): jackknife_samples/configurations length mismatch "
                    f"jackknife_samples={len(jackknife_samples)} configurations={len(configurations)}"
                )

        if configurations is not None:
            _check_unique_cfgs(f"add_entry({tag!r})", configurations)

        self.database[tag] = Entry(
            central_value=central_value, jackknife_samples=jackknife_samples, samples=samples, weights=weights,
            configurations=configurations, bootstrap_samples=bootstrap_samples, metadata=metadata, bin_size=bin_size,
        )

    def remove_entry(self, tag):
        """Drop the entry at ``tag``."""
        if tag in self.database:
            del self.database[tag]
        else:
            message(f"remove_entry: {tag!r} not in database.")

    def rename_entry(self, old, new):
        """Move the entry from ``old`` to ``new``; an existing ``new`` raises :class:`DuplicateTagError`."""
        if old not in self.database:
            message(f"rename_entry: {old!r} not in database.")
            return
        if new in self.database:
            raise DuplicateTagError(f"rename_entry({old!r} -> {new!r}): target tag already exists")
        self.database[new] = self.database[old]
        del self.database[old]

    def __repr__(self):
        return f"<DB num_entries={len(self.database)}>"

    def __str__(self):
        """Multi-line overview: entry counts per category."""
        counts = {"raw data": 0, "binned data": 0, "derived (jackknife)": 0, "result (central/bootstrap)": 0}
        for entry in self.database.values():
            if entry.samples is not None and entry.bin_size == 1:
                counts["raw data"] += 1
            elif entry.samples is not None:
                counts["binned data"] += 1
            elif entry.jackknife_samples is not None:
                counts["derived (jackknife)"] += 1
            else:
                counts["result (central/bootstrap)"] += 1
        lines = [f"DB with {len(self.database)} entries:"]
        for cat, n in counts.items():
            lines.append(f"  {cat:20s} {n}")
        return "\n".join(lines)

    ################################ QUERIES ###################################

    def get_tags(self, pattern=".*"):
        """Tags matching the regex ``pattern`` (via :func:`re.search`)."""
        return [tag for tag in self.database if re.search(pattern, tag)]

    def print_tags(self, pattern=".*"):
        """Print the tags matching ``pattern``, one per line, sorted."""
        print(*sorted(self.get_tags(pattern)), sep="\n")

    def get_configurations(self, tag):
        """Configuration labels of the entry at ``tag`` as a list."""
        return list(self.database[tag].configurations)

    ################################ TRANSFORM #################################

    def transform(self, tag, f, store_as=None):
        """Apply ``f`` to ``central_value``, every jackknife and (if present)
        every bootstrap sample of the entry at ``tag``.

        If ``store_as`` is given, the result is stored under that tag and
        nothing is returned; otherwise the
        ``(central_value, jackknife_samples, bootstrap_samples)`` tuple is returned.
        """
        entry = self.database[tag]
        central_value = f(entry.central_value)
        jks = self.transform_jackknife(tag, f) if entry.jackknife_samples is not None else None
        bss = self.transform_bootstrap(tag, f) if entry.bootstrap_samples is not None else None
        if store_as is not None:
            self.add_entry(store_as, central_value=central_value, jackknife_samples=jks,
                           configurations=entry.configurations, bootstrap_samples=bss)
            return
        return central_value, jks, bss

    def transform_jackknife(self, tag, f):
        """Return ``f`` applied to every jackknife sample of the entry at ``tag``."""
        return np.array([f(jk) for jk in self.database[tag].jackknife_samples])

    def transform_bootstrap(self, tag, f, bootstraps=None):
        """Return ``f`` applied to every bootstrap sample of the entry at ``tag``.

        If the entry has no stored ``bootstrap_samples`` (raw-data entry with
        samples+weights), they are computed on the fly via :meth:`bootstrap_samples`;
        pass the ``(num_bs, num_configurations)`` bootstrap-index matrix as ``bootstraps``.
        """
        entry = self.database[tag]
        if entry.bootstrap_samples is not None:
            bss = entry.bootstrap_samples
        else:
            if bootstraps is None:
                raise ValueError(f"transform_bootstrap({tag!r}): entry has no stored bootstrap_samples; pass bootstraps=<index matrix>")
            bss = self.bootstrap_samples(tag, bootstraps)
        return np.array([f(b) for b in bss])

    ################################ COMBINE ###################################

    def combine(self, *tags, f, store_as=None):
        """Combine multiple entries cfg-wise by applying ``f`` across them.

        Cfg sets may differ — the union is taken in encounter order and
        entries missing a cfg contribute their ``central_value`` (= "no
        fluctuation at this cfg"). ``bootstrap_samples`` are aligned by bootstrap
        index and combined only if every input has them.

        If ``store_as`` is given, the result is stored under that tag and
        nothing is returned; otherwise the
        ``(central_value, jackknife_samples, bootstrap_samples)`` tuple is returned.
        """
        entries = [self.database[tag] for tag in tags]
        for tag, entry in zip(tags, entries):
            if entry.configurations is None:
                raise ValueError(f"combine({tag!r}): input entry has no configurations")
            if entry.jackknife_samples is None:
                raise ValueError(f"combine({tag!r}): input entry has no jackknife_samples")

        # Union cfgs in encounter order — preserves tags[0]'s ordering.
        seen = set()
        union_cfgs = []
        for entry in entries:
            for c in entry.configurations:
                if c not in seen:
                    seen.add(c)
                    union_cfgs.append(c)
        union_cfgs = np.array(union_cfgs)

        idx_maps = [{c: i for i, c in enumerate(entry.configurations)} for entry in entries]

        central_value = f(*[entry.central_value for entry in entries])
        jks = np.array([
            f(*[entry.jackknife_samples[idx_maps[i][c]] if c in idx_maps[i] else entry.central_value
                for i, entry in enumerate(entries)])
            for c in union_cfgs
        ])
        bss = None
        if all(entry.bootstrap_samples is not None for entry in entries):
            num_bs = entries[0].bootstrap_samples.shape[0]
            bss = np.array([f(*[entry.bootstrap_samples[i] for entry in entries]) for i in range(num_bs)])

        if store_as is not None:
            self.add_entry(store_as, central_value=central_value, jackknife_samples=jks,
                           configurations=union_cfgs, bootstrap_samples=bss)
            return
        return central_value, jks, bss

    ################################ BINNING ###################################

    def bin_entry(self, tag, bin_size):
        """Bin the data entry at ``tag`` into bins of ``bin_size`` configs.

        Returns a dict of the binned ``samples``, ``weights``, ``configurations``,
        ``bin_size`` and ``metadata``, ready to splat into :meth:`add_entry` --
        the caller picks the destination tag. Nothing is stored here.

        Configurations become synthetic ``f"{src}-bin{i}"`` labels; the trailing
        incomplete bin is truncated.
        """
        if bin_size <= 1:
            raise ValueError(f"bin_entry({tag!r}): bin_size must be > 1, got {bin_size}")
        src_entry = self.database[tag]
        if src_entry.bin_size != 1:
            raise ValueError(f"bin_entry({tag!r}): entry is already binned (bin_size={src_entry.bin_size})")
        if src_entry.samples is None or src_entry.weights is None:
            raise ValueError(f"bin_entry({tag!r}): entry must carry samples and weights")
        # statistics.bin truncates a trailing incomplete bin.
        num_bins = len(src_entry.samples) // bin_size
        prefix = tag.split("/")[0]
        return {
            "samples": statistics.bin(src_entry.samples, bin_size, weights=src_entry.weights),
            "weights": statistics.bin(src_entry.weights, bin_size, weights=np.ones(len(src_entry.weights))),
            "configurations": np.array([f"{prefix}-bin{i}" for i in range(num_bins)]),
            "metadata": src_entry.metadata,
            "bin_size": bin_size,
        }

    ################################ CONCATENATION #############################

    def concatenate_entries(self, tags):
        """Join unbinned data entries per config along the data axis.

        Returns a dict of ``samples``, ``central_value``, ``weights``,
        ``configurations`` and ``metadata``, ready to splat into :meth:`add_entry`.
        Nothing is stored here.
        """
        entries = [self.database[tag] for tag in tags]
        for tag, entry in zip(tags, entries):
            if entry.bin_size != 1 or entry.samples is None:
                raise ValueError(f"concatenate_entries: {tag!r} must be an unbinned data entry")
        first = entries[0]
        for tag, entry in zip(tags[1:], entries[1:]):
            if not np.array_equal(entry.configurations, first.configurations):
                raise ValueError(f"concatenate_entries: configurations of {tag!r} and {tags[0]!r} differ")
            if not np.array_equal(entry.weights, first.weights):
                raise ValueError(f"concatenate_entries: weights of {tag!r} and {tags[0]!r} differ")
            if entry.samples.shape != first.samples.shape:
                raise ValueError(f"concatenate_entries: samples shapes of {tag!r} and {tags[0]!r} differ")
        return {
            "samples": np.concatenate([e.samples for e in entries], axis=1),
            "central_value": np.concatenate([e.central_value for e in entries]),
            "weights": first.weights,
            "configurations": first.configurations,
            "metadata": {"tags": tuple(tags)},
        }

    ################################ STATISTICS ################################

    def jackknife_variance(self, tag):
        """Variance of the entry's stored jackknife samples."""
        return jackknife.variance(self.database[tag].jackknife_samples)

    def jackknife_covariance(self, tag):
        """Covariance of the entry's stored jackknife samples."""
        return jackknife.covariance(self.database[tag].jackknife_samples)

    def bootstrap_samples(self, tag, bootstraps):
        """Compute bootstrap samples of ``tag``'s samples.

        ``bootstraps`` is the ``(num_bs, num_configurations)`` index matrix produced by
        :func:`statpy.statistics.bootstrap.generate_bootstraps` (or read
        from a ``.boot.txt`` file via :func:`parse_bootstrap_file`).
        ``tag`` must point at an unbinned entry — the indices reference
        cfg positions, not bin positions.
        """
        entry = self.database[tag]
        return bootstrap.mean_sample(entry.samples, bootstraps, weights=entry.weights)

    def bootstrap_variance(self, tag):
        """Variance of the entry's stored bootstrap samples."""
        return bootstrap.variance(self.database[tag].bootstrap_samples)

    def bootstrap_covariance(self, tag):
        """Covariance of the entry's stored bootstrap samples."""
        return bootstrap.covariance(self.database[tag].bootstrap_samples)
