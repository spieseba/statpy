import os
import pickle
import struct
import zlib
from collections.abc import Mapping

import statpy
from statpy.log import message
from statpy.variable import Layout, Variable, _check_unique_cfgs

# Database file: magic number (format version) + u32 CRC32 + pickled {tag: variable state dict}.
# Change the magic number on any format change. Retired: v1 JSON, v2 SPDB, v3 SPD3, v4 SPD4, v5 SPD5.
_MAGIC = b"SPD6"
_commit_logged = False


class DuplicateTagError(ValueError):
    """An operation would write a tag that already exists in the Database."""


class Database(Mapping):
    """Dict-like store of Variables by tag, saved to and loaded from a database file."""

    def __init__(self, *args):
        """Create a Database from database files or Databases with disjoint tags."""
        global _commit_logged
        self._variables = {}
        if not _commit_logged:
            message(f"Initialized database with statpy commit hash {statpy.__commit__}.")
            _commit_logged = True
        for src in args:
            if isinstance(src, str):
                self.load(src)
            elif isinstance(src, Database):
                dup = [t for t in src._variables if t in self._variables]
                if dup:
                    raise DuplicateTagError(
                        f"Database merge: {len(dup)} overlapping tag(s): {dup[:5]}"
                        + ("..." if len(dup) > 5 else "")
                    )
                for t, variable in src._variables.items():
                    if variable.layout is not None:
                        _check_unique_cfgs(f"Database merge: variable {t!r}", variable.layout.configurations)
                for t, variable in src._variables.items():
                    self._variables[t] = Variable(**vars(variable))

    def load(self, src):
        """Load a database file and add its variables; tags must not overlap."""
        message(f"Load {src}")
        if not os.path.isfile(src):
            raise FileNotFoundError(f"{src} not found")
        with open(src, "rb") as f:
            header = f.read(8)
            if len(header) < 8 or header[:4] != _MAGIC:
                raise ValueError(
                    f"{src}: not a statpy database file, or an outdated format "
                    f"(expected magic {_MAGIC.decode()}); regenerate if outdated"
                )
            (crc_expected,) = struct.unpack("<I", header[4:8])
            payload = f.read()
        if (zlib.crc32(payload) & 0xFFFFFFFF) != crc_expected:
            raise ValueError(f"{src}: CRC mismatch -- file corrupted")
        src_db = pickle.loads(payload)
        dup = [t for t in src_db if t in self._variables]
        if dup:
            raise DuplicateTagError(
                f"load({src!r}): {len(dup)} overlapping tag(s): {dup[:5]}"
                + ("..." if len(dup) > 5 else "")
            )
        # Validate every variable before inserting any
        for t, state in src_db.items():
            if state.get("configurations") is not None:
                _check_unique_cfgs(f"load({src!r}): variable {t!r}", state["configurations"])
        variables = {}
        for t, state in src_db.items():
            state = state.copy()
            configurations = state.pop("configurations")
            weights = state.pop("weights")
            layout = Layout(configurations, weights) if configurations is not None else None
            variables[t] = Variable(layout=layout, **state)
        self._variables.update(variables)

    def save(self, dst):
        """Write the entire database (all fields, including observations) to ``dst``."""
        payload = pickle.dumps(
            {t: {
                "central_value": variable.central_value,
                "jackknife_replicates": variable.jackknife_replicates,
                "observations": variable.observations,
                "weights": variable.layout.weights if variable.layout is not None else None,
                "configurations": variable.layout.configurations if variable.layout is not None else None,
                "bootstrap_replicates": variable.bootstrap_replicates,
                "metadata": variable.metadata,
            } for t, variable in self._variables.items()},
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        with open(dst, "wb") as f:
            f.write(_MAGIC)
            f.write(struct.pack("<I", crc))
            f.write(payload)

    def __getitem__(self, tag):
        return self._variables[tag]

    def __iter__(self):
        return iter(self._variables)

    def __len__(self):
        return len(self._variables)

    def __setitem__(self, tag, variable):
        if tag in self._variables:
            raise DuplicateTagError(f"Database[{tag!r}]: tag already exists")
        if not isinstance(variable, Variable):
            raise TypeError("Database values must be Variable instances")
        self._variables[tag] = variable

    def __delitem__(self, tag):
        del self._variables[tag]

    def rename(self, old, new):
        """Move a variable to an unused tag."""
        variable = self._variables[old]
        if new in self._variables:
            raise DuplicateTagError(f"rename({old!r} -> {new!r}): target tag already exists")
        self._variables[new] = variable
        del self._variables[old]

    def __repr__(self):
        return f"<Database num_variables={len(self._variables)}>"

    def __str__(self):
        """Multi-line overview: variable counts per category."""
        counts = {"data": 0, "derived (jackknife)": 0, "result (central/bootstrap)": 0}
        for variable in self._variables.values():
            if variable.observations is not None:
                counts["data"] += 1
            elif variable.jackknife_replicates is not None:
                counts["derived (jackknife)"] += 1
            else:
                counts["result (central/bootstrap)"] += 1
        lines = [f"Database with {len(self._variables)} variables:"]
        for cat, n in counts.items():
            lines.append(f"  {cat:20s} {n}")
        return "\n".join(lines)
