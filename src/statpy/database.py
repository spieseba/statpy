import pickle
import struct
import zlib
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import statpy
from statpy.log import message
from statpy.variable import Layout, Variable

# Database file: magic number (format version) + u32 CRC32 + pickled {tag: variable state dict}.
# Change the magic number on any format change. Retired: v1 JSON, v2 SPDB, v3 SPD3, v4 SPD4, v5 SPD5,
# v6 SPD6, v7 SPD7, v8 SPD8, v9 SPD9.
_MAGIC = b"SP10"
_commit_logged = False


class DuplicateTagError(ValueError):
    """An operation would write a tag that already exists in the Database."""


class Database(Mapping[str, Variable]):
    """Dict-like store of Variables by tag, saved to and loaded from a database file."""

    # ---------------------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------------------

    def __init__(self, *args: "Path | Database") -> None:
        """Create a Database from database files (``Path``) or Databases with disjoint tags."""
        global _commit_logged
        self._variables: dict[str, Variable] = {}
        if not _commit_logged:
            message(f"Initialized database with statpy commit hash {statpy.__commit__}.")
            _commit_logged = True
        for src in args:
            if isinstance(src, Path):
                self.load(src)
            elif isinstance(src, Database):
                self._check_new_tags(src._variables, "Database merge")
                for t, variable in src._variables.items():
                    self._variables[t] = Variable(**vars(variable))
            else:
                raise TypeError(f"Database: sources must be Path or Database, got {type(src).__name__}")

    def __repr__(self) -> str:
        return f"<Database num_variables={len(self._variables)}>"

    # ---------------------------------------------------------------------------
    # File I/O
    # ---------------------------------------------------------------------------

    def load(self, src: Path) -> None:
        """Load the database file ``src`` (``Path``) and add its variables; tags must not overlap."""
        message(f"Load {src}")
        if not src.is_file():
            raise FileNotFoundError(f"{src} not found")
        with src.open("rb") as f:
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
        self._check_new_tags(src_db, f"load({src})")
        variables = {}
        for t, state in src_db.items():
            state = state.copy()
            ensemble = state.pop("ensemble")
            monte_carlo_times = state.pop("monte_carlo_times")
            replicas = state.pop("replicas")
            weights = state.pop("weights")
            layout = Layout(ensemble, monte_carlo_times, replicas=replicas, weights=weights)
            variables[t] = Variable(layout=layout, **state)
        self._variables.update(variables)

    def _check_new_tags(self, tags: Iterable[str], context: str) -> None:
        dup = [t for t in tags if t in self._variables]
        if dup:
            raise DuplicateTagError(f"{context}: {len(dup)} overlapping tag(s): {dup[:5]}"
                                    + ("..." if len(dup) > 5 else ""))

    def save(self, dst: Path) -> None:
        """Write the entire database (all fields, including observations) to ``dst``."""
        payload = pickle.dumps(
            {t: {
                "observations": variable.observations,
                "ensemble": variable.layout.ensemble,
                "monte_carlo_times": variable.layout.monte_carlo_times,
                "replicas": variable.layout.replicas,
                "weights": variable.layout.weights,
                "central_value": variable.central_value,
                "replica_central_values": variable.replica_central_values,
                "fluctuations": variable.fluctuations,
                "jackknife_replicates": variable.jackknife_replicates,
                "bootstrap_replicates": variable.bootstrap_replicates,
            } for t, variable in self._variables.items()},
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        with dst.open("wb") as f:
            f.write(_MAGIC)
            f.write(struct.pack("<I", crc))
            f.write(payload)

    # ---------------------------------------------------------------------------
    # Mapping interface
    # ---------------------------------------------------------------------------

    def __getitem__(self, tag: str) -> Variable:
        return self._variables[tag]

    def __iter__(self) -> Iterator[str]:
        return iter(self._variables)

    def __len__(self) -> int:
        return len(self._variables)

    def __setitem__(self, tag: str, variable: Variable) -> None:
        if tag in self._variables:
            raise DuplicateTagError(f"Database[{tag!r}]: tag already exists")
        if not isinstance(variable, Variable):
            raise TypeError("Database values must be Variable instances")
        self._variables[tag] = variable

    def __delitem__(self, tag: str) -> None:
        del self._variables[tag]

    def rename(self, old: str, new: str) -> None:
        """Move a variable to an unused tag."""
        variable = self._variables[old]
        if new in self._variables:
            raise DuplicateTagError(f"rename({old!r} -> {new!r}): target tag already exists")
        self._variables[new] = variable
        del self._variables[old]
