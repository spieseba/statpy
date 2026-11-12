import warnings
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import h5py
import numpy as np

import statpy
from statpy.log import message
from statpy.variable import Layout, Variable

# Database file: HDF5 with root attribute ``statpy_commit``; one group per tag with one dataset per field that is
# not None, ``ensembles`` as UTF-8 bytes. A field whose meaning changes gets a new name.
_VARIABLE_FIELDS = ("observations", "central_value", "replica_central_values", "fluctuations",
                    "jackknife_replicates", "bootstrap_replicates")
_LAYOUT_FIELDS = ("ensembles", "monte_carlo_times", "replicas", "weights")
# Scalar datasets cannot carry filters
_DATASET_FILTERS = {"fletcher32": True, "shuffle": True, "compression": "gzip", "compression_opts": 4}
_commit_logged = False


class DuplicateTagError(ValueError):
    """An operation would write a tag that already exists in the Database."""


class Database(Mapping[str, Variable]):
    """Dict-like store of Variables by tag, saved to and loaded from a database file.

    Tags must not contain ``/``; use ``.`` to separate their parts (e.g. ``"pion.mass"``).
    """

    # ---------------------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------------------

    def __init__(self, *args: Path | Database) -> None:
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
        with h5py.File(src, "r") as f:
            if "statpy_commit" not in f.attrs:
                raise ValueError(f"{src}: not a statpy database file")
            self._check_new_tags(f, f"load({src})")
            variables = {}
            for tag, group in f.items():
                unknown = sorted(set(group) - set(_VARIABLE_FIELDS) - set(_LAYOUT_FIELDS))
                if unknown:
                    warnings.warn(f"{src}: {tag}: fields unknown to this statpy version ignored: {unknown}",
                                  stacklevel=2)
                missing = sorted({"central_value", *_LAYOUT_FIELDS} - set(group))
                if missing:
                    raise ValueError(f"{src}: {tag}: no {missing}; written by statpy commit "
                                     f"{f.attrs['statpy_commit']}, which this version cannot read")
                fields = {name: group[name][...] for name in group if name not in unknown}
                layout = Layout(np.char.decode(fields.pop("ensembles"), "utf-8"), fields.pop("monte_carlo_times"),
                                replicas=fields.pop("replicas"), weights=fields.pop("weights"))
                variables[tag] = Variable(layout=layout, **fields)
        self._variables.update(variables)

    def _check_new_tags(self, tags: Iterable[str], context: str) -> None:
        dup = [t for t in tags if t in self._variables]
        if dup:
            raise DuplicateTagError(f"{context}: {len(dup)} overlapping tag(s): {dup[:5]}"
                                    + ("..." if len(dup) > 5 else ""))

    @staticmethod
    def _check_tag(tag: str) -> None:
        if "/" in tag:
            raise ValueError(f"tag {tag!r} contains '/'; use '.' to separate its parts")

    def save(self, dst: Path) -> None:
        """Write the entire database (all fields, including observations) to ``dst``."""
        with h5py.File(dst, "w") as f:
            f.attrs["statpy_commit"] = statpy.__commit__ or "unknown"
            for tag, variable in self._variables.items():
                group = f.create_group(tag)
                fields = ({name: getattr(variable, name) for name in _VARIABLE_FIELDS}
                          | {name: getattr(variable.layout, name) for name in _LAYOUT_FIELDS})
                fields["ensembles"] = np.char.encode(variable.layout.ensembles, "utf-8")
                for name, value in fields.items():
                    if value is not None:
                        group.create_dataset(name, data=value, **(_DATASET_FILTERS if np.ndim(value) > 0 else {}))

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
        self._check_tag(tag)
        if tag in self._variables:
            raise DuplicateTagError(f"Database[{tag!r}]: tag already exists")
        if not isinstance(variable, Variable):
            raise TypeError("Database values must be Variable instances")
        self._variables[tag] = variable

    def __delitem__(self, tag: str) -> None:
        del self._variables[tag]

    def rename(self, old: str, new: str) -> None:
        """Move a variable to an unused tag."""
        self._check_tag(new)
        variable = self._variables[old]
        if new in self._variables:
            raise DuplicateTagError(f"rename({old!r} -> {new!r}): target tag already exists")
        self._variables[new] = variable
        del self._variables[old]
