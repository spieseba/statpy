"""Tests for Database save and load."""

import numpy as np

from statpy.database import Database
from statpy.variable import Layout, Variable


def test_save_load_keeps_every_field(tmp_path):
    layout = Layout("E", np.array([0, 2, 3, 0, 1, 2]), replicas=np.array([0, 0, 0, 1, 1, 1]),
                    weights=np.array([1.0, 2, 1, 1, 2, 1]))
    bootstrap_indices = np.random.default_rng(0).integers(6, size=(4, 6))
    variable = Variable.from_data(np.arange(12.0).reshape(6, 2), layout, bootstrap_indices=bootstrap_indices)
    db = Database()
    db["a"] = variable
    db.save(tmp_path / "db")
    loaded = Database(tmp_path / "db")["a"]
    for name, value in vars(variable).items():
        if name == "layout":
            for field, array in vars(layout).items():
                np.testing.assert_array_equal(getattr(loaded.layout, field), array)
        else:
            np.testing.assert_array_equal(getattr(loaded, name), value)
