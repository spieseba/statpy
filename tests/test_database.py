"""Tests for Database save and load."""

import numpy as np
import pytest

from statpy.database import Database
from statpy.variable import Layout, Variable


# Several replicas give no jackknife and bootstrap replicates, so both cases are saved
@pytest.mark.parametrize("monte_carlo_times, replicas", [(np.array([0, 2, 3, 4, 5, 6]), np.zeros(6, dtype=int)),
                                                         (np.array([0, 2, 3, 0, 1, 2]), np.array([0, 0, 0, 1, 1, 1]))])
def test_save_load_keeps_every_field(tmp_path, monte_carlo_times, replicas):
    layout = Layout("E", monte_carlo_times, replicas=replicas, weights=np.array([1.0, 2, 1, 1, 2, 1]))
    one_replica = len(np.unique(replicas)) == 1
    bootstrap_indices = np.random.default_rng(0).integers(6, size=(4, 6)) if one_replica else None
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


def test_save_load_keeps_external_inputs(tmp_path):
    c = Variable.from_external("c", [1.0, 2.0], [[0.04, 0.01], [0.01, 0.09]], num_bootstraps=4)
    db = Database()
    db["propagated"] = Variable.derive(lambda c: c[0] * c, c)
    db["unpropagated"] = Variable.derive(lambda c: c[0] * c, c, jacobians=None)
    db.save(tmp_path / "db")
    loaded = Database(tmp_path / "db")
    for tag, variable in db.items():
        external, loaded_external = variable.external_inputs["c"], loaded[tag].external_inputs["c"]
        np.testing.assert_array_equal(loaded_external.covariance, external.covariance)
        np.testing.assert_array_equal(loaded_external.jacobian, external.jacobian)
        np.testing.assert_array_equal(loaded[tag].bootstrap_replicates, variable.bootstrap_replicates)
    np.testing.assert_array_equal(loaded["propagated"].jackknife_variance(), db["propagated"].jackknife_variance())
    assert loaded["unpropagated"].external_inputs["c"].jacobian is None
