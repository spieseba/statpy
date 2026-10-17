"""Tests for Layout's Monte Carlo times and replicas."""

import numpy as np
import pytest

from statpy.database import Database
from statpy.variable import Layout, Variable

CONFIGURATIONS = np.array([f"c{i}" for i in range(5)])


def test_monte_carlo_times_increase_within_each_replica():
    replicas = np.array([0, 0, 0, 1, 1])
    Layout(CONFIGURATIONS, monte_carlo_times=np.array([0, 2, 3, 0, 1]), replicas=replicas)
    with pytest.raises(ValueError, match="replica 1"):
        Layout(CONFIGURATIONS, monte_carlo_times=np.array([0, 2, 3, 1, 1]), replicas=replicas)


def test_save_load_keeps_monte_carlo_times_and_replicas(tmp_path):
    monte_carlo_times = np.array([0, 2, 3, 0, 1])
    replicas = np.array([0, 0, 0, 1, 1])
    db = Database()
    db["a"] = Variable.from_data(np.arange(5.0), Layout(CONFIGURATIONS, np.ones(5),
                                                        monte_carlo_times=monte_carlo_times, replicas=replicas))
    db.save(str(tmp_path / "db"))
    layout = Database(str(tmp_path / "db"))["a"].layout
    np.testing.assert_array_equal(layout.monte_carlo_times, monte_carlo_times)
    np.testing.assert_array_equal(layout.replicas, replicas)
