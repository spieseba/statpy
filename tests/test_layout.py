"""Tests for Layout's Monte Carlo times and replicas."""

import numpy as np
import pytest

from statpy.variable import Layout, Variable


def test_monte_carlo_times_increase_within_each_replica():
    replicas = np.array([0, 0, 0, 1, 1])
    Layout("E", np.array([0, 2, 3, 0, 1]), replicas=replicas)
    with pytest.raises(ValueError, match="replica 1"):
        Layout("E", np.array([0, 2, 3, 1, 1]), replicas=replicas)


def test_several_replicas_have_no_jackknife_or_bootstrap():
    layout = Layout("E", np.arange(4), replicas=np.array([0, 0, 1, 1]))
    assert Variable.from_data(np.arange(4.0), layout).jackknife_replicates is None
    with pytest.raises(ValueError):
        Variable.from_data(np.arange(4.0), layout, bootstrap_indices=np.zeros((2, 4), dtype=int))
