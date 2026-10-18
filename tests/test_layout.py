"""Tests for Layout's Monte Carlo times and replicas."""

import numpy as np
import pytest

from statpy.variable import Layout


def test_monte_carlo_times_increase_within_each_replica():
    replicas = np.array([0, 0, 0, 1, 1])
    Layout("E", np.array([0, 2, 3, 0, 1]), replicas=replicas)
    with pytest.raises(ValueError, match="replica 1"):
        Layout("E", np.array([0, 2, 3, 1, 1]), replicas=replicas)
