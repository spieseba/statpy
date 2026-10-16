"""Tests for central values of weighted data variables."""

import numpy as np

from statpy.database import Database
from statpy.variable import Layout, Variable


def test_from_data_uses_full_weighted_sample_mean():
    sample = np.array([[0.0, 2.0], [10.0, 6.0], [4.0, 8.0]])
    weights = np.array([1.0, 3.0, 2.0])
    cfgs = np.array(["c0", "c1", "c2"])

    db = Database()
    db["weighted"] = Variable.from_data(sample, Layout(cfgs, weights))

    # (1*[0,2] + 3*[10,6] + 2*[4,8]) / 6
    expected = np.array([19 / 3, 6.0])
    np.testing.assert_allclose(db["weighted"].central_value, expected)

    # For unequal weights, averaging the delete-one estimates is not the
    # full-sample estimator. This guards against reintroducing that shortcut.
    assert not np.allclose(expected, np.mean(db["weighted"].jackknife_replicates, axis=0))
