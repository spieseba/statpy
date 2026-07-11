"""Tests for central values of weighted data entries."""

import numpy as np

from statpy.database.core import DB


def test_add_entry_uses_full_weighted_sample_mean():
    sample = np.array([[0.0, 2.0], [10.0, 6.0], [4.0, 8.0]])
    weights = np.array([1.0, 3.0, 2.0])
    cfgs = np.array(["c0", "c1", "c2"])

    db = DB()
    db.add_entry("weighted", sample=sample, weights=weights, cfgs=cfgs)

    # (1*[0,2] + 3*[10,6] + 2*[4,8]) / 6
    expected = np.array([19 / 3, 6.0])
    np.testing.assert_allclose(db.database["weighted"].mean, expected)

    # For unequal weights, averaging the delete-one estimates is not the
    # full-sample estimator. This guards against reintroducing that shortcut.
    assert not np.allclose(expected, np.mean(db.database["weighted"].jks, axis=0))
