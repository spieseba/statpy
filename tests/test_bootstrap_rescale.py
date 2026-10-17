"""Tests for bootstrap.rescale."""

import numpy as np

from statpy.statistics import bootstrap


def test_rescale_scales_spread_about_mean():
    bss = np.array([1.0, 3.0])  # mean 2
    np.testing.assert_allclose(bootstrap.rescale(bss, 2.0), [0.0, 4.0])
    np.testing.assert_allclose(np.mean(bootstrap.rescale(bss, 3.0)), 2.0)
