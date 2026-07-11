"""Tests for bootstrap.rescale: shape/dtype robustness and scaling behavior."""

import numpy as np

from statpy.statistics import bootstrap


def test_rescale_preserves_shape_and_dtype_combinations():
    rng = np.random.default_rng(0)
    for shape in [(100,), (100, 3), (100, 2, 2)]:
        for dtype in [np.float64, np.float32]:
            bss = rng.normal(size=shape).astype(dtype)
            assert bootstrap.rescale(bss, 2.0).shape == shape


def test_rescale_scales_spread_about_mean():
    bss = np.array([1.0, 3.0])  # mean 2
    np.testing.assert_allclose(bootstrap.rescale(bss, 2.0), [0.0, 4.0])
    np.testing.assert_allclose(np.mean(bootstrap.rescale(bss, 3.0)), 2.0)
