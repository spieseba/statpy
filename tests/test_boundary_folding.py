"""Tests for _fold_meson_boundary: fold values and non-mutation of the input."""

import numpy as np

from statpy.qcd.correlator._masking import _fold_meson_boundary


def test_fold_does_not_mutate_input():
    # np.flip yields a view; the antisymmetric sign flip must not write
    # through it into the caller's array (e.g. a DB entry's mean/jks).
    arr = np.arange(8.0)
    _fold_meson_boundary(arr, antisymmetric=True)
    np.testing.assert_array_equal(arr, np.arange(8.0))


def test_fold_values():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    np.testing.assert_allclose(
        _fold_meson_boundary(arr, antisymmetric=False), (arr[:4] + arr[4:][::-1]) / 2
    )
    np.testing.assert_allclose(
        _fold_meson_boundary(arr, antisymmetric=True), (arr[:4] - arr[4:][::-1]) / 2
    )
