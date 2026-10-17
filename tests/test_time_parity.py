"""Forward/backward folding and averaging by time parity, without mutating the input."""

import numpy as np
import pytest

from statpy.qcd.correlator._masking import _get_masked_meson_sample
from statpy.qcd.correlator.primitives import meson_fold_correlator


@pytest.mark.parametrize("sign", [1, -1])
def test_fold_values(sign):
    arr = np.arange(8)
    expected = [0, 4, 4, 4] if sign == 1 else [0, -3, -2, -1]
    np.testing.assert_array_equal(meson_fold_correlator(arr, sign), expected)


@pytest.mark.parametrize("time_parity", [1, -1])
def test_masked_average_does_not_mutate_source(time_parity):
    sample = np.arange(32.0).reshape(2, 2, 8)[:, :, ::2]
    original = sample.copy()
    sample.setflags(write=False)

    result = _get_masked_meson_sample(sample, 4, 4, time_parity)

    expected_backward = original[:, :, [0, 3, 2, 1]]
    expected_backward[:, :, 1:] *= time_parity
    np.testing.assert_array_equal(result[:, :2], original)
    np.testing.assert_array_equal(result[:, 2:], expected_backward)
    np.testing.assert_array_equal(sample, original)
    assert not np.shares_memory(result.data, sample)


@pytest.mark.parametrize("time_parity", [1, -1])
def test_fold_does_not_mutate_source(time_parity):
    arr = np.arange(16.0)[::2]
    original = arr.copy()
    arr.setflags(write=False)
    result = meson_fold_correlator(arr, time_parity)
    np.testing.assert_array_equal(arr, original)
    assert not np.shares_memory(result, arr)
