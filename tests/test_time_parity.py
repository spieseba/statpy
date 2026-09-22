"""Time-parity validation and non-mutating forward/backward averaging."""

import numpy as np
import pytest

from statpy.qcd.correlator._masking import (
    _fold_meson_boundary,
    _get_masked_meson_sample,
)
from statpy.qcd.correlator.averaging import (
    meson_fold_correlator_entry,
    obc_meson_boundary_average,
    obc_meson_correlator_average,
)
from statpy.qcd.correlator.primitives import meson_fold_correlator


@pytest.mark.parametrize("time_parity", [
    True, False, np.bool_(True), np.bool_(False),
    0, 2, -2, 0.5, np.nan, np.inf, None, "1", 1j,
    [1], np.array(1), np.array([1, -1]),
])
@pytest.mark.parametrize("call", [
    pytest.param(lambda p: meson_fold_correlator(None, p), id="fold"),
    pytest.param(lambda p: _fold_meson_boundary(None, p), id="boundary-fold"),
    pytest.param(lambda p: _get_masked_meson_sample(None, 4, 4, p), id="mask"),
    pytest.param(
        lambda p: meson_fold_correlator_entry(None, "input", "output", p),
        id="fold-entry",
    ),
    pytest.param(
        lambda p: obc_meson_correlator_average(None, [], None, "output", time_parity=p),
        id="source-average",
    ),
    pytest.param(
        lambda p: obc_meson_boundary_average(None, [], 1, 1, time_parity=p),
        id="boundary-average",
    ),
])
def test_invalid_parity_rejected_before_accessing_data(call, time_parity):
    # None inputs ensure validation happens before array/DB access or writes.
    with pytest.raises(ValueError, match="time_parity must be"):
        call(time_parity)


@pytest.mark.parametrize("scalar", [int, float, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("sign", [1, -1])
def test_numeric_signs_accepted(scalar, sign):
    arr = np.arange(8)
    expected = [0, 4, 4, 4] if sign == 1 else [0, -3, -2, -1]
    np.testing.assert_array_equal(meson_fold_correlator(arr, scalar(sign)), expected)


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
