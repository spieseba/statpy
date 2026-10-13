"""Tests for gamma."""

import numpy as np
import pytest

from statpy.statistics import gamma


def test_lag_sum_fft_matches_direct_sum():
    rng = np.random.default_rng(0)
    for shape, max_lag in [((50,), 10), ((50, 3), 49), ((5, 3), 8)]:
        values = rng.normal(size=shape)
        np.testing.assert_allclose(gamma._lag_sum(values, max_lag),
                                   gamma._lag_sum(values, max_lag, fft=False), atol=1e-12)


def test_autocovariance_matches_direct_pair_sum():
    rng = np.random.default_rng(0)
    # Chain 0 has gaps at 3, 6, 7; chain 1 reuses positions, so pairs across chains would show.
    chains = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    positions = np.array([0, 1, 2, 4, 5, 8, 3, 4, 6, 7])
    fluctuations = rng.normal(size=(10, 2))
    # Default max_lag: 9 // 2 - 1 = 3; every lag up to it has pairs.
    expected = []
    for lag in range(4):
        pairs = [(i, j) for i in range(10) for j in range(10)
                 if chains[i] == chains[j] and positions[j] - positions[i] == lag]
        # Mean over pairs = product sum / number of pairs.
        expected.append(np.mean([fluctuations[i] * fluctuations[j] for i, j in pairs], axis=0))
    np.testing.assert_allclose(gamma.autocovariance(fluctuations, chains, positions), expected)


def test_integrated_autocorrelation_times_by_hand():
    gamma_values = np.array([[1.0, 0.5, 0.25],
                             [2.0, 1.0, 0.5]]).T
    expected = np.array([[0.5, 1.0, 1.25],
                         [0.5, 1.0, 1.25]]).T
    np.testing.assert_allclose(gamma.integrated_autocorrelation_times(gamma_values), expected)


def test_window():
    tau_ints = np.array([[0.5, 0.4] + [1.0] * 8,    # tau_int <= 1/2 at W = 1
                         [0.5] + [1.0] * 9,         # g(7) > 0 > g(8)
                         [0.5] + [50.0] * 9]).T     # no sign change: largest W = 9
    with pytest.warns(UserWarning, match="No window found"):
        np.testing.assert_array_equal(gamma.window(tau_ints, 10_000), [1, 8, 9])


def test_variance_and_errors_by_hand():
    gamma_values = np.array([[2.0, 4.0]])           # only gamma[0] enters
    tau_ints = np.array([[0.5, 1.0, 1.5, 2.0],
                         [0.5, 3.0, 4.5, 5.0]]).T
    W = np.array([1, 2])                            # tau_int(W) = 1.0 and 4.5
    # 2 tau_int gamma[0] (1 + (2W+1)/N) / N with N = 200
    np.testing.assert_allclose(gamma.variance(gamma_values, tau_ints, W, 200), [0.0203, 0.1845])
    # 2 tau_int sqrt(|W + 1/2 - tau_int| / N); component 1 needs the abs
    np.testing.assert_allclose(gamma.error_tau_int(tau_ints, W, 200), [0.1, 0.9])
    # error sqrt((W + 1/2) / N) with N = 150
    np.testing.assert_allclose(gamma.error_of_error(np.array([2.0, 3.0]), np.array([1, 13]), 150), [0.2, 0.9])
