"""Tests for gamma: lag sums and autocovariance against direct sums."""

import numpy as np

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
