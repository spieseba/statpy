"""Tests for ar1."""

import numpy as np

from statpy.simulation import ar1


def test_generate_chains_matches_exact_autocorrelation():
    tau = 4.0
    a = (2 * tau - 1) / (2 * tau + 1)
    num_chains = 10_000
    chains = ar1.generate_chains(np.full(num_chains, tau), 20, np.random.default_rng(0))
    # covariance between times s and t over independent chains; Wolff eq. 70: rho(|t - s|) = a^|t - s|, unit variance
    times = np.arange(20)
    # 3 sigma of the largest statistical error, sqrt(2 / num_chains) at s = t
    np.testing.assert_allclose(np.cov(chains), a ** np.abs(np.subtract.outer(times, times)), atol=3 * np.sqrt(2 / num_chains))
