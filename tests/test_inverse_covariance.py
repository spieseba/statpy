"""Tests for the correlated-fit inverse covariance."""

import numpy as np

from statpy.qcd.correlator.fits import _inverse_covariance


def test_large_dynamic_range_is_accepted_and_inverted():
    # Periodic correlator window: errors large at both ends, tiny at the midpoint.
    Nt, t = 128, np.arange(16, 113)
    n = len(t)
    sigma = np.exp(-0.9 * t) + np.exp(-0.9 * (Nt - t))
    corr = 0.9 ** np.abs(t[:, None] - t[None, :])
    cov = corr * np.outer(sigma, sigma)
    assert not np.all(np.linalg.eigvals(cov) > 0)  # previous check rejects it

    inverse, reason = _inverse_covariance(cov, n_samples=1000)
    assert reason is None
    np.testing.assert_allclose(inverse * np.outer(sigma, sigma) @ corr, np.eye(n), atol=1e-10)


def test_indefinite_matrix_is_rejected():
    corr = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
    inverse, reason = _inverse_covariance(corr, n_samples=100)
    assert inverse is None and reason.startswith("not positive definite")


def test_too_few_samples_is_rejected():
    rng = np.random.default_rng(0)
    cov = np.cov(rng.normal(size=(10, 12)), rowvar=False)
    inverse, reason = _inverse_covariance(cov, n_samples=10)
    assert inverse is None and reason == "singular: 12 points from 10 samples"
