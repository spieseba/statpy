"""Tests for jackknife.delayed_binning (delayed binning, arxiv:2410.17053).

Delayed binning reconstructs the binned jackknife sample from the unbinned one,
so an analysis pipeline can be run once on the unbinned sample and only then
binned for error estimation. We check this against binning the raw data up front,
for a linear and a non-linear pipeline. Runs under pytest or as a plain script:

    python tests/test_delayed_binning.py
"""
import numpy as np

from statpy.statistics import jackknife
from statpy.statistics.core import bin as bin_data


def _linear(y):     return 2 * y[..., 0] + 3 * y[..., 1]      # linear combination
def _nonlinear(y):  return y[..., 0] / y[..., 1]              # a ratio, cf. the notebook's Z(t)


def _data(N, seed=0):
    # positive mean so the ratio pipeline is well-behaved
    return 5.0 + np.random.default_rng(seed).standard_normal((N, 2))


def _sample(x):
    return jackknife.mean_sample(x, np.ones(len(x)))


def _uniform_formula(jks, bin_size):
    """Binned jackknife sample from the unweighted delayed-binning formula."""
    N = len(jks)
    num_bins = N // bin_size
    mean = jks.mean(axis=0)
    bin_sums = jks[:num_bins * bin_size].reshape(num_bins, bin_size, *jks.shape[1:]).sum(axis=1)
    return mean + (bin_sums - bin_size * mean) * (N - 1) / (N - bin_size)


def _variances(x, bin_size, g):
    """Binned variance of g(x): delayed binning vs. binning the raw data up front."""
    delayed = jackknife.delayed_binning(g(_sample(x)), bin_size, np.ones(len(x)))
    upfront = g(_sample(bin_data(x, bin_size, np.ones(len(x)))))
    return jackknife.variance(delayed), jackknife.variance(upfront)


def _mean_reldev(N, bin_size, g, seeds=8):
    """Mean over seeds of the relative deviation between the two binned variances."""
    devs = []
    for s in range(seeds):
        vd, vu = _variances(_data(N, seed=s), bin_size, g)
        devs.append(abs(vd - vu) / abs(vu))
    return float(np.mean(devs))


def test_uniform_weights_match_uniform_formula():
    jks = _sample(_data(1003))   # 1003 % 4 != 0: trailing configs truncated
    expected = _uniform_formula(jks, 4)
    for c in (1.0, 2.0):
        weights = np.full(len(jks), c)
        np.testing.assert_allclose(jackknife.delayed_binning(jks, 4, weights), expected, rtol=1e-12, atol=0)


def test_weighted_matches_leave_bin_out_mean():
    N, bin_size = 1003, 4        # trailing configs truncated, but kept in every mean
    x = _data(N)
    w = np.random.default_rng(1).uniform(0.5, 1.5, N)
    W, S = w.sum(), (w[:, None] * x).sum(axis=0)
    expected = np.array([
        (S - (w[i:i + bin_size, None] * x[i:i + bin_size]).sum(axis=0)) / (W - w[i:i + bin_size].sum())
        for i in range(0, N - N % bin_size, bin_size)
    ])
    jks = jackknife.mean_sample(x, w)
    np.testing.assert_allclose(jackknife.delayed_binning(jks, bin_size, w), expected, rtol=1e-12, atol=0)


def test_exact_when_bin_size_divides():
    x = _data(1000)
    # identity pipeline: the delayed-binned sample equals the up-front binned one
    for bin_size in (2, 4, 5, 8):  # all divide 1000
        delayed = jackknife.delayed_binning(_sample(x), bin_size, np.ones(len(x)))
        upfront = _sample(bin_data(x, bin_size, np.ones(len(x))))
        np.testing.assert_allclose(delayed, upfront, rtol=1e-10, atol=0)
    # linear pipeline: variances are exact too
    for bin_size in (2, 4, 5, 8):
        vd, vu = _variances(x, bin_size, _linear)
        np.testing.assert_allclose(vd, vu, rtol=1e-10, atol=0)


def test_deviation_decreases_with_sample_size():
    # When bin_size does not divide N, delayed and up-front binning differ only by
    # a finite-N boundary effect (the truncated trailing configs) that vanishes
    # as N grows. Hold N % bin_size fixed so the truncated fraction shrinks like
    # 1/N, and check the (seed-averaged) deviation falls off accordingly -- for a
    # linear and a non-linear (ratio) pipeline.
    bin_size = 7
    Ns = [7 * 40 + 1, 7 * 160 + 1, 7 * 640 + 1]  # 281, 1121, 4481: all == 1 (mod 7), x4 apart
    for g in (_linear, _nonlinear):
        devs = [_mean_reldev(N, bin_size, g) for N in Ns]
        assert devs[0] > devs[1] > devs[2], (g.__name__, devs)   # strictly decreasing
        assert devs[2] < devs[0] / 4, (g.__name__, devs)         # ~1/N over the x16 range


if __name__ == "__main__":
    test_uniform_weights_match_uniform_formula()
    test_weighted_matches_leave_bin_out_mean()
    test_exact_when_bin_size_divides()
    test_deviation_decreases_with_sample_size()
    print("OK: delayed_binning uniform formula + weighted + exact-when-divides + convergence with N")
