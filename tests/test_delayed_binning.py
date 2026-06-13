"""Tests for jackknife.delayed_binning (delayed binning, arxiv:2410.17053).

Delayed binning reconstructs the binned jackknife sample from the unbinned one,
so an analysis pipeline can be run once on the unbinned sample and only then
binned for error estimation. We check this against binning the raw data up front,
for a linear and a non-linear pipeline. Runs under pytest or as a plain script:

    python tests/test_delayed_binning.py
"""
import warnings

import numpy as np

from statpy.statistics import jackknife
from statpy.statistics.core import bin as bin_data


def _linear(y):     return 2 * y[..., 0] + 3 * y[..., 1]      # linear combination
def _nonlinear(y):  return y[..., 0] / y[..., 1]              # a ratio, cf. the notebook's Z(t)


def _data(N, seed=0):
    # positive mean so the ratio pipeline is well-behaved
    return 5.0 + np.random.default_rng(seed).standard_normal((N, 2))


def _variances(x, binsize, g):
    """Binned variance of g(x): delayed binning vs. binning the raw data up front."""
    delayed = jackknife.delayed_binning(g(jackknife.sample(x)), binsize)
    upfront = g(jackknife.sample(bin_data(x, binsize)))
    return jackknife.variance(delayed), jackknife.variance(upfront)


def _mean_reldev(N, binsize, g, seeds=8):
    """Mean over seeds of the relative deviation between the two binned variances."""
    devs = []
    for s in range(seeds):
        vd, vu = _variances(_data(N, seed=s), binsize, g)
        devs.append(abs(vd - vu) / abs(vu))
    return float(np.mean(devs))


def test_uniform_weights_match_none_and_do_not_warn():
    jks = jackknife.sample(_data(1000))
    ref = jackknife.delayed_binning(jks, 4)
    # all-ones and any other uniform value must take the scalar branch (no warning)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ones = jackknife.delayed_binning(jks, 4, weights=np.ones(len(jks)))
        twos = jackknife.delayed_binning(jks, 4, weights=np.full(len(jks), 2.0))
    np.testing.assert_allclose(ones, ref, rtol=1e-12, atol=0)
    np.testing.assert_allclose(twos, ref, rtol=1e-12, atol=0)


def test_exact_when_binsize_divides():
    x = _data(1000)
    # identity pipeline: the delayed-binned sample equals the up-front binned one
    for binsize in (2, 4, 5, 8):  # all divide 1000
        delayed = jackknife.delayed_binning(jackknife.sample(x), binsize)
        upfront = jackknife.sample(bin_data(x, binsize))
        np.testing.assert_allclose(delayed, upfront, rtol=1e-10, atol=0)
    # linear pipeline: variances are exact too
    for binsize in (2, 4, 5, 8):
        vd, vu = _variances(x, binsize, _linear)
        np.testing.assert_allclose(vd, vu, rtol=1e-10, atol=0)


def test_deviation_decreases_with_sample_size():
    # When binsize does not divide N, delayed and up-front binning differ only by
    # a finite-N boundary effect (the truncated trailing configs) that vanishes
    # as N grows. Hold N % binsize fixed so the truncated fraction shrinks like
    # 1/N, and check the (seed-averaged) deviation falls off accordingly -- for a
    # linear and a non-linear (ratio) pipeline.
    binsize = 7
    Ns = [7 * 40 + 1, 7 * 160 + 1, 7 * 640 + 1]  # 281, 1121, 4481: all == 1 (mod 7), x4 apart
    for g in (_linear, _nonlinear):
        devs = [_mean_reldev(N, binsize, g) for N in Ns]
        assert devs[0] > devs[1] > devs[2], (g.__name__, devs)   # strictly decreasing
        assert devs[2] < devs[0] / 4, (g.__name__, devs)         # ~1/N over the x16 range


if __name__ == "__main__":
    test_uniform_weights_match_none_and_do_not_warn()
    test_exact_when_binsize_divides()
    test_deviation_decreases_with_sample_size()
    print("OK: delayed_binning uniform-weights + exact-when-divides + convergence with N")
