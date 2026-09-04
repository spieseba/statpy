"""Test for combined_corr_chi2 (joint two-correlator fit).

The accelerated kernel evaluates both blocks in one vectorized expression
over the concatenated time array, encoding the block structure per point
(amplitude index and backward-propagator sign). We check the chi^2 exactly
as the fit builds it (via ``fits._make_combined_chi2``) against a
straightforward per-block reference implementation.
"""
from itertools import product

import numpy as np
import pytest

from statpy.qcd.correlator.fits import _combined_model_name, _make_combined_chi2


def reference_combined_chi2(models, t0, t1, p, y, W, Nt):
    """Build each block explicitly, concatenate, contract."""
    A0, A1, m = p
    sign = {"cosh": 1.0, "sinh": -1.0, "exp": 0.0}
    block0 = A0 * (np.exp(-m * t0) + sign[models[0]] * np.exp(-m * (Nt - t0)))
    block1 = A1 * (np.exp(-m * t1) + sign[models[1]] * np.exp(-m * (Nt - t1)))
    residual = np.concatenate((block0, block1)) - y
    return residual @ W @ residual


def test_accelerated_chi2_matches_reference():
    Nt = 32
    rng = np.random.default_rng(1)
    for n0, n1 in [(1, 1), (3, 5), (8, 8)]:
        t0 = np.arange(5, 5 + n0)
        t1 = np.arange(5, 5 + n1)   # overlapping times: a block mix-up cannot hide
        t = np.hstack((t0, t1))
        n = n0 + n1
        for models in product(("cosh", "sinh", "exp"), repeat=2):
            for _ in range(20):
                p = rng.uniform(0.01, 2.0, 3)          # A0, A1, m -- all positive
                y = rng.standard_normal(n)
                a = rng.standard_normal((n, n))
                W = a @ a.T                            # symmetric weight matrix
                chi2 = _make_combined_chi2(models, W, (n0, n1), Nt)
                ref = reference_combined_chi2(models, t0, t1, p, y, W, Nt)
                # not required bitwise: numpy and numba may use different exp()
                np.testing.assert_allclose(chi2(t, p, y), ref, rtol=1e-12, atol=0)


def test_combined_model_name_rejects_invalid_models():
    with pytest.raises(ValueError):
        _combined_model_name(("cosh",))
    with pytest.raises(ValueError):
        _combined_model_name(("cosh", "sinh", "exp"))
    with pytest.raises(ValueError):
        _combined_model_name(("cosh", "unknown"))
