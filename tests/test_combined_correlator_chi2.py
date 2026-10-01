"""Test combined_correlator_chi2 against a per-block reference."""
from itertools import product

import numpy as np
import pytest

from statpy.qcd.correlator.fits import _make_combined_chi2


def reference_combined_chi2(models, t0, t1, p, y, W, Nt):
    """Build each block explicitly, concatenate, contract."""
    A0, A1, m = p
    sign = {"cosh": 1.0, "sinh": -1.0, "exp": 0.0}
    block0 = A0 * (np.exp(-m * t0) + sign[models[0]] * np.exp(-m * (Nt - t0)))
    block1 = A1 * (np.exp(-m * t1) + sign[models[1]] * np.exp(-m * (Nt - t1)))
    residual = np.concatenate((block0, block1)) - y
    return residual @ W @ residual


@pytest.mark.parametrize("L", [32, 16, 20])   # unfolded, folded, shortened by OBC averaging
def test_combined_chi2_matches_reference(L):
    Nt = 32
    rng = np.random.default_rng(1)
    for n0, n1 in [(1, 1), (3, 5), (8, 8)]:
        t0 = np.arange(5, 5 + n0)
        t1 = np.arange(5, 5 + n1)   # overlapping times: a block mix-up cannot hide
        t = np.concatenate((t0, L + t1))
        n = n0 + n1
        for models in product(("cosh", "sinh", "exp"), repeat=2):
            for _ in range(20):
                p = rng.uniform(0.01, 2.0, 3)
                y = rng.standard_normal(n)
                a = rng.standard_normal((n, n))
                W = a @ a.T
                chi2 = _make_combined_chi2(f"combined-{models[0]}-{models[1]}", W, Nt, L)
                ref = reference_combined_chi2(models, t0, t1, p, y, W, Nt)
                np.testing.assert_allclose(chi2(t, p, y), ref, rtol=1e-12, atol=0)


def test_combined_chi2_rejects_invalid_models():
    for fit_model in ("combined-cosh", "combined-cosh-sinh-exp", "combined-cosh-unknown"):
        with pytest.raises(ValueError):
            _make_combined_chi2(fit_model, np.eye(2), 32, 32)
