"""Test for combined_corr_chi2 (joint two-correlator fit, e.g. PSPS + PSA4I).

The accelerated kernel evaluates both blocks in one vectorized expression
over the concatenated time array, encoding the block structure per point
(amplitude index and backward-propagator sign). We check the chi^2 exactly
as the fit builds it (via ``fits._make_combined_chi2``) against a
straightforward per-block reference implementation. Runs under pytest or
as a plain script:

    python tests/test_combined_corr_chi2.py
"""
import numpy as np

from statpy.qcd.correlator.fits import _make_combined_chi2


def reference_combined_chi2(model, t_PS, t_A4I, p, y, W, Nt):
    """Textbook formulation: build each block explicitly, concatenate, contract."""
    A_PS, A_A4I, m = p
    if model == "combined-cosh-sinh":
        block_PS = A_PS * (np.exp(-m * t_PS) + np.exp(-m * (Nt - t_PS)))       # cosh
        block_A4I = A_A4I * (np.exp(-m * t_A4I) - np.exp(-m * (Nt - t_A4I)))   # sinh
    else:  # combined-exp-exp
        block_PS = A_PS * np.exp(-m * t_PS)                                    # exp (open BC)
        block_A4I = A_A4I * np.exp(-m * t_A4I)
    residual = np.concatenate((block_PS, block_A4I)) - y
    return residual @ W @ residual


def test_accelerated_chi2_matches_reference():
    Nt = 32
    rng = np.random.default_rng(1)
    for n_PS, n_A4I in [(1, 1), (3, 5), (8, 8)]:
        t_PS = np.arange(5, 5 + n_PS)
        t_A4I = np.arange(5, 5 + n_A4I)   # overlapping times: a block mix-up cannot hide
        t = np.hstack((t_PS, t_A4I))
        n = n_PS + n_A4I
        for model in ("combined-cosh-sinh", "combined-exp-exp"):
            for _ in range(20):
                p = rng.uniform(0.01, 2.0, 3)          # A_PS, A_A4I, m -- all positive
                y = rng.standard_normal(n)
                a = rng.standard_normal((n, n))
                W = a @ a.T                            # symmetric weight matrix
                chi2 = _make_combined_chi2(model, W, n_PS, n_A4I, Nt)
                ref = reference_combined_chi2(model, t_PS, t_A4I, p, y, W, Nt)
                # not required bitwise: numpy and numba may use different exp()
                np.testing.assert_allclose(chi2(t, p, y), ref, rtol=1e-12, atol=0)


if __name__ == "__main__":
    test_accelerated_chi2_matches_reference()
    print("OK: accelerated combined chi2 matches per-block reference")
