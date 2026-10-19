"""Least-squares fit of exact data recovers the parameters of the central value and every replicate."""
import numpy as np
from numba import njit

from statpy.variable import Layout, Variable

T = np.arange(10)
TRUE_P = np.array([1.5, 0.3])
N_CFG = 8
N_BOOT = 5


@njit
def _model(t, p):
    return p[0] * np.exp(-p[1] * t)


def test_fit_recovers_parameters_of_central_value_and_replicates():
    rng = np.random.default_rng(3)
    jackknife_p = TRUE_P * (1 + 0.01 * rng.standard_normal((N_CFG, 2)))
    bootstrap_p = TRUE_P * (1 + 0.01 * rng.standard_normal((N_BOOT, 2)))
    v = Variable.from_results(
        central_value=_model(T, TRUE_P), layout=Layout("test", np.arange(N_CFG)),
        jackknife_replicates=np.array([_model(T, p) for p in jackknife_p]),
        bootstrap_replicates=np.array([_model(T, p) for p in bootstrap_p]),
    )
    fit = v.least_squares_fit(_model, T, [1.0, 0.2], tol=1e-8)
    np.testing.assert_allclose(fit.parameters.central_value, TRUE_P, rtol=1e-5)
    assert fit.parameters.jackknife_replicates is not None and fit.parameters.bootstrap_replicates is not None
    np.testing.assert_allclose(fit.parameters.jackknife_replicates, jackknife_p, rtol=1e-5)
    np.testing.assert_allclose(fit.parameters.bootstrap_replicates, bootstrap_p, rtol=1e-5)
    assert fit.chi2 < 1e-8
    assert fit.dof == len(T) - 2
