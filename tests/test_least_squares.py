"""Least-squares fit of exact data recovers the parameters of the central value and every replicate."""
import jax.numpy as jnp
import numpy as np
import pytest
from numba import njit

from statpy.simulation import ar1
from statpy.statistics import least_squares
from statpy.variable import Layout, Variable

T = np.arange(10)
TRUE_P = np.array([1.5, 0.3])
N_CFG = 8
N_BOOT = 5


@njit
def _model(t, p):
    return p[0] * np.exp(-p[1] * t)


def _jnp_model(t, p):
    return p[0] * jnp.exp(-p[1] * t)


@pytest.mark.parametrize("numba_model", [None, _model])
def test_fit_recovers_parameters_of_central_value_and_replicates(numba_model):
    rng = np.random.default_rng(3)
    jackknife_p = TRUE_P * (1 + 0.01 * rng.standard_normal((N_CFG, 2)))
    bootstrap_p = TRUE_P * (1 + 0.01 * rng.standard_normal((N_BOOT, 2)))
    v = Variable.from_results(
        layout=Layout("test", np.arange(N_CFG)), central_value=_model(T, TRUE_P),
        jackknife_replicates=np.array([_model(T, p) for p in jackknife_p]),
        bootstrap_replicates=np.array([_model(T, p) for p in bootstrap_p]),
    )
    fit = v.least_squares_fit(_jnp_model, T, [1.0, 0.2], numba_model=numba_model, tolerance=1e-8)
    np.testing.assert_allclose(fit.parameters.central_value, TRUE_P, rtol=1e-5)
    assert fit.parameters.jackknife_replicates is not None and fit.parameters.bootstrap_replicates is not None
    np.testing.assert_allclose(fit.parameters.jackknife_replicates, jackknife_p, rtol=1e-5)
    np.testing.assert_allclose(fit.parameters.bootstrap_replicates, bootstrap_p, rtol=1e-5)
    assert fit.chi2 < 1e-8
    assert fit.dof == len(T) - 2


def test_parameter_jacobian_matches_exact_reference_at_zero_residual():
    """At zero residual, dp/dy = (J^T W J)^-1 J^T W with J = dmodel/dp."""
    y = np.asarray(_jnp_model(T, TRUE_P))
    weight_matrix = np.diag(1 / (0.01 * y) ** 2)
    J = np.stack([np.exp(-TRUE_P[1] * T), -TRUE_P[0] * T * np.exp(-TRUE_P[1] * T)], axis=1)
    expected = np.linalg.solve(J.T @ weight_matrix @ J, J.T @ weight_matrix)
    np.testing.assert_allclose(least_squares.parameter_jacobian(_jnp_model, T, y, TRUE_P, weight_matrix), expected,
                               rtol=1e-12)


def test_fit_fluctuations_match_exact_error_and_tau_int():
    # Model plus chains as in Wolff eq. 71: y(t) = G(t) + c(t) (nu_shared + nu_t), fitted on part of the data
    tau_shared, tau_own, num_configurations = 4.0, 8.0, 16000
    c = 0.02 * np.exp(-TRUE_P[1] * T)
    rng = np.random.default_rng(0)
    nu_shared = ar1.generate_chains(np.array(tau_shared), num_configurations, rng)[:, None]
    nu_own = ar1.generate_chains(np.full(len(T), tau_own), num_configurations, rng)
    data = Variable.from_data(np.asarray(_jnp_model(T, TRUE_P)) + c * (nu_shared + nu_own), Layout("test", np.arange(num_configurations)))
    # without jackknife replicates: no refits
    v = Variable.from_results(layout=data.layout, central_value=data.central_value, fluctuations=data.fluctuations)
    fit_range = T[2:]
    weight_matrix = np.diag(1 / c[fit_range] ** 2)
    fit = v.least_squares_fit(_jnp_model, fit_range, TRUE_P, weight_matrix=weight_matrix, tolerance=1e-8)
    result = fit.parameters.gamma_method()

    # exact values, leading order in 1/N: dp/dy = (J^T W J)^-1 J^T W at the true parameters
    J = np.stack([np.exp(-TRUE_P[1] * fit_range), -TRUE_P[0] * fit_range * np.exp(-TRUE_P[1] * fit_range)], axis=1)
    per_chain = np.linalg.solve(J.T @ weight_matrix @ J, J.T @ weight_matrix) * c[fit_range]
    v_shared, v_own = per_chain.sum(axis=1) ** 2, (per_chain ** 2).sum(axis=1)
    tau_int = (tau_shared * v_shared + tau_own * v_own) / (v_shared + v_own)
    error = np.sqrt(2 * (tau_shared * v_shared + tau_own * v_own) / num_configurations)
    assert np.all(np.abs(result.tau_int - tau_int) < 3 * result.error_tau_int)
    assert np.all(np.abs(result.error - error) < 3 * result.error_of_error)
