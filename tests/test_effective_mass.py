"""Root-solve effective masses on exact cosh and sinh correlators."""

import jax
import numpy as np
import pytest

from statpy.qcd.correlator.effective_observables import effective_mass
from statpy.statistics import gamma

Nt, m = 48, 0.3


@pytest.mark.parametrize(("kernel", "undefined"), [("cosh", []), ("sinh", [Nt // 2 - 1, Nt // 2, Nt - 1])])
def test_solve_returns_mass_and_nan_where_undefined(kernel, undefined):
    corr = getattr(np, kernel)(m * (np.arange(Nt) - Nt / 2))
    mass = np.asarray(effective_mass(corr, estimator=f"{kernel}_solve", Nt=Nt))
    np.testing.assert_array_equal(np.flatnonzero(np.isnan(mass)), undefined)
    np.testing.assert_allclose(np.delete(mass, undefined), m, rtol=1e-13)
    # a sign flip and a zero must give NaN at their neighbours, not stall the bisection
    corr[10], corr[30] = -corr[10], 0.0
    mass = np.asarray(effective_mass(corr, estimator=f"{kernel}_solve", Nt=Nt))
    np.testing.assert_array_equal(np.flatnonzero(np.isnan(mass)), sorted({9, 10, 29, 30, *undefined}))


@pytest.mark.parametrize("kernel", ["cosh", "sinh"])
@pytest.mark.parametrize("jacobian", [jax.jacfwd, jax.jacrev])
def test_solve_jacobian_matches_central_differences(kernel, jacobian):
    corr = getattr(np, kernel)(m * (np.arange(Nt) - Nt / 2))
    corr[30] = 0.0  # zero entries (also sinh at Nt/2) must not spread NaN into valid rows
    def func(c):
        return effective_mass(c, estimator=f"{kernel}_solve", Nt=Nt)
    valid = ~np.isnan(np.asarray(func(corr)))
    numerical = gamma.numerical_jacobians(func, [corr], step_sizes=[1e-6 * np.abs(corr)])[0]
    np.testing.assert_allclose(np.asarray(jacobian(func)(corr))[valid], numerical[valid], atol=1e-8)
