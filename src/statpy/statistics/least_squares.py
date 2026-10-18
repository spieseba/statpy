"""Least-squares fits minimized with Migrad (iminuit)."""

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from iminuit import Minuit
from numba import njit
from numba.core.registry import CPUDispatcher
from scipy import stats
from scipy.linalg import cho_factor, cho_solve

if TYPE_CHECKING:
    from statpy.variable import Variable


class ConvergenceError(Exception):
    pass


@dataclass(frozen=True)
class LeastSquaresFit:
    """Least-squares fit result: parameters as a Variable and goodness of fit."""
    parameters: "Variable"
    chi2: float
    dof: int
    p_value: float


@functools.cache
def chi2(model):
    """Return the numba function ``chi2(t, p, y, W) = r @ W @ r`` with ``r = model(t, p) - y``."""
    if not isinstance(model, CPUDispatcher):
        raise TypeError(f"chi2: model must be a numba function, got {type(model).__name__}")

    @njit
    def model_chi2(t, p, y, W):
        residuals = model(t, p) - y
        return residuals @ W @ residuals
    return model_chi2


def minimize(objective, y, p0, *, tol, max_calls):
    """Return the parameters minimizing ``objective(p, y)`` with Migrad, starting from ``p0``."""
    minuit = Minuit(lambda p: objective(p, y), p0)
    minuit.tol = tol
    minuit.migrad(ncall=max_calls)
    if not minuit.valid:
        raise ConvergenceError("Migrad did not converge")
    parameters = np.array(minuit.values)
    if not np.isfinite(parameters).all():
        raise ConvergenceError(f"Migrad produced non-finite parameters: {parameters}")
    return parameters


def fit_replicates(objective, replicates, p0, *, tol, max_calls):
    """Return the parameters minimizing ``objective`` for each replicate, each fit starting from ``p0``."""
    return np.array([minimize(objective, y, p0, tol=tol, max_calls=max_calls) for y in replicates])


def p_value(chi2_value, dof):
    """Return the probability of a chi^2 of at least ``chi2_value`` with ``dof`` degrees of freedom."""
    return stats.chi2.sf(chi2_value, dof)


def inverse_covariance(cov, num_samples):
    """Return ``(inverse, None)``, or ``(None, reason)`` if ``cov`` is not invertible."""
    if len(cov) >= num_samples:
        return None, f"singular: {len(cov)} points from {num_samples} samples"
    sigma = np.sqrt(np.diag(cov))
    if not np.all(np.isfinite(sigma) & (sigma > 0)):
        return None, "not positive definite: non-positive or non-finite variance"
    scale = np.outer(sigma, sigma)
    try:
        factor = cho_factor(cov / scale, lower=True)
    except np.linalg.LinAlgError:
        return None, "not positive definite"
    return cho_solve(factor, np.eye(len(cov))) / scale, None


def aic(chi2, dof, num_parameters):
    """Return chi2 - dof + num_parameters, following arXiv:2211.03744."""
    return chi2 - dof + num_parameters


def model_prediction_variance(t, parameters, covariance, gradient):
    """Return the model prediction variance from its parameter gradient and covariance."""
    g = gradient(t, parameters)
    return g @ covariance @ g
