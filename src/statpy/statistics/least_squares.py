"""Least-squares fits minimized with Migrad (iminuit)."""

import functools
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
from iminuit import Minuit
from numba import njit
from numba.core.registry import CPUDispatcher
from numpy.typing import ArrayLike
from scipy import stats
from scipy.linalg import cho_factor, cho_solve

from statpy.plot import legend_above, style_axes

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from statpy.variable import Variable


Model = Callable[[np.ndarray, np.ndarray], np.ndarray | jax.Array]


class ConvergenceError(Exception):
    pass


class HessianNotPositiveDefiniteError(Exception):
    """The chi^2 Hessian is not positive definite: the parameters are not at a strict minimum."""


@dataclass(frozen=True)
class LeastSquaresFit:
    """Least-squares fit result: parameters as a Variable, goodness of fit, and the model, fit range and data."""
    parameters: Variable
    chi2: float
    dof: int
    p_value: float
    model: Model
    fit_range: np.ndarray
    data: Variable

    def summarize(self, S: float = 1.5, components: Mapping[str, int | tuple[int, ...]] | Iterable[int | tuple[int, ...]] | None = None) -> None:
        """Print the fit range, chi^2/dof and p-value, then ``parameters.summarize(S, components)``."""
        runs = np.split(self.fit_range, np.flatnonzero(np.diff(self.fit_range) != 1) + 1)
        print(f"fit range {', '.join(f'{run[0]}..{run[-1]}' for run in runs)} ({len(self.fit_range)} points), "
              f"chi2/dof = {self.chi2:.3g}/{self.dof} = {self.chi2 / self.dof:.3g}, p-value = {self.p_value:.2f}")
        self.parameters.summarize(S, components)

    def plot(self, ax: Axes, error: Literal["gamma", "jackknife", "bootstrap"] = "gamma",
             plot_range: ArrayLike | None = None) -> Axes:
        """Plot the data at the indices ``plot_range`` (default all) with ``error`` bars into ``ax``, and the fitted
        model with its ``error`` band on the part of the fit range inside ``plot_range``."""
        indices = np.arange(len(self.data.central_value)) if plot_range is None else np.asarray(plot_range)
        points = ax.errorbar(indices, self.data.central_value[indices], _variable_error(self.data, error)[indices], fmt="o",
                             markersize=5, markeredgecolor="white", markeredgewidth=0.6, elinewidth=1, capsize=3, zorder=3,
                             label=f"data ({error})")
        color = points[0].get_color()
        prediction = self.parameters.derive(lambda parameters: self.model(self.fit_range, parameters), self.parameters)
        in_plot = np.isin(self.fit_range, indices)
        x, y, band = self.fit_range[in_plot], prediction.central_value[in_plot], _variable_error(prediction, error)[in_plot]
        # NaN at jumps of the fit range, so the line and band break there
        jumps = np.flatnonzero(np.diff(x) > np.diff(self.fit_range).min()) + 1
        x, y, band = (np.insert(values, jumps, np.nan) for values in (x.astype(float), y, band))
        ax.plot(x, y, color=color, linewidth=1.2, label=f"fit ({error})")
        ax.fill_between(x, y - band, y + band, color=color, alpha=0.3, linewidth=0, label=f"fit ({error})")
        # One legend entry per label; the fit symbol shows line and band together
        entries = {}
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            entries.setdefault(label, []).append(handle)
        legend_above(ax, [tuple(handles) for handles in entries.values()], list(entries), ncols=2)
        ax.set(xlabel="$x$", ylabel="$y$")
        style_axes(ax)
        return ax


def _variable_error(variable: Variable, error: str) -> np.ndarray:
    """Return the ``"gamma"``, ``"jackknife"`` or ``"bootstrap"`` error of ``variable``; raises ``ValueError`` if absent."""
    if error == "gamma":
        return variable.gamma_method().error
    if error == "jackknife":
        return np.sqrt(variable.jackknife_variance())
    if error == "bootstrap":
        return np.sqrt(variable.bootstrap_variance())
    raise ValueError(f"Unknown error: {error}")


# ---------------------------------------------------------------------------
# Weight matrix
# ---------------------------------------------------------------------------

def inverse_covariance(cov: np.ndarray, num_samples: int) -> tuple[np.ndarray | None, str | None]:
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


# ---------------------------------------------------------------------------
# Minimization
# ---------------------------------------------------------------------------

@jax.jit(static_argnums=0)
def chi2_gradient(model: Model, x: np.ndarray, y: np.ndarray, parameters: np.ndarray,
                  weight_matrix: np.ndarray) -> jax.Array:
    """Return d chi2 / dp, shape (parameters), compiled once per model and input shape."""
    chi2_function = chi2(model)
    return jax.grad(chi2_function, argnums=2)(x, y, parameters, weight_matrix)


@functools.cache
def chi2(model: Model) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray | jax.Array]:
    """Return ``chi2(x, y, parameters, weight_matrix) = r @ weight_matrix @ r`` with
    ``r = model(x, parameters) - y``; compiled with numba for a numba ``model``, otherwise with ``jax.jit``."""
    def model_chi2(x: np.ndarray, y: np.ndarray, parameters: np.ndarray, weight_matrix: np.ndarray) -> np.ndarray | jax.Array:
        residuals = model(x, parameters) - y
        return residuals @ weight_matrix @ residuals
    return njit(model_chi2) if isinstance(model, CPUDispatcher) else jax.jit(model_chi2)


def minimize(objective: Callable[[np.ndarray, np.ndarray], float], y: np.ndarray, initial_parameters: np.ndarray, *,
             tolerance: float | None, max_calls: int | None, objective_gradient: Callable[[np.ndarray, np.ndarray], np.ndarray | jax.Array] | None) -> np.ndarray:
    """Return the parameters minimizing ``objective(y, parameters)`` with Migrad, starting from
    ``initial_parameters``."""
    grad = (lambda parameters: objective_gradient(y, parameters)) if objective_gradient is not None else None
    minuit = Minuit(lambda parameters: objective(y, parameters), initial_parameters, grad=grad)  # pyrefly: ignore[bad-argument-type]  # iminuit types fcn with float arguments only
    minuit.tol = tolerance
    minuit.migrad(ncall=max_calls)
    if not minuit.valid:
        raise ConvergenceError("Migrad did not converge")
    parameters = np.array(minuit.values)
    if not np.isfinite(parameters).all():
        raise ConvergenceError(f"Migrad produced non-finite parameters: {parameters}")
    return parameters


def fit_replicates(objective: Callable[[np.ndarray, np.ndarray], float], replicates: np.ndarray,
                   initial_parameters: np.ndarray, *, tolerance: float | None, max_calls: int | None,
                   objective_gradient: Callable[[np.ndarray, np.ndarray], np.ndarray | jax.Array] | None) -> np.ndarray:
    """Return the parameters minimizing ``objective`` for each replicate, each fit starting from ``initial_parameters``."""
    return np.array([minimize(objective, y, initial_parameters, tolerance=tolerance, max_calls=max_calls, objective_gradient=objective_gradient)
                     for y in replicates])


# ---------------------------------------------------------------------------
# Fit quality
# ---------------------------------------------------------------------------

def p_value(chi2_value: float, dof: int) -> float:
    """Return the probability of a chi^2 of at least ``chi2_value`` with ``dof`` degrees of freedom."""
    return float(stats.chi2.sf(chi2_value, dof))


def aic(chi2_value: float, dof: int, num_parameters: int) -> float:
    """Return chi2 - dof + num_parameters, following arXiv:2211.03744."""
    return chi2_value - dof + num_parameters


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------

@jax.jit(static_argnums=0)
def _chi2_second_derivatives(model: Model, x: np.ndarray, y: np.ndarray, parameters: np.ndarray,
                             weight_matrix: np.ndarray) -> tuple[jax.Array, jax.Array]:
    """Return d^2 chi2 / dp^2, shape (parameters, parameters), and d^2 chi2 / dp dy, shape (parameters, data points),
    compiled once per model and input shape."""
    chi2_function = chi2(model)
    hessian = jax.hessian(chi2_function, argnums=2)(x, y, parameters, weight_matrix)
    mixed_derivatives = jax.jacfwd(jax.grad(chi2_function, argnums=2), argnums=1)(x, y, parameters, weight_matrix)
    return hessian, mixed_derivatives


def parameter_jacobian(model: Model, x: np.ndarray, y: np.ndarray, parameters: np.ndarray,
                       weight_matrix: np.ndarray) -> np.ndarray:
    """Return the Jacobian of the fitted ``parameters`` with respect to the data ``y``, shape (parameters, data
    points), at the chi^2 minimum ``parameters``; ``model`` must be traceable by JAX.

    Raises ``HessianNotPositiveDefiniteError`` if the chi^2 Hessian is not positive definite; warns if it is
    ill-conditioned.
    """
    hessian, mixed_derivatives = _chi2_second_derivatives(model, x, y, parameters, weight_matrix)
    diagonal = jnp.diag(hessian)
    if jnp.any(diagonal <= 0):
        raise HessianNotPositiveDefiniteError(f"parameter_jacobian: chi^2 Hessian has a non-positive diagonal: {diagonal}")
    scaling = jnp.diag(1 / jnp.sqrt(diagonal))
    eigenvalues, eigenvectors = jnp.linalg.eigh(scaling @ hessian @ scaling)  # unit diagonal
    if eigenvalues[0] <= 0:
        raise HessianNotPositiveDefiniteError(f"parameter_jacobian: chi^2 Hessian has a non-positive eigenvalue: {eigenvalues}")
    condition_number = eigenvalues[-1] / eigenvalues[0]
    relative_error = condition_number * jnp.finfo(jnp.float64).eps  # error bound of the solve
    if relative_error > 1e-8:
        warnings.warn(f"parameter_jacobian: ill-conditioned chi^2 Hessian (condition number {condition_number:.1e}); "
                      f"Jacobian accurate to fewer than 8 digits (relative error about {relative_error:.0e})",
                      stacklevel=2)
    inverse_hessian = scaling @ eigenvectors @ jnp.diag(1 / eigenvalues) @ eigenvectors.T @ scaling
    return np.asarray(-inverse_hessian @ mixed_derivatives)


def model_prediction_variance(t: float, parameters: np.ndarray, covariance: np.ndarray,
                              gradient: Callable[[float, np.ndarray], np.ndarray]) -> float:
    """Return the model prediction variance from its parameter gradient and covariance."""
    g = gradient(t, parameters)
    return float(g @ covariance @ g)
