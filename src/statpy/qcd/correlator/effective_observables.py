"""Effective masses and amplitudes of correlators."""
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def effective_mass(correlator: np.ndarray | jax.Array, *, estimator: str, Nt: int | None = None) -> jax.Array:
    """Return effective masses of a 1D correlator.

    estimator selects the formula:
      "log": forward log ratio, log(C(t)/C(t+1)).
      "log_symmetric": centered log ratio, log(C(t-1)/C(t+1))/2.
      "arccosh": three-point recurrence, valid for cosh and sinh.
      "cosh_midpoint": abs(arccosh(C(t-1)/C(Nt/2)) - arccosh(C(t+1)/C(Nt/2)))/2.
      "cosh_solve": solves cosh(m(t-Nt/2))/cosh(m(t+1-Nt/2)) = C(t)/C(t+1) for m.
      "sinh_solve": solves sinh(m(t-Nt/2))/sinh(m(t+1-Nt/2)) = C(t)/C(t+1) for m.
      (*_solve: Gattringer & Lang, Lect. Notes Phys. 788 (2010), p. 145.)
    Nt is required for cosh_solve and sinh_solve (also for folded input);
    cosh_midpoint uses the correlator length and needs unfolded data.
    Times start at zero; slices whose neighbors lie outside the array are NaN.
    Undefined points (e.g. arccosh argument < 1) return NaN or inf without warning.
    """
    if estimator == "cosh_midpoint":
        Nt = len(correlator) if Nt is None else Nt
        midpoint = correlator[Nt // 2]
        previous, following = correlator[:-2], correlator[2:]
        mass = jnp.abs(jnp.arccosh(previous / midpoint) - jnp.arccosh(following / midpoint)) / 2
        return jnp.pad(mass, (1, 1), constant_values=jnp.nan)
    if estimator == "log":
        current, following = correlator[:-1], correlator[1:]
        mass = jnp.log(current / following)
        return jnp.pad(mass, (0, 1), constant_values=jnp.nan)
    if estimator == "log_symmetric":
        previous, following = correlator[:-2], correlator[2:]
        mass = jnp.log(previous / following) / 2
        return jnp.pad(mass, (1, 1), constant_values=jnp.nan)
    if estimator == "arccosh":
        previous, current, following = correlator[:-2], correlator[1:-1], correlator[2:]
        mass = jnp.arccosh((previous + following) / (2 * current))
        return jnp.pad(mass, (1, 1), constant_values=jnp.nan)
    if estimator in ("cosh_solve", "sinh_solve"):
        if Nt is None:
            raise ValueError(f"{estimator} needs Nt")
        mass = _solve_effective_mass(correlator, kernel=estimator.removesuffix("_solve"), Nt=Nt)
        return jnp.pad(mass, (0, 1), constant_values=jnp.nan)

    raise ValueError(f"Unknown effective-mass estimator: {estimator}")


@partial(jax.jit, static_argnames=("kernel", "Nt"))
def _solve_effective_mass(correlator: np.ndarray | jax.Array, *, kernel: str, Nt: int) -> jax.Array:
    """Solve kernel(m|t-Nt/2|)/kernel(m|t+1-Nt/2|) = C(t)/C(t+1) per t = 0, ..., len - 2; kernel is "cosh" or "sinh"."""
    current, following = correlator[:-1], correlator[1:]
    t = jnp.arange(len(current))
    a = jnp.abs(t - Nt / 2)
    b = jnp.abs(t + 1 - Nt / 2)
    nonzero = (current != 0) & (following != 0)
    # 1 instead of 0 keeps reverse-mode derivatives finite
    log_ratio = jnp.log(jnp.where(nonzero, current, 1) / jnp.where(nonzero, following, 1))
    if kernel == "cosh":
        # log(cosh[m(t - Nt/2)] / cosh[m(t + 1 - Nt/2)]) - log(C(t) / C(t+1))
        def objective_function(m: jax.Array, a: jax.Array, b: jax.Array, log_ratio: jax.Array) -> jax.Array:
            return (
                m * (a - b)
                + jnp.log1p(jnp.exp(-2 * m * a))
                - jnp.log1p(jnp.exp(-2 * m * b))
                - log_ratio
            )
        lower = jnp.abs(log_ratio)
        upper = lower + jnp.log(2)
        valid = nonzero & (a != b)
    else:
        # log(sinh[m(t - Nt/2)] / sinh[m(t + 1 - Nt/2)]) - log(C(t)/C(t+1))
        def objective_function(m: jax.Array, a: jax.Array, b: jax.Array, log_ratio: jax.Array) -> jax.Array:
            return (
                m * (a - b)
                + jnp.log(-jnp.expm1(-2 * m * a))
                - jnp.log(-jnp.expm1(-2 * m * b))
                - log_ratio
            )
        upper = jnp.abs(log_ratio)
        lower = upper - jnp.abs(jnp.log(a/b))
        valid = nonzero & (a != b) & (a != 0) & (b != 0) & (lower > 0)
        lower = jnp.where(valid, lower, jnp.nan)
    root, bracketed = _find_root(objective_function, (lower, upper), args=(a, b, log_ratio))
    mass = jnp.where(bracketed & valid, root, jnp.nan)
    return mass


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _find_root(objective_function: Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array],
               bracket: tuple[jax.Array, jax.Array], args: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
    """Bisect objective_function in bracket = (lower, upper) per element; return roots and where the bracket has a sign change."""
    lower, upper = bracket
    sign_f_lower = jnp.sign(objective_function(lower, *args))
    sign_f_upper = jnp.sign(objective_function(upper, *args))
    bracketed = sign_f_lower * sign_f_upper <= 0  # signs differ or an end is a root

    def shrinking(state: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        lower, upper, sign_f_middle = state
        middle = (lower + upper) / 2
        # stop when no double lies between the ends; |sign| == 1 stops exact roots (0) and NaN
        return jnp.any((lower < middle) & (middle < upper) & (jnp.abs(sign_f_middle) == 1))

    def halve(state: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
        lower, upper, _ = state
        middle = (lower + upper) / 2
        sign_f_middle = jnp.sign(objective_function(middle, *args))
        lower = jnp.where(sign_f_middle == sign_f_lower, middle, lower)
        upper = jnp.where(sign_f_middle == sign_f_upper, middle, upper)
        return lower, upper, sign_f_middle

    lower, upper, _ = jax.lax.while_loop(shrinking, halve, (lower, upper, jnp.ones_like(lower)))
    return (lower + upper) / 2, bracketed


@_find_root.defjvp
def _find_root_jvp(objective_function, primals, tangents):
    """Implicit derivative of the root: dm = -(df/dargs . dargs) / (df/dm); the bracket has none."""
    bracket, args = primals
    _, args_tangents = tangents
    root, bracketed = _find_root(objective_function, bracket, args)
    _, df_dm = jax.jvp(lambda m: objective_function(m, *args), (root,), (jnp.ones_like(root),))
    _, df_dargs = jax.jvp(lambda *args: objective_function(root, *args), args, args_tangents)
    # zero where unbracketed; built from values only, so NaN stays out of reverse mode
    k = jnp.where(bracketed, -1 / df_dm, 0)
    return (root, bracketed), (k * df_dargs, np.zeros(bracketed.shape, jax.dtypes.float0))


def effective_amplitude(correlator: np.ndarray, mass: float, *, kernel: str, Nt: int | None = None) -> np.ndarray:
    """Divide a 1D correlator by a single-state kernel, preserving its sign.

    kernel selects the time dependence:
      "exp": exp(-mass*t).
      "cosh"/"sinh": exp(-mass*t) plus/minus exp(-mass*(Nt-t)).
    Nt is required for cosh and sinh (also for folded data).
    Times start at zero; exponential sums set the normalization.
    """
    correlator = np.asarray(correlator)
    t = np.arange(len(correlator))
    forward = np.exp(-mass * t)
    if kernel == "exp":
        return correlator / forward
    if kernel not in ("cosh", "sinh"):
        raise ValueError(f"Unknown effective-amplitude kernel: {kernel}")
    if Nt is None:
        raise ValueError(f"{kernel} kernel needs Nt")
    backward = np.exp(-mass * (Nt - t))
    return correlator / (forward + backward) if kernel == "cosh" else correlator / (forward - backward)
