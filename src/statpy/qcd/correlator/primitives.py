"""Pure correlator / effective-mass primitives. No database state."""
from collections.abc import Callable
from functools import partial
from numbers import Real

import jax
import jax.numpy as jnp
import numpy as np

from statpy.log import message

# ---------------------------------------------------------------------------
# Effective mass and amplitude
# ---------------------------------------------------------------------------

def effective_mass(corr: np.ndarray | jax.Array, *, estimator: str, Nt: int | None = None) -> jax.Array:
    """Return effective masses of a 1D correlator.

    estimator selects the formula:
      "log": forward log ratio, log(C(t)/C(t+1)).
      "log_symmetric": centered log ratio, log(C(t-1)/C(t+1))/2.
      "arccosh": three-point recurrence, valid for cosh and sinh.
      "cosh_midpoint": abs(acosh(C(t-1)/Cmid) - acosh(C(t+1)/Cmid))/2.
      "cosh_solve": solves cosh(m(t-Nt/2))/cosh(m(t+1-Nt/2)) = C(t)/C(t+1) for m.
      "sinh_solve": solves sinh(m(t-Nt/2))/sinh(m(t+1-Nt/2)) = C(t)/C(t+1) for m.
      (*_solve: Gattringer & Lang, Lect. Notes Phys. 788 (2010), p. 145.)
    Nt is required for cosh_solve and sinh_solve (also for folded input);
    cosh_midpoint uses the correlator length and needs unfolded data.
    Times start at zero; neighbor formulas wrap endpoints as with np.roll.
    Undefined points (e.g. arccosh argument < 1) return NaN or inf without warning.
    """
    if estimator == "cosh_midpoint":
        Nt = len(corr) if Nt is None else Nt
        midpoint = corr[Nt // 2]
        return jnp.abs(
            jnp.arccosh(jnp.roll(corr, 1) / midpoint)
            - jnp.arccosh(jnp.roll(corr, -1) / midpoint)
        ) / 2
    if estimator == "log":
        return jnp.log(corr / jnp.roll(corr, -1))
    if estimator == "log_symmetric":
        return jnp.log(jnp.roll(corr, 1) / jnp.roll(corr, -1)) / 2
    if estimator == "arccosh":
        return jnp.arccosh(0.5 * (jnp.roll(corr, -1) + jnp.roll(corr, 1)) / corr)
    if estimator in ("cosh_solve", "sinh_solve"):
        if Nt is None:
            raise ValueError(f"{estimator} needs Nt")
        return _solve_effective_mass(corr, kernel=estimator.removesuffix("_solve"), Nt=Nt)

    raise ValueError(f"Unknown effective-mass estimator: {estimator}")


@partial(jax.jit, static_argnames=("kernel", "Nt"))
def _solve_effective_mass(corr: np.ndarray | jax.Array, *, kernel: str, Nt: int) -> jax.Array:
    """Solve kernel(m|t-Nt/2|)/kernel(m|t+1-Nt/2|) = C(t)/C(t+1) per t; kernel is "cosh" or "sinh"."""
    t = jnp.arange(len(corr))
    a = jnp.abs(t - Nt / 2)
    b = jnp.abs(t + 1 - Nt / 2)
    next_corr = jnp.roll(corr, -1)
    nonzero = (corr != 0) & (next_corr != 0)
    # 1 instead of 0 keeps reverse-mode derivatives finite
    log_ratio = jnp.log(jnp.where(nonzero, corr, 1) / jnp.where(nonzero, next_corr, 1))
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
               init: tuple[jax.Array, jax.Array], args: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
    """Bisect objective_function in [lower, upper] per element; return roots and where the bracket has a sign change."""
    lower, upper = init
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
    init, args = primals
    _, args_tangents = tangents
    root, bracketed = _find_root(objective_function, init, args)
    _, df_dm = jax.jvp(lambda m: objective_function(m, *args), (root,), (jnp.ones_like(root),))
    _, df_dargs = jax.jvp(lambda *args: objective_function(root, *args), args, args_tangents)
    # zero where unbracketed; built from values only, so NaN stays out of reverse mode
    k = jnp.where(bracketed, -1 / df_dm, 0)
    return (root, bracketed), (k * df_dargs, np.zeros(bracketed.shape, jax.dtypes.float0))


def effective_amplitude(corr: np.ndarray, mass: float, *, kernel: str, Nt: int | None = None) -> np.ndarray:
    """Divide a 1D correlator by a single-state kernel, preserving its sign.

    kernel selects the time dependence:
      "exp": exp(-mass*t).
      "cosh"/"sinh": exp(-mass*t) plus/minus exp(-mass*(Nt-t)).
    Nt is required for cosh and sinh (also for folded data).
    Times start at zero; exponential sums set the normalization.
    """
    corr = np.asarray(corr)
    t = np.arange(len(corr))
    forward = np.exp(-mass * t)
    if kernel == "exp":
        return corr / forward
    if kernel not in ("cosh", "sinh"):
        raise ValueError(f"Unknown effective-amplitude kernel: {kernel}")
    if Nt is None:
        raise ValueError(f"{kernel} kernel needs Nt")
    backward = np.exp(-mass * (Nt - t))
    return corr / (forward + backward) if kernel == "cosh" else corr / (forward - backward)


# ---------------------------------------------------------------------------
# Folding
# ---------------------------------------------------------------------------

def meson_fold_correlator(corr: np.ndarray, time_parity: int = 1) -> np.ndarray:
    """Fold a meson correlator around T/2; time_parity is +1 (even) or -1 (odd)."""
    time_parity = _validate_time_parity(time_parity)
    half = len(corr) // 2
    first_half = corr[:half]
    second_half = np.roll(np.flip(corr[half:]), 1) * time_parity
    second_half[0] = first_half[0]
    return np.mean([first_half, second_half], axis=0)


def _validate_time_parity(time_parity: int) -> int:
    """Return the sign for even (+1) or odd (-1) time parity; reject booleans."""
    if (
        isinstance(time_parity, (bool, np.bool_))
        or not isinstance(time_parity, Real)
        or time_parity not in (-1, 1)
    ):
        raise ValueError("time_parity must be +1 (even) or -1 (odd), not a boolean")
    return int(time_parity)


# ---------------------------------------------------------------------------
# Signal-to-noise cut
# ---------------------------------------------------------------------------

def get_tmax_signal_to_noise(mean: np.ndarray, var: np.ndarray, min_signal_to_noise: float, tmin: int | None = None,
                             debug: bool = False) -> int:
    """Return the first failing slice (exclusive endpoint), or the data length.

    A failure is S/N below the required threshold or NaN. Search all slices
    unless tmin is supplied, in which case only t > tmin is considered.
    """
    signal_to_noise = mean / var**.5
    times = np.arange(len(signal_to_noise))
    invalid = (signal_to_noise < min_signal_to_noise) | np.isnan(signal_to_noise)
    if tmin is not None:
        invalid &= times > tmin
    candidates = times[invalid]

    tmax = int(candidates[0]) if candidates.size else len(mean)
    if not candidates.size:
        reason = "End of data; no failing slice in search region"
    else:
        reason = "First NaN" if np.isnan(signal_to_noise[tmax]) else "First value below threshold"
    search_region = "all time slices" if tmin is None else f"t > {tmin}"
    message(
        "Signal-to-noise cutoff\n"
        f"  {'Threshold':<14} {min_signal_to_noise}\n"
        f"  {'Search region':<14} {search_region}\n"
        f"  {'tmax':<14} {tmax} (exclusive)\n"
        f"  {'Reason':<14} {reason}"
    )
    if debug:
        message(f"Signal-to-noise values ({len(signal_to_noise)} slices):\n{signal_to_noise}")
    return tmax
