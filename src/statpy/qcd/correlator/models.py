"""Model factories for correlator and effective-mass fits."""
import functools

import jax
import jax.numpy as jnp
import numpy as np

from statpy.statistics.least_squares import Model


@functools.cache
def exp() -> Model:
    """Return the model ``A * exp(-mt)``; p = [A, m]."""
    def model(t: np.ndarray, p: np.ndarray) -> jax.Array:
        return p[0] * jnp.exp(-p[1]*t)
    return model


@functools.cache
def cosh(Nt: int) -> Model:
    """Return the model ``A * [exp(-mt) + exp(-m(Nt-t))]``; p = [A, m]."""
    def model(t: np.ndarray, p: np.ndarray) -> jax.Array:
        return p[0] * ( jnp.exp(-p[1]*t) + jnp.exp(-p[1]*(Nt-t)) )
    return model


@functools.cache
def sinh(Nt: int) -> Model:
    """Return the model ``A * [exp(-mt) - exp(-m(Nt-t))]``; p = [A, m]."""
    def model(t: np.ndarray, p: np.ndarray) -> jax.Array:
        return p[0] * ( jnp.exp(-p[1]*t) - jnp.exp(-p[1]*(Nt-t)) )
    return model


@functools.cache
def double_exp() -> Model:
    """Return the model ``A0 * exp(-m0t) + A1 * exp(-m1t)``; p = [A0, m0, A1, m1]."""
    def model(t: np.ndarray, p: np.ndarray) -> jax.Array:
        return p[0] * jnp.exp(-p[1]*t) + p[2] * jnp.exp(-p[3]*t)
    return model


@functools.cache
def double_cosh(Nt: int) -> Model:
    """Return the model ``A0 * [exp(-m0t) + exp(-m0(Nt-t))] + A1 * [exp(-m1t) + exp(-m1(Nt-t))]``;
    p = [A0, m0, A1, m1]."""
    def model(t: np.ndarray, p: np.ndarray) -> jax.Array:
        return p[0] * ( jnp.exp(-p[1]*t) + jnp.exp(-p[1]*(Nt-t)) ) + p[2] * ( jnp.exp(-p[3]*t) + jnp.exp(-p[3]*(Nt-t)) )
    return model


@functools.cache
def double_sinh(Nt: int) -> Model:
    """Return the model ``A0 * [exp(-m0t) - exp(-m0(Nt-t))] + A1 * [exp(-m1t) - exp(-m1(Nt-t))]``;
    p = [A0, m0, A1, m1]."""
    def model(t: np.ndarray, p: np.ndarray) -> jax.Array:
        return p[0] * ( jnp.exp(-p[1]*t) - jnp.exp(-p[1]*(Nt-t)) ) + p[2] * ( jnp.exp(-p[3]*t) - jnp.exp(-p[3]*(Nt-t)) )
    return model


@functools.cache
def const_plus_exp() -> Model:
    """Return the model ``A * exp(-mt) + c``; p = [A, m, c]."""
    def model(t: np.ndarray, p: np.ndarray) -> jax.Array:
        return p[0] * jnp.exp(-p[1] * t) + p[2]
    return model


@functools.cache
def combined(Nt: int, L: int, signs: tuple[float, float]) -> Model:
    """Return the model of two joined correlators with amplitudes A0, A1 and a shared mass m;
    p = [A0, A1, m].

    Correlator i is ``Ai * [exp(-mt) + si * exp(-m(Nt-t))]`` with ``(s0, s1) = signs``, each +1 cosh,
    -1 sinh, 0 exp. An entry ``t < L`` is slice ``t`` of correlator 0, an entry ``t >= L`` is slice
    ``t - L`` of correlator 1.
    """
    s0, s1 = signs

    def model(t: np.ndarray, p: np.ndarray) -> jax.Array:
        A0, A1, m = p[0], p[1], p[2]
        first_correlator = t < L
        tau = jnp.where(first_correlator, t, t - L)
        A = jnp.where(first_correlator, A0, A1)
        s = jnp.where(first_correlator, s0, s1)
        return A * (jnp.exp(-m * tau) + s * jnp.exp(-m * (Nt - tau)))
    return model
