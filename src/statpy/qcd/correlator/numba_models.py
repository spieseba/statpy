"""Numba model factories for correlator and effective-mass fits."""
import functools

import numpy as np
from numba import njit

from statpy.statistics.least_squares import Model


@functools.cache
def exp() -> Model:
    """Return the numba model ``A * exp(-mt)``; p = [A, m]."""
    @njit(cache=True)
    def model(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] * np.exp(-p[1]*t)
    return model


@functools.cache
def cosh(Nt: int) -> Model:
    """Return the numba model ``A * [exp(-mt) + exp(-m(Nt-t))]``; p = [A, m]."""
    @njit(cache=True)
    def model(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] * ( np.exp(-p[1]*t) + np.exp(-p[1]*(Nt-t)) )
    return model


@functools.cache
def sinh(Nt: int) -> Model:
    """Return the numba model ``A * [exp(-mt) - exp(-m(Nt-t))]``; p = [A, m]."""
    @njit(cache=True)
    def model(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] * ( np.exp(-p[1]*t) - np.exp(-p[1]*(Nt-t)) )
    return model


@functools.cache
def double_exp() -> Model:
    """Return the numba model ``A0 * exp(-m0t) + A1 * exp(-m1t)``; p = [A0, m0, A1, m1]."""
    @njit(cache=True)
    def model(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] * np.exp(-p[1]*t) + p[2] * np.exp(-p[3]*t)
    return model


@functools.cache
def double_cosh(Nt: int) -> Model:
    """Return the numba model ``A0 * [exp(-m0t) + exp(-m0(Nt-t))] + A1 * [exp(-m1t) + exp(-m1(Nt-t))]``;
    p = [A0, m0, A1, m1]."""
    @njit(cache=True)
    def model(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] * ( np.exp(-p[1]*t) + np.exp(-p[1]*(Nt-t)) ) + p[2] * ( np.exp(-p[3]*t) + np.exp(-p[3]*(Nt-t)) )
    return model


@functools.cache
def double_sinh(Nt: int) -> Model:
    """Return the numba model ``A0 * [exp(-m0t) - exp(-m0(Nt-t))] + A1 * [exp(-m1t) - exp(-m1(Nt-t))]``;
    p = [A0, m0, A1, m1]."""
    @njit(cache=True)
    def model(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] * ( np.exp(-p[1]*t) - np.exp(-p[1]*(Nt-t)) ) + p[2] * ( np.exp(-p[3]*t) - np.exp(-p[3]*(Nt-t)) )
    return model


@functools.cache
def const_plus_exp() -> Model:
    """Return the numba model ``A * exp(-mt) + c``; p = [A, m, c]."""
    @njit(cache=True)
    def model(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] * np.exp(-p[1] * t) + p[2]
    return model


@functools.cache
def combined(Nt: int, L: int, signs: tuple[float, float]) -> Model:
    """Return the numba model of two joined correlators with amplitudes A0, A1 and a shared mass m;
    p = [A0, A1, m].

    Correlator i is ``Ai * [exp(-mt) + si * exp(-m(Nt-t))]`` with ``(s0, s1) = signs``, each +1 cosh,
    -1 sinh, 0 exp. An entry ``t < L`` is slice ``t`` of correlator 0, an entry ``t >= L`` is slice
    ``t - L`` of correlator 1.
    """
    s0, s1 = signs

    @njit(cache=True)
    def model(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        A0, A1, m = p[0], p[1], p[2]
        first_correlator = t < L
        tau = np.where(first_correlator, t, t - L)
        A = np.where(first_correlator, A0, A1)
        s = np.where(first_correlator, s0, s1)
        return A * (np.exp(-m * tau) + s * np.exp(-m * (Nt - tau)))
    return model
