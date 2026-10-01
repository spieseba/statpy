"""Fit model classes and chi^2 functions for correlator and effective-mass fits."""
import numpy as np
from numba import njit

FIT_MODEL_FORMULAS = {
    "cosh": "A * [exp(-mt) + exp(-m(Nt-t))]; A = p[0], m = p[1]",
    "sinh": "A * [exp(-mt) - exp(-m(Nt-t))]; A = p[0], m = p[1]",
    "exp": "A * exp(-mt); A = p[0], m = p[1]",
    "double-cosh": "A0 * [exp(-m0t) + exp(-m0(Nt-t))] + A1 * [exp(-m1t) + exp(-m1(Nt-t))]; A0 = p[0], m0 = p[1], A1 = p[2], m1 = p[3]",
    "double-sinh": "A0 * [exp(-m0t) - exp(-m0(Nt-t))] + A1 * [exp(-m1t) - exp(-m1(Nt-t))]; A0 = p[0], m0 = p[1], A1 = p[2], m1 = p[3]",
    "double-exp": "A0 * exp(-m0t) + A1 * exp(-m1t); A0 = p[0], m0 = p[1], A1 = p[2], m1 = p[3]",
}

_BLOCK_FORMULAS = {
    "cosh": "A{i} * [exp(-mt) + exp(-m(Nt-t))]",
    "sinh": "A{i} * [exp(-mt) - exp(-m(Nt-t))]",
    "exp": "A{i} * exp(-mt)",
}
FIT_MODEL_FORMULAS |= {
    f"combined-{m0}-{m1}": f"C0(t) = {_BLOCK_FORMULAS[m0].format(i=0)}, C1(t) = {_BLOCK_FORMULAS[m1].format(i=1)}; A0 = p[0], A1 = p[1], m = p[2]"
    for m0 in _BLOCK_FORMULAS for m1 in _BLOCK_FORMULAS
}


# ---------------------------------------------------------------------------
# cosh model to fit correlator with periodic boundary conditions
# ---------------------------------------------------------------------------

# C(t) = A * [exp(-mt) + exp(-m(Nt-t))]; A = p[0], m = p[1]
class CoshModel:
    def __init__(self, Nt):
        self.Nt = Nt
    def __call__(self, t, p):
        return p[0] * ( np.exp(-p[1]*t) + np.exp(-p[1]*(self.Nt-t)) )
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t) + np.exp(-p[1]*(self.Nt-t)), p[0] * (np.exp(-p[1]*t) * (-t) + np.exp(-p[1]*(self.Nt-t)) * (t-self.Nt))])

@njit(cache=True)
def cosh_chi2(t, p, y, W, Nt):
    model = p[0] * ( np.exp(-p[1]*t) + np.exp(-p[1]*(Nt-t)) )
    return (model - y) @ W @ (model - y)


# ---------------------------------------------------------------------------
# double cosh model to fit correlator with periodic boundary conditions
# ---------------------------------------------------------------------------

# C(t) = A0 * [exp(-m0 t) + exp(-m0(Nt-t))] + A1 * [exp(-m1 t) + exp(-m1(Nt-t))]; A0 = p[0], m0 = p[1], A1 = p[2], m1 = p[3]
class DoubleCoshModel:
    def __init__(self, Nt):
        self.Nt = Nt
    def __call__(self, t, p):
        return p[0] * ( np.exp(-p[1]*t) + np.exp(-p[1]*(self.Nt-t)) ) + p[2] * ( np.exp(-p[3]*t) + np.exp(-p[3]*(self.Nt-t)) )
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t) + np.exp(-p[1]*(self.Nt-t)), p[0] * (np.exp(-p[1]*t) * (-t) + np.exp(-p[1]*(self.Nt-t)) * (t-self.Nt)),
                         np.exp(-p[3]*t) + np.exp(-p[3]*(self.Nt-t)), p[2] * (np.exp(-p[3]*t) * (-t) + np.exp(-p[3]*(self.Nt-t)) * (t-self.Nt))])

@njit(cache=True)
def double_cosh_chi2(t, p, y, W, Nt):
    model = p[0] * ( np.exp(-p[1]*t) + np.exp(-p[1]*(Nt-t)) ) + p[2] * ( np.exp(-p[3]*t) + np.exp(-p[3]*(Nt-t)) )
    return (model - y) @ W @ (model - y)


# ---------------------------------------------------------------------------
# sinh model to fit correlator with periodic boundary conditions
# ---------------------------------------------------------------------------

# C(t) = A * [exp(-mt) - exp(-m(Nt-t))]; A = p[0], m = p[1]
class SinhModel:
    def __init__(self, Nt):
        self.Nt = Nt
    def __call__(self, t, p):
        return p[0] * ( np.exp(-p[1]*t) - np.exp(-p[1]*(self.Nt-t)) )
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t) - np.exp(-p[1]*(self.Nt-t)), p[0] * (np.exp(-p[1]*t) * (-t) - np.exp(-p[1]*(self.Nt-t)) * (t-self.Nt))])

@njit(cache=True)
def sinh_chi2(t, p, y, W, Nt):
    model = p[0] * ( np.exp(-p[1]*t) - np.exp(-p[1]*(Nt-t)) )
    return (model - y) @ W @ (model - y)


# ---------------------------------------------------------------------------
# double sinh model to fit correlator with periodic boundary conditions
# ---------------------------------------------------------------------------

# C(t) = A0 * [exp(-m0 t) - exp(-m0(Nt-t))] + A1 * [exp(-m1 t) - exp(-m1(Nt-t))]; A0 = p[0], m0 = p[1], A1 = p[2], m1 = p[3]
class DoubleSinhModel:
    def __init__(self, Nt):
        self.Nt = Nt
    def __call__(self, t, p):
        return p[0] * ( np.exp(-p[1]*t) - np.exp(-p[1]*(self.Nt-t)) ) + p[2] * ( np.exp(-p[3]*t) - np.exp(-p[3]*(self.Nt-t)) )
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t) - np.exp(-p[1]*(self.Nt-t)),
                    p[0] * (np.exp(-p[1]*t) * (-t) - np.exp(-p[1]*(self.Nt-t)) * (t-self.Nt)),
                    np.exp(-p[3]*t) - np.exp(-p[3]*(self.Nt-t)),
                    p[2] * (np.exp(-p[3]*t) * (-t) - np.exp(-p[3]*(self.Nt-t)) * (t-self.Nt))])

@njit(cache=True)
def double_sinh_chi2(t, p, y, W, Nt):
    model = p[0] * ( np.exp(-p[1]*t) - np.exp(-p[1]*(Nt-t)) ) + p[2] * ( np.exp(-p[3]*t) - np.exp(-p[3]*(Nt-t)) )
    return (model - y) @ W @ (model - y)


# ---------------------------------------------------------------------------
# exp model to fit correlator with open boundary conditions
# ---------------------------------------------------------------------------

# f(t) = A * exp(-mt); A = p[0], m = p[1]
class ExpModel:
    def __init__(self):
        pass
    def __call__(self, t, p):
        return p[0] * np.exp(-p[1]*t)
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t), p[0] * np.exp(-p[1]*t) * (-t)])

@njit(cache=True)
def exp_chi2(t, p, y, W):
    model = p[0] * np.exp(-p[1]*t)
    return (model - y) @ W @ (model - y)


# ---------------------------------------------------------------------------
# double exp model to fit correlator with open boundary conditions
# ---------------------------------------------------------------------------

# C(t) = A0 * exp(-m0t) + A1 * exp(-m1t); A0 = p[0], m0 = p[1], A1 = p[2], m1 = p[3]
class DoubleExpModel:
    def __init__(self):
        pass
    def __call__(self, t, p):
        return p[0] * np.exp(-p[1]*t) + p[2] * np.exp(-p[3]*t)
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t), p[0] * np.exp(-p[1]*t) * (-t), np.exp(-p[3]*t), p[2] * np.exp(-p[3]*t) * (-t)])

@njit(cache=True)
def double_exp_chi2(t, p, y, W):
    model = p[0] * np.exp(-p[1]*t) + p[2] * np.exp(-p[3]*t)
    return (model - y) @ W @ (model - y)


# ---------------------------------------------------------------------------
# const plus exp model to fit effective mass plateau
# ---------------------------------------------------------------------------

class ConstPlusExpModel:
    def __call__(self, t, p):
        return p[0] * np.exp(-p[1] * t) + p[2]

def const_plus_exp_chi2(t, p, y, W):
    with np.errstate(over='ignore', invalid='ignore'):
        model = p[0] * np.exp(-p[1] * t) + p[2]
        return (model - y) @ W @ (model - y)


# ---------------------------------------------------------------------------
# combined two-correlator fit (independent amplitudes, shared ground-state mass)
# ---------------------------------------------------------------------------

# C0(t) = A0 * [exp(-mt) + s0 * exp(-m(Nt-t))],  C1(t) = A1 * [exp(-mt) + s1 * exp(-m(Nt-t))]
# s = +1 cosh, -1 sinh, 0 exp; A0 = p[0], A1 = p[1], m = p[2]
@njit(cache=True)
def combined_correlator_chi2(t, p, y, W, Nt, L, s0, s1):
    """chi^2 of two correlators with separate amplitudes and a shared mass.

    t indexes both correlators joined into one array of length 2*L, L being the length
    of one correlator (e.g. Nt unfolded, Nt/2 folded, shorter after OBC averaging):
    block-0 time slices, then block-1 time slices + L, e.g. concatenate([t0, L + t1]).
    y and W follow the same order. Nt is the period used by cosh and sinh blocks.
    """
    A0, A1, m = p[0], p[1], p[2]
    t0, t1 = t[t < L], t[t >= L] - L
    block0 = A0 * (np.exp(-m * t0) + s0 * np.exp(-m * (Nt - t0)))
    block1 = A1 * (np.exp(-m * t1) + s1 * np.exp(-m * (Nt - t1)))
    model = np.concatenate((block0, block1))
    return (model - y) @ W @ (model - y)
