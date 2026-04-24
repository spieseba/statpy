"""Fit model classes and chi^2 functions for correlator and effective-mass fits."""
import numpy as np
from numba import njit


fit_model_dict = {
    "cosh": "A * [exp(-mt) + exp(-m(Nt-t))]; A = p[0]; m = p[1]",
    "sinh": "A * [exp(-mt) - exp(-m(Nt-t))]; A = p[0]; m = p[1]",
    "exp": "A * exp(-mt); A = p[0]; m = p[1]",
    "double-cosh": "A0 * [exp(-m0t) + exp(-m0(Nt-t))] + A1 * [exp(-m1t) + exp(-m1(Nt-t))]; A0 = p[0], m0 = p[1], A1 = p[2]; m1 = p[3]",
    "double-sinh": "A0 * [exp(-m0t) - exp(-m0(Nt-t))] + A1 * [exp(-m1t) - exp(-m1(Nt-t))]; A0 = p[0], m0 = p[1], A1 = p[2]; m1 = p[3]",
    "double-exp": "A0 * exp(-m0t) + A1 * exp(-m1t); A0 = p[0], m0 = p[1], A1 = p[2]; m1 = p[3]",
    "combined-cosh-sinh": "A0 * [exp(-mt) + exp(-m(Nt-t))], A1 * [exp(-mt) - exp(-m(Nt-t))]; A0 = p[0], A1 = p[1], m = p[2]",
    "combined-exp-exp": "A0 * exp(-mt), A1 * exp(-mt); A0 = p[0], A1 = p[1], m = p[2]",
}


############################## cosh model to fit correlator with periodic boundary conditions ###########################

# C(t) = A * [exp(-mt) + exp(-m(Nt-t))]; A = p[0]; m = p[1]
class cosh_model:
    def __init__(self, Nt):
        self.Nt = Nt
    def __call__(self, t, p):
        return p[0] * ( np.exp(-p[1]*t) + np.exp(-p[1]*(self.Nt-t)) )
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t) + np.exp(-p[1]*(self.Nt-t)), p[0] * (np.exp(-p[1]*t) * (-t) + np.exp(-p[1]*(self.Nt-t)) * (t-self.Nt))])

@njit
def cosh_chi2(t, p, y, W, Nt):
    model = p[0] * ( np.exp(-p[1]*t) + np.exp(-p[1]*(Nt-t)) )
    return (model - y) @ W @ (model - y)


########################## double cosh model to fit correlator with periodic boundary conditions ########################

# C(t) = A0 * [exp(-m0 t) + exp(-m0(Nt-t))] + A1 * [exp(-m1 t) + exp(-m1(Nt-t))]; A0 = p[0], m0 = p[1], A1 = p[2]; m1 = p[3]
class double_cosh_model():
    def __init__(self, Nt):
        self.Nt = Nt
    def __call__(self, t, p):
        return p[0] * ( np.exp(-p[1]*t) + np.exp(-p[1]*(self.Nt-t)) ) + p[2] * ( np.exp(-p[3]*t) + np.exp(-p[3]*(self.Nt-t)) )
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t) + np.exp(-p[1]*(self.Nt-t)), p[0] * (np.exp(-p[1]*t) * (-t) + np.exp(-p[1]*(self.Nt-t)) * (t-self.Nt)),
                         np.exp(-p[3]*t) + np.exp(-p[3]*(self.Nt-t)), p[2] * (np.exp(-p[3]*t) * (-t) + np.exp(-p[3]*(self.Nt-t)) * (t-self.Nt))])

@njit
def double_cosh_chi2(t, p, y, W, Nt):
    model = p[0] * ( np.exp(-p[1]*t) + np.exp(-p[1]*(Nt-t)) ) + p[2] * ( np.exp(-p[3]*t) + np.exp(-p[3]*(Nt-t)) )
    return (model - y) @ W @ (model - y)


############################## sinh model to fit correlator with periodic boundary conditions ###########################

# C(t) = A * [exp(-mt) - exp(-m(Nt-t))]; A = p[0]; m = p[1]
class sinh_model:
    def __init__(self, Nt):
        self.Nt = Nt
    def __call__(self, t, p):
        return p[0] * ( np.exp(-p[1]*t) - np.exp(-p[1]*(self.Nt-t)) )
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t) - np.exp(-p[1]*(self.Nt-t)), p[0] * (np.exp(-p[1]*t) * (-t) - np.exp(-p[1]*(self.Nt-t)) * (t-self.Nt))])

@njit
def sinh_chi2(t, p, y, W, Nt):
    model = p[0] * ( np.exp(-p[1]*t) - np.exp(-p[1]*(Nt-t)) )
    return (model - y) @ W @ (model - y)


########################## double sinh model to fit correlator with periodic boundary conditions ########################

# C(t) = A0 * [exp(-m0 t) - exp(-m0(Nt-t))] + A1 * [exp(-m1 t) - exp(-m1(Nt-t))]; A0 = p[0], m0 = p[1], A1 = p[2]; m1 = p[3]
class double_sinh_model():
    def __init__(self, Nt):
        self.Nt = Nt
    def __call__(self, t, p):
        return p[0] * ( np.exp(-p[1]*t) - np.exp(-p[1]*(self.Nt-t)) ) + p[2] * ( np.exp(-p[3]*t) - np.exp(-p[3]*(self.Nt-t)) )
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t) - np.exp(-p[1]*(self.Nt-t)),
                    p[0] * (np.exp(-p[1]*t) * (-t) - np.exp(-p[1]*(self.Nt-t)) * (t-self.Nt)),
                    np.exp(-p[3]*t) + np.exp(-p[3]*(self.Nt-t)),
                    p[2] * (np.exp(-p[3]*t) * (-t) - np.exp(-p[3]*(self.Nt-t)) * (t-self.Nt))])

@njit
def double_sinh_chi2(t, p, y, W, Nt):
    model = p[0] * ( np.exp(-p[1]*t) - np.exp(-p[1]*(Nt-t)) ) + p[2] * ( np.exp(-p[3]*t) - np.exp(-p[3]*(Nt-t)) )
    return (model - y) @ W @ (model - y)


################################ exp model to fit correlator with open boundary conditions ##############################

# f(t) = A * exp(-mt); A = p[0]; m = p[1]
class exp_model:
    def __init__(self):
        pass
    def __call__(self, t, p):
        return p[0] * np.exp(-p[1]*t)
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t), p[0] * np.exp(-p[1]*t) * (-t)])

@njit
def exp_chi2(t, p, y, W):
    model = p[0] * np.exp(-p[1]*t)
    return (model - y) @ W @ (model - y)


############################ double exp model to fit correlator with open boundary conditions ###########################

# C(t) = A0 * exp(-m0t) + A1 * exp(-m1t); A0 = p[0], m0 = p[1], A1 = p[2]; m1 = p[3]
class double_exp_model:
    def __init__(self):
        pass
    def __call__(self, t, p):
        return p[0] * np.exp(-p[1]*t) + p[2] * np.exp(-p[3]*t)
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t), p[0] * np.exp(-p[1]*t) * (-t), np.exp(-p[3]*t), p[2] * np.exp(-p[3]*t) * (-t)], dtype=object)

@njit
def double_exp_chi2(t, p, y, W):
    model = p[0] * np.exp(-p[1]*t) + p[2] * np.exp(-p[3]*t)
    return (model - y) @ W @ (model - y)


####################################### const model to fit effective mass plateau #######################################

class const_model:
        def __init__(self):
            pass
        def __call__(self, t, p):
            return p[0]
        def parameter_gradient(self, t, p):
            return np.array([np.ones_like(t)])

@njit
def const_chi2(t, p, y, W):
    return (p[0] - y) @ W @ (p[0] - y)


####################################### const plus exp model to fit effective mass plateau #######################################

def const_plus_exp(t, p):
    return p[0] * np.exp(-p[1] * t) + p[2]

def const_plus_exp_chi2(t, p, y, W):
    model = p[0] * np.exp(-p[1] * t) + p[2]
    return (model - y) @ W @ (model - y)


#################################################### combined models ####################################################

############## periodic boundary conditions #############

# C0(t) = A0 * [exp(-mt) + exp(-m(Nt-t))]; A0 = p[0]; m = p[2]
# C1(t) = A1 * [exp(-mt) - exp(-m(Nt-t))]; A1 = p[1]; m = p[2]
class combined_cosh_sinh_model:
    def __init__(self, Nt, t0, t1):
        self.Nt = Nt
        self.t0 = t0
        self.t1 = t1
    def __call__(self, t, p):
        f0 = p[0] * ( np.exp(-p[2]*self.t0) + np.exp(-p[2]*(self.Nt-self.t0)) )
        f1 = p[1] * ( np.exp(-p[2]*self.t1) - np.exp(-p[2]*(self.Nt-self.t1)) )
        return np.hstack((f0,f1))

@njit
def combined_cosh_sinh_chi2(t0, t1, p, y, W, Nt):
    f0 = p[0] * ( np.exp(-p[2]*t0) + np.exp(-p[2]*(Nt-t0)) )
    f1 = p[1] * ( np.exp(-p[2]*t1) - np.exp(-p[2]*(Nt-t1)) )
    model = np.hstack((f0,f1))
    return (model - y) @ W @ (model - y)

################ open boundary conditions ###############

# C0(t) = A0 * exp(-mt); A0 = p[0]; m = p[2]
# C1(t) = A1 * exp(-mt); A1 = p[1]; m = p[2]
class combined_exp_exp_model:
    def __init__(self, t0, t1):
        self.t0 = t0
        self.t1 = t1
    def __call__(self, t, p):
        f0 = p[0] * np.exp(-p[2]*self.t0)
        f1 = p[1] * np.exp(-p[2]*self.t1)
        return np.hstack((f0,f1))

@njit
def combined_exp_exp_model_chi2(t0, t1, p, y, W):
    f0 = p[0] * np.exp(-p[2]*t0)
    f1 = p[1] * np.exp(-p[2]*t1)
    model = np.hstack((f0,f1))
    return (model - y) @ W @ (model - y)
