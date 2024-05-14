import numpy as np
from numba import njit

#########################################################################################################################
################################################# EFFECTIVE MASS CURVES #################################################
#########################################################################################################################

# open boundary conditions
def effective_mass_log(Ct, tmin, tmax):
    return np.array([np.log(Ct[t] / Ct[t+1]) for t in range(tmin,tmax)]) 

# periodic boundary conditions
def effective_mass_acosh(Ct, tmin, tmax):
    return np.array([np.arccosh(0.5 * (Ct[t+1] + Ct[t-1]) / Ct[t]) for t in range(tmin,tmax)])





#########################################################################################################################
##################################################### FITTING ###########################################################
#########################################################################################################################


#################################################### FIT MODELS #########################################################

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
    tmp = p[0] * ( np.exp(-p[1]*t) + np.exp(-p[1]*(Nt-t)) )
    return (tmp - y) @ W @ (tmp - y)


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
    return ( (p[0] * np.exp(-p[1]*t)) - y ) @ W @ ( (p[0] * np.exp(-p[1]*t)) - y )


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


######################################## const + exp model to fit effective mass ########################################

class const_plus_exp_model:
        def __init__(self):
            pass  
        def __call__(self, t, p):
            return p[0] + p[1] * np.exp(-p[2]*t)
        def parameter_gradient(self, t, p):
            return np.array([np.ones_like(t), np.exp(-p[2]*t), p[1] * np.exp(-p[2]*t) * (-t)])

@njit
def const_plus_exp_chi2(t, p, y, W):
    return ( (p[0] + p[1] * np.exp(-p[2]*t)) - y ) @ W @ ( (p[0] + p[1] * np.exp(-p[2]*t)) - y )


######################################## const + cosh model to fit effective mass ########################################

class const_plus_cosh_model:
        def __init__(self, Nt):
            self.Nt = Nt
            pass  
        def __call__(self, t, p):
            return p[0] + p[1] * ( np.exp(-p[2]*t) + np.exp(-p[2]*(self.Nt-t)) )
        def parameter_gradient(self, t, p):
            return np.array([np.ones_like(t), np.exp(-p[2]*t) + np.exp(-p[2]*(self.Nt-t)), p[1] * (np.exp(-p[2]*t) * (-t) + np.exp(-p[2]*t) * (self.Nt-t))])

@njit
def const_plus_cosh_chi2(t, p, y, W, Nt):
    model = p[0] + p[1] * ( np.exp(-p[2]*t) + np.exp(-p[2]*(Nt-t)) )
    return (model - y) @ W @ (model - y)
