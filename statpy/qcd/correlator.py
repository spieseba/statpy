import numpy as np
import statpy as sp
from scipy.linalg import block_diag

########################################### EFFECTIVE MASS CURVES ###########################################

# open boundary conditions
def effective_mass_log(Ct, tmax, tmin=0):
    Ct = Ct.real
    return np.array([np.log(Ct[t] / Ct[t+1]) for t in range(tmin,tmax)])

# periodic boundary conditions
def effective_mass_acosh(Ct, tmax, tmin=1):
    Ct = Ct.real
    return np.array([np.arccosh(0.5 * (Ct[t+1] + Ct[t-1]) / Ct[t]) for t in range(tmin,tmax)])

################################################## FITTING #################################################

#################################### FIT MODELS ############################################

################ open boundary conditions ###############

# C(t) = A * exp(-mt); A = p[0]; m = p[1] 
class exp_model:
    def __init__(self):
        pass   
    def __call__(self, t, p):
        return p[0] * np.exp(-p[1]*t)
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t), p[0] * np.exp(-p[1]*t) * (-t)], dtype=object)

############## periodic boundary conditions #############

# C(t) = A * [exp(-mt) + exp(-m(T-t))]; A = p[0]; m = p[1] 
class cosh_model:
    def __init__(self, T):
        self.T = T 
    def __call__(self, t, p):
        return p[0] * ( np.exp(-p[1]*t) + np.exp(-p[1]*(self.T-t)) )
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t) + np.exp(-p[1]*(self.T-t)), p[0] * (np.exp(-p[1]*t) * (-t) + np.exp(-p[1]*(self.T-t)) * (t-self.T))], dtype=object)    

# C(t) = A * [exp(-mt) - exp(-m(T-t))]; A = p[0]; m = p[1]  
class sinh_model:
    def __init__(self, T):
        self.T = T 
    def __call__(self, t, p):
        return p[0] * ( np.exp(-p[1]*t) - np.exp(-p[1]*(self.T-t)) )
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t) - np.exp(-p[1]*(self.T-t)), p[0] * (np.exp(-p[1]*t) * (-t) - np.exp(-p[1]*(self.T-t)) * (t-self.T))], dtype=object)  

# C(t) = A0 * [exp(-m0 t) + exp(-m0(T-t))] + A1 * [exp(-m1 t) + exp(-m1(T-t))]; A0 = p[0], m0 = p[1], A1 = p[2]; m1 = p[3] 
class double_cosh_model():
    def __init__(self, T):
        self.T = T
    def __call__(self, t, p):
        return p[0] * ( np.exp(-p[1]*t) + np.exp(-p[1]*(self.T-t)) ) + p[2] * ( np.exp(-p[3]*t) + np.exp(-p[3]*(self.T-t)) )
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t) + np.exp(-p[1]*(self.T-t)), 
                    p[0] * (np.exp(-p[1]*t) * (-t) + np.exp(-p[1]*(self.T-t)) * (t-self.T)),
                    np.exp(-p[3]*t) + np.exp(-p[3]*(self.T-t)), 
                    p[2] * (np.exp(-p[3]*t) * (-t) + np.exp(-p[3]*(self.T-t)) * (t-self.T))], dtype=object) 
    
# C(t) = A0 * [exp(-m0 t) - exp(-m0(T-t))] + A1 * [exp(-m1 t) - exp(-m1(T-t))]; A0 = p[0], m0 = p[1], A1 = p[2]; m1 = p[3] 
class double_sinh_model():
    def __init__(self, T):
        self.T = T
    def __call__(self, t, p):
        return p[0] * ( np.exp(-p[1]*t) - np.exp(-p[1]*(self.T-t)) ) + p[2] * ( np.exp(-p[3]*t) - np.exp(-p[3]*(self.T-t)) )
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t) - np.exp(-p[1]*(self.T-t)), 
                    p[0] * (np.exp(-p[1]*t) * (-t) - np.exp(-p[1]*(self.T-t)) * (t-self.T)),
                    np.exp(-p[3]*t) + np.exp(-p[3]*(self.T-t)), 
                    p[2] * (np.exp(-p[3]*t) * (-t) - np.exp(-p[3]*(self.T-t)) * (t-self.T))], dtype=object) 
    
######### model to fit effective mass plateau #########
class const_model:
        def __init__(self):
            pass  
        def __call__(self, t, p):
            return p[0]
        def parameter_gradient(self, t, p):
            return np.array([1.0], dtype=object)

##############################################################################################

def fit(t, Ct, Ct_jks, Ct_cov, p0, model, fit_method, fit_params, jks_fit_method=None, jks_fit_params=None, verbosity=0):
    # mean fit
    fitter = sp.fitting.Fitter(t, Ct_cov, model, lambda x: x, fit_method, fit_params)
    best_parameter, chi2, _ = fitter.estimate_parameters(fitter.chi_squared, Ct, p0)
    # jks fits
    if jks_fit_method == None: jks_fit_method = fit_method; jks_fit_params = fit_params
    jks_fitter = sp.fitting.Fitter(t, Ct_cov, model, lambda x: x, jks_fit_method, jks_fit_params)
    best_parameter_jks = {}
    for cfg in  Ct_jks:
        best_parameter_jks[cfg], _, _ = jks_fitter.estimate_parameters(fitter.chi_squared, Ct_jks[cfg], best_parameter)
    best_parameter_cov = sp.statistics.jackknife.covariance_jks(best_parameter, best_parameter_jks)
    if verbosity >=1: 
        print(f"jackknife parameter covariance is ", best_parameter_cov) 
    dof = len(t) - len(best_parameter)
    p = fitter.get_pvalue(chi2, dof)
    if verbosity >= 0:
        for i in range(len(best_parameter)):
            print(f"parameter[{i}] = {best_parameter[i]} +- {best_parameter_cov[i][i]**0.5}")
        print(f"chi2 / dof = {chi2} / {dof} = {chi2/dof}, i.e., p = {p}")
    return best_parameter, best_parameter_cov, best_parameter_jks 

############################################## COMBINED FITTING ############################################

#################################### FIT MODELS ############################################

# C0(t) = A0 * [exp(-mt) + exp(-m(T-t))]; A0 = p[0]; m = p[2]
# C1(t) = A1 * [exp(-mt) - exp(-m(T-t))]; A1 = p[1]; m = p[2]  
class combined_cosh_sinh_model:
    def __init__(self, T):
        self.T = T 
    def __call__(self, t, p):
        f = p[0] * ( np.exp(-p[2]*t) + np.exp(-p[2]*(self.T-t)) ) 
        g = p[1] * ( np.exp(-p[2]*t) - np.exp(-p[2]*(self.T-t)) ) 
        return np.hstack((f,g)) 
    def parameter_gradient(self, t, p):
        df = np.array([np.exp(-p[2]*t) + np.exp(-p[2]*(self.T-t)), 0, p[0] * (np.exp(-p[2]*t) * (-t) + np.exp(-p[2]*(self.T-t)) * (t-self.T))], dtype=object)  
        dg = np.array([0, np.exp(-p[2]*t) - np.exp(-p[2]*(self.T-t)), p[1] * (np.exp(-p[2]*t) * (-t) - np.exp(-p[2]*(self.T-t)) * (t-self.T))], dtype=object)  
        return np.array([df, dg]) 