import numpy as np
import statpy as sp


########################################### MODELS ###########################################

# C(t) = A * exp(-mt); A = p[0]; m = p[1] 
class exp_model:
    def __init__(self):
        pass   
    def __call__(self, t, p):
        return p[0] * np.exp(-p[1]*t)
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t), p[0] * np.exp(-p[1]*t) * (-t)], dtype=object)

# C(t) = A * [exp(-mt) + exp(-m(T-t))]; A = p[0]; m = p[1] 
class symmetric_exp_model:
    def __init__(self, T):
        self.T = T 
    def __call__(self, t, p):
        return p[0] * ( np.exp(-p[1]*t) + np.exp(-p[1]*(self.T-t)) )
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t) + np.exp(-p[1]*(self.T-t)), p[0] * (np.exp(-p[1]*t) * (-t) + np.exp(-p[1]*(self.T-t)) * (t-self.T))], dtype=object)    

class const_model:
        def __init__(self):
            pass  
        def __call__(self, t, p):
            return p[0]
        def parameter_gradient(self, t, p):
            return np.array([1.0], dtype=object)

##############################################################################################

def effective_mass_log(Ct, tmax, shift=0):
    Ct = np.roll(Ct, shift).real
    return np.array([np.log(Ct[t] / Ct[t+1]) for t in range(tmax)])

def effective_mass_acosh(Ct, tmax, shift=0):
    Ct = np.roll(Ct, shift).real
    return np.array([np.arccosh(0.5 * (Ct[t+1] + Ct[t-1]) / Ct[t]) for t in range(1,tmax)])

def correlator_exp_fit(t, Ct, cov, p0, weights=None, symmetric=False, Nt=0, method="Nelder-Mead", minimizer_params={}, shift=0, verbose=True):
    assert method in ["Nelder-Mead", "Migrad", "Levenberg-Marquardt"]
    Ct = np.roll(Ct, shift).real
    cov = np.roll(np.roll(cov, shift, axis=0), shift, axis=1).real
    if symmetric:
        assert Nt != 0
        model = symmetric_exp_model(Nt)
    else:
        model = exp_model()

    if verbose:
        print("*** correlator fit ***")
        print("fit window:", t)

    if method in ["Nelder-Mead", "Migrad"]:
        fitter = sp.fitting.fit(t, Ct, cov, model, p0, lambda x: x, weights=weights, method=method, minimizer_params=minimizer_params)
    else:
        fitter = sp.fitting.LM_fit(t, Ct, cov, model, p0, lambda x: x, weights=weights, minimizer_params=minimizer_params)
    fitter.fit(verbose)

    return fitter.best_parameter, fitter.best_parameter_cov, fitter.jk_parameter ,fitter.fit_err, fitter.p, fitter.chi2, fitter.dof, model

def const_fit(t, y, cov, p0, method="Nelder-Mead", minimizer_params={}, error=True, verbose=True):
    assert method in ["Nelder-Mead", "Migrad", "Levenberg-Marquardt"]

    model = const_model()

    if method in ["Nelder-Mead", "Migrad"]:
        fitter = sp.fitting.fit(t, y, cov, model, p0, estimator=lambda x: x, weights=None, method=method, minimizer_params=minimizer_params)
        fitter.fit(verbose, error)
    else:
        fitter = sp.fitting.LM_fit(t, y, cov, model, p0, estimator=lambda x: x, weights=None, minimizer_params=minimizer_params)
        fitter.fit(verbose, error)

    if error:
        return fitter.best_parameter, fitter.best_parameter_cov, fitter.fit_err, fitter.p, fitter.chi2, fitter.dof, model
    
    return fitter.best_parameter, fitter.p, fitter.chi2, fitter.dof, model