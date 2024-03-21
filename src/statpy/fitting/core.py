import numpy as np
import scipy.optimize as opt
from scipy.integrate import quad
from scipy.special import gamma
from iminuit import Minuit
from statpy.log import message

# default Nelder-Mead parameter
nm_parameter = {
    "tol": None,
    "maxiter": None
}

class ConvergenceError(Exception):
    pass

class Fitter:
    """
    fit class using Nelder-Mead provided by scipy or Migrad algorithm provided by iminuit package

        Parameters:
        -----------
                chi_squared (function): chi2 squared function of the fit which takes independent variable t, model parameter array p and sample y as input. Returns a number.
                method (string): minimization method. Can be "Nelder-Mead", "Migrad". Default is "Nelder-Mead".
    """
    def __init__(self, method="Nelder-Mead", minimizer_params=None):
        assert method in ["Nelder-Mead", "Migrad"]
        self.method = method
        self.min_params = {} if minimizer_params is None else minimizer_params

    def estimate_parameters(self, t, f, y, p0):
        f2 = lambda first,second: f(t, first, second)
        if self.method == "Nelder-Mead":
            return self._opt_NelderMead(f2, y, p0)
        if self.method == "Migrad":
            return self._opt_Migrad(f2, y, p0)
         
    def _opt_NelderMead(self, f, y, p0):
        for param, value in self.min_params.items():
            if param in nm_parameter:
                nm_parameter[param] = value
        opt_res = opt.minimize(lambda p: f(p, y), p0, method="Nelder-Mead", tol=nm_parameter["tol" ], options={"maxiter": nm_parameter["maxiter"]})
        if opt_res.success is not True:
            raise ConvergenceError("Nelder-Mead did not converge")
        assert opt_res.success == True
        return opt_res.x, opt_res.fun, None

    def _opt_Migrad(self, f, y, p0):
        m = Minuit(lambda p: f(p, y), p0)
        m.migrad()
        if m.valid is not True:
            raise ConvergenceError("Migrad did not converge")
        return np.array(m.values), m.fval, None
    
def get_pvalue(chi2, dof):
    return quad(lambda x: 2**(-dof/2)/gamma(dof/2)*x**(dof/2-1)*np.exp(-x/2), chi2, np.inf)[0]
    
def model_prediction_var(t, best_parameter, best_parameter_cov, model_parameter_gradient):
    return model_parameter_gradient(t, best_parameter) @ best_parameter_cov @ model_parameter_gradient(t, best_parameter)

##################################################################################################################################################################
########################################################################## STATPY DB #############################################################################
##################################################################################################################################################################

def fit(db, t, tag, p0, chi2_func, fit_method, fit_params, jks_fit_method, jks_fit_params, perform_jks_fit=True, dst_tag=None, verbosity=0):
    if isinstance(p0, list): p0 = np.array(p0); assert isinstance(p0, np.ndarray)
    fitter = Fitter(fit_method, fit_params); jks_fitter = Fitter(jks_fit_method, jks_fit_params)
    best_parameter = db.combine_mean(tag, f=lambda y: fitter.estimate_parameters(t, chi2_func, y[t], p0)[0]) 
    best_parameter_jks = db.combine_jks(tag, f=lambda y: jks_fitter.estimate_parameters(t, chi2_func, y[t], best_parameter)[0]) if perform_jks_fit else None
    chi2 = chi2_func(t, best_parameter, db.database[tag].mean[t])
    dof = len(t) - len(best_parameter)
    pval = get_pvalue(chi2, dof)
    misc = {"t": t, "chi2": chi2, "dof": dof, "pval": pval}
    if dst_tag is None:
        return best_parameter, best_parameter_jks, misc
    db.add_leaf(dst_tag, best_parameter, best_parameter_jks, None, misc)
    best_parameter_cov = db.jackknife_covariance(dst_tag)
    if verbosity >= 0:
        for i in range(len(best_parameter)):
            message(f"parameter[{i}] = {best_parameter[i]} +- {best_parameter_cov[i][i]**0.5} (STAT)")
        message(f"chi2 / dof = {chi2} / {dof} = {chi2/dof}, i.e., p = {pval}")  

def fitMultiple(db, t_tags, y_tags, p0, chi2_func, fit_method, fit_params, jks_fit_method, jks_fit_params, perform_jks_fit=True, dst_tag=None, verbosity=0):
    if isinstance(p0, list): p0 = np.array(p0); assert isinstance(p0, np.ndarray)
    tags = np.concatenate((t_tags, y_tags))
    fitter = Fitter(fit_method, fit_params); jks_fitter = Fitter(jks_fit_method, jks_fit_params)
    def estimate_parameters(f, t, y, p):
        t = np.array(t); y = np.array(y)
        return f.estimate_parameters(t, chi2_func, y, p)[0]
    best_parameter = db.combine_mean(*tags, f=lambda *tags: estimate_parameters(fitter, tags[:len(t_tags)], tags[len(t_tags):], p0)) 
    best_parameter_jks = db.combine_jks(*tags, f=lambda *tags: estimate_parameters(jks_fitter, tags[:len(t_tags)], tags[len(t_tags):], best_parameter)) if perform_jks_fit else None
    chi2 = chi2_func(np.array([db.database[tag].mean for tag in t_tags]), best_parameter, np.array([db.database[tag].mean for tag in y_tags]))
    dof = len(t_tags) - len(best_parameter)
    pval = get_pvalue(chi2, dof)
    misc = {"t_tags": t_tags, "y_tags": y_tags, "chi2": chi2, "dof": dof, "pval": pval}
    if dst_tag is None:
        return best_parameter, best_parameter_jks, misc
    db.add_leaf(dst_tag, best_parameter, best_parameter_jks, None, misc)
    best_parameter_cov = db.jackknife_covariance(dst_tag)
    if verbosity >= 0:
        for i in range(len(best_parameter)):
            message(f"parameter[{i}] = {best_parameter[i]} +- {best_parameter_cov[i][i]**0.5} (STAT)")
        message(f"chi2 / dof = {chi2} / {dof} = {chi2/dof}, i.e., p = {pval}")  