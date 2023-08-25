#!/usr/bin/env python3

import numpy as np
import scipy.optimize as opt
from scipy.integrate import quad
from scipy.special import gamma
from iminuit import Minuit
from .levenberg_marquardt import LevenbergMarquardt 

# default Nelder-Mead parameter
nm_parameter = {
    "tol": None,
    "maxiter": None
}

class Fitter:
    """
    fit class using Nelder-Mead provided by scipy or Migrad algorithm provided by iminuit package

        Parameters:
        -----------
                C (numpy 2D array): Covariance matrix of the data.
                model (function): fit function which takes independent variable t, model parameter array p as an input and returns real number.
                estimator (function): function which takes sample y as an input and returns desired quantity.
                method (string): minimization method. Can be "Nelder-Mead", "Migrad" or "Levenberg-Marquardt". Default is "Nelder-Mead".
    """
    def __init__(self, C, model, method="Nelder-Mead", minimizer_params=None):
        self.C = C
        self.W = np.linalg.inv(C)
        self.model = model
        assert method in ["Nelder-Mead", "Migrad", "Levenberg-Marquardt"]
        self.method = method
        if minimizer_params == None:
            self.min_params = {}
        else:
            self.min_params = minimizer_params
        self.boolean = True
 
    def chi_squared(self, t, p, y):
        return (self.model(t, p) - y) @ self.W @ (self.model(t, p) - y)

    def get_pvalue(self, chi2, dof):
        return quad(lambda x: 2**(-dof/2)/gamma(dof/2)*x**(dof/2-1)*np.exp(-x/2), chi2, np.inf)[0]

    def model_prediction_var(self, t, p, cov_p, dmodel_dp):
        return dmodel_dp(t,p) @ cov_p @ dmodel_dp(t,p)

    def estimate_parameters(self, t, f, y, p0):
        f2 = lambda first,second: f(t, first, second)
        if self.method == "Nelder-Mead":
            return self._opt_NelderMead(f2, y, p0)
        if self.method == "Migrad":
            return self._opt_Migrad(f2, y, p0)
        if self.method == "Levenberg-Marquardt":
            return self._opt_LevenbergMarquardt(t, y, p0) # uses chi squared
         
    def _opt_NelderMead(self, f, y, p0):
        for param, value in self.min_params.items():
            if param in nm_parameter:
                nm_parameter[param] = value
        opt_res = opt.minimize(lambda p: f(p, y), p0, method="Nelder-Mead", tol=nm_parameter["tol" ], options={"maxiter": nm_parameter["maxiter"]})
        assert opt_res.success == True
        return opt_res.x, opt_res.fun, None

    def _opt_Migrad(self, f, y, p0):
        m = Minuit(lambda p: f(p, y), p0)
        m.migrad()
        assert m.valid == True
        return np.array(m.values), m.fval, None
    
    def _opt_LevenbergMarquardt(self, t, y, p0):
        p, chi2, iterations, success, J = LevenbergMarquardt(t, y, self.W, self.model, p0, self.min_params)()
        assert success == True
        return p, chi2, J


def model_prediction_var(t, best_parameter, best_parameter_cov, model_parameter_gradient):
    return model_parameter_gradient(t, best_parameter) @ best_parameter_cov @ model_parameter_gradient(t, best_parameter)

##############################################################################################################################
##############################################################################################################################
####################################################### JKS SYSTEM ###########################################################
##############################################################################################################################
##############################################################################################################################

def fit(db, t, tag, cov, p0, model, fit_method, fit_params, jks_fit_method, jks_fit_params, binsize, dst_tag, sys_tags=None, verbosity=0):
    fitter = Fitter(cov, model, fit_method, fit_params)
    db.combine_mean(tag, f=lambda y: fitter.estimate_parameters(t, fitter.chi_squared, y[t], p0)[0], dst_tag=dst_tag) 
    best_parameter = db.database[dst_tag].mean
    jks_fitter = Fitter(cov, model, jks_fit_method, jks_fit_params)
    db.combine_jks(tag, f=lambda y: jks_fitter.estimate_parameters(t, fitter.chi_squared, y[t], best_parameter)[0], dst_tag=dst_tag) 
    best_parameter_cov = db.jackknife_covariance(dst_tag, binsize, pavg=True)
    if sys_tags is not None:
        for sys_tag in sys_tags:
            mean_shifted = db.get_mean_shifted(tag, f=lambda y: fitter.estimate_parameters(t, fitter.chi_squared, y[t], p0)[0], sys_tag=sys_tag)
            db.propagate_sys_var(mean_shifted, dst_tag=dst_tag, sys_tag=sys_tag)
    if verbosity >=1: 
        print(f"jackknife parameter covariance is ", best_parameter_cov) 
    chi2 = fitter.chi_squared(t, best_parameter, db.database[tag].mean[t])
    dof = len(t) - len(best_parameter)
    pval = fitter.get_pvalue(chi2, dof)
    if db.database[dst_tag].misc == None: db.database[dst_tag].misc = {}
    db.database[dst_tag].misc["t"] = t
    db.database[dst_tag].misc["best_parameter_cov"] = best_parameter_cov 
    db.database[dst_tag].misc["chi2"] = chi2; db.database[dst_tag].misc["dof"] = dof; db.database[dst_tag].misc["pval"] = pval
    if verbosity >= 0:
        for i in range(len(best_parameter)):
            print(f"parameter[{i}] = {best_parameter[i]} +- {best_parameter_cov[i][i]**0.5} (STAT) +- {db.get_sys_var(dst_tag)[i]**.5} (SYS) [{(db.get_tot_var(dst_tag, binsize))[i]**.5} (STAT + SYS)]")
        print(f"chi2 / dof = {chi2} / {dof} = {chi2/dof}, i.e., p = {pval}")  

def fit_multiple(db, t_tags, y_tags, cov, p0, model, fit_method, fit_params, jks_fit_method, jks_fit_params, binsize, dst_tag, sys_tags=None, verbosity=0):
    tags = np.concatenate((t_tags, y_tags))
    fitter = Fitter(cov, model, fit_method, fit_params)
    jks_fitter = Fitter(cov, model, jks_fit_method, jks_fit_params)
    def estimate_parameters(f, t, y, p):
        t = np.array(t); y = np.array(y)
        return f.estimate_parameters(t, f.chi_squared, y, p)[0]
    db.combine_mean(*tags, f=lambda *tags: estimate_parameters(fitter, tags[:len(t_tags)], tags[len(t_tags):], p0), dst_tag=dst_tag) 
    best_parameter = db.database[dst_tag].mean
    db.combine_jks(*tags, f=lambda *tags: estimate_parameters(jks_fitter, tags[:len(t_tags)], tags[len(t_tags):], best_parameter), dst_tag=dst_tag) 
    best_parameter_cov = db.jackknife_covariance(dst_tag, binsize, pavg=True)
    if sys_tags is not None:
        for sys_tag in sys_tags:
            mean_shifted = db.get_mean_shifted(*tags, f=lambda *tags: estimate_parameters(fitter, tags[:len(t_tags)], tags[len(t_tags):], p0) , sys_tag=sys_tag)
            db.propagate_sys_var(mean_shifted, dst_tag=dst_tag, sys_tag=sys_tag)
    if verbosity >=1: 
        print(f"jackknife parameter covariance is ", best_parameter_cov) 
    chi2 = fitter.chi_squared(np.array([db.database[tag].mean for tag in t_tags]), best_parameter, np.array([db.database[tag].mean for tag in y_tags]))
    dof = len(t_tags) - len(best_parameter)
    pval = fitter.get_pvalue(chi2, dof)
    if db.database[dst_tag].misc == None: db.database[dst_tag].misc = {}
    db.database[dst_tag].misc["t_tags"] = t_tags
    db.database[dst_tag].misc["y_tags"] = y_tags
    db.database[dst_tag].misc["best_parameter_cov"] = best_parameter_cov 
    db.database[dst_tag].misc["chi2"] = chi2; db.database[dst_tag].misc["dof"] = dof; db.database[dst_tag].misc["pval"] = pval
    if verbosity >= 0:
        for i in range(len(best_parameter)):
            print(f"parameter[{i}] = {best_parameter[i]} +- {best_parameter_cov[i][i]**0.5} (STAT) +- {db.get_sys_var(dst_tag)[i]**.5} (SYS) [{(db.get_tot_var(dst_tag, binsize))[i]**.5} (STAT + SYS)]")
        print(f"chi2 / dof = {chi2} / {dof} = {chi2/dof}, i.e., p = {pval}")  