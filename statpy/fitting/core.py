#!/usr/bin/env python3

import numpy as np
import scipy.optimize as opt
from scipy.integrate import quad
from scipy.special import gamma
from iminuit import Minuit
from .levenberg_marquardt import LevenbergMarquardt #, Jacobian, param_cov_lm, fit_std_err_lm
from ..statistics.jackknife import samples, covariance

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
                t (numpy array): independent variable t
                y (numpy array): Markov chains at each t or numpy array of means. Note: pass array of means encapsulated in another array.
                C (numpy 2D array): Covariance matrix of the data.
                model (function): fit function which takes independent variable t, model parameter array p as an input and returns real number.
                estimator (function): function which takes sample (y or y[i]) as an input and returns desired quantity.
                method (string): minimization method. Can be "Nelder-Mead", "Migrad" or "Levenberg-Marquardt". Default is "Nelder-Mead".
    """
    def __init__(self, t, C, model, estimator, method="Nelder-Mead", minimizer_params=None):
        self.t = t; self.C = C
        self.W = np.linalg.inv(C)
        self.model = model
        self.estimator = estimator
        assert method in ["Nelder-Mead", "Migrad", "Levenberg-Marquardt"]
        self.method = method
        if minimizer_params == None:
            self.min_params = {}
        else:
            self.min_params = minimizer_params
        self.boolean = True
 
    def chi_squared(self, p, y):
        return (self.model(self.t, p) - y) @ self.W @ (self.model(self.t, p) - y)

    def get_pvalue(self, chi2, dof):
        return quad(lambda x: 2**(-dof/2)/gamma(dof/2)*x**(dof/2-1)*np.exp(-x/2), chi2, np.inf)[0]

    def model_prediction_var(self, t, p, cov_p, dmodel_dp):
        return dmodel_dp(t,p) @ cov_p @ dmodel_dp(t,p)

    def estimate_parameters(self, f, y, p0):
        if self.method == "Nelder-Mead":
            return self._opt_NelderMead(f, y, p0)
        if self.method == "Migrad":
            return self._opt_Migrad(f, y, p0)
        if self.method == "Levenberg-Marquardt":
            return self._opt_LevenbergMarquardt(y, p0) # uses chi squared
         
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
    
    def _opt_LevenbergMarquardt(self, y, p0):
        p, chi2, iterations, success, J = LevenbergMarquardt(self.t, y, self.W, self.model, p0, self.min_params)()
        assert success == True
        return p, chi2, J
        
###################################################################################################################################

    def jackknife(self, parameter_estimator, verbosity=0):
        self.best_parameter_jks = samples(lambda y: parameter_estimator(self.estimator(y))[0], self.y)
        cov = covariance(lambda y: parameter_estimator(self.estimator(y))[0], self.y, self.best_parameter_jks)
        if verbosity >=1:
                print(f"jackknife parameter covariance is ", cov)
        return cov

    def fit(self, y, p0, verbosity=0):
        self.y = y
        self.y_est = self.estimator(self.y)
        self.best_parameter, self.chi2, self.J = self.estimate_parameters(self.chi_squared, self.y_est, p0)
        self.best_parameter_cov = self.jackknife(lambda y: self.estimate_parameters(self.chi_squared, y, self.best_parameter), verbosity)
        #if self.method == "Levenberg-Marquardt":
        #    self.best_parameter_cov_lm = param_cov_lm(self.J, self.W)
        #    self.fit_err_lm = lambda trange: fit_std_err_lm(jacobian(self.model, trange, self.best_parameter, delta=1e-5)(), self.best_parameter_cov_lm)  
        self.dof = len(self.t) - len(self.best_parameter)
        self.pval = self.get_pvalue(self.chi2, self.dof)
        self.fit_var = lambda t: self.model_prediction_var(t, self.best_parameter, self.best_parameter_cov, self.model.parameter_gradient)
        if verbosity >= 0:    
            for i in range(len(self.best_parameter)):
                print(f"parameter[{i}] = {self.best_parameter[i]} +- {self.best_parameter_cov[i][i]**0.5}")
            print(f"chi2 / dof = {self.chi2} / {self.dof} = {self.chi2/self.dof}, i.e., p = {self.pval}")

###################################################################################################################################

    def jackknife_indep_samples(self, parameter_estimator, verbosity=0):
        self.best_parameter_jks = np.array([samples(lambda yi: parameter_estimator(np.array(list(self.y_est[:i]) + [self.estimator(yi)] + list(self.y_est[i+1:])))[0], self.y_samples[i]) for i in range(len(self.t))]) 
        covs = np.array([covariance(lambda yi: parameter_estimator(np.array(list(self.y_est[:i]) + [self.estimator(yi)] + list(self.y_est[i+1:])))[0], self.y_samples[i], self.best_parameter_jks[i]) for i in range(len(self.t))]) 
        if verbosity >= 1:
            for i in range(len(self.t)):
                print(f"jackknife parameter covariance from t[{i}] is ", covs[i])
        return sum(covs)    

    def fit_indep_samples(self, y_samples, p0, verbosity=0):
        self.y_samples = y_samples
        self.y_est = np.array([self.estimator(np.mean(self.y_samples[i], axis=0)) for i in range(len(self.t))])
        self.best_parameter, self.chi2, J = self.estimate_parameters(self.chi_squared, self.y_est, p0)
        self.best_parameter_cov = self.jackknife_indep_samples(lambda y: self.estimate_parameters(self.chi_squared, y, self.best_parameter), verbosity)
        #if self.method == "Levenberg-Marquardt":
        #    self.best_parameter_cov_lm = param_cov_lm(self.J, self.W)
        #    self.fit_err_lm = lambda trange: fit_std_err_lm(jacobian(self.model, trange, self.best_parameter, delta=1e-5)(), self.best_parameter_cov_lm)  
        self.dof = len(self.t) - len(self.best_parameter)
        self.p = self.get_pvalue(self.chi2, self.dof)
        self.fit_var = lambda t: self.model_prediction_var(t, self.best_parameter, self.best_parameter_cov, self.model.parameter_gradient)
        if verbosity >= 0:
            for i in range(len(self.best_parameter)):
                print(f"parameter[{i}] = {self.best_parameter[i]} +- {self.best_parameter_cov[i][i]**0.5}")
            print(f"chi2 / dof = {self.chi2} / {self.dof} = {self.chi2/self.dof}, i.e., p = {self.p}") 