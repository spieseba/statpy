import numpy as np
from iminuit import Minuit
from .levenberg_marquardt import LevenbergMarquardt, jacobian, param_cov_lm, fit_std_err_lm
from ..statistics.jackknife import samples, covariance_samples
import scipy.optimize as opt
from scipy.integrate import quad
from scipy.special import gamma

def get_p_value(chi2, dof):
    return quad(lambda x: 2**(-dof/2)/gamma(dof/2)*x**(dof/2-1)*np.exp(-x/2), chi2, np.inf)[0]

def fit_std_err(t, p, dmodel_dp, cov_p):
    return (dmodel_dp(t,p) @ cov_p @ dmodel_dp(t,p))**0.5

# default Nelder-Mead parameter
nm_parameter = {
    "tol": None,
    "maxiter": None
}

class fit:
    """
    fit class using Nelder-Mead provided by scipy or Migrad algorithm provided by iminuit package

        Parameters:
        -----------
                t (numpy array): independent variable t
                y (numpy array): Markov chains at each t or numpy array of means. Note: pass array of means encapsulated in another array.
                C (numpy 2D array): Covariance matrix of the data.
                model (function): fit function which takes independent variable t, model parameter array p as an input and returns real number.
                p0 (numpy array): array containing the start values of the parameter vector.
                estimator (function): function which takes sample (y or y[i]) as an input and returns desired quantity.
                method (string): minimization method. Can be "Nelder-Mead" or "Migrad". Default is "Nelder-Mead".
    """
    def __init__(self, t, y, C, model, p0, estimator, method="Nelder-Mead", minimizer_params={}):
        self.t = t; self.y = y; self.C = C
        self.W = np.linalg.inv(C)
        self.model = model
        self.p0 = p0
        self.estimator = estimator
        assert method in ["Nelder-Mead", "Migrad"]
        self.method = method
        self.min_params = minimizer_params
        self.boolean = True

    def chi_squared(self, p, y):
        return (self.model(self.t, p) - y) @ self.W @ (self.model(self.t, p) - y)

    def _opt_NelderMead(self, f, y, p0):
        for param, value in self.min_params.items():
            if param in nm_parameter:
                nm_parameter[param] = value
        opt_res = opt.minimize(lambda p: f(p, y), p0, method="Nelder-Mead", tol=nm_parameter["tol" ], options={"maxiter": nm_parameter["maxiter"]})
        assert opt_res.success == True
        return opt_res.x, opt_res.fun

    def _opt_Migrad(self, f, y, p0):
        m = Minuit(lambda p: f(p, y), p0)
        m.migrad()
        assert m.valid == True
        self.boolean = False
        
        return np.array(m.values), m.fval
        
    def estimate_parameters(self, f, y, p0):
        if self.method == "Nelder-Mead":
            return self._opt_NelderMead(f, y, p0)
        if self.method == "Migrad":
            return self._opt_Migrad(f, y, p0)
         
###################################################################################################################################

    def jackknife(self, parameter_estimator, verbose=False):
        self.jk_parameter = samples(lambda y: parameter_estimator(self.estimator(y))[0], self.y)
        self.covariance = covariance_samples(lambda y: parameter_estimator(self.estimator(y))[0], self.y, self.jk_parameter)
        if verbose:
                print(f"jackknife parameter covariance is ", self.covariance)
        return self.covariance 

    def fit(self, verbose=True, error=True):
        self.y_est = self.estimator(np.mean(self.y, axis=0))
        self.best_parameter, self.chi2 = self.estimate_parameters(self.chi_squared, self.y_est, self.p0)
        if error:
            self.best_parameter_cov = self.jackknife(lambda y: self.estimate_parameters(self.chi_squared, y, self.best_parameter), verbose)
        else:
            self.best_parameter_cov = np.zeros((len(self.best_parameter), len(self.best_parameter)))
        self.dof = len(self.t) - len(self.best_parameter)
        self.p = get_p_value(self.chi2, self.dof)
        self.fit_err =  lambda trange: fit_std_err(trange, self.best_parameter, self.model.parameter_gradient, self.best_parameter_cov)
        if verbose:    
            for i in range(len(self.best_parameter)):
                print(f"parameter[{i}] = {self.best_parameter[i]} +- {self.best_parameter_cov[i][i]**0.5}")
            print(f"chi2 / dof = {self.chi2} / {self.dof} = {self.chi2/self.dof}, i.e., p = {self.p}")

###################################################################################################################################

    def multi_mc_jackknife(self, parameter_estimator, verbose=False):
        self.jk_parameter = np.array([samples(lambda yi: parameter_estimator(np.array(list(self.y_est[:i]) + [self.estimator(yi)] + list(self.y_est[i+1:])))[0], self.y[i]) for i in range(len(self.t))]) 
        self.covariances = np.array([covariance_samples(lambda yi: parameter_estimator(np.array(list(self.y_est[:i]) + [self.estimator(yi)] + list(self.y_est[i+1:])))[0], self.y[i], self.jk_parameter[i]) for i in range(len(self.t))]) 
        if verbose:
            for i in range(len(self.t)):
                print(f"jackknife parameter covariance from t[{i}] is ", self.covariances[i])
        return sum(self.covariances)    

    def multi_mc_fit(self, verbose=True):
        self.y_est = np.array([self.estimator(np.mean(self.y[i], axis=0)) for i in range(len(self.t))])
        self.best_parameter, self.chi2 = self.estimate_parameters(self.chi_squared, self.y_est, self.p0)
        self.best_parameter_cov = self.multi_mc_jackknife(lambda y: self.estimate_parameters(self.chi_squared, y, self.best_parameter), verbose)
        self.dof = len(self.t) - len(self.best_parameter)
        self.p = get_p_value(self.chi2, self.dof)
        self.fit_err =  lambda trange: fit_std_err(trange, self.best_parameter, self.model.parameter_gradient, self.best_parameter_cov)
        if verbose:
            for i in range(len(self.best_parameter)):
                print(f"parameter[{i}] = {self.best_parameter[i]} +- {self.best_parameter_cov[i][i]**0.5} (jackknife)")
            print(f"chi2 / dof = {self.chi2} / {self.dof} = {self.chi2/self.dof}, i.e., p = {self.p}")


class LM_fit:
    """
    fit class using self-written implementation of the Levenberg-Marquardt minimization algorithm described in https://people.duke.edu/~hpgavin/ce281/lm.pdf. 

        Parameters:
        -----------
                t (numpy array): independent variable t
                y (numpy array): Markov chains at each t or numpy array of means. Note: pass array of means encapsulated in another array.
                C (numpy 2D array): Covariance matrix of the data.
                model (function: p -> scalar): fit function which takes independent variable t, model parameter array p as an input and returns real number.
                p0 (numpy array): array containing the start values of the parameter vector.
                estimator (function): function which takes sample (y or y[i]) as an input and returns desired quantity.
                minimizer_params (dict): user specified parameters to overwrite default parameters of the minimization algorithm. Default is {}.
                    C (2D numpy array): 2D array containing the covariance matrix of the data if known. Default is None.
                    delta (float): fractional increment of p for numerical derivatives. Default is 1e-3.
                    lmbd0 (float): damping parameter which determines whether Levenberg-Marquardt update resembles gradient descent or Gauss-Newton update. Default is 1e-2.
                    maxiter (int): maximum number of iterations. Default is 10 * N_parameter
                    eps1 (float): convergence tolerance for gradient: max|J^T W (y - y_hat)| < eps1. Default is 1e-3.
                    eps2 (float): convergence tolerance for parameters: max|h_i / p_i| < eps2. Default is 1e-3.
                    eps3 (float): convergence tolerance for reduced chi2: chi^2/(N_data - N_parameter + 1) < eps3. Default is 1e-1.
                    eps4 (float): acceptance of a L-M step. Default us 1e-1.
                    Lup (float): factor for increasing lambda. Default is 11.
                    Ldown (float): factor for decreasing lambda. Default is 9.
                    update_type (int): Determines update method of L-M-algorithm. Default is 3.
                                        1: Levenberg-Marquardt lambda update
                                        2: Quadratic update
                                        3: Nielsen's lambda update equations
    """
    def __init__(self, t, y, C, model, p0, estimator, minimizer_params={}):
        self.t = t; self.y = y; self.C = C
        self.W = np.linalg.inv(C)
        self.model = model
        self.p0 = p0
        self.estimator = estimator
        self.min = LevenbergMarquardt
        self.min_params = minimizer_params

    def chi_squared(self, p, y):
        return (self.model(self.t, p) - y) @ self.W @ (self.model(self.t, p) - y)

    def estimate_parameters(self, t, y_est, W, model, p0, verbose=False):
        p, chi2, iterations, success, J = self.min(t, y_est, W, model, p0, self.min_params)()
        assert success == True, f"Fitter did not converge after {iterations+1} iterations"
        if verbose:
            print(f"converged after {iterations+1} iterations")
        return p, chi2, J

####################################################################################################################################################################

    def jackknife(self, p0, verbose=False):
        self.jk_parameter = samples(lambda y: self.estimate_parameters(self.t, self.estimator(y), self.W, self.model, p0)[0], self.y)
        self.covariance = covariance_samples(lambda y: self.estimate_parameters(self.t, self.estimator(y), self.W, self.model, p0)[0], self.y, self.jk_parameter)
        if verbose:
                print(f"jackknife parameter covariance is ", self.covariance)
        return self.covariance 

    def fit(self, verbose=True, error=True):
        self.y_est = self.estimator(np.mean(self.y, axis=0))
        self.best_parameter, self.chi2, self.J = self.estimate_parameters(self.t, self.y_est, self.W, self.model, self.p0, verbose)
        if error:
            self.best_parameter_cov = self.jackknife(self.best_parameter, verbose)
            
        else:
            self.best_parameter_cov = np.zeros((len(self.best_parameter), len(self.best_parameter)))
        self.best_parameter_cov_lm = param_cov_lm(self.J, self.W)
        self.dof = len(self.t) - len(self.best_parameter)
        self.p = get_p_value(self.chi2, self.dof)
        self.fit_err_lm = lambda trange: fit_std_err_lm(jacobian(self.model, trange, self.best_parameter, delta=1e-5)(), 
                            self.best_parameter_cov_lm)
        self.fit_err =  lambda trange: fit_std_err(trange, self.best_parameter, self.model.parameter_gradient, self.best_parameter_cov)
        if verbose:
            for i in range(len(self.best_parameter)):
                print(f"parameter[{i}] = {self.best_parameter[i]} +- {self.best_parameter_cov[i][i]**0.5} (jackknife)") #, {self.best_parameter_cov_lm[i][i]**0.5} (error propagation)")
            print(f"chi2 / dof = {self.chi2} / {self.dof} = {self.chi2/self.dof}, i.e., p = {self.p}")

####################################################################################################################################################################

    def multi_mc_jackknife(self, verbose=False):
        self.jk_parameter = np.array([samples(lambda yi: self.estimate_parameters(self.t, np.array(list(self.y_est[:i]) + [self.estimator(yi)] + list(self.y_est[i+1:])), self.W, self.model, self.best_parameter)[0], self.y[i]) for i in range(len(self.t))]) 
        self.covariances = np.array([covariance_samples(lambda yi: self.estimate_parameters(self.t, np.array(list(self.y_est[:i]) + [self.estimator(yi)] + list(self.y_est[i+1:])), self.W, self.model, self.best_parameter)[0], self.y[i], self.jk_parameter[i]) for i in range(len(self.t))]) 
        if verbose:
            for i in range(len(self.t)):
                print(f"jackknife parameter covariance from t[{i}] is ", self.covariances[i])
        return sum(self.covariances)    


    def multi_mc_fit(self, verbose=True):
        self.y_est = np.array([self.estimator(np.mean(self.y[i], axis=0)) for i in range(len(self.t))])
        self.best_parameter, self.chi2, self.J = self.estimate_parameters(self.t, self.y_est, self.W, self.model, self.p0, verbose)
        self.best_parameter_cov = self.multi_mc_jackknife(verbose)
        self.best_parameter_cov_lm = param_cov_lm(self.J, self.W)
        self.dof = len(self.t) - len(self.best_parameter)
        self.p = get_p_value(self.chi2, self.dof)
        self.fit_err =  lambda trange: fit_std_err(trange, self.best_parameter, self.model.parameter_gradient, self.best_parameter_cov)
        self.fit_err_lm = lambda trange: fit_std_err_lm(jacobian(self.model, trange, self.best_parameter, delta=1e-5)(), 
                            self.best_parameter_cov_lm)
        if verbose:
            for i in range(len(self.best_parameter)):
                print(f"parameter[{i}] = {self.best_parameter[i]} +- {self.best_parameter_cov[i][i]**0.5} (jackknife)")
            print(f"chi2 / dof = {self.chi2} / {self.dof} = {self.chi2/self.dof}, i.e., p = {self.p}")

#####################################################################################
