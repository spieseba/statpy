import numpy as np
import matplotlib.pyplot as plt
from .levenberg_marquardt import LevenbergMarquardt, jacobian, param_cov_lm, fit_std_err_lm
from ..statistics.jackknife import jackknife
from scipy.integrate import quad
from scipy.special import gamma


class fit:
    """
    Fit class

        Parameters:
        -----------
                
                t (numpy array): independent variable t
                y (numpy array): Markov chains at each t or numpy array of means. Note: pass array of means encapsulated in another array.
                stds (numpy array): standard deviations measurements
                model (function): fit function which takes independent variable t, model parameter array p as an input and returns real number.
                p0 (numpy array): array containing the start values of the parameter vector.
                minimizer (class): minimization algorithm. Default is Levenberg-Marquardt algorithm.
                minimizer_params (dict): user specified parameters to overwrite default parameters of the minimization algorithm. See docs of minimization algorithm for possible parameters. Default is {}.
    """
    def __init__(self, t, y, stds, model, p0, minimizer=LevenbergMarquardt, minimizer_params={}):
        self.t = t; self.y = y; self.stds = stds
        self.W = np.diag(1.0 / self.stds**2)
        self.model = model
        self.p0 = p0
        self.min = minimizer 
        self.min_params = minimizer_params

    def estimate_parameters(self, data, model, verbose=False):
        p, chi2, iterations, success, J = self.min(data, model, self.p0, self.min_params)()
        assert success == True, f"Fitter did not converge after {iterations+1} iterations"
        if verbose:
            print(f"converged after {iterations+1} iterations")
        return p, chi2, J

    def get_p_value(self, chi2, dof):
        return quad(lambda x: 2**(-dof/2)/gamma(dof/2)*x**(dof/2-1)*np.exp(-x/2), chi2, np.inf)[0]

    def fit_std_err(self, t, p, dmodel_dp, cov_p):
        return (dmodel_dp(t,p) @ cov_p @ dmodel_dp(t,p))**0.5

    def jackknife_samples(self, f, x):
        self.fx = f(x)
        jk = jackknife(x)
        return np.array([f(jk.sample(k)) for k in range(jk.N)])

    def jackknife_covariance(self, params_jk, fx):
        N = len(params_jk)
        def outer_sqr(a):
            return np.outer(a,a)
        return np.array([outer_sqr((params_jk[k] - fx)) for k in range(N)]) * (N - 1)

    def jackknife(self, y_est, estimator, verbose=False):
        self.params_jk = np.array([
                self.jackknife_samples(lambda yi: self.estimate_parameters((self.t, np.array(list(y_est[:i]) + [estimator(yi)] + list(y_est[i+1:])), self.stds), self.model)[0], self.y[:,i]) for i in range(len(self.t)) 
            ]) 
        self.covariance_jk = np.array([self.jackknife_covariance(self.params_jk[i], self.fx) for i in range(len(self.t))
            ])
        self.covariances = np.array([np.mean(self.covariance_jk[i], axis=0) for i in range(len(self.t))])
        if verbose:
            for i in range(len(self.t)):
                print(f"covariance from t[{i}] is ", self.covariances[i])
        return sum(self.covariances)     

    def __call__(self, jackknife_resampling=True, estimator=lambda x: np.mean(x, axis=0)):
        self.y_est = estimator(self.y)
        self.best_parameter, self.chi2, self.J = self.estimate_parameters((self.t, self.y_est, self.stds), self.model, verbose=True)
        self.best_parameter_cov_lm = param_cov_lm(self.J, self.W)
        self.dof = len(self.t) - len(self.best_parameter)
        self.p = self.get_p_value(self.chi2, self.dof)
        self.fit_err_lm = lambda trange: fit_std_err_lm(jacobian(self.model, trange, self.best_parameter, delta=1e-5)(), 
                            self.best_parameter_cov_lm)
        if jackknife_resampling:
            self.best_parameter_cov = self.jackknife(self.y_est, estimator, verbose=True)
            self.fit_err =  lambda trange: self.fit_std_err(trange, 
                                                        self.best_parameter, self.model.parameter_gradient, self.best_parameter_cov)
            for i in range(len(self.best_parameter)):
                print(f"parameter[{i}] = {self.best_parameter[i]} +- {self.best_parameter_cov[i][i]**0.5} (jackknife), {self.best_parameter_cov_lm[i][i]**0.5} (error propagation)")
            print(f"chi2 / dof = {self.chi2} / {self.dof}, i.e., p = {self.p}")
        else:
            for i in range(len(self.best_parameter)):
                print(f"parameter[{i}] = {self.best_parameter[i]} +- {self.best_parameter_cov_lm[i][i]**0.5} (error propagation)")
            print(f"chi2 / dof = {self.chi2} / {self.dof}, i.e., p = {self.p}")