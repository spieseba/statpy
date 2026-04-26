import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
from iminuit import Minuit
from statpy.log import message

class ConvergenceError(Exception):
    pass

class Fitter:
    """
    fit class using Nelder-Mead provided by scipy or Migrad algorithm provided by iminuit package

        Parameters:
        -----------
                chi_squared (function): chi2 squared function of the fit which takes independent variable t, model parameter array p and sample y as input. Returns a number.
                method (string): minimization method. Can be "Migrad", "Nelder-Mead", or "Simplex". Default is "Migrad".
    """
    def __init__(self, method="Migrad", minimizer_params=None):
        assert method in ["Migrad", "Nelder-Mead", "Simplex"]
        self.method = method
        self.min_params = {"tol": None, "maxiter": None} if minimizer_params is None else minimizer_params

    def estimate_parameters(self, t, f, y, p0):
        f2 = f if t is None else lambda first,second: f(t, first, second)
        if self.method == "Migrad":
            return self._opt_Migrad(f2, y, p0)
        elif self.method == "Nelder-Mead":
            return self._opt_NelderMead(f2, y, p0)
        elif self.method == "Simplex":
            return self._opt_simplex(f2, y, p0)
        else:
            raise AssertionError("Unknown minimization method")
         
    def _opt_NelderMead(self, f, y, p0):
        opt_res = opt.minimize(lambda p: f(p, y), p0, method="Nelder-Mead", tol=self.min_params["tol"], options={"maxiter": self.min_params["maxiter"]})
        if opt_res.success is not True:
            raise ConvergenceError("Nelder-Mead did not converge")
        return opt_res.x, opt_res.fun, None

    def _opt_Migrad(self, f, y, p0):
        m = Minuit(lambda p: f(p, y), p0)
        m.tol = self.min_params["tol"]
        m.migrad(ncall=self.min_params["maxiter"])
        if m.valid is not True:
            raise ConvergenceError("Migrad did not converge")
        return np.array(m.values), m.fval, None
    
    def _opt_simplex(self, f, y, p0):
        m = Minuit(lambda p: f(p, y), p0)
        m.tol = self.min_params["tol"]
        m.simplex(ncall=self.min_params["maxiter"])
        if m.valid is not True:
            raise ConvergenceError("Simplex did not converge")
        return np.array(m.values), m.fval, None
    
def get_pvalue(chi2_value, dof):
    return stats.chi2.sf(chi2_value, dof)

def model_prediction_var(t, best_parameter, best_parameter_cov, model_parameter_gradient):
    return model_parameter_gradient(t, best_parameter) @ best_parameter_cov @ model_parameter_gradient(t, best_parameter)

def print_fit_results(best_parameter, best_parameter_cov, misc, silent=False):
    if not silent:
        if best_parameter_cov is not None:
            for i in range(len(best_parameter)):
                message(f"parameter[{i}] = {best_parameter[i]} +- {best_parameter_cov[i][i]**0.5}")
        else:
            message(f"parameter = {best_parameter}")
        if misc is not None: 
            message(f"chi2 / dof = {misc['chi2']} / {misc['dof']} = {misc['chi2']/misc['dof']}, i.e., p = {misc['pval']:.2f}")  


# use same def as in arxiv:2211.03744
def compute_AIC(chi2, Ndof, k):
    # chi2: chi2 value of the fit
    # Ndof: number of degrees of freedom
    # k: number of parameters in the model
    return chi2 - Ndof + k
        