import numpy as np
import scipy.optimize as opt
#from scipy.integrate import quad
#from scipy.special import gamma
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
    #quad(lambda x: 2**(-dof/2)/gamma(dof/2)*x**(dof/2-1)*np.exp(-x/2), chi2, np.inf)[0]
    
def model_prediction_var(t, best_parameter, best_parameter_cov, model_parameter_gradient):
    return model_parameter_gradient(t, best_parameter) @ best_parameter_cov @ model_parameter_gradient(t, best_parameter)

##################################################################################################################################################################
########################################################################## STATPY DB #############################################################################
##################################################################################################################################################################

def fit(db, t, tag, p0, chi2_func, fit_method, fit_params, perform_jks_fit=True, eval_offset=True, dst_tag=None, verbosity=0):
    # --- input validation ---
    if isinstance(p0, (list, tuple)):
        p0 = np.asarray(p0, dtype=float)
    elif not isinstance(p0, np.ndarray):
        raise TypeError(f"'p0' must be list, tuple, or np.ndarray, got {type(p0).__name__}")
    if p0.ndim != 1 or p0.size == 0:
        raise ValueError(f"'p0' must be a non-empty 1-D array, got shape {p0.shape}")
    if tag not in db.database:
        raise KeyError(f"tag {tag!r} not in database")
    n_data = len(db.database[tag].mean)
    if not eval_offset and len(t) != n_data:
        raise ValueError(
            f"with eval_offset=False, len(t)={len(t)} must equal data length {n_data} for tag {tag!r}"
        )
    dof = len(t) - len(p0)
    if dof <= 0:
        raise ValueError(
            f"non-positive degrees of freedom: len(t)={len(t)}, n_params={len(p0)}, dof={dof}"
        )

    # --- run fits ---
    t_eval = t if eval_offset else np.arange(len(t))
    fitter = Fitter(fit_method, fit_params)
    try:
        best_parameter = db.combine_mean(tag, f=lambda y: fitter.estimate_parameters(t, chi2_func, y[t_eval], p0)[0])
    except ConvergenceError as e:
        raise ConvergenceError(f"mean fit for tag {tag!r} did not converge: {e}") from e
    if not np.isfinite(best_parameter).all():
        raise ConvergenceError(f"mean fit for tag {tag!r} produced non-finite parameters: {best_parameter}")
    if perform_jks_fit:
        try:
            best_parameter_jks = db.combine_jks(tag, f=lambda y: fitter.estimate_parameters(t, chi2_func, y[t_eval], best_parameter)[0])
        except ConvergenceError as e:
            raise ConvergenceError(f"jackknife fit for tag {tag!r} did not converge: {e}") from e
    else:
        best_parameter_jks = None

    chi2 = chi2_func(t, best_parameter, db.database[tag].mean[t_eval])
    if not np.isfinite(chi2):
        raise ConvergenceError(f"non-finite chi^2 = {chi2} for tag {tag!r}")
    pval = get_pvalue(chi2, dof)
    misc = {"t": t, "chi2": chi2, "dof": dof, "pval": pval}

    if dst_tag is not None:
        db.add_leaf(dst_tag, best_parameter, best_parameter_jks, None, misc)
        best_parameter_cov = db.jackknife_covariance(dst_tag)
        print_fit_results(best_parameter, best_parameter_cov, misc, verbosity)
    return best_parameter, best_parameter_jks, misc

def print_fit_results(best_parameter, best_parameter_cov, misc, verbosity=0):
    if verbosity >= 0:
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
        