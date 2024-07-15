import numpy as np
import scipy.optimize as opt
#from scipy.integrate import quad
#from scipy.special import gamma
import scipy.stats as stats
from scipy.linalg import block_diag
from iminuit import Minuit

from statpy.fitting.levenberg_marquardt import LevenbergMarquardt 
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
        assert opt_res.success == True
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
    #return quad(lambda x: 2**(-dof/2)/gamma(dof/2)*x**(dof/2-1)*np.exp(-x/2), chi2, np.inf)[0]
    
def model_prediction_var(t, best_parameter, best_parameter_cov, model_parameter_gradient):
    return model_parameter_gradient(t, best_parameter) @ best_parameter_cov @ model_parameter_gradient(t, best_parameter)

# Akaike Information Criterion
def get_AIC(k, chi2):
    # k denotes the number of parameters in the model
    # P(M) = exp(-AIC)
    return (2.0 * k + chi2) 

##################################################################################################################################################################
########################################################################## STATPY DB #############################################################################
##################################################################################################################################################################

# Standard fitting procedure: dependent variables t assumed to be no random variables, observable[t] is fitted
def fit(db, t, tag, p0, chi2_func, fit_method, fit_params, jks_fit_method, jks_fit_params, binsize, dst_tag, verbosity=0):
    if isinstance(p0, list): p0 = np.array(p0); assert isinstance(p0, np.ndarray)
    fitter = Fitter(fit_method, fit_params); jks_fitter = Fitter(jks_fit_method, jks_fit_params)
    best_parameter = db.combine_mean(tag, f=lambda y: fitter.estimate_parameters(t, chi2_func, y[t], p0)[0]) 
    best_parameter_jks = db.combine_jks(tag, f=lambda y: jks_fitter.estimate_parameters(t, chi2_func, y[t], best_parameter)[0]) 
    misc = db.propagate_systematics(tag, f=lambda y: fitter.estimate_parameters(t, chi2_func, y[t], best_parameter)[0])
    chi2 = chi2_func(t, best_parameter, db.database[tag].mean[t])
    dof = len(t) - len(best_parameter)
    pval = get_pvalue(chi2, dof)
    misc["t"] = t
    misc["chi2"] = chi2; misc["dof"] = dof; misc["pval"] = pval
    misc["AIC"] = get_AIC(len(best_parameter), chi2); misc["P(M)"] = np.exp(-misc["AIC"]/2.0) 
    db.add_leaf(dst_tag, best_parameter, best_parameter_jks, None, misc)
    best_parameter_cov = db.jackknife_covariance(dst_tag, binsize)
    if verbosity >= 0:
        for i in range(len(best_parameter)):
            message(f"parameter[{i}] = {best_parameter[i]} +- {best_parameter_cov[i][i]**0.5} (STAT) +- {db.get_sys_var(dst_tag)[i]**.5} (SYS) [{(db.get_tot_var(dst_tag, binsize))[i]**.5} (STAT + SYS)]")
        message(f"chi2 / dof = {chi2} / {dof} = {chi2/dof}, i.e., p = {pval}")  
        message(f"P(M) = exp(-AIC) = exp([2 * k + chi2] / 2) = exp(-{misc['AIC']} / 2) = {misc['P(M)']}")

def fitMultipleEnsembles(db, t_tags, y_tags, p0, chi2_func, fit_method, fit_params, jks_fit_method, jks_fit_params, binsize, dst_tag, verbosity=0):
    if isinstance(p0, list): p0 = np.array(p0); assert isinstance(p0, np.ndarray)
    tags = np.concatenate((t_tags, y_tags))
    fitter = Fitter(fit_method, fit_params); jks_fitter = Fitter(jks_fit_method, jks_fit_params)
    def estimate_parameters(f, t, y, p):
        t = np.array(t); y = np.array(y)
        return f.estimate_parameters(t, chi2_func, y, p)[0]
    best_parameter = db.combine_mean(*tags, f=lambda *tags: estimate_parameters(fitter, tags[:len(t_tags)], tags[len(t_tags):], p0)) 
    best_parameter_jks = db.combine_jks(*tags, f=lambda *tags: estimate_parameters(jks_fitter, tags[:len(t_tags)], tags[len(t_tags):], best_parameter)) 
    misc = db.propagate_systematics(*tags, f=lambda *tags: estimate_parameters(fitter, tags[:len(t_tags)], tags[len(t_tags):], best_parameter)) 
    chi2 = chi2_func(np.array([db.database[tag].mean for tag in t_tags]), best_parameter, np.array([db.database[tag].mean for tag in y_tags]))
    dof = len(t_tags) - len(best_parameter)
    pval = get_pvalue(chi2, dof)
    misc["t_tags"] = t_tags
    misc["y_tags"] = y_tags
    misc["chi2"] = chi2; misc["dof"] = dof; misc["pval"] = pval
    misc["AIC"] = get_AIC(len(best_parameter), chi2); misc["P(M)"] = np.exp(-misc["AIC"]/2.0)
    db.add_leaf(dst_tag, best_parameter, best_parameter_jks, None, misc)
    best_parameter_cov = db.jackknife_covariance(dst_tag, binsize)
    if verbosity >= 0:
        for i in range(len(best_parameter)):
            message(f"parameter[{i}] = {best_parameter[i]} +- {best_parameter_cov[i][i]**0.5} (STAT) +- {db.get_sys_var(dst_tag)[i]**.5} (SYS) [{(db.get_tot_var(dst_tag, binsize))[i]**.5} (STAT + SYS)]")
        message(f"chi2 / dof = {chi2} / {dof} = {chi2/dof}, i.e., p = {pval}")  
        message(f"P(M) = exp(-AIC / 2) = exp([2 * k + chi2] / 2) = exp(-{misc['AIC']} / 2) = {misc['P(M)']}")

# x_tags: 2D array which contains lists of x_tags for each ensemble
# y_tags: 2D array which contains lists of y_tags for each ensemble
# p0: initial guess for the y parameters
# chi2_func: chi2 function which takes x+y parameters and returns chi2
def fitMultipleEnsemblesl2Norm(db, x_tags, y_tags, p0, chi2_func, fit_method, fit_params, jks_fit_method, jks_fit_params, binsizes, dst_tag, verbosity=0):
    assert _is_2D_list(x_tags) and _is_2D_list(y_tags)
    tags_2D = []
    for x_e_tags, y_e_tags in zip(x_tags, y_tags):
        tags_2D.append(y_e_tags + x_e_tags) 
    message(f"l2-norm fit with tags = [")
    for e_tags in tags_2D:
        message(f"\t\t\t{e_tags}")
    message(f"\t\t\t]")
    message(f"Compute inverses of block covariances")
    dst_tags = [f"tmp_concat_{'-'.join(e_tags)}" for e_tags in tags_2D]
    Ws = []
    for e_tags, e_dst_tag in zip(tags_2D,dst_tags):
        db.combine(*e_tags, f=lambda *xs:np.array([x for x in xs]), dst_tag=e_dst_tag)
        cov = db.jackknife_covariance(e_dst_tag, binsize=max([binsizes[e_tag] for e_tag in e_tags]))
        Ws.append(np.linalg.inv(cov))
        db.remove_leaf(e_dst_tag, verbosity=-1)
    W_block_diag = block_diag(*Ws)
    p0 = np.array(p0)
    q0 = np.array([db.database[x_tag].mean for x_tag in np.array(x_tags).flatten()])
    message(f"p0 = {p0}")
    message(f"q0 = {q0}")
    p0_tot = np.concatenate((p0, q0))
    tags_flattened = np.array(tags_2D).flatten()
    # mean, jks, misc fits
    fitter = Fitter(method=fit_method, minimizer_params=fit_params)
    jks_fitter = Fitter(method=jks_fit_method if jks_fit_method is not None else fit_method, minimizer_params=jks_fit_params if jks_fit_params is not None else fit_params)
    def estimate_parameters(fit, x, p):
        return fit.estimate_parameters(t=None, f=lambda p,y: chi2_func(p,y,W_block_diag), y=x, p0=p)[0]
    best_parameter = db.combine_mean(*tags_flattened, f=lambda *xs: estimate_parameters(fitter, np.array([x for x in xs]), p0_tot))
    best_parameter_jks = db.combine_jks(*tags_flattened, f=lambda *xs: estimate_parameters(jks_fitter, np.array([x for x in xs]), best_parameter)) 
    misc = db.propagate_systematics(*tags_flattened, f=lambda *xs: estimate_parameters(fitter, np.array([x for x in xs]), best_parameter)) 
    # chi2, dof, pval
    chi2 = chi2_func(best_parameter, np.array([db.database[tag].mean for tag in tags_flattened]), W_block_diag)
    dof = len(tags_flattened) - len(best_parameter)
    pval = get_pvalue(chi2, dof)
    # add to database
    misc["tags"] = tags_2D; misc["chi2"] = chi2; misc["dof"] = dof; misc["pval"] = pval
    misc["AIC"] = get_AIC(len(best_parameter), chi2); misc["P(M)"] = np.exp(-misc["AIC"]/2.0)
    db.add_leaf(dst_tag, best_parameter, best_parameter_jks, None, misc)
    # display results
    best_parameter_cov = db.jackknife_covariance(dst_tag, binsize=[max([binsizes[e_tag] for e_tag in e_tags]) for e_tags in tags_2D])
    if verbosity >= 0:
        for i in range(len(best_parameter)):
            message(f"parameter[{i}] = {best_parameter[i]} +- {best_parameter_cov[i][i]**0.5} (STAT) +- {db.get_sys_var(dst_tag)[i]**.5} (SYS) [{(db.get_tot_var(dst_tag, [max([binsizes[e_tag] for e_tag in e_tags]) for e_tags in tags_2D]))[i]**.5} (STAT + SYS)]")
        message(f"chi2 / dof = {chi2} / {dof} = {chi2/dof}, i.e., p = {pval}")  
        message(f"P(M) = exp(-AIC / 2) = exp([2 * k + chi2] / 2) = exp(-{misc['AIC']} / 2) = {misc['P(M)']}")

def _is_2D_list(lst):
    if not isinstance(lst, list):
        return False
    return all(isinstance(sublst, list) for sublst in lst)

##################################################################################################################################################################
##################################################################################################################################################################
########################################################################## FITTER V1 #############################################################################
##################################################################################################################################################################
##################################################################################################################################################################
class FitterV1:
    """
    fit class using Nelder-Mead provided by scipy or Migrad algorithm provided by iminuit package

        Parameters:
        -----------
                C (numpy 2D array): Covariance matrix of the data.
                model (function): fit function which takes independent variable t, model parameter array p as an input and returns real number.
                estimator (function): function which takes sample y as an input and returns desired quantity.
                method (string): minimization method. Can be "Migrad", "Nelder-Mead", "Simplex" or "Levenberg-Marquardt". Default is "Migrad".
    """
    def __init__(self, C, model, method="Migrad", minimizer_params=None):
        self.C = C
        self.W = np.linalg.inv(C)
        self.model = model
        assert method in ["Nelder-Mead", "Migrad", "Simplex", "Levenberg-Marquardt"]
        self.method = method
        self.min_params = {"tol": None, "maxiter": None} if minimizer_params is None else minimizer_params
        self.boolean = True
 
    def chi_squared(self, t, p, y):
        return (self.model(t, p) - y) @ self.W @ (self.model(t, p) - y)

    def get_pvalue(self, chi2_value, dof):
        return stats.chi2.sf(chi2_value, dof)
        #return quad(lambda x: 2**(-dof/2)/gamma(dof/2)*x**(dof/2-1)*np.exp(-x/2), chi2, np.inf)[0]

    def model_prediction_var(self, t, p, cov_p, dmodel_dp):
        return dmodel_dp(t,p) @ cov_p @ dmodel_dp(t,p)
    
    def estimate_parameters(self, t, f, y, p0):
        f2 = f if t is None else lambda first,second: f(t, first, second)
        if self.method == "Migrad":
            return self._opt_Migrad(f2, y, p0)
        elif self.method == "Nelder-Mead":
            return self._opt_NelderMead(f2, y, p0)
        elif self.method == "Simplex":
            return self._opt_simplex(f2, y, p0)
        elif self.method == "Levenberg-Marquardt":
            return self._opt_LevenbergMarquardt(t, y, p0) # uses chi squared
        else:
            raise AssertionError("Unknown minimization method")

    def _opt_NelderMead(self, f, y, p0):
        opt_res = opt.minimize(lambda p: f(p, y), p0, method="Nelder-Mead", tol=self.min_params["tol"], options={"maxiter": self.min_params["maxiter"]})
        if opt_res.success is not True:
            raise ConvergenceError("Nelder-Mead did not converge")
        assert opt_res.success == True
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
    
    def _opt_LevenbergMarquardt(self, t, y, p0):
        p, chi2, iterations, success, J = LevenbergMarquardt(t, y, self.W, self.model, p0, self.min_params)()
        if success is not True:
            raise ConvergenceError("Levenberg-Marquardt did not converge")
        return p, chi2, J

##################################################################################################################################################################
########################################################################## STATPY DB #############################################################################
##################################################################################################################################################################

def fitV1(db, t, tag, cov, p0, model, fit_method, fit_params, jks_fit_method, jks_fit_params, binsize, dst_tag, verbosity=0):
    assert len(p0) == len(model.parameter_gradient(t, p0)), f"len(p0) = {len(p0)} != len(best_parameter) = {len(model.parameter_gradient(t, p0))}"
    if isinstance(p0, list): p0 = np.array(p0); assert isinstance(p0, np.ndarray)
    fitter = FitterV1(cov, model, fit_method, fit_params)
    best_parameter = db.combine_mean(tag, f=lambda y: fitter.estimate_parameters(t, fitter.chi_squared, y[t], p0)[0]) 
    jks_fitter = FitterV1(cov, model, jks_fit_method, jks_fit_params)
    best_parameter_jks = db.combine_jks(tag, f=lambda y: jks_fitter.estimate_parameters(t, fitter.chi_squared, y[t], best_parameter)[0]) 
    misc = db.propagate_systematics(tag, f=lambda y: fitter.estimate_parameters(t, fitter.chi_squared, y[t], best_parameter)[0])
    chi2 = fitter.chi_squared(t, best_parameter, db.database[tag].mean[t])
    dof = len(t) - len(best_parameter)
    pval = fitter.get_pvalue(chi2, dof)
    misc["t"] = t
    misc["chi2"] = chi2; misc["dof"] = dof; misc["pval"] = pval
    misc["AIC"] = get_AIC(len(best_parameter), chi2); misc["P(M)"] = np.exp(-misc["AIC"]/2.0)
    db.add_leaf(dst_tag, best_parameter, best_parameter_jks, None, misc)
    best_parameter_cov = db.jackknife_covariance(dst_tag, binsize)
    if verbosity >= 0:
        for i in range(len(best_parameter)):
            message(f"parameter[{i}] = {best_parameter[i]} +- {best_parameter_cov[i][i]**0.5} (STAT) +- {db.get_sys_var(dst_tag)[i]**.5} (SYS) [{(db.get_tot_var(dst_tag, binsize))[i]**.5} (STAT + SYS)]")
        message(f"chi2 / dof = {chi2} / {dof} = {chi2/dof}, i.e., p = {pval}")  
        message(f"P(M) = exp(-AIC / 2) = exp([2 * k + chi2] / 2) = exp(-{misc['AIC']} / 2) = {misc['P(M)']}")
 
def fitMultipleEnsemblesV1(db, t_tags, y_tags, cov, p0, model, fit_method, fit_params, jks_fit_method, jks_fit_params, binsize, dst_tag, verbosity=0):
    assert len(p0) == len(model.parameter_gradient(1, p0)), f"len(p0) = {len(p0)} != len(best_parameter) = {len(model.parameter_gradient(1, p0))}"
    if isinstance(p0, list): p0 = np.array(p0); assert isinstance(p0, np.ndarray)
    tags = np.concatenate((t_tags, y_tags))
    fitter = FitterV1(cov, model, fit_method, fit_params)
    jks_fitter = FitterV1(cov, model, jks_fit_method, jks_fit_params)
    def estimate_parameters(f, t, y, p):
        t = np.array(t); y = np.array(y)
        return f.estimate_parameters(t, f.chi_squared, y, p)[0]
    best_parameter = db.combine_mean(*tags, f=lambda *tags: estimate_parameters(fitter, tags[:len(t_tags)], tags[len(t_tags):], p0)) 
    best_parameter_jks = db.combine_jks(*tags, f=lambda *tags: estimate_parameters(jks_fitter, tags[:len(t_tags)], tags[len(t_tags):], best_parameter)) 
    misc = db.propagate_systematics(*tags, f=lambda *tags: estimate_parameters(fitter, tags[:len(t_tags)], tags[len(t_tags):], best_parameter)) 
    chi2 = fitter.chi_squared(np.array([db.database[tag].mean for tag in t_tags]), best_parameter, np.array([db.database[tag].mean for tag in y_tags]))
    dof = len(t_tags) - len(best_parameter)
    pval = fitter.get_pvalue(chi2, dof)
    misc["t_tags"] = t_tags
    misc["y_tags"] = y_tags
    misc["chi2"] = chi2; misc["dof"] = dof; misc["pval"] = pval
    misc["AIC"] = get_AIC(len(best_parameter), chi2); misc["P(M)"] = np.exp(-misc["AIC"]/2.0)
    db.add_leaf(dst_tag, best_parameter, best_parameter_jks, None, misc)
    best_parameter_cov = db.jackknife_covariance(dst_tag, binsize)
    if verbosity >= 0:
        for i in range(len(best_parameter)):
            message(f"parameter[{i}] = {best_parameter[i]} +- {best_parameter_cov[i][i]**0.5} (STAT) +- {db.get_sys_var(dst_tag)[i]**.5} (SYS) [{(db.get_tot_var(dst_tag, binsize))[i]**.5} (STAT + SYS)]")
        message(f"chi2 / dof = {chi2} / {dof} = {chi2/dof}, i.e., p = {pval}")  
        message(f"P(M) = exp(-AIC / 2) = exp([2 * k + chi2] / 2) = exp(-{misc['AIC']} / 2) = {misc['P(M)']}")
