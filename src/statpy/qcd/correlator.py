import numpy as np
from statpy.log import message
from statpy.fitting.core import Fitter, ConvergenceError
from statpy.statistics import jackknife, bootstrap
from statpy.fitting.core import fit, print_fit_results, get_pvalue
from numba import njit

### periodic boundary conditions ###
def effective_mass_acosh1(Ct):
    return np.arccosh(0.5 * (np.roll(Ct, -1) + np.roll(Ct, 1)) / Ct)

# spectrum paper
def effective_mass_acosh2(Ct, a=1):
    Nt = len(Ct)
    return np.abs((np.arccosh(np.roll(Ct,a)/Ct[Nt//2]) - np.arccosh(np.roll(Ct,-a)/Ct[Nt//2]))) / (2. * a)

def effective_mass_asinh(Ct):
    Nt = len(Ct)
    eff_m = np.arcsinh(Ct/Ct[Nt-1])
    return np.abs(np.roll(eff_m, -1) - eff_m)

### open boundary conditions ###
def effective_mass_log1(Ct):
    return np.log(Ct / np.roll(Ct, -1))

# spectrum paper 
def effective_mass_log2(Ct, a=2):
    return np.log(np.roll(Ct, a//2) / np.roll(Ct, -a//2)) / a


# cosh
def effective_amplitude_cosh(Ct, m):
    Nt = len(Ct)
    return Ct / np.array([(np.exp(-m*t)) + np.exp(-m*(Nt-t)) for t in range(Nt)])

# sinh
def effective_amplitude_sinh(Ct, m):
    Nt = len(Ct)
    return Ct / np.array([(np.exp(-m*t)) - np.exp(-m*(Nt-t)) for t in range(Nt)])

# exp
def effective_amplitude_exp(Ct, m):
    Nt = len(Ct)
    return Ct / np.array([(np.exp(-m*t)) for t in range(Nt)])

#########################################################################################################################
##################################################### FITTING ###########################################################
#########################################################################################################################


#################################################### FIT MODELS #########################################################

############################## cosh model to fit correlator with periodic boundary conditions ###########################

# C(t) = A * [exp(-mt) + exp(-m(Nt-t))]; A = p[0]; m = p[1] 
class cosh_model:
    def __init__(self, Nt):
        self.Nt = Nt 
    def __call__(self, t, p):
        return p[0] * ( np.exp(-p[1]*t) + np.exp(-p[1]*(self.Nt-t)) )
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t) + np.exp(-p[1]*(self.Nt-t)), p[0] * (np.exp(-p[1]*t) * (-t) + np.exp(-p[1]*(self.Nt-t)) * (t-self.Nt))])    

@njit
def cosh_chi2(t, p, y, W, Nt):
    model = p[0] * ( np.exp(-p[1]*t) + np.exp(-p[1]*(Nt-t)) )
    return (model - y) @ W @ (model - y)


########################## double cosh model to fit correlator with periodic boundary conditions ########################

# C(t) = A0 * [exp(-m0 t) + exp(-m0(Nt-t))] + A1 * [exp(-m1 t) + exp(-m1(Nt-t))]; A0 = p[0], m0 = p[1], A1 = p[2]; m1 = p[3] 
class double_cosh_model():
    def __init__(self, Nt):
        self.Nt = Nt
    def __call__(self, t, p):
        return p[0] * ( np.exp(-p[1]*t) + np.exp(-p[1]*(self.Nt-t)) ) + p[2] * ( np.exp(-p[3]*t) + np.exp(-p[3]*(self.Nt-t)) )
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t) + np.exp(-p[1]*(self.Nt-t)), p[0] * (np.exp(-p[1]*t) * (-t) + np.exp(-p[1]*(self.Nt-t)) * (t-self.Nt)),
                         np.exp(-p[3]*t) + np.exp(-p[3]*(self.Nt-t)), p[2] * (np.exp(-p[3]*t) * (-t) + np.exp(-p[3]*(self.Nt-t)) * (t-self.Nt))]) 
    
@njit
def double_cosh_chi2(t, p, y, W, Nt):
    model = p[0] * ( np.exp(-p[1]*t) + np.exp(-p[1]*(Nt-t)) ) + p[2] * ( np.exp(-p[3]*t) + np.exp(-p[3]*(Nt-t)) )
    return (model - y) @ W @ (model - y)

############################## sinh model to fit correlator with periodic boundary conditions ###########################

# C(t) = A * [exp(-mt) - exp(-m(Nt-t))]; A = p[0]; m = p[1]  
class sinh_model:
    def __init__(self, Nt):
        self.Nt = Nt 
    def __call__(self, t, p):
        return p[0] * ( np.exp(-p[1]*t) - np.exp(-p[1]*(self.Nt-t)) )
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t) - np.exp(-p[1]*(self.Nt-t)), p[0] * (np.exp(-p[1]*t) * (-t) - np.exp(-p[1]*(self.Nt-t)) * (t-self.Nt))])  
  
@njit
def sinh_chi2(t, p, y, W, Nt):
    model = p[0] * ( np.exp(-p[1]*t) - np.exp(-p[1]*(Nt-t)) )
    return (model - y) @ W @ (model - y)


########################## double sinh model to fit correlator with periodic boundary conditions ########################

# C(t) = A0 * [exp(-m0 t) - exp(-m0(Nt-t))] + A1 * [exp(-m1 t) - exp(-m1(Nt-t))]; A0 = p[0], m0 = p[1], A1 = p[2]; m1 = p[3] 
class double_sinh_model():
    def __init__(self, Nt):
        self.Nt = Nt
    def __call__(self, t, p):
        return p[0] * ( np.exp(-p[1]*t) - np.exp(-p[1]*(self.Nt-t)) ) + p[2] * ( np.exp(-p[3]*t) - np.exp(-p[3]*(self.Nt-t)) )
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t) - np.exp(-p[1]*(self.Nt-t)), 
                    p[0] * (np.exp(-p[1]*t) * (-t) - np.exp(-p[1]*(self.Nt-t)) * (t-self.Nt)),
                    np.exp(-p[3]*t) + np.exp(-p[3]*(self.Nt-t)), 
                    p[2] * (np.exp(-p[3]*t) * (-t) - np.exp(-p[3]*(self.Nt-t)) * (t-self.Nt))]) 
    
@njit
def double_sinh_chi2(t, p, y, W, Nt):
    model = p[0] * ( np.exp(-p[1]*t) - np.exp(-p[1]*(Nt-t)) ) + p[2] * ( np.exp(-p[3]*t) - np.exp(-p[3]*(Nt-t)) )
    return (model - y) @ W @ (model - y)


################################ exp model to fit correlator with open boundary conditions ##############################

# f(t) = A * exp(-mt); A = p[0]; m = p[1] 
class exp_model:
    def __init__(self):
        pass   
    def __call__(self, t, p):
        return p[0] * np.exp(-p[1]*t)
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t), p[0] * np.exp(-p[1]*t) * (-t)])
    
@njit
def exp_chi2(t, p, y, W):
    model = p[0] * np.exp(-p[1]*t)
    return (model - y) @ W @ (model - y)


############################ double exp model to fit correlator with open boundary conditions ###########################

# C(t) = A0 * exp(-m0t) + A1 * exp(-m1t); A0 = p[0], m0 = p[1], A1 = p[2]; m1 = p[3] 
class double_exp_model:
    def __init__(self):
        pass   
    def __call__(self, t, p):
        return p[0] * np.exp(-p[1]*t) + p[2] * np.exp(-p[3]*t)
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t), p[0] * np.exp(-p[1]*t) * (-t), np.exp(-p[3]*t), p[2] * np.exp(-p[3]*t) * (-t)], dtype=object)  
    
@njit
def double_exp_chi2(t, p, y, W):
    model = p[0] * np.exp(-p[1]*t) + p[2] * np.exp(-p[3]*t)
    return (model - y) @ W @ (model - y)


####################################### const model to fit effective mass plateau #######################################

class const_model:
        def __init__(self):
            pass  
        def __call__(self, t, p):
            return p[0]
        def parameter_gradient(self, t, p):
            return np.array([np.ones_like(t)])
        
@njit
def const_chi2(t, p, y, W):
    return (p[0] - y) @ W @ (p[0] - y)

#################################################### combined models ####################################################

############## periodic boundary conditions #############

# C0(t) = A0 * [exp(-mt) + exp(-m(Nt-t))]; A0 = p[0]; m = p[2]
# C1(t) = A1 * [exp(-mt) - exp(-m(Nt-t))]; A1 = p[1]; m = p[2]  
class combined_cosh_sinh_model:
    def __init__(self, Nt, t0, t1):
        self.Nt = Nt 
        self.t0 = t0
        self.t1 = t1
    def __call__(self, t, p):
        f0 = p[0] * ( np.exp(-p[2]*self.t0) + np.exp(-p[2]*(self.Nt-self.t0)) ) 
        f1 = p[1] * ( np.exp(-p[2]*self.t1) - np.exp(-p[2]*(self.Nt-self.t1)) ) 
        return np.hstack((f0,f1)) 

@njit
def combined_cosh_sinh_chi2(t0, t1, p, y, W, Nt):
    f0 = p[0] * ( np.exp(-p[2]*t0) + np.exp(-p[2]*(Nt-t0)) ) 
    f1 = p[1] * ( np.exp(-p[2]*t1) - np.exp(-p[2]*(Nt-t1)) ) 
    model = np.hstack((f0,f1))
    return (model - y) @ W @ (model - y)

################ open boundary conditions ###############

# C0(t) = A0 * exp(-mt); A0 = p[0]; m = p[2]
# C1(t) = A1 * exp(-mt); A1 = p[1]; m = p[2]
class combined_exp_exp_model:
    def __init__(self, t0, t1):
        self.t0 = t0
        self.t1 = t1
    def __call__(self, t, p):
        f0 = p[0] * np.exp(-p[2]*self.t0) 
        f1 = p[1] * np.exp(-p[2]*self.t1)
        return np.hstack((f0,f1)) 
    
@njit
def combined_exp_exp_model_chi2(t0, t1, p, y, W):
    f0 = p[0] * np.exp(-p[2]*t0) 
    f1 = p[1] * np.exp(-p[2]*t1)
    model = np.hstack((f0,f1)) 
    return (model - y) @ W @ (model - y)


##############################################################################################################################
##############################################################################################################################
##################################################### LATTICE CHARM ##########################################################
##############################################################################################################################
##############################################################################################################################
    
class LatticeCharmToolkit():
    def __init__(self, db, fit_method="Nelder-Mead", fit_params={"maxiter":1000, "tol":1e-07}, res_fit_method=None, res_fit_params=None):
        self.db = db
        self.fit_method = fit_method
        self.fit_params = fit_params
        self.res_fit_method = self.fit_method if res_fit_method is None else res_fit_method
        self.res_fit_params = self.fit_params if res_fit_params is None else res_fit_params

    # Wolfgangs hdf5 geometry 
    def ptsrc_avg(self, Ct_tag, dst_tag):
        self.db.combine_sample(Ct_tag, f=lambda x: np.mean(x, axis=0), dst_tag=dst_tag)

    def tsrc_avg(self, Ctsrc_tags, dst_tag):
        srcs_pos = sorted([int(k.split("_")[4].split("tsrc")[1]) for k in Ctsrc_tags]) 
        tmin = min(srcs_pos); tmax = max(srcs_pos)
        A4_in_tag = "A4" in Ctsrc_tags[0]
        self.db.combine_sample(*Ctsrc_tags, f=lambda *Cts: self._avg_obc_srcs(srcs_pos, tmin, tmax, *Cts, antiperiodic=A4_in_tag), dst_tag=dst_tag)

    def _avg_obc_srcs(self, srcs, tmin, tmax, *Cts, antiperiodic=False):
        Ct_arr = np.ma.empty((2 * len(Cts), max(tmax - srcs[0], srcs[-1] - tmin))); Ct_arr.mask = True    
        # forward average
        tmax_srcs_fw = tmax - np.array(srcs) 
        for idx, Ct, tmax_src in zip(np.arange(len(Cts)), Cts, tmax_srcs_fw):
            Ct_arr[idx, :tmax_src] = Ct[:tmax_src]
        # backward average
        tmax_srcs_bw = np.array(srcs) - tmin
        for idx, Ct, tmax_src in zip(len(Cts) + np.arange(len(Cts)), Cts, tmax_srcs_bw):
            Ct_arr[idx, :tmax_src] = np.roll(np.flip(Ct), 1)[:tmax_src]
            if antiperiodic: Ct_arr[idx, 1:tmax_src] *= -1.
        return Ct_arr.mean(axis=0) 

    
    # determine improved PSA4 according to https://arxiv.org/pdf/1502.04999.pdf
    def determine_PSA4I(self, tag_PSPS_sml, tag_PSA4_sml, beta):
        def compute_cA(beta):
            p0 = 9.2056; p1 = -13.9847
            return - 0.006033 * 6./beta * (1 + np.exp(p0 + p1*beta/6.))
        def derivative(f):
            return 0.5 * (np.roll(f, -1) - np.roll(f, 1)) 
        def compute_PSA4I(PS_A4, PS_PS, beta):
            PS_A4I = PS_A4 - compute_cA(beta) * derivative(PS_PS)
            PS_A4I[0] = 0.; PS_A4I[-1] = 0
            return PS_A4I
        tag_PSA4I = tag_PSA4_sml.replace("PSA4", "PSA4I")
        self.db.combine_sample(tag_PSA4_sml, tag_PSPS_sml, f=lambda x,y: compute_PSA4I(x, y, beta), dst_tag=tag_PSA4I)
 

    def fit_range_fit(self, tag, binsize, initial_fit_ranges, p0, model_type, verbosity):
        def _sort_params(p):
            if p[3] <  p[1]: return [p[2], p[3], p[0], p[1]]
            else: return p
        message(f"CORRELATOR: {tag}")
        message(f"P0 = {p0}")
        message(f"BINSIZE = {binsize}", verbosity)
        message(f"MODEL = {model_type}")
        binned_tag = self.db.add_binned_leaf(tag, binsize)
        cov = self.db.jackknife_covariance(binned_tag); var = np.diag(cov)        
        Nt = len(self.db.database[binned_tag].mean) 
        model_func = {"double-cosh": double_cosh_model(Nt),
                      "double-sinh": double_sinh_model(Nt),
                      "double-exp": double_exp_model()}[model_type]       
        fit_range_dict = None
        fit_range = initial_fit_ranges[0]
        for t in initial_fit_ranges:
            message(f"INITIAL FIT RANGE: [[{t[0]},{t[-1]}]]", verbosity)
            W = np.linalg.inv(np.diag(var[t]))
            chi2_func = {"double-cosh": lambda t,p,y: double_cosh_chi2(t, p, y, W, Nt),
                         "double-sinh": lambda t,p,y: double_sinh_chi2(t, p, y, W, Nt),
                         "double-exp": lambda t,p,y: double_exp_chi2(t, p, y, W)}[model_type]
            try:
                best_parameter, best_parameter_jks, misc = fit(self.db, t, binned_tag, p0, chi2_func, self.fit_method, self.fit_params, self.res_fit_method, self.res_fit_params)
            except ConvergenceError as ce:
                message(f"{ce} -> JUMP TO NEXT FIT RANGE")
                message("---------------------------------------------------------------------------------", verbosity) 
                message("---------------------------------------------------------------------------------", verbosity) 
                continue 
            best_parameter = _sort_params(best_parameter); best_parameter_jks = {cfg: _sort_params(best_parameter_jks[cfg]) for cfg in best_parameter_jks}
            best_parameter_cov = jackknife.covariance(self.db.as_array(best_parameter_jks))
            print_fit_results(best_parameter, best_parameter_cov, misc, verbosity)
            message("------------------------------ CORRELATED MEAN FIT ------------------------------", verbosity)
            try:
                W_correlated = np.linalg.inv(cov[t][:,t])
                chi2_func_correlated = {"double-cosh": lambda t,p,y: double_cosh_chi2(t, p, y, W_correlated, Nt),
                                        "double-sinh": lambda t,p,y: double_sinh_chi2(t, p, y, W_correlated, Nt),
                                        "double-exp": lambda t,p,y: double_exp_chi2(t, p, y, W_correlated)}[model_type]
                best_parameter_correlated, _, misc_correlated = fit(self.db, t, binned_tag, p0, chi2_func_correlated, self.fit_method, self.fit_params, self.res_fit_method, self.res_fit_params, perform_jks_fit=False)
                best_parameter_correlated = _sort_params(best_parameter_correlated)
                print_fit_results(best_parameter_correlated, None, misc_correlated, verbosity)
                correlated_converged = True
            except ConvergenceError as ce:
                correlated_converged = False
                message(f"{ce} for correlated mean fit") 
            message("---------------------------------------------------------------------------------", verbosity) 
            criterion = np.abs([model_func(i, [0, 0, best_parameter[2], best_parameter[3]]) for i in t]) < var[t]**.5/4.
            t_crit = t[criterion]
            if len(t_crit) < 3:
                message(f"DETERMINED FIT RANGE {t_crit} HAS FEWER THAN 3 ELEMENTS", verbosity)
                message(f"---> STORED FIT RANGE IS NOT UPDATED", verbosity)
                message("---------------------------------------------------------------------------------", verbosity) 
                message("---------------------------------------------------------------------------------", verbosity) 
                continue
            else:
                message(f"DETERMINED FIT RANGE [[{t_crit[0]},{t_crit[-1]}]]", verbosity)
            if len(t_crit) <= len(fit_range):
                message(f"---> STORED FIT RANGE IS UPDATED", verbosity)
                misc["fit_range_crit"] = t_crit
                best_parameter = best_parameter[:2]; best_parameter_jks = {cfg:jk[:2] for cfg,jk in best_parameter_jks.items()}
                fit_range_dict = {"tag": f"{binned_tag}/fit_range_fit", "mean":best_parameter, "jks":best_parameter_jks, "sample":None, "misc":misc}
            message("---------------------------------------------------------------------------------", verbosity) 
            message("---------------------------------------------------------------------------------", verbosity) 
        if correlated_converged: self.db.add_leaf(tag=f"{binned_tag}/correlated_fit_range_mean_fit", mean=best_parameter_correlated, jks=None, sample=None, misc=misc_correlated)
        self.db.add_leaf(**fit_range_dict)
        return misc["fit_range_crit"], best_parameter

    def correlator_fit(self, tag, binsize, fit_range, p0, model_type, verbosity):
        message(f"CORRELATOR: {tag}")
        message(f"P0 = {p0}")
        message(f"FIT RANGE {fit_range}") 
        message(f"MODEL = {model_type}") 
        for b in range(1, binsize+1):
            message(f"BINSIZE = {b}", verbosity)
            binned_tag = self.db.add_binned_leaf(tag, b)
            message("--------------------------------- JACKKNIFE FIT ---------------------------------", verbosity)
            var = self.db.jackknife_variance(binned_tag)  
            Nt = len(self.db.database[binned_tag].mean) 
            W = np.linalg.inv(np.diag(var[fit_range]))
            chi2_func = {"cosh": lambda t,p,y: cosh_chi2(t, p, y, W, Nt),
                         "sinh": lambda t,p,y: sinh_chi2(t, p, y, W, Nt),
                         "exp": lambda t,p,y: exp_chi2(t, p, y, W)}[model_type]
            best_parameter, best_parameter_jks, misc = fit(self.db, fit_range, binned_tag, p0, chi2_func, self.fit_method, self.fit_params, self.res_fit_method, self.res_fit_params)
            best_parameter_cov = jackknife.covariance(self.db.as_array(best_parameter_jks)) 
            print_fit_results(best_parameter, best_parameter_cov, misc, verbosity)
            if b in [1,binsize]:
                message("------------------------------ CORRELATED MEAN FIT ------------------------------", verbosity)
                try:
                    W_correlated = np.linalg.inv(self.db.jackknife_covariance(binned_tag)[fit_range][:,fit_range])
                    chi2_func_correlated = {"cosh": lambda t,p,y: cosh_chi2(t, p, y, W_correlated, Nt),
                                            "sinh": lambda t,p,y: sinh_chi2(t, p, y, W_correlated, Nt),
                                            "exp": lambda t,p,y: exp_chi2(t, p, y, W_correlated)}[model_type]
                    best_parameter_correlated, _, misc_correlated = fit(self.db, fit_range, binned_tag, p0, chi2_func_correlated, self.fit_method, self.fit_params, self.res_fit_method, self.res_fit_params, perform_jks_fit=False)
                    print_fit_results(best_parameter_correlated, None, misc_correlated, verbosity)
                    self.db.add_leaf(tag=f"{binned_tag}/{model_type}_correlated_mean_fit", mean=best_parameter_correlated, jks=None, sample=None, misc=misc_correlated)
                except ConvergenceError as ce:
                    message(f"{ce} for correlated mean fit") 
            if b == 1:
                message("--------------------------------- BOOTSTRAP FIT ---------------------------------", verbosity)
                bss = self.db.bss(binned_tag); mean_bss = self.db.database[binned_tag].mean
                W_bss = np.linalg.inv(np.diag(bootstrap.variance(bss)[fit_range]))
                chi2_func_bss = {"cosh": lambda t,p,y: cosh_chi2(t, p, y, W_bss, Nt),
                                 "sinh": lambda t,p,y: sinh_chi2(t, p, y, W_bss, Nt),
                                 "exp": lambda t,p,y: exp_chi2(t, p, y, W_bss)}[model_type]
                best_parameter_bmean, best_parameter_bss, misc_bss = self._fit_bootstrap(fit_range, mean_bss, bss, best_parameter, chi2_func_bss)
                print_fit_results(best_parameter_bmean, best_parameter_bss, misc_bss)
                misc_bss["bss"] = best_parameter_bss
                self.db.add_leaf(tag=f"{binned_tag}/{model_type}_bootstrap_fit", mean=best_parameter_bmean, jks=None, sample=None, misc=misc_bss)
            self.db.add_leaf(tag=f"{binned_tag}/{model_type}_fit", mean=best_parameter, jks=best_parameter_jks, sample=None, misc=misc)
            message("---------------------------------------------------------------------------------", verbosity) 
            message("---------------------------------------------------------------------------------", verbosity) 

    def _fit_bootstrap(self, t, mean, bss, p0, chi2_func, eval_offset=True):
        t_eval = t if eval_offset else np.arange(len(t))
        if not eval_offset: assert len(t) == len(mean)
        fitter = Fitter(self.fit_method, self.fit_params); fit_func = lambda y: fitter.estimate_parameters(t, chi2_func, y[t_eval], p0)[0]
        best_parameter = fit_func(mean)
        fitter_bss = Fitter(self.res_fit_method, self.res_fit_params); fit_func_bss = lambda y: fitter_bss.estimate_parameters(t, chi2_func, y[t_eval], best_parameter)[0]
        best_parameter_bss = self.db.combine_bss(bss, f=fit_func_bss)
        chi2 = chi2_func(t, best_parameter, mean[t_eval])
        dof = len(t) - len(best_parameter)
        pval = get_pvalue(chi2, dof)
        misc = {"t": t, "chi2": chi2, "dof": dof, "pval": pval}
        return best_parameter, best_parameter_bss, misc

    def extract_mass(self, correlator_fit_tags):
        for tag in correlator_fit_tags:
            lf = self.db.database[tag]
            self.db.add_leaf(f"{tag}/am", mean=lf.mean[1], jks={cfg:jk[1] for cfg,jk in lf.jks.items()}, sample=None, misc=None)
            if "binsize" not in tag:
                bootstrap_tag = tag.replace("fit", "bootstrap_fit"); lf_bs = self.db.database[bootstrap_tag]  
                self.db.add_leaf(f"{bootstrap_tag}/am", mean=lf_bs.mean[1], jks=None, sample=None, misc={"bss": lf_bs.misc["bss"]})
    
    def correlator_combined_fit(self, tag_PS, tag_A4I, fit_range_PS, fit_range_A4I, binsize, p0, model_type_combined, verbosity=0):
        message("------------------ COMBINED CORRELATOR FIT PSPS/PSA4I ---------------------") 
        model_type_PS = model_type_combined.split("-")[1]
        model_type_A4I = model_type_combined.split("-")[2]
        message(f"PSPS correlator: {tag_PS}")
        message(f"PSPS - FIT RANGE {fit_range_PS}") 
        message(f"PSPS - model: {model_type_PS}")
        message(f"PSA4I correlator: {tag_A4I}")
        message(f"PSA4I - FIT RANGE {fit_range_A4I}") 
        message(f"PSA4I - model: {model_type_A4I}")
        message(f"combined model: {model_type_combined}")
        message(f"P0 = {p0}")

        Nt = len(self.db.database[tag_PS].mean)
        fit_range_combined = np.hstack((fit_range_PS, fit_range_A4I))
        combined_tag = f"{tag_PS};{tag_A4I.split("/")[1]}"
        self.db.combine_sample(tag_PS, tag_A4I, f=lambda x,y: np.hstack((x[fit_range_PS],y[fit_range_A4I])), dst_tag=combined_tag)
        for b in range(1, binsize+1):
            message(f"BINSIZE = {b}", verbosity)
            binned_tag = self.db.add_binned_leaf(combined_tag, b)
            message("--------------------------------- JACKKNIFE FIT ---------------------------------", verbosity)
            var = self.db.jackknife_variance(binned_tag)  
            W = np.linalg.inv(np.diag(var))
            chi2_func = {"combined-cosh-sinh": lambda t,p,y: combined_cosh_sinh_chi2(t[:len(fit_range_PS)], t[len(fit_range_PS):], p, y, W, Nt),
                         "combined-exp-exp": lambda t,p,y: combined_exp_exp_model_chi2(t[:len(fit_range_PS)], t[len(fit_range_PS):], p, y, W)}[model_type_combined]
            best_parameter, best_parameter_jks, misc = fit(self.db, fit_range_combined, binned_tag, p0, chi2_func, self.fit_method, self.fit_params, self.res_fit_method, self.res_fit_params, eval_offset=False)
            best_parameter_cov = jackknife.covariance(self.db.as_array(best_parameter_jks)) 
            print_fit_results(best_parameter, best_parameter_cov, misc, verbosity)
            if b in [1,binsize]:
                message("------------------------------ CORRELATED MEAN FIT ------------------------------", verbosity)
                try:
                    W_correlated = np.linalg.inv(self.db.jackknife_covariance(binned_tag))
                    chi2_func_correlated = {"combined-cosh-sinh": lambda t,p,y: combined_cosh_sinh_chi2(t[:len(fit_range_PS)], t[len(fit_range_PS):], p, y, W_correlated, Nt),
                                            "combined-exp-exp": lambda t,p,y: combined_exp_exp_model_chi2(t[:len(fit_range_PS)], t[len(fit_range_PS):], p, y, W_correlated)}[model_type_combined]
                    best_parameter_correlated, _, misc_correlated = fit(self.db, fit_range_combined, binned_tag, best_parameter, chi2_func_correlated, self.fit_method, self.fit_params, self.res_fit_method, self.res_fit_params, perform_jks_fit=False, eval_offset=False)
                    print_fit_results(best_parameter_correlated, None, misc_correlated, verbosity)
                    self.db.add_leaf(tag=f"{binned_tag}/{model_type_combined}_correlated_mean_fit", mean=best_parameter_correlated, jks=None, sample=None, misc=misc_correlated)
                except ConvergenceError as ce:
                    message(f"{ce} for correlated mean fit") 
            if b == 1:
                message("--------------------------------- BOOTSTRAP FIT ---------------------------------", verbosity)
                bss = self.db.bss(binned_tag); mean_bss = self.db.database[binned_tag].mean
                W_bss = np.linalg.inv(np.diag(bootstrap.variance(bss)))
                chi2_func_bss = {"combined-cosh-sinh": lambda t,p,y: combined_cosh_sinh_chi2(t[:len(fit_range_PS)], t[len(fit_range_PS):], p, y, W_bss, Nt),
                                 "combined-exp-exp": lambda t,p,y: combined_exp_exp_model_chi2(t[:len(fit_range_PS)], t[len(fit_range_PS):], p, y, W_bss)}[model_type_combined]
                best_parameter_bmean, best_parameter_bss, misc_bss = self._fit_bootstrap(fit_range_combined, mean_bss, bss, best_parameter, chi2_func_bss, eval_offset=False)
                print_fit_results(best_parameter_bmean, best_parameter_bss, misc_bss)
                misc_bss["bss"] = best_parameter_bss
                self.db.add_leaf(tag=f"{binned_tag}/{model_type_combined}_bootstrap_fit", mean=best_parameter_bmean, jks=None, sample=None, misc=misc_bss)
            self.db.add_leaf(tag=f"{binned_tag}/{model_type_combined}_fit", mean=best_parameter, jks=best_parameter_jks, sample=None, misc=misc)
            message("------------------------------ BARE DECAY CONSTANT ------------------------------")
            self.db.combine(f"{binned_tag}/{model_type_combined}_fit", f=bare_decay_constant, dst_tag=f"{binned_tag}/{model_type_combined}_fit/f_bare")
            if b == 1:
                bootstrap_tag = f"{binned_tag}/{model_type_combined}_bootstrap_fit"
                f_bare_bss_mean = bare_decay_constant(self.db.database[bootstrap_tag].mean)
                f_bare_bss = self.db.combine_bss(self.db.database[bootstrap_tag].misc["bss"], f=bare_decay_constant)
                self.db.add_leaf(tag=f"{bootstrap_tag}/f_bare", mean=f_bare_bss_mean, jks=None, sample=None, misc={"bss": f_bare_bss})
                f_bare_bs_str = f"         {f_bare_bss_mean:.8f} +- {bootstrap.variance(f_bare_bss)**.5:.8f} (bootstrap)"
            message(f"f_bare = {self.db.database[f"{binned_tag}/{model_type_combined}_fit/f_bare"].mean:.8f} +- {self.db.jackknife_variance(f"{binned_tag}/{model_type_combined}_fit/f_bare")**.5:.8f} (jackknife)")
            if b == 1: message(f_bare_bs_str)
            message("---------------------------------------------------------------------------------", verbosity) 
            message("---------------------------------------------------------------------------------", verbosity) 
           
def bare_decay_constant(p):
    # p[0] = A_PSPS, p[1] = A_PSA4I, p[2] = m
    return np.sqrt(2.) * p[1] / np.sqrt(p[0] * p[2])
