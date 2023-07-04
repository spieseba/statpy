#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from ..fitting.core import Fitter, fit, fit_multiple, model_prediction_var
from ..statistics import jackknife

########################################### EFFECTIVE MASS CURVES ###########################################

# open boundary conditions
def effective_mass_log(Ct, tmin, tmax):
    return np.array([np.log(Ct[t] / Ct[t+1]) for t in range(tmin,tmax)]) 

# periodic boundary conditions
def effective_mass_acosh(Ct, tmin, tmax):
    return np.array([np.arccosh(0.5 * (Ct[t+1] + Ct[t-1]) / Ct[t]) for t in range(tmin,tmax)])

########################################### EFFECTIVE AMPLITUDE CURVES ###########################################

# cosh
def effective_amplitude_cosh(Ct, t, m, Nt):
    return Ct / ( np.exp(-m*t) + np.exp(-m*(Nt-t)) ) 

# sinh
def effective_amplitude_sinh(Ct, t, m, Nt):
    return Ct / ( np.exp(-m*t) - np.exp(-m*(Nt-t)) ) 

# exp
def effective_amplitude_exp(Ct, t, m):
    return Ct / np.exp(-m*t)

################################################## FITTING #################################################

#################################### FIT MODELS ############################################

############## periodic boundary conditions #############

# C(t) = A * [exp(-mt) + exp(-m(Nt-t))]; A = p[0]; m = p[1] 
class cosh_model:
    def __init__(self, Nt):
        self.Nt = Nt 
    def __call__(self, t, p):
        return p[0] * ( np.exp(-p[1]*t) + np.exp(-p[1]*(self.Nt-t)) )
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t) + np.exp(-p[1]*(self.Nt-t)), p[0] * (np.exp(-p[1]*t) * (-t) + np.exp(-p[1]*(self.Nt-t)) * (t-self.Nt))], dtype=object)    

# C(t) = A * [exp(-mt) - exp(-m(Nt-t))]; A = p[0]; m = p[1]  
class sinh_model:
    def __init__(self, Nt):
        self.Nt = Nt 
    def __call__(self, t, p):
        return p[0] * ( np.exp(-p[1]*t) - np.exp(-p[1]*(self.Nt-t)) )
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t) - np.exp(-p[1]*(self.Nt-t)), p[0] * (np.exp(-p[1]*t) * (-t) - np.exp(-p[1]*(self.Nt-t)) * (t-self.Nt))], dtype=object)  

# C(t) = A0 * [exp(-m0 t) + exp(-m0(Nt-t))] + A1 * [exp(-m1 t) + exp(-m1(Nt-t))]; A0 = p[0], m0 = p[1], A1 = p[2]; m1 = p[3] 
class double_cosh_model():
    def __init__(self, Nt):
        self.Nt = Nt
    def __call__(self, t, p):
        return p[0] * ( np.exp(-p[1]*t) + np.exp(-p[1]*(self.Nt-t)) ) + p[2] * ( np.exp(-p[3]*t) + np.exp(-p[3]*(self.Nt-t)) )
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t) + np.exp(-p[1]*(self.Nt-t)), 
                    p[0] * (np.exp(-p[1]*t) * (-t) + np.exp(-p[1]*(self.Nt-t)) * (t-self.Nt)),
                    np.exp(-p[3]*t) + np.exp(-p[3]*(self.Nt-t)), 
                    p[2] * (np.exp(-p[3]*t) * (-t) + np.exp(-p[3]*(self.Nt-t)) * (t-self.Nt))], dtype=object) 
    
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
                    p[2] * (np.exp(-p[3]*t) * (-t) - np.exp(-p[3]*(self.Nt-t)) * (t-self.Nt))], dtype=object) 

################ open boundary conditions ###############

# C(t) = A * exp(-mt); A = p[0]; m = p[1] 
class exp_model:
    def __init__(self):
        pass   
    def __call__(self, t, p):
        return p[0] * np.exp(-p[1]*t)
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t), p[0] * np.exp(-p[1]*t) * (-t)], dtype=object)
    
# C(t) = A0 * exp(-m0t) + A1 * exp(-m1t); A0 = p[0], m0 = p[1], A1 = p[2]; m1 = p[3] 
class double_exp_model:
    def __init__(self):
        pass   
    def __call__(self, t, p):
        return p[0] * np.exp(-p[1]*t) + p[2] * np.exp(-p[3]*t)
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t), p[0] * np.exp(-p[1]*t) * (-t), np.exp(-p[3]*t), p[2] * np.exp(-p[3]*t) * (-t)], dtype=object)
    
############################################## COMBINED FITTING ############################################

#################################### FIT MODELS ############################################

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
    def cosh(self, t, p):
        return p[0] * ( np.exp(-p[2]*t) + np.exp(-p[2]*(self.Nt-t)) ) 
    def sinh(self, t, p):
        return p[1] * ( np.exp(-p[2]*t) - np.exp(-p[2]*(self.Nt-t)) )  
    def parameter_gradient(self, t, p):
        df0 = np.array([np.exp(-p[2]*t) + np.exp(-p[2]*(self.Nt-t)), 0, p[0] * (np.exp(-p[2]*t) * (-t) + np.exp(-p[2]*(self.Nt-t)) * (t-self.Nt))], dtype=object)  
        df1 = np.array([0, np.exp(-p[2]*t) - np.exp(-p[2]*(self.Nt-t)), p[1] * (np.exp(-p[2]*t) * (-t) - np.exp(-p[2]*(self.Nt-t)) * (t-self.Nt))], dtype=object)  
        return np.array([df0, df1]) 
    
    # potentially second function
    
######### const model to fit effective mass plateau #########
class const_model:
        def __init__(self):
            pass  
        def __call__(self, t, p):
            return p[0]
        def parameter_gradient(self, t, p):
            return np.array([1.0], dtype=object)


##############################################################################################################################
##############################################################################################################################
####################################################### JKS SYSTEM ###########################################################
##############################################################################################################################
##############################################################################################################################

def effective_mass_curve_fit(db, tag, t0_min, t0_max, nt, cov, p0, bc, method, minimizer_params, binsize, dst_tag, sys_tags=None, verbosity=0):
   assert bc in ["pbc", "obc"]
   model = {"pbc": cosh_model(len(db.database[tag].mean)), "obc": exp_model()}[bc]
   for t0 in range(t0_min, t0_max):
       t = np.arange(nt) + t0
       if verbosity >=0: db.message(f"fit window: {t}")
       fit(db, t, tag, cov[t][:,t], p0, model, method, minimizer_params, binsize, dst_tag + f"={t0}", sys_tags, verbosity)
       db.database[dst_tag + f"={t0}"].mean = db.database[dst_tag + f"={t0}"].mean[1]
       db.database[dst_tag + f"={t0}"].jks = {cfg:val[1] for cfg, val in db.database[dst_tag + f"={t0}"].jks.items()} 
       db.database[dst_tag + f"={t0}"].info["best_parameter_cov"] = db.database[dst_tag + f"={t0}"].info["best_parameter_cov"][1][1]
       for sys in sys_tags:
           db.database[dst_tag + f"={t0}"].info[f"MEAN_SHIFTED_{sys}"] = db.database[dst_tag + f"={t0}"].info[f"MEAN_SHIFTED_{sys}"][1]
           db.database[dst_tag + f"={t0}"].info[f"SYS_VAR_{sys}"] = db.database[dst_tag + f"={t0}"].info[f"SYS_VAR_{sys}"][1]

def effective_mass_const_fit(db, ts, tags, cov, p0, method, minimizer_params, binsize, dst_tag, sys_tags=None, verbosity=0):
    model = const_model()
    # add t Leafs
    for t in ts: db.add_Leaf(f"tmp_t{t}", mean=t, jks={}, sample=None, nrwf=None, info=None)
    fit_multiple(db, [f"tmp_t{t}" for t in ts], tags, cov, p0, model, method, minimizer_params, binsize, dst_tag, sys_tags, verbosity)
    # cleanup t Leafs
    db.remove(*[f"tmp_t{t}" for t in ts])


##############################################################################################################################
##############################################################################################################################
##################################################### LATTICE CHARM ##########################################################
##############################################################################################################################
##############################################################################################################################

def lc_fit(t, Ct, Ct_jks, Ct_cov, p0, model, fit_method, fit_params, jks_fit_method=None, jks_fit_params=None):
    # mean fit
    fitter = Fitter(Ct_cov, model, fit_method, fit_params)
    best_parameter, chi2, _ = fitter.estimate_parameters(t, fitter.chi_squared, Ct, p0)
    # jks fits
    if jks_fit_method == None: jks_fit_method = fit_method; jks_fit_params = fit_params
    jks_fitter = Fitter(Ct_cov, model, jks_fit_method, jks_fit_params)
    best_parameter_jks = {}
    for cfg in  Ct_jks:
        best_parameter_jks[cfg], _, _ = jks_fitter.estimate_parameters(t, fitter.chi_squared, Ct_jks[cfg], best_parameter)
    dof = len(t) - len(best_parameter)
    pval = fitter.get_pvalue(chi2, dof) 
    return best_parameter, best_parameter_jks, chi2, dof, pval

def lc_fit_range(db, Ct_tag, binsize, initial_fit_range, ds, p0, model_type, fit_method, fit_params=None, jks_fit_method=None, jks_fit_params=None, verbosity=0, make_plots=True):
    def sort_params(p):
        if p[3] <  p[1]: return [p[2], p[3], p[0], p[1]]
        else: return p
    if jks_fit_method == None:
        jks_fit_method = fit_method; jks_fit_params = fit_params
    jks = db.sample_jks(Ct_tag, binsize)
    mean = np.mean(jks, axis=0)
    var = jackknife.variance_jks(mean, jks)
    Nt = len(mean)
    model = {"double-cosh": double_cosh_model(Nt), "double-sinh": double_sinh_model(Nt), "double-exp": double_exp_model()}[model_type]
    fit_range = np.arange(Nt)
    for d in ds:
        assert d > 1
        t = initial_fit_range[d:-d+1]
        if model_type == "double-exp":
            t = initial_fit_range[d:]
        db.message(f"fit range: {t}", verbosity)
        y = mean[t]; y_jks = {cfg:Ct[t] for cfg,Ct in enumerate(jks)}; cov = np.diag(var)[t][:,t]
        best_parameter, best_parameter_jks, chi2, dof, pval = lc_fit(t, y, y_jks, cov, p0, model, fit_method, fit_params, jks_fit_method, jks_fit_params)
        best_parameter = sort_params(best_parameter)
        for cfg in best_parameter_jks:
            best_parameter_jks[cfg] = sort_params(best_parameter_jks[cfg])   
        best_parameter_cov = jackknife.covariance_jks(best_parameter, db.as_array(best_parameter_jks)) 
        for i in range(len(best_parameter)):
            db.message(f"parameter[{i}] = {best_parameter[i]} +- {best_parameter_cov[i][i]**0.5}", verbosity)
        db.message(f"chi2 / dof = {chi2} / {dof} = {chi2/dof}, i.e., p = {pval}", verbosity)
        criterion = np.abs([model(i, [0, 0, best_parameter[2], best_parameter[3]]) for i in t]) < var[t]**.5/4.
        reduced_t = t[criterion]
        db.message(f"reduced fit range {reduced_t}", verbosity)
        if len(reduced_t) < len(fit_range): fit_range = reduced_t
        if verbosity >= 0:     
            print("----------------------------------------------------------------------------------------------------------------------------------")
            print("----------------------------------------------------------------------------------------------------------------------------------")
        if make_plots:
            lcp = LatticeCharmPlots(mean, jks, t, reduced_t, model_type, best_parameter, best_parameter_jks)
            lcp.plot()
            #lc_plot(db, Ct_tag, mean, jks, binsize, t, model_type, model, best_parameter, best_parameter_jks, best_parameter_cov, reduced_t)
    db.message(f"FINAL REDUCED FIT RANGE: {fit_range}", verbosity)
    return fit_range

def lc_spectroscopy(db, Ct_tag, b_max, fit_range, p0, model_type, correlated=False, fit_method="Migrad", fit_params=None, jks_fit_method=None, jks_fit_params=None, make_plot=True, verbosity=0):
    A_dict = {}; A_var_dict = {}
    m_dict = {}; m_var_dict = {}
    # estimate cov using unbinned sample
    if correlated:
        cov = db.sample_jackknife_covariance(Ct_tag, binsize=1)
    else:
        cov = np.diag(db.sample_jackknife_variance(Ct_tag, binsize=1))
    for b in range(1,b_max+1):
        db.message(f"BINSIZE = {b}\n", verbosity)
        jks = db.sample_jks(Ct_tag, binsize=b)
        mean = np.mean(jks, axis=0)
        t = fit_range
        y = mean[t]; y_jks = {cfg:Ct[t] for cfg,Ct in enumerate(jks)}
        model = {"cosh": cosh_model(len(mean)), "sinh": sinh_model(len(mean)), "exp": exp_model()}[model_type]
        best_parameter, best_parameter_jks, chi2, dof, pval = lc_fit(t, y, y_jks, cov[t][:,t], p0, model, fit_method, fit_params, jks_fit_method, jks_fit_params)
        best_parameter_cov = jackknife.covariance_jks(best_parameter, best_parameter_jks)
        for i in range(len(best_parameter)):
            db.message(f"parameter[{i}] = {best_parameter[i]} +- {best_parameter_cov[i][i]**0.5}", verbosity)
        db.message(f"chi2 / dof = {chi2} / {dof} = {chi2/dof}, i.e., p = {pval}", verbosity)
        A_dict[b] = best_parameter[0]; A_var_dict[b] = best_parameter_cov[0][0]
        m_dict[b] = best_parameter[1]; m_var_dict[b] = best_parameter_cov[1][1]
        if verbosity >=0:
            print("\n-----------------------------------------------------------------------------------------")
            print("-----------------------------------------------------------------------------------------\n")
        if b == b_max:
            lcp = LatticeCharmPlots(mean, jks, t, None, model_type, best_parameter, best_parameter_jks) 
            lcp.plot()
            #lc_plot(db, Ct_tag, mean, jks, b_max, t, model_type, model, best_parameter, best_parameter_jks, best_parameter_cov)
    return A_dict, A_var_dict, m_dict, m_var_dict  

def lc_combined_cosh_sinh_fit(db, Ct_tag_PSPS, Ct_tag_PSA4, B, fit_range_PSPS, fit_range_PSA4, p0, correlated=False, fit_method="Migrad", fit_params=None, jks_fit_method=None, jks_fit_params=None, verbosity=0, make_plot=True):
    A_PS_dict = {}; A_PS_var_dict = {}; A_PS_jks_dict = {}
    A_A4_dict = {}; A_A4_var_dict = {}; A_A4_jks_dict = {}
    m_dict = {}; m_var_dict = {}; m_jks_dict = {}
    # estimate cov using unbinned sample
    jks_ub_PSPS = db.sample_jks(Ct_tag_PSPS, 1); jks_ub_PSA4 = db.sample_jks(Ct_tag_PSA4, 1)
    jks_ub = np.array([np.hstack((jks_ub_PSPS[cfg][fit_range_PSPS],jks_ub_PSA4[cfg][fit_range_PSA4])) for cfg in range(len(jks_ub_PSPS))])
    cov = jackknife.covariance_jks(np.mean(jks_ub, axis=0), jks_ub)
    if not correlated:
        cov = np.diag(np.diag(cov))
    for b in range(1, B+1):
        print(f"BINSIZE = {b}\n")
        Ct_jks_PSPS = db.sample_jks(Ct_tag_PSPS, binsize=b); Ct_jks_PSA4 = db.sample_jks(Ct_tag_PSA4, binsize=b)
        Ct_mean_PSPS = np.mean(Ct_jks_PSPS, axis=0); Ct_mean_PSA4 = np.mean(Ct_jks_PSA4, axis=0)
        jks_arr = np.array([np.hstack((Ct_jks_PSPS[cfg][fit_range_PSPS], Ct_jks_PSA4[cfg][fit_range_PSA4])) for cfg in range(len(Ct_jks_PSPS))])
        jks = {cfg:jks_arr[cfg] for cfg in range(len(jks_arr))}
        mean = np.mean(jks_arr, axis=0)
        model = combined_cosh_sinh_model(len(Ct_jks_PSPS[0]), fit_range_PSPS, fit_range_PSA4)
        best_parameter, best_parameter_jks, chi2, dof, pval = lc_fit(np.hstack((fit_range_PSPS, fit_range_PSA4)), mean, jks, cov, p0, model, fit_method, fit_params, jks_fit_method, jks_fit_params)
        best_parameter_cov = jackknife.covariance_jks(best_parameter, best_parameter_jks)
        for i in range(len(best_parameter)):
            db.message(f"parameter[{i}] = {best_parameter[i]} +- {best_parameter_cov[i][i]**0.5}", verbosity)
        db.message(f"chi2 / dof = {chi2} / {dof} = {chi2/dof}, i.e., p = {pval}", verbosity)
        A_PS_dict[b] = best_parameter[0]; A_PS_var_dict[b] = best_parameter_cov[0][0]
        A_PS_jks_dict[b] = {cfg:p[0] for cfg,p in best_parameter_jks.items()}
        A_A4_dict[b] = best_parameter[1]; A_A4_var_dict[b] = best_parameter_cov[1][1]
        A_A4_jks_dict[b] = {cfg:p[1] for cfg,p in best_parameter_jks.items()}
        m_dict[b] = best_parameter[2]; m_var_dict[b] = best_parameter_cov[2][2]
        m_jks_dict[b] = {cfg:p[2] for cfg,p in best_parameter_jks.items()}
        if verbosity >=0:
            print("\n-----------------------------------------------------------------------------------------")
            print("-----------------------------------------------------------------------------------------\n")
        if b == B:
            lcp = LatticeCharmPlots()
            #lc_combined_plot(db, Ct_tag_PSPS, Ct_mean_PSPS, Ct_jks_PSPS, fit_range_PSPS, Ct_tag_PSA4, Ct_mean_PSA4, Ct_jks_PSA4, fit_range_PSA4, 
            #                 B, model, best_parameter, best_parameter_jks, best_parameter_cov)
    return A_PS_dict, A_PS_var_dict, A_PS_jks_dict, A_A4_dict, A_A4_var_dict, A_A4_jks_dict, m_dict, m_var_dict, m_jks_dict 


### TO DO: combined exp fit

############################################## PLOTS  ############################################


class LatticeCharmPlots():

    def __init__(self, Ct_mean, Ct_jks, fit_range, reduced_fit_range, model_type, best_parameter, best_parameter_jks, Ct_mean2=None, Ct_jks2=None, fit_range2=None):
        self.Ct_mean = Ct_mean
        self.Nt = len(Ct_mean)
        self.Ct_jks = Ct_jks
        self.Ct_var = jackknife.variance_jks(Ct_mean, Ct_jks)
        self.fit_range = fit_range
        self.fit_range_dt = np.arange(fit_range[0], fit_range[-1], 0.01)
        self.reduced_fit_range = reduced_fit_range
        self.model_type = model_type
        self.model = self._get_model()
        self.best_parameter = np.array(best_parameter)
        self.best_parameter_jks = best_parameter_jks
        self.best_parameter_cov = jackknife.covariance_jks(best_parameter, np.array(list(best_parameter_jks.values())))

    def plot(self): 
        fig, ((ax0, ax_not), (ax1, ax2)) = plt.subplots(nrows=2, ncols=2)
        ax_not.set_axis_off()
        self._add_correlator(ax0)
        self._add_amplitude(ax1)
        self._add_mass(ax2)
        plt.tight_layout()
        plt.plot()

    def combined_plot(self):
        
        fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(nrows=3, ncols=2, figsize=(12,12))


#    # PSPS correlator
#    Nt = len(Ct_mean_PSPS)
#    ## correlator ##
#    color = "C0"
#    ax0.set_xlabel(r"source-sink separation $t/a$")
#    ax0.set_ylabel(r"$C(t)$")     
#    ax0.errorbar(np.arange(1,Nt-1), Ct_mean_PSPS[1:-1], jackknife.variance_jks(Ct_mean_PSPS, Ct_jks_PSPS)[1:-1]**0.5, linestyle="", capsize=3, color=color, label="data")
#    # correlator fit
#    d = 3
#    trange = np.arange(d, Nt-d, 0.01)
#    color = "C1"
#    fy_PSPS = np.array([model_combined.cosh(t, best_parameter) for t in trange])
#    fy_std_PSPS = np.array([model_prediction_var(t, best_parameter, best_parameter_cov, lambda x,y: model_combined.parameter_gradient(x,y)[0]) for t in trange])**.5
#    ax0.plot(trange, fy_PSPS, color=color, lw=.5, label=r"$C_{PSPS}(t) = A_0 (e^{-m t} + e^{-m (T-t)})$ - fit")
#    ax0.fill_between(trange, fy_PSPS-fy_std_PSPS, fy_PSPS+fy_std_PSPS, alpha=0.5, color=color)
#    ax0.legend(loc="upper left")
#    # fit range marker
#    ax0.axvline(fit_range_PSPS[0], color="gray", linestyle="--")
#    ax0.axvline(fit_range_PSPS[-1], color="gray", linestyle="--")
#    # set title
#    ax0.set_title(Ct_tag_PSPS)


    def _get_model(self):
        if self.model_type == "cosh":
            return cosh_model(self.Nt)
        if self.model_type == "double-cosh":
            return double_cosh_model(self.Nt)
        elif self.model_type == "sinh":
            return sinh_model(self.Nt)
        elif self.model_type == "double-sinh":
            return double_sinh_model(self.Nt)
        elif self.model_type == "exp":
            return exp_model()
        elif self.model_type == "double-exp":
            return double_exp_model()
        #elif self.model_type == "combined-cosh-sinh":
        #    return combined_cosh_sinh_model(self.Nt, )
        else:
            raise Exception("Model not available")

    def _add_fit_range_marker(self, ax):
        ax.axvline(self.fit_range[0], color="gray", linestyle="--", label="fit range")
        ax.axvline(self.fit_range[-1], color="gray", linestyle="--")
        if self.reduced_fit_range is not None:
            ax.axvline(self.reduced_fit_range[0], color="black", linestyle="--", label="reduced fit range")
            ax.axvline(self.reduced_fit_range[-1], color="black", linestyle="--")

    def _add_correlator(self, ax):
        ax.set_xlabel(r"source-sink separation $t/a$")
        ax.set_ylabel(r"$C(t)$")    
        # data
        color = "C0"
        ax.errorbar(np.arange(self.Nt), self.Ct_mean, self.Ct_var**0.5, linestyle="", capsize=3, color=color, label="C(t) data")
        # fit
        color = "C1"
        fy = np.array([self.model(t, self.best_parameter) for t in self.fit_range_dt])
        fy_err = np.array([model_prediction_var(t, self.best_parameter, self.best_parameter_cov, self.model.parameter_gradient) for t in self.fit_range_dt])**.5
        ax.plot(self.fit_range_dt, fy, color=color, lw=.5, label=self._get_Ct_fit_model_label())
        ax.fill_between(self.fit_range_dt, fy-fy_err, fy+fy_err, alpha=0.5, color=color)
        # misc
        self._add_fit_range_marker(ax)
        ax.set_xlim(self.fit_range[0]-1, self.fit_range[-1]+1)
        ax.legend()

    def _add_amplitude(self, ax):
        ax.set_xlabel(r"source-sink separation $t/a$")
        ax.set_ylabel("$A$")
        # data   
        color = "C2"
        At = self._effective_amplitude()(self.Ct_mean[self.fit_range], self.fit_range, self.best_parameter[1])
        At_var = jackknife.variance_jks(At, np.array([self._effective_amplitude()(self.Ct_jks[j][self.fit_range], self.fit_range, self.best_parameter_jks[j][1]) for j in self.best_parameter_jks]))
        ax.errorbar(self.fit_range, At, At_var**.5, linestyle="", capsize=3, color=color, label=self._get_effective_amplitude_label())
        # fit
        color = "C3"
        At_fit, At_fit_var = self._local_amp()
        ax.plot(self.fit_range_dt, At_fit, color=color, lw=.5, label=self._get_amplitude_fit_label())
        ax.fill_between(self.fit_range_dt, At_fit-At_fit_var**.5, At_fit+At_fit_var**.5, alpha=0.5, color=color)
        # misc
        self._add_fit_range_marker(ax)
        ax.set_xlim(self.fit_range[0]-1, self.fit_range[-1]+1)
        ax.legend()

#    # best amplitude
#    color = "C4"
#    A_arr = np.array([best_parameter[0] for t in trange])
#    ax1.plot(trange, A_arr, color=color, lw=.5, label=r"$A_0 = $" + f"{best_parameter[0]:.4g} +- {best_parameter_cov[0][0]**.5:.4g}")
#    ax1.fill_between(trange, A_arr-best_parameter_cov[0][0]**.5, A_arr+best_parameter_cov[0][0]**.5, alpha=0.5, color=color)

    def _add_mass(self, ax):
        ax.set_xlabel(r"source-sink separation $t/a$")
        ax.set_ylabel(r"$m/GeV$")
        # data
        color = "C2"
        mt = self._effective_mass()(self.Ct_mean, self.fit_range[0], self.fit_range[-1]+1)
        mt_var = jackknife.variance_jks(mt, np.array([self._effective_mass()(self.Ct_jks[j], self.fit_range[0], self.fit_range[-1]+1) for j in self.best_parameter_jks]))
        ax.errorbar(self.fit_range, mt, mt_var**.5, linestyle="", capsize=3, color=color, label=self._get_effective_mass_label())
        # fit 
        color = "C3"
        mt_fit, mt_fit_var = self._local_mass() 
        ax.plot(self.fit_range_dt[1:-1], mt_fit, color=color, lw=.5, label=self._get_mass_fit_label()) 
        ax.fill_between(self.fit_range_dt[1:-1], mt_fit-mt_fit_var**.5, mt_fit+mt_fit_var**.5, alpha=0.5, color=color)
        # misc 
        self._add_fit_range_marker(ax)
        ax.set_xlim(self.fit_range[0]-1, self.fit_range[-1]+1)
        ax.legend()

#    # best mass
#    color = "C4"
#    m_arr = np.array([best_parameter[1] for t in trange])
#    ax2.plot(trange, m_arr, color=color, lw=.5, label=r"$m_0 = $" + f"{best_parameter[1]:.4g} +- {best_parameter_cov[1][1]**.5:.4g}")
#    ax2.fill_between(trange, m_arr-best_parameter_cov[1][1]**.5, m_arr+best_parameter_cov[1][1]**.5, alpha=0.5, color=color)

    ######### AMPLITUDE #########

    def _effective_amplitude(self):
        if self.model_type in ["cosh", "double-cosh"]:
            return lambda Ct, t, m: effective_amplitude_cosh(Ct, t, m, self.Nt)
        elif self.model_type in ["sinh", "double-sinh"]:
            return lambda Ct, t, m: effective_amplitude_sinh(Ct, t, m, self.Nt)
        elif self.model_type in ["exp", "double-exp"]:
            return lambda Ct, t, m: effective_amplitude_exp(Ct, t, m)
        else:
            raise Exception("Effective amplitude function not found")

    def _local_amp(self):
        C_fit_range_dt = np.array([self.model(t, self.best_parameter) for t in self.fit_range_dt])
        At = self._effective_amplitude()(C_fit_range_dt, self.fit_range_dt, self.best_parameter[1]) 
        At_var = jackknife.variance_jks(At, np.array([self._effective_amplitude()(np.array([self.model(t, self.best_parameter_jks[j]) for t in self.fit_range_dt]), 
                                                      self.fit_range_dt, self.best_parameter_jks[j][1]) for j in self.best_parameter_jks]) )
        return At, At_var

    ######### MASS #########

    def _effective_mass(self):
        if self.model_type in ["cosh", "double-cosh", "sinh", "double-sinh"]:
            return lambda Ct, tmin, tmax: effective_mass_acosh(Ct, tmin, tmax)
        elif self.model_type in ["exp", "double-exp"]:
            return lambda Ct, tmin, tmax: effective_mass_log(Ct, tmin, tmax)
        else:
            raise Exception("Effective mass function not found")
        
    def _local_mass(self):
        C_fit_range_dt = np.array([self.model(t, self.best_parameter) for t in self.fit_range_dt])
        dt = self.fit_range_dt[1] - self.fit_range_dt[0]
        mt = self._effective_mass()(C_fit_range_dt, 1, len(self.fit_range_dt)-1) / dt
        mt_var = jackknife.variance_jks(mt, np.array([self._effective_mass()(np.array([self.model(t, self.best_parameter_jks[j]) for t in self.fit_range_dt]), 
                                                               1, len(self.fit_range_dt)-1)/dt for j in self.best_parameter_jks]) )
        return mt, mt_var

    ######### LABELS #########

    def _get_Ct_fit_model_label(self):
        if self.model_type == "cosh":
            return r"$C(t) = A_0 (e^{-m_0 t} + e^{-m_0 (T-t)})$"
        elif self.model_type == "sinh":
            return r"$C(t) = A_0 (e^{-m_0 t} - e^{-m_0 (T-t)})$"
        elif self.model_type == "double-cosh":
            return r"$C(t) = A_0 (e^{-m_0 t} + e^{-m_0 (T-t)}) + A_1 (e^{-m_1 t} + e^{-m_1 (T-t)})$"
        elif self.model_type == "double-sinh":
            return r"$C(t) = A_0 (e^{-m_0 t} - e^{-m_0 (T-t)}) + A_1 (e^{-m_1 t} - e^{-m_1 (T-t)})$"
        elif self.model_type == "exp":
            return r"$C(t) = A_0 e^{-m_0 t}$"
        elif self.model_type == "double-exp":
            return r"$C(t) = A_0 e^{-m_0 t} + A_1 e^{-m_1 t}$"
        else:
            raise Exception("Label for Ct fit model not found")
        
    def _get_effective_amplitude_label(self):
        if self.model_type in ["cosh", "double-cosh"]:
            return r"$A(t) = \frac{C(t)}{e^{-m_0 t} + e^{-m_0 (T-t)}}$"
        elif self.model_type in ["sinh", "double-sinh"]:
            return r"$A(t) = \frac{C(t)}{e^{-m_0 t} - e^{-m_0 (T-t)}}$"
        elif self.model_type in ["exp", "double-exp"]:
            return r"$A(t) = \frac{C(t)}{e^{-m_0 t}}$"
        else:
            raise Exception("Label for effective amplitude not found") 
        
    def _get_amplitude_fit_label(self):
        if self.model_type in ["cosh", "double-cosh"]:
            return r"$A(t) = \frac{C_{fit}(t)}{e^{-m_0 t} + e^{-m_0 (T-t)}}$"
        elif self.model_type in ["sinh", "double-sinh"]:
            return r"$A(t) = \frac{C_{fit}(t)}{e^{-m_0 t} - e^{-m_0 (T-t)}}$"
        elif self.model_type in ["exp", "double-exp"]:
            return r"$A(t) = \frac{C_{fit}(t)}{e^{-m_0 t}}$"
        else:
            raise Exception("Label for effective amplitude not found") 

    def _get_effective_mass_label(self):
        if self.model_type in ["cosh", "double-cosh", "sinh", "double-sinh"]:
            return r"$m(t) = cosh^{-1}\left( \frac{C(t+1) + C(t-1)}{2 C(t)} \right)$"
        elif self.model_type in ["exp", "double-exp"]:
            return r"$m(t) = log\left( \frac{C(t)}{C(t+1)} \right)$"
        else:
            raise Exception("Label for effective mass not found") 
    
    def _get_mass_fit_label(self):
        if self.model_type in ["cosh", "double-cosh", "sinh", "double-sinh"]:
            return r"$m(t) = cosh^{-1}\left( \frac{C_{fit}(t+1) + C_{fit}(t-1)}{2 C_{fit}(t)} \right)$"
        elif self.model_type in ["exp", "double-exp"]:
            return r"$m(t) = log\left( \frac{C_{fit}(t)}{C_{fit}(t+1)} \right)$"
        else:
            raise Exception("Label for effective mass not found") 



##############################################################

#def lc_combined_cosh_sinh_plot(db, Ct_tag_PSPS, Ct_mean_PSPS, Ct_jks_PSPS, fit_range_PSPS, Ct_tag_PSA4, Ct_mean_PSA4, Ct_jks_PSA4, fit_range_PSA4, binsize, 
#                    model_combined, best_parameter, best_parameter_jks, best_parameter_cov):
#    
#    fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(nrows=3, ncols=2, figsize=(12,12))
#
#    # PSPS correlator
#    Nt = len(Ct_mean_PSPS)
#    ## correlator ##
#    color = "C0"
#    ax0.set_xlabel(r"source-sink separation $t/a$")
#    ax0.set_ylabel(r"$C(t)$")     
#    ax0.errorbar(np.arange(1,Nt-1), Ct_mean_PSPS[1:-1], jackknife.variance_jks(Ct_mean_PSPS, Ct_jks_PSPS)[1:-1]**0.5, linestyle="", capsize=3, color=color, label="data")
#    # correlator fit
#    d = 3
#    trange = np.arange(d, Nt-d, 0.01)
#    color = "C1"
#    fy_PSPS = np.array([model_combined.cosh(t, best_parameter) for t in trange])
#    fy_std_PSPS = np.array([model_prediction_var(t, best_parameter, best_parameter_cov, lambda x,y: model_combined.parameter_gradient(x,y)[0]) for t in trange])**.5
#    ax0.plot(trange, fy_PSPS, color=color, lw=.5, label=r"$C_{PSPS}(t) = A_0 (e^{-m t} + e^{-m (T-t)})$ - fit")
#    ax0.fill_between(trange, fy_PSPS-fy_std_PSPS, fy_PSPS+fy_std_PSPS, alpha=0.5, color=color)
#    ax0.legend(loc="upper left")
#    # fit range marker
#    ax0.axvline(fit_range_PSPS[0], color="gray", linestyle="--")
#    ax0.axvline(fit_range_PSPS[-1], color="gray", linestyle="--")
#    # set title
#    ax0.set_title(Ct_tag_PSPS)
#    ## local Amplitude ##
#    # data
#    color = "C2"
#    At_func_PSPS = lambda x, y: effective_amp_cosh(x[d:Nt-d], np.arange(d, Nt-d), y, Nt)
#    At_data_PSPS = At_func_PSPS(Ct_mean_PSPS, best_parameter[2])
#    At_jks_PSPS = {}
#    for j in best_parameter_jks:
#        At_jks_PSPS[j] = At_func_PSPS(Ct_jks_PSPS[j], best_parameter_jks[j][2]) 
#    At_var_data_PSPS = jackknife.variance_jks(At_data_PSPS, np.array(list(At_jks_PSPS.values())))
#    ax2.set_xlabel(r"source-sink separation $t/a$")
#    ax2.set_ylabel("$m/GeV$")
#    ax2.errorbar(np.arange(d, Nt-d), At_data_PSPS, At_var_data_PSPS**.5, linestyle="", capsize=3, color=color, label=r"$A_0(t)$ data")
#    # fit
#    color = "C3"
#    At_fit_PSPS, At_var_fit_PSPS = local_amp(trange, model_combined.cosh, "cosh", best_parameter, best_parameter_jks, Nt, mass_idx=2)
#    ax2.set_xlabel(r"source-sink separation $t/a$")
#    ax2.set_ylabel("$A$")
#    ax2.plot(trange[1:-1], At_fit_PSPS, color=color, lw=.5, label=r"$A_0(t) = \frac{C_{PSPS}(t)}{e^{-m t} + e^{-m (T-t)}}$")
#    ax2.fill_between(trange[1:-1], At_fit_PSPS-At_var_fit_PSPS**.5, At_fit_PSPS+At_var_fit_PSPS**.5, alpha=0.5, color=color)
#    # best amplitude
#    color = "C4"
#    A_arr_PSPS = np.array([best_parameter[0] for t in trange])
#    ax2.plot(trange, A_arr_PSPS, color=color, lw=.5, label=r"$A_0 = $" + f"{best_parameter[0]:.4g} +- {best_parameter_cov[0][0]**.5:.4g}")
#    ax2.fill_between(trange, A_arr_PSPS-best_parameter_cov[0][0]**.5, A_arr_PSPS+best_parameter_cov[0][0]**.5, alpha=0.5, color=color)
#    ax2.legend(loc="upper right")
#    # fit range marker
#    ax2.axvline(fit_range_PSPS[0], color="gray", linestyle="--")
#    ax2.axvline(fit_range_PSPS[-1], color="gray", linestyle="--")
#    ## local mass ## 
#    # data
#    color = "C2"
#    mt_func_PSPS = lambda x: effective_mass_acosh(x, tmin=1, tmax=Nt-1)
#    mt_data_PSPS = mt_func_PSPS(Ct_mean_PSPS)
#    mt_var_data_PSPS = db.sample_jackknife_variance(Ct_tag_PSPS, binsize, mt_func_PSPS)
#    ax4.set_xlabel(r"source-sink separation $t/a$")
#    ax4.set_ylabel("$m/GeV$")
#    ax4.errorbar(np.arange(1, Nt-1), mt_data_PSPS, mt_var_data_PSPS**.5, linestyle="", capsize=3, color=color, label=r"$m(t)$ data")
#    # fit 
#    color = "C3"
#    mt_fit_PSPS, mt_var_fit_PSPS = local_mass(trange, model_combined.cosh, best_parameter, best_parameter_jks)
#    ax4.plot(trange[1:-1], mt_fit_PSPS, color=color, lw=.5, label=r"$m(t) = cosh^{-1}\left( \frac{C_{PSPS}(t+1) + C_{PSPS}(t-1)}{2 C_{PSPS}(t)} \right)$")
#    ax4.fill_between(trange[1:-1], mt_fit_PSPS-mt_var_fit_PSPS**.5, mt_fit_PSPS+mt_var_fit_PSPS**.5, alpha=0.5, color=color)
#    # mass from fit
#    color = "C4"
#    m_arr = np.array([best_parameter[2] for t in trange])
#    ax4.plot(trange, m_arr, color=color, lw=.5, label=r"$m = $" + f"{best_parameter[2]:.4g} +- {best_parameter_cov[2][2]**.5:.4g}")
#    ax4.fill_between(trange, m_arr-best_parameter_cov[2][2]**.5, m_arr+best_parameter_cov[2][2]**.5, alpha=0.5, color=color)
#    ax4.legend(loc="upper right")
#    # fit range marker
#    ax4.axvline(fit_range_PSPS[0], color="gray", linestyle="--")
#    ax4.axvline(fit_range_PSPS[-1], color="gray", linestyle="--")


#    ## correlator ##
#    color = "C0"
#    ax1.set_xlabel(r"source-sink separation $t/a$")
#    ax1.set_ylabel(r"$C_{PSA4}(t)$")     
#    ax1.errorbar(np.arange(1,Nt-1), Ct_mean_PSA4[1:-1], jackknife.variance_jks(Ct_mean_PSA4, Ct_jks_PSA4)[1:-1]**0.5, linestyle="", capsize=3, color=color, label="data")
#    # correlator fit
#    d = 3
#    trange = np.arange(d, Nt-d, 0.01)
#    color = "C1"
#    fy_PSA4 = np.array([model_combined.sinh(t, best_parameter) for t in trange])
#    fy_std_PSA4 = np.array([model_prediction_var(t, best_parameter, best_parameter_cov, lambda x,y: model_combined.parameter_gradient(x,y)[1]) for t in trange])**.5
#    ax1.plot(trange, fy_PSA4, color=color, lw=.5, label=r"$C_{PSA4}(t) = A_1 (e^{-m t} - e^{-m (T-t)})$ - fit")
#    ax1.fill_between(trange, fy_PSA4-fy_std_PSA4, fy_PSA4+fy_std_PSA4, alpha=0.5, color=color)
#    ax1.legend(loc="upper left")
#    # fit range marker
#    ax1.axvline(fit_range_PSA4[0], color="gray", linestyle="--")
#    ax1.axvline(fit_range_PSA4[-1], color="gray", linestyle="--")
#    # set title
#    ax1.set_title(Ct_tag_PSA4)
#    ## local Amplitude ##
#    # data
#    color = "C2"
#    At_func_PSA4 = lambda x, y: effective_amp_sinh(x[d:Nt-d], np.arange(d, Nt-d), y, Nt)
#    At_data_PSA4 = At_func_PSA4(Ct_mean_PSA4, best_parameter[2])
#    At_jks_PSA4 = {}
#    for j in best_parameter_jks:
#        At_jks_PSA4[j] = At_func_PSA4(Ct_jks_PSA4[j], best_parameter_jks[j][2]) 
#    At_var_data_PSA4 = jackknife.variance_jks(At_data_PSA4, np.array(list(At_jks_PSA4.values())))
#    ax3.set_xlabel(r"source-sink separation $t/a$")
#    ax3.set_ylabel("$m/GeV$")
#    ax3.errorbar(np.arange(d, Nt-d), At_data_PSA4, At_var_data_PSA4**.5, linestyle="", capsize=3, color=color, label=r"$A_1(t)$ data")
#    # fit
#    color = "C3"
#    At_fit_PSA4, At_var_fit_PSA4 = local_amp(trange, model_combined.sinh, "sinh", best_parameter, best_parameter_jks, Nt, mass_idx=2)
#    ax3.set_xlabel(r"source-sink separation $t/a$")
#    ax3.set_ylabel("$A$")
#    ax3.plot(trange[1:-1], At_fit_PSA4, color=color, lw=.5, label=r"$A_1(t) = \frac{C_{PSA4}(t)}{e^{-m t} - e^{-m (T-t)}}$")
#    ax3.fill_between(trange[1:-1], At_fit_PSA4-At_var_fit_PSA4**.5, At_fit_PSA4+At_var_fit_PSA4**.5, alpha=0.5, color=color)
#    # best amplitude
#    color = "C4"
#    A_arr_PSA4 = np.array([best_parameter[1] for t in trange])
#    ax3.plot(trange, A_arr_PSA4, color=color, lw=.5, label=r"$A_1 = $" + f"{best_parameter[1]:.4g} +- {best_parameter_cov[1][1]**.5:.4g}")
#    ax3.fill_between(trange, A_arr_PSA4-best_parameter_cov[1][1]**.5, A_arr_PSA4+best_parameter_cov[1][1]**.5, alpha=0.5, color=color)
#    ax3.legend(loc="upper right")
#    # fit range marker
#    ax3.axvline(fit_range_PSA4[0], color="gray", linestyle="--")
#    ax3.axvline(fit_range_PSA4[-1], color="gray", linestyle="--")
#    ## local mass ## 
#    # data
#    color = "C2"
#    mt_func_PSA4 = lambda x: effective_mass_acosh(x, tmin=1, tmax=Nt-1)
#    mt_data_PSA4 = mt_func_PSA4(Ct_mean_PSA4)
#    mt_var_data_PSA4 = db.sample_jackknife_variance(Ct_tag_PSA4, binsize, mt_func_PSA4)
#    ax5.set_xlabel(r"source-sink separation $t/a$")
#    ax5.set_ylabel("$m/GeV$")
#    ax5.errorbar(np.arange(1, Nt-1), mt_data_PSA4, mt_var_data_PSA4**.5, linestyle="", capsize=3, color=color, label=r"$m(t)$ data")
#    # fit 
#    color = "C3"
#    mt_fit_PSA4, mt_var_fit_PSA4 = local_mass(trange, model_combined.sinh, best_parameter, best_parameter_jks)
#    ax5.plot(trange[1:-1], mt_fit_PSA4, color=color, lw=.5, label=r"$m(t) = cosh^{-1}\left( \frac{C_{PSA4}(t+1) + C_{PSA4}(t-1)}{2 C_{PSA4}(t)} \right)$")
#    ax5.fill_between(trange[1:-1], mt_fit_PSA4-mt_var_fit_PSPS**.5, mt_fit_PSA4+mt_var_fit_PSA4**.5, alpha=0.5, color=color)
#    # mass from fit
#    color = "C4"
#    m_arr = np.array([best_parameter[2] for t in trange])
#    ax5.plot(trange, m_arr, color=color, lw=.5, label=r"$m = $" + f"{best_parameter[2]:.4g} +- {best_parameter_cov[2][2]**.5:.4g}")
#    ax5.fill_between(trange, m_arr-best_parameter_cov[2][2]**.5, m_arr+best_parameter_cov[2][2]**.5, alpha=0.5, color=color)
#    ax5.legend(loc="upper right")
#    # fit range marker
#    ax5.axvline(fit_range_PSA4[0], color="gray", linestyle="--")
#    ax5.axvline(fit_range_PSA4[-1], color="gray", linestyle="--")

#    plt.suptitle(f"combined fit")            
#    plt.tight_layout()
#    plt.plot()