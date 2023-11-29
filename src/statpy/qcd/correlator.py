#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from ..fitting.core import Fitter, ConvergenceError, fit, fit_multiple, model_prediction_var
from ..statistics import jackknife, bootstrap
from ..database.leafs import Leaf
# import multiprocessing module and overwrite its Pickle class using dill
import dill, multiprocessing
dill.Pickler.dumps, dill.Pickler.loads = dill.dumps, dill.loads
multiprocessing.reduction.ForkingPickler = dill.Pickler
multiprocessing.reduction.dump = dill.dump

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

################ open boundary conditions ###############

# C0(t) = A0 * exp(-mt); A0 = p[0]; m = p[2]
# C1(t) = A1 * exp(-mt); A1 = p[1]; m = p[2]
class combined_exp_model:
    def __init__(self, t0, t1):
        self.t0 = t0
        self.t1 = t1
    def __call__(self, t, p):
        f0 = p[0] * np.exp(-p[2]*self.t0) 
        f1 = p[1] * np.exp(-p[2]*self.t1)
        return np.hstack((f0,f1)) 
    
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

def effective_mass_curve_fit(db, tag, t0_min, t0_max, dt, cov, p0, bc, fit_method, fit_params, jks_fit_method, jks_fit_params, binsize, dst_tag, sys_tags=None, verbosity=0):
   assert bc in ["pbc", "obc"]
   model = {"pbc": cosh_model(len(db.database[tag].mean)), "obc": exp_model()}[bc]
   for t0 in range(t0_min, t0_max):
       t = np.arange(dt) + t0
       if verbosity >=0: db.message(f"fit window: {t}")
       fit(db, t, tag, cov[t][:,t], p0, model, fit_method, fit_params, jks_fit_method, jks_fit_params, binsize, dst_tag + f"={t0}", sys_tags, verbosity)
       db.database[dst_tag + f"={t0}"].mean = db.database[dst_tag + f"={t0}"].mean[1]
       db.database[dst_tag + f"={t0}"].jks = {cfg:val[1] for cfg, val in db.database[dst_tag + f"={t0}"].jks.items()} 
       db.database[dst_tag + f"={t0}"].misc["best_parameter_cov"] = db.database[dst_tag + f"={t0}"].misc["best_parameter_cov"][1][1]
       for sys in sys_tags:
           db.database[dst_tag + f"={t0}"].misc[f"MEAN_SHIFTED_{sys}"] = db.database[dst_tag + f"={t0}"].misc[f"MEAN_SHIFTED_{sys}"][1]
           db.database[dst_tag + f"={t0}"].misc[f"SYS_VAR_{sys}"] = db.database[dst_tag + f"={t0}"].misc[f"SYS_VAR_{sys}"][1]

def effective_mass_const_fit(db, ts, tags, cov, p0, fit_method, fit_params, jks_fit_method, jks_fit_params, binsize, dst_tag, sys_tags=None, verbosity=0):
    model = const_model()
    # add t Leafs
    for t in ts: db.add_Leaf(f"tmp_t{t}", mean=t, jks={}, sample=None, misc=None)
    fit_multiple(db, [f"tmp_t{t}" for t in ts], tags, cov, p0, model, fit_method, fit_params, jks_fit_method, jks_fit_params, binsize, dst_tag, sys_tags, verbosity)
    # cleanup t Leafs
    db.remove(*[f"tmp_t{t}" for t in ts])

def spectroscopy(db, tag, bc, t0_min, t0_max, dt, ts, p0, binsize, fit_method="Nelder-Mead", fit_params={"tol":1e-7, "maxiter":1000}, jks_fit_method="Migrad", jks_fit_params=None, verbosity=-1):
    effective_mass_curve_fit(db, tag, t0_min, t0_max, dt, np.diag(db.jackknife_variance(tag, binsize=1)), p0, bc,
                             fit_method, fit_params, jks_fit_method, jks_fit_params, binsize, 
                             dst_tag=f"{tag}/am_t", sys_tags=db.get_sys_tags(tag), verbosity=verbosity-1)
    
    effective_mass_const_fit(db, ts, [f"{tag}/am_t={t}" for t in ts], np.diag([db.jackknife_variance(f"{tag}/am_t={t}", binsize=1) for t in ts]), p0[1],
                             fit_method, fit_params, jks_fit_method, jks_fit_params, binsize, 
                             dst_tag=f"{tag}/am", sys_tags=db.get_sys_tags(*[f"{tag}/am_t={t}" for t in ts]), verbosity=verbosity)

##############################################################################################################################
##############################################################################################################################
##################################################### LATTICE CHARM ##########################################################
##############################################################################################################################
##############################################################################################################################

class CorrelatorData():
    def __init__(self, mean, jks, model_type, model_parameter_gradient, fit_range, reduced_fit_range, best_parameter, best_parameter_jks):
        self.mean = mean
        self.Nt = len(mean)
        self.jks = jks
        self.var = jackknife.variance_jks(jks)
        self.model_type = model_type
        self.model = self._get_model()
        if model_parameter_gradient is not None:
            self.model_parameter_gradient = model_parameter_gradient
        else:
            self.model_parameter_gradient = self.model.parameter_gradient
        self.fit_range = fit_range
        self.fit_range_dt = np.arange(fit_range[0], fit_range[-1], step=0.01)
        self.reduced_fit_range = reduced_fit_range
        self.best_parameter = best_parameter
        self.best_parameter_jks = best_parameter_jks
        self.best_parameter_cov = jackknife.covariance_jks(np.array(list(best_parameter_jks.values())))

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
        else:
            raise Exception("Model not available")

class LatticeCharmSpectroscopy():
    def __init__(self, db, fit_method="Nelder-Mead", fit_params={"maxiter":1000, "tol":1e-07}, res_fit_method="Migrad", res_fit_params=None, num_proc=None):
        self.db = db
        self.fit_method = fit_method
        self.fit_params = fit_params
        self.res_fit_method = res_fit_method
        self.res_fit_params = res_fit_params
        self.num_proc = num_proc
 
    def obc_tsrc_avg(self, Ctsrc_tags, tmin, tmax, dst_tag, cleanup=True):
        srcs = sorted([int(k.split("_")[4].split("tsrc")[1]) for k in Ctsrc_tags]) 
        A4_in_tag = "A4" in Ctsrc_tags[0]
        self.db.combine_sample(*Ctsrc_tags, f=lambda *Cts: self._avg_obc_srcs(srcs, tmin, tmax, *Cts, antiperiodic=A4_in_tag), dst_tag=dst_tag,
                               sorting_key=lambda x: int(x[0].split("-")[-1]))
        if cleanup:
            self.db.remove(*Ctsrc_tags)
        self.db.init_sample_means(dst_tag)
        self.db.init_sample_jks(dst_tag)

    def fit_PSPS(self, tag, binsize, fit_ranges, p0, bc, make_plot=True, PSPS_spectroscopy=False, figsize=None, Ct_scale=None, verbosity=0):
        if bc == "pbc":
            fit_range_model_type = "double-cosh"
            spectroscopy_model_type = "cosh"
        elif bc == "obc":
            fit_range_model_type = "double-exp"
            spectroscopy_model_type = "exp"
        else:
            raise ValueError(bc)         
        self.db.message("-------------------- DETERMINE FIT RANGE FOR PSPS CORRELATOR --------------------")  
        self.fit_range_PSPS, self.p0_PSPS  = self._get_fit_range(tag, binsize, fit_ranges, p0, fit_range_model_type, 
                                                  self.fit_method, self.fit_params, self.res_fit_method, self.res_fit_params, make_plot, figsize, Ct_scale, verbosity)   
        self.db.message("---------------------------------------------------------------------------------") 
        self.db.message(f"REDUCED PSPS FIT RANGE: {self.fit_range_PSPS}")
        if PSPS_spectroscopy:
            self.db.message("---------------------------------------------------------------------------------") 
            self.db.message("---------------------------------------------------------------------------------") 
            self.db.message("---------------------------------------------------------------------------------\n") 
            self.db.message("------------------ FIT PSPS CORRELATOR WITH REDUCED FIT RANGE --------------------") 
            self.A_PSPS_single, self.m_PSPS_single, self.jks_PSPS_single = self._spectroscopy(tag, binsize, self.fit_range_PSPS, self.p0_PSPS, spectroscopy_model_type, False,
                                                                                        self.fit_method, self.fit_params, self.res_fit_method, self.res_fit_params, make_plot, figsize, verbosity)
 
    def determine_PSA4I(self, tag_PSPS_sml, tag_PSA4_sml, beta):
        # https://arxiv.org/pdf/1502.04999.pdf
        def compute_cA(beta):
            p0 = 9.2056; p1 = -13.9847
            return - 0.006033 * 6./beta * (1 + np.exp(p0 + p1*beta/6.))
        def derivative(f):
            return 0.5 * (np.roll(f, -1) - np.roll(f, 1)) 
        def compute_PS_A4I(PS_A4, PS_PS, beta):
            PS_A4I = PS_A4 - compute_cA(beta) * derivative(PS_PS)
            PS_A4I[0] = 0.; PS_A4I[-1] = 0
            return PS_A4I
        tag_PSA4I = tag_PSA4_sml.replace("PSA4", "PSA4I")
        self.db.combine_sample(tag_PSA4_sml, tag_PSPS_sml, f=lambda x,y: compute_PS_A4I(x, y, beta), dst_tag=tag_PSA4I)
        self.db.init_sample_means(tag_PSA4I)
        self.db.init_sample_jks(tag_PSA4I)

    def fit_PSA4I(self, tag, binsize, fit_ranges, p0, bc, make_plot=True, PSA4I_spectroscopy=False, figsize=None, Ct_scale=None, verbosity=0):
        if bc == "pbc":
            fit_range_model_type = "double-sinh"
            spectroscopy_model_type = "sinh"
        elif bc == "obc":
            fit_range_model_type = "double-exp"
            spectroscopy_model_type = "exp"
        else:
            raise ValueError(bc)         
        self.db.message("-------------------- DETERMINE FIT RANGE FOR PSA4I CORRELATOR --------------------")  
        self.fit_range_PSA4I, self.p0_PSA4I  = self._get_fit_range(tag, binsize, fit_ranges, p0, fit_range_model_type, 
                                                  self.fit_method, self.fit_params, self.res_fit_method, self.res_fit_params, make_plot, figsize, Ct_scale, verbosity) 
        self.db.message("---------------------------------------------------------------------------------") 
        self.db.message(f"REDUCED FIT RANGE: {self.fit_range_PSA4I}")
        if PSA4I_spectroscopy:
            self.db.message("---------------------------------------------------------------------------------") 
            self.db.message("---------------------------------------------------------------------------------") 
            self.db.message("---------------------------------------------------------------------------------\n") 
            self.db.message("------------------ FIT PSA4I CORRELATOR WITH REDUCED FIT RANGE --------------------") 
            self.A_PSA4I_single, self.m_PSA4I_single, self.jks_PSA4I_single = self._spectroscopy(tag, binsize, self.fit_range_PSA4I, self.p0_PSA4I, spectroscopy_model_type, False, self.fit_method, self.fit_params, self.res_fit_method, self.res_fit_params, make_plot, figsize, verbosity)
    
    def fit_combined(self, tag_PSPS, fit_range_PSPS, tag_PSA4I, fit_range_PSA4I, binsize, p0, bc, correlated=False, make_plot=True, figsize=None, Ct_scale=None, verbosity=0):
        self.db.message("------------------ COMBINED FIT PSPS/PSA4I CORRELATORs WITH REDUCED FIT RANGES --------------------") 
        self.db.message(f"PSPS correlator: {tag_PSPS}")
        self.db.message(f"PSPS - REDUCED FIT RANGE {fit_range_PSPS}") 
        self.db.message(f"PSA4I correlator: {tag_PSA4I}")
        self.db.message(f"PSA4I - REDUCED FIT RANGE {fit_range_PSA4I}") 
        if bc == "pbc":
            model_type_combined = "combined-cosh-sinh"
            model_type_PSPS = "cosh"
            model_type_PSA4I = "sinh"
        elif bc == "obc":
            model_type_combined = "combined-exp"
            model_type_PSPS = "exp"
            model_type_PSA4I = "exp"
        best_lf = Leaf(mean=None, jks=None, sample=None,
                       misc={"fit_range_PSPS":fit_range_PSPS, "fit_range_PSA4I":fit_range_PSA4I, 
                             "model": model_type_combined, "fit_type": {0: "uncorrelated", 1: "correlated"}[int(correlated)], 
                             "best_parameter": {}, "jks":{}, "chi2": {}, "chi2 / dof":{}, "p":{}})
        for b in range(1, binsize+1):
            self.db.message(f"BINSIZE = {b}", verbosity)
            self.db.message("--------------------------------- JACKKNIFE FIT ---------------------------------", verbosity)
            jks_PSPS = self.db.sample_jks(tag_PSPS, b, sorting_key=lambda x: (int(x[0].split("r")[-1].split("-")[0]),int(x[0].split("-")[-1]))) 
            jks_PSA4I = self.db.sample_jks(tag_PSA4I, b, sorting_key=lambda x: (int(x[0].split("r")[-1].split("-")[0]),int(x[0].split("-")[-1]))) 
            mean_PSPS = np.mean(jks_PSPS, axis=0); mean_PSA4I = np.mean(jks_PSA4I, axis=0)
            jks_arr = np.array([np.hstack((jks_PSPS[cfg][fit_range_PSPS], jks_PSA4I[cfg][fit_range_PSA4I])) for cfg in range(len(jks_PSPS))])
            jks = {cfg:jks_arr[cfg] for cfg in range(len(jks_arr))}
            mean = np.mean(jks_arr, axis=0)
            cov = jackknife.covariance_jks(jks_arr) if correlated else np.diag(jackknife.variance_jks(jks_arr))
            Nt = len(mean_PSPS)
            model = self._get_model(model_type_combined, Nt, fit_range_PSPS, fit_range_PSA4I)
            best_parameter, best_parameter_jks, chi2, dof, pval = self._fit(np.hstack((fit_range_PSPS, fit_range_PSA4I)), mean, jks, cov, p0, model, self.fit_method, self.fit_params, self.res_fit_method, self.res_fit_params)
            best_parameter_cov = jackknife.covariance_jks(self.db.as_array(best_parameter_jks, sorting_key=None))
            # store jks fit results in db
            best_lf.misc["best_parameter"][b] = best_parameter
            best_lf.misc["jks"][b] = best_parameter_jks
            best_lf.misc["chi2"][b] = chi2; best_lf.misc["chi2 / dof"][b] = chi2/dof; best_lf.misc["p"][b] = pval
            if b == 1:
                best_lf.mean = best_parameter
                best_lf.jks = best_parameter_jks
            # print jk fit results
            for i in range(len(best_parameter)):
                self.db.message(f"parameter[{i}] = {best_parameter[i]} +- {best_parameter_cov[i][i]**0.5} (jackknife)", verbosity)
            self.db.message(f"chi2 / dof = {chi2} / {dof} = {chi2/dof}, i.e., p = {pval}", verbosity)

            # perform bootstrap fit for binsize = 1
            if b == 1:
                self.db.message("--------------------------------- BOOTSTRAP FIT ---------------------------------", verbosity)
                bss_PSPS = self.db.database[tag_PSPS].misc["bss"]; bss_PSA4I = self.db.database[tag_PSA4I].misc["bss"]
                bss = np.array([np.hstack((bss_PSPS[k][fit_range_PSPS],bss_PSA4I[k][fit_range_PSA4I])) for k in range(len(bss_PSPS))])
                mean_bss = np.mean(bss, axis=0)
                cov_bss = bootstrap.covariance_bss(bss) if correlated else np.diag(bootstrap.variance_bss(bss))
                best_parameter_bs, best_parameter_bss, chi2_bss, dof_bss, pval_bss = self._fit(np.hstack((fit_range_PSPS, fit_range_PSA4I)), mean, bss, cov_bss, p0, model, self.fit_method, self.fit_params, self.res_fit_method, self.res_fit_params)
                best_parameter_cov_bss = bootstrap.covariance_bss(best_parameter_bss)
                # print bs fit results
                for i in range(len(best_parameter_bs)):
                    self.db.message(f"parameter[{i}] = {best_parameter_bs[i]} +- {best_parameter_cov_bss[i][i]**0.5} (bootstrap)", verbosity)
                self.db.message(f"chi2 / dof = {chi2_bss} / {dof_bss} = {chi2_bss/dof_bss}, i.e., p = {pval_bss}", verbosity)
                best_lf.misc["bss"] = best_parameter_bss 
            self.db.message("---------------------------------------------------------------------------------", verbosity) 
            self.db.message("---------------------------------------------------------------------------------", verbosity) 
            # plot fit result for binsize b = B
            if b == binsize:
                best_parameter_PSPS = np.array([best_parameter[0], best_parameter[2]])
                best_parameter_jks_PSPS = {cfg:np.array([best_parameter_jks[cfg][0], best_parameter_jks[cfg][2]]) for cfg in best_parameter_jks}
                Ct_data_PSPS = CorrelatorData(mean_PSPS, jks_PSPS, model_type_PSPS, None, fit_range_PSPS, None, best_parameter_PSPS, best_parameter_jks_PSPS)
                best_parameter_PSA4I = np.array([best_parameter[1], best_parameter[2]])
                best_parameter_jks_PSA4I = {cfg:np.array([best_parameter_jks[cfg][1], best_parameter_jks[cfg][2]]) for cfg in best_parameter_jks}
                Ct_data_PSA4I = CorrelatorData(mean_PSA4I, jks_PSA4I, model_type_PSA4I, None, fit_range_PSA4I, None, best_parameter_PSA4I, best_parameter_jks_PSA4I)
                if make_plot:
                    lcp = LatticeCharmPlots()
                    lcp.make_plot(Ct_data_PSPS, Ct_data_PSA4I, f"Combined fit for {tag_PSPS} and {tag_PSA4I} ", figsize, Ct_scale)
        self.db.database[f"{tag_PSPS};{tag_PSA4I.split('/')[1]}/combined_fit"] = best_lf
        # store mass as separate leaf
        m_mean = self.db.database[f"{tag_PSPS};{tag_PSA4I.split('/')[1]}/combined_fit"].mean[2]
        m_jks = {}
        for b in range(1, binsize+1):
            m_jks[b] = {j:p[2] for j,p in self.db.database[f"{tag_PSPS};{tag_PSA4I.split('/')[1]}/combined_fit"].misc["jks"][b].items()} 
        m_bss = np.array([p[2] for p in best_lf.misc["bss"]])
        self.db.add_Leaf(tag=f"{tag_PSPS};{tag_PSA4I.split('/')[1]}/combined_fit/m", mean=m_mean, jks=None, sample=None, misc={"jks":m_jks, "bss":m_bss})
        # compute decay constant
        self.db.message(f"BARE DECAY CONSTANT ESTIMATE FOR BINSIZE {binsize}:")
        self.compute_decay_constant(f"{tag_PSPS};{tag_PSA4I.split('/')[1]}/combined_fit", binsize)
            
    def compute_decay_constant(self, combined_fit_tag, B):
        def bare_decay_constant(A_PSPS, A_PSA4I, m):
            return np.sqrt(2.) * A_PSA4I / np.sqrt(A_PSPS * m)
        combined_lf = self.db.database[combined_fit_tag]
        f_bare = bare_decay_constant(*combined_lf.mean)   
        f_bare_jks = {}
        for b in range(1,B+1):
            f_bare_jks[b] = {j:bare_decay_constant(*combined_lf.misc["jks"][b][j]) for j in combined_lf.misc["jks"][b]}
        f_bare_var = jackknife.variance_jks(self.db.as_array(f_bare_jks[B], sorting_key=None))
        f_bare_bss = np.array([bare_decay_constant(*combined_lf.misc["bss"][k]) for k in range(len(combined_lf.misc["bss"]))])
        self.db.message(f"f_bare = sqrt(2) A_A4I / sqrt(m A_PS) = {np.mean(self.db.as_array(f_bare_jks[B], sorting_key=None)):.8f}  +- {f_bare_var**.5:.8f} (jackknife)")
        self.db.add_Leaf(tag=f"{combined_fit_tag}/f_bare", mean=f_bare, jks=None, sample=None, 
                         misc={"jks":f_bare_jks, "bss":f_bare_bss})
 
    def _fit(self, t, y, res, cov, p0, model, fit_method, fit_params, res_fit_method, res_fit_params, num_proc=None):
        if num_proc is None: num_proc = self.num_proc
        # mean fit
        fitter = Fitter(cov, model, fit_method, fit_params)
        try:
            best_parameter, chi2, _ = fitter.estimate_parameters(t, fitter.chi_squared, y, p0)
        except ConvergenceError as ce:
            raise ConvergenceError(f"{ce} for mean")
        # res fits
        if res_fit_method is None: res_fit_method = fit_method; res_fit_params = fit_params
        res_fitter = Fitter(cov, model, res_fit_method, res_fit_params)
        if num_proc is None:
            if isinstance(res, dict):
                best_parameter_res = {}
                idxs = list(res.keys())
            if isinstance(res, list) or isinstance(res, np.ndarray):
                best_parameter_res = np.zeros(shape=(len(res), len(res[0])))
                idxs = range(len(res))
            for idx in idxs:
                best_parameter_res[idx], _, _ = res_fitter.estimate_parameters(t, fitter.chi_squared, res[idx], best_parameter)
        else:
            self.db.message(f"Spawn {num_proc} processes to compute resampled sample", verbosity=self.db.verbosity)
            if isinstance(res, dict):
                def dict_wrapper(idx, y_i):
                    best_parameter_i, _, _ = res_fitter.estimate_parameters(t, fitter.chi_squared, y_i, best_parameter)
                    return idx, best_parameter_i
                with multiprocessing.Pool(num_proc) as pool:
                    best_parameter_res = dict(pool.starmap(dict_wrapper, [(idx, y_k) for idx, y_k in res.items()]))
            if isinstance(res, list) or isinstance(res, np.ndarray):
                def arr_wrapper(y_k):
                    best_parameter_k, _, _ = res_fitter.estimate_parameters(t, fitter.chi_squared, y_k, best_parameter)
                    return best_parameter_k
                with multiprocessing.Pool(num_proc) as pool:
                    best_parameter_res = np.array(pool.starmap(arr_wrapper, [(y_k,) for y_k in res]))
        dof = len(t) - len(best_parameter)
        pval = fitter.get_pvalue(chi2, dof) 
        return best_parameter, best_parameter_res, chi2, dof, pval

    def _get_model(self, model_type, Nt=None, t0=None, t1=None):
        if model_type == "cosh":
            return cosh_model(Nt)
        if model_type == "double-cosh":
            return double_cosh_model(Nt)
        elif model_type == "sinh":
            return sinh_model(Nt)
        elif model_type == "double-sinh":
            return double_sinh_model(Nt)
        elif model_type == "exp":
            return exp_model()
        elif model_type == "double-exp":
            return double_exp_model()
        elif model_type == "combined-cosh-sinh":
            return combined_cosh_sinh_model(Nt, t0, t1)
        elif model_type == "combined-exp":
            return combined_exp_model(t0, t1)
        else:
            raise Exception("Model not available")
        
    def _sort_params(self, p):
        if p[3] <  p[1]: return [p[2], p[3], p[0], p[1]]
        else: return p

    def _get_fit_range(self, tag, binsize, fit_ranges, p0, model_type, fit_method, fit_params, res_fit_method, res_fit_params, make_plot, figsize, Ct_scale, verbosity):
        jks = self.db.sample_jks(tag, binsize, sorting_key=lambda x: (int(x[0].split("r")[-1].split("-")[0]),int(x[0].split("-")[-1])))
        mean = np.mean(jks, axis=0)
        var = jackknife.variance_jks(jks)
        Nt = len(mean)
        model = self._get_model(model_type, Nt)
        fit_range = np.arange(Nt)
        p_fit_range = np.zeros_like(p0)
        for t in fit_ranges:
            self.db.message(f"initial fit range: {t}", verbosity)
            y = mean[t]; y_jks = {cfg:Ct[t] for cfg,Ct in enumerate(jks)}; cov = np.diag(var)[t][:,t]
            try:
                best_parameter, best_parameter_jks, chi2, dof, pval = self._fit(t, y, y_jks, cov, p0, model, fit_method, fit_params, res_fit_method, res_fit_params)
            except ConvergenceError as ce:
                self.db.message(f"{ce} -> jump to next fit range")
                self.db.message("---------------------------------------------------------------------------------", verbosity) 
                self.db.message("---------------------------------------------------------------------------------", verbosity) 
                continue
            best_parameter = self._sort_params(best_parameter)
            best_parameter_jks = {cfg:self._sort_params(best_parameter_jks[cfg]) for cfg in best_parameter_jks}
            best_parameter_cov = jackknife.covariance_jks(self.db.as_array(best_parameter_jks, sorting_key=None)) 
            for i in range(len(best_parameter)):
                self.db.message(f"parameter[{i}] = {best_parameter[i]} +- {best_parameter_cov[i][i]**0.5}", verbosity)
            self.db.message(f"chi2 / dof = {chi2} / {dof} = {chi2/dof}, i.e., p = {pval}", verbosity)
            criterion = np.abs([model(i, [0, 0, best_parameter[2], best_parameter[3]]) for i in t]) < var[t]**.5/4.
            reduced_t = t[criterion]
            if len(reduced_t) < 1:
                self.db.message(f"reduced fit range {reduced_t} -> jump to next fit range", verbosity)
                self.db.message("---------------------------------------------------------------------------------", verbosity) 
                self.db.message("---------------------------------------------------------------------------------", verbosity) 
                continue
            else:
                self.db.message(f"reduced fit range {reduced_t}", verbosity)
            if len(reduced_t) < len(fit_range): 
                fit_range = reduced_t
                p_fit_range = best_parameter[:2]
                self.db.add_Leaf(tag=f"{tag}/fit_range_fit", mean=best_parameter, jks=best_parameter_jks, sample=None, 
                                 misc={"initial_fit_range": t, "fit_range":fit_range, "model": model_type,
                                       "fit_type": "uncorrelated", "chi2":chi2, "chi2 / dof":chi2/dof, "p":pval})
            self.db.message("---------------------------------------------------------------------------------", verbosity) 
            self.db.message("---------------------------------------------------------------------------------", verbosity) 
        if make_plot:
            best_lf = self.db.database[f"{tag}/fit_range_fit"]
            Ct_data = CorrelatorData(mean, jks, model_type, None, 
                                     best_lf.misc["initial_fit_range"], best_lf.misc["fit_range"], best_lf.mean, best_lf.jks)
            LCP = LatticeCharmPlots() 
            LCP.make_plot(Ct_data, None, f"Fit range determination plot for {tag}", figsize, Ct_scale)
        return fit_range, p_fit_range
    
    def _spectroscopy(self, tag, B, fit_range, p0, model_type, correlated, fit_method, fit_params, res_fit_method, res_fit_params, make_plot, figsize, verbosity):
        if correlated: cov = self.db.sample_jackknife_covariance(tag, binsize=1)
        else: cov = np.diag(self.db.sample_jackknife_variance(tag, binsize=1))
        A = {}; A_var = {}
        m = {}; m_var = {}
        for b in range(1, B+1):
            self.db.message(f"BINSIZE = {b}", verbosity)
            jks = self.db.sample_jks(tag, binsize=b)
            mean = np.mean(jks, axis=0)
            Nt = len(mean)
            t = fit_range
            y = mean[t]; y_jks = {cfg:Ct[t] for cfg,Ct in enumerate(jks)}
            model = self._get_model(model_type, Nt)
            best_parameter, best_parameter_jks, chi2, dof, pval = self._fit(t, y, y_jks, cov[t][:,t], p0, model, fit_method, fit_params, res_fit_method, res_fit_params)
            best_parameter_cov = jackknife.covariance_jks(best_parameter_jks)
            A[b] = best_parameter[0]
            m[b] = best_parameter[1]
            for i in range(len(best_parameter)):
                self.db.message(f"parameter[{i}] = {best_parameter[i]} +- {best_parameter_cov[i][i]**0.5}", verbosity)
            self.db.message(f"chi2 / dof = {chi2} / {dof} = {chi2/dof}, i.e., p = {pval}", verbosity)
            self.db.message("---------------------------------------------------------------------------------", verbosity) 
            self.db.message("---------------------------------------------------------------------------------", verbosity) 
            if b == B and make_plot:
                Ct_data = CorrelatorData(mean, jks, model_type, None, t, None, best_parameter, best_parameter_jks)
                lcp = LatticeCharmPlots() 
                lcp.make_plot(Ct_data, None, f"Spectroscopy plot for {tag} with fit range [{t[0],t[-1]}]", figsize)
        return A, m, best_parameter_jks
    
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
    
############################################## PLOTS  ############################################ 

class LatticeCharmPlots():
    def make_plot(self, Ct_data0, Ct_data1=None, title=None, figsize=None, Ct_scale=None):
        if Ct_scale is None: Ct_scale = "linear"
        if Ct_data1 is not None:
            self._combined_plot(Ct_data0, Ct_data1, title, figsize, Ct_scale)
        else:
            self._single_plot(Ct_data0, title, figsize, Ct_scale)
    
    def _single_plot(self, Ct_data, title, figsize, Ct_scale): 
        fig, ((ax0, ax_not), (ax1, ax2)) = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        ax_not.set_axis_off()
        self._add_correlator(ax0, Ct_data, Ct_scale)
        self._add_amplitude(ax1, Ct_data)
        self._add_mass(ax2, Ct_data)
        plt.suptitle(title)
        plt.tight_layout()
        plt.plot()

    def _combined_plot(self, Ct_data0, Ct_data1, title, figsize, Ct_scale):
        fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(nrows=3, ncols=2, figsize=figsize)
        # correlator 1
        self._add_correlator(ax0, Ct_data0, Ct_scale)
        self._add_amplitude(ax2, Ct_data0)
        self._add_mass(ax4, Ct_data0)
        # correlator 2
        self._add_correlator(ax1, Ct_data1, Ct_scale)
        self._add_amplitude(ax3, Ct_data1)
        self._add_mass(ax5, Ct_data1)
        plt.suptitle(title)
        plt.tight_layout()
        plt.plot()

    def _add_fit_range_marker(self, ax, Ct):
        ax.axvline(Ct.fit_range[0], color="gray", linestyle="--", label="fit range")
        ax.axvline(Ct.fit_range[-1], color="gray", linestyle="--")
        if Ct.reduced_fit_range is not None:
            ax.axvline(Ct.reduced_fit_range[0], color="black", linestyle="--", label="reduced fit range")
            ax.axvline(Ct.reduced_fit_range[-1], color="black", linestyle="--")

    def _add_correlator(self, ax, Ct, Ct_scale):
        ax.set_xlabel(r"source-sink separation $t/a$")
        ax.set_ylabel(r"$C(t)$")   
        ax.set_yscale(Ct_scale) 
        # data
        color = "C0"
        ax.errorbar(np.arange(Ct.Nt), Ct.mean, Ct.var**0.5, linestyle="", capsize=3, color=color, label="C(t) data")
        # fit
        color = "C1"
        fy = np.array([Ct.model(t, Ct.best_parameter) for t in Ct.fit_range_dt])
        fy_err = np.array([model_prediction_var(t, Ct.best_parameter, Ct.best_parameter_cov, Ct.model_parameter_gradient) for t in Ct.fit_range_dt])**.5
        ax.plot(Ct.fit_range_dt, fy, color=color, lw=.5, label=self._get_Ct_fit_model_label(Ct))
        ax.fill_between(Ct.fit_range_dt, fy-fy_err, fy+fy_err, alpha=0.5, color=color)
        # misc
        self._add_fit_range_marker(ax, Ct)
        #ax.set_xlim(Ct.fit_range[0]-1, Ct.fit_range[-1]+1)
        ax.set_xlim(0, Ct.Nt)
        ax.legend()

    def _add_amplitude(self, ax, Ct):
        ax.set_xlabel(r"source-sink separation $t/a$")
        ax.set_ylabel("$A$")
        # data   
        color = "C2"
        At = self._effective_amplitude(Ct)(Ct.mean[Ct.fit_range], Ct.fit_range, Ct.best_parameter[1])
        At_var = jackknife.variance_jks(np.array([self._effective_amplitude(Ct)(Ct.jks[j][Ct.fit_range], Ct.fit_range, Ct.best_parameter_jks[j][1]) for j in Ct.best_parameter_jks]))
        ax.errorbar(Ct.fit_range, At, At_var**.5, linestyle="", capsize=3, color=color, label=self._get_effective_amplitude_label(Ct))
        # fit
        color = "C3"
        At_fit, At_fit_var = self._local_amp(Ct)
        ax.plot(Ct.fit_range_dt, At_fit, color=color, lw=.5, label=self._get_amplitude_fit_label(Ct))
        ax.fill_between(Ct.fit_range_dt, At_fit-At_fit_var**.5, At_fit+At_fit_var**.5, alpha=0.5, color=color)
        # misc
        self._add_fit_range_marker(ax, Ct)
        ax.set_xlim(Ct.fit_range[0]-1, Ct.fit_range[-1]+1)
        ax.legend()
        ## best amplitude
        #color = "C4"
        #A_arr = np.array([best_parameter[0] for t in trange])
        #ax1.plot(trange, A_arr, color=color, lw=.5, label=r"$A_0 = $" + f"{best_parameter[0]:.4g} +- {best_parameter_cov[0][0]**.5:.4g}")
        #ax1.fill_between(trange, A_arr-best_parameter_cov[0][0]**.5, A_arr+best_parameter_cov[0][0]**.5, alpha=0.5, color=color)

    def _add_mass(self, ax, Ct):
        ax.set_xlabel(r"source-sink separation $t/a$")
        ax.set_ylabel(r"$m/GeV$")
        # data
        color = "C2"
        mt = self._effective_mass(Ct)(Ct.mean, Ct.fit_range[0], Ct.fit_range[-1]+1)
        mt_var = jackknife.variance_jks(np.array([self._effective_mass(Ct)(Ct.jks[j], Ct.fit_range[0], Ct.fit_range[-1]+1) for j in Ct.best_parameter_jks]))
        ax.errorbar(Ct.fit_range, mt, mt_var**.5, linestyle="", capsize=3, color=color, label=self._get_effective_mass_label(Ct))
        # fit 
        color = "C3"
        mt_fit, mt_fit_var = self._local_mass(Ct) 
        ax.plot(Ct.fit_range_dt[1:-1], mt_fit, color=color, lw=.5, label=self._get_mass_fit_label(Ct)) 
        ax.fill_between(Ct.fit_range_dt[1:-1], mt_fit-mt_fit_var**.5, mt_fit+mt_fit_var**.5, alpha=0.5, color=color)
        # misc 
        self._add_fit_range_marker(ax, Ct)
        ax.set_xlim(Ct.fit_range[0]-1, Ct.fit_range[-1]+1)
        ax.legend()
        ## best mass
        #color = "C4"
        #m_arr = np.array([best_parameter[1] for t in trange])
        #ax2.plot(trange, m_arr, color=color, lw=.5, label=r"$m_0 = $" + f"{best_parameter[1]:.4g} +- {best_parameter_cov[1][1]**.5:.4g}")
        #ax2.fill_between(trange, m_arr-best_parameter_cov[1][1]**.5, m_arr+best_parameter_cov[1][1]**.5, alpha=0.5, color=color)

    ######### AMPLITUDE #########

    def _effective_amplitude(self, Ct):
        if Ct.model_type in ["cosh", "double-cosh"]:
            return lambda C, t, m: effective_amplitude_cosh(C, t, m, Ct.Nt)
        elif Ct.model_type in ["sinh", "double-sinh"]:
            return lambda C, t, m: effective_amplitude_sinh(C, t, m, Ct.Nt)
        elif Ct.model_type in ["exp", "double-exp"]:
            return lambda C, t, m: effective_amplitude_exp(C, t, m)
        else:
            raise Exception("Effective amplitude function not found")

    def _local_amp(self, Ct):
        C_fit_range_dt = np.array([Ct.model(t, Ct.best_parameter) for t in Ct.fit_range_dt])
        At = self._effective_amplitude(Ct)(C_fit_range_dt, Ct.fit_range_dt, Ct.best_parameter[1]) 
        At_var = jackknife.variance_jks(np.array([self._effective_amplitude(Ct)(np.array([Ct.model(t, Ct.best_parameter_jks[j]) for t in Ct.fit_range_dt]), 
                                                      Ct.fit_range_dt, Ct.best_parameter_jks[j][1]) for j in Ct.best_parameter_jks]) )
        return At, At_var

    ######### MASS #########

    def _effective_mass(self, Ct):
        if Ct.model_type in ["cosh", "double-cosh", "sinh", "double-sinh"]:
            return lambda C, tmin, tmax: effective_mass_acosh(C, tmin, tmax)
        elif Ct.model_type in ["exp", "double-exp"]:
            return lambda C, tmin, tmax: effective_mass_log(C, tmin, tmax)
        else:
            raise Exception("Effective mass function not found")
        
    def _local_mass(self, Ct):
        C_fit_range_dt = np.array([Ct.model(t, Ct.best_parameter) for t in Ct.fit_range_dt])
        dt = Ct.fit_range_dt[1] - Ct.fit_range_dt[0]
        mt = self._effective_mass(Ct)(C_fit_range_dt, 1, len(Ct.fit_range_dt)-1) / dt
        mt_var = jackknife.variance_jks(np.array([self._effective_mass(Ct)(np.array([Ct.model(t, Ct.best_parameter_jks[j]) for t in Ct.fit_range_dt]), 
                                                               1, len(Ct.fit_range_dt)-1)/dt for j in Ct.best_parameter_jks]) )
        return mt, mt_var

    ######### LABELS #########

    def _get_Ct_fit_model_label(self, Ct):
        if Ct.model_type == "cosh":
            return r"$C_{fit}(t) = A (e^{-m t} + e^{-m (T-t)})$"
        elif Ct.model_type == "sinh":
            return r"$C_{fit}(t) = A (e^{-m t} - e^{-m (T-t)})$"
        elif Ct.model_type == "double-cosh":
            return r"$C_{fit}(t) = A (e^{-m t} + e^{-m (T-t)}) + A' (e^{-m' t} + e^{-m' (T-t)})$"
        elif Ct.model_type == "double-sinh":
            return r"$C_{fit}(t) = A (e^{-m t} - e^{-m (T-t)}) + A' (e^{-m' t} - e^{-m' (T-t)})$"
        elif Ct.model_type == "exp":
            return r"$C_{fit}(t) = A e^{-m t}$"
        elif Ct.model_type == "double-exp":
            return r"$C_{fit}(t) = A e^{-m t} + A' e^{-m' t}$"
        else:
            raise Exception("Label for Ct fit model not found")
        
    def _get_effective_amplitude_label(self, Ct):
        if Ct.model_type in ["cosh", "double-cosh"]:
            return r"$A(t) = \frac{C(t)}{e^{-m t} + e^{-m (T-t)}}$"
        elif Ct.model_type in ["sinh", "double-sinh"]:
            return r"$A(t) = \frac{C(t)}{e^{-m t} - e^{-m (T-t)}}$"
        elif Ct.model_type in ["exp", "double-exp"]:
            return r"$A(t) = \frac{C(t)}{e^{-m t}}$"
        else:
            raise Exception("Label for effective amplitude not found") 
        
    def _get_amplitude_fit_label(self, Ct):
        if Ct.model_type in ["cosh", "double-cosh"]:
            return r"$A(t) = \frac{C_{fit}(t)}{e^{-m t} + e^{-m (T-t)}}$"
        elif Ct.model_type in ["sinh", "double-sinh"]:
            return r"$A(t) = \frac{C_{fit}(t)}{e^{-m t} - e^{-m (T-t)}}$"
        elif Ct.model_type in ["exp", "double-exp"]:
            return r"$A(t) = \frac{C_{fit}(t)}{e^{-m t}}$"
        else:
            raise Exception("Label for effective amplitude not found") 

    def _get_effective_mass_label(self, Ct):
        if Ct.model_type in ["cosh", "double-cosh", "sinh", "double-sinh"]:
            return r"$m(t) = cosh^{-1}\left( \frac{C(t+1) + C(t-1)}{2 C(t)} \right)$"
        elif Ct.model_type in ["exp", "double-exp"]:
            return r"$m(t) = log\left( \frac{C(t)}{C(t+1)} \right)$"
        else:
            raise Exception("Label for effective mass not found") 
    
    def _get_mass_fit_label(self, Ct):
        if Ct.model_type in ["cosh", "double-cosh", "sinh", "double-sinh"]:
            return r"$m(t) = cosh^{-1}\left( \frac{C_{fit}(t+1) + C_{fit}(t-1)}{2 C_{fit}(t)} \right)$"
        elif Ct.model_type in ["exp", "double-exp"]:
            return r"$m(t) = log\left( \frac{C_{fit}(t)}{C_{fit}(t+1)} \right)$"
        else:
            raise Exception("Label for effective mass not found") 