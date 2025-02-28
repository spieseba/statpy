import numpy as np
from statpy.log import message
from statpy.fitting.core import Fitter, ConvergenceError
from statpy.statistics import jackknife, bootstrap
from statpy.fitting.core import fit, print_fit_results, get_pvalue, compute_AIC
from numba import njit
from math import isnan
import re, warnings
from sys import exit

### periodic boundary conditions ###
def effective_mass_acosh1(Ct, ax=0):
    return np.arccosh(0.5 * (np.roll(Ct, -1, axis=ax) + np.roll(Ct, 1, axis=ax)) / Ct)

# spectrum paper
def effective_mass_acosh2(Ct, a=1):
    Nt = len(Ct)
    return np.abs((np.arccosh(np.roll(Ct,a)/Ct[Nt//2]) - np.arccosh(np.roll(Ct,-a)/Ct[Nt//2]))) / (2. * a)

def effective_mass_asinh(Ct):
    Nt = len(Ct)
    eff_m = np.arcsinh(Ct/Ct[Nt-1])
    return np.abs(np.roll(eff_m, -1) - eff_m)

### open boundary conditions ###
def effective_mass_log1(Ct, ax=0):
    return np.log(Ct / np.roll(Ct, -1, axis=ax))

# spectrum paper 
def effective_mass_log2(Ct, ax=0):
    return np.log(np.roll(Ct, 1, axis=ax) / np.roll(Ct, -1, axis=ax)) / 2

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

fit_model_dict = {
    "cosh": "A * [exp(-mt) + exp(-m(Nt-t))]; A = p[0]; m = p[1]",
    "sinh": "A * [exp(-mt) - exp(-m(Nt-t))]; A = p[0]; m = p[1]",
    "exp": "A * exp(-mt); A = p[0]; m = p[1]",
    "double-cosh": "A0 * [exp(-m0t) + exp(-m0(Nt-t))] + A1 * [exp(-m1t) + exp(-m1(Nt-t))]; A0 = p[0], m0 = p[1], A1 = p[2]; m1 = p[3]",
    "double-sinh": "A0 * [exp(-m0t) - exp(-m0(Nt-t))] + A1 * [exp(-m1t) - exp(-m1(Nt-t))]; A0 = p[0], m0 = p[1], A1 = p[2]; m1 = p[3]",
    "double-exp": "A0 * exp(-m0t) + A1 * exp(-m1t); A0 = p[0], m0 = p[1], A1 = p[2]; m1 = p[3]",
    "combined-cosh-sinh": "A0 * [exp(-mt) + exp(-m(Nt-t))], A1 * [exp(-mt) - exp(-m(Nt-t))]; A0 = p[0], A1 = p[1], m = p[2]",
    "combined-exp-exp": "A0 * exp(-mt), A1 * exp(-mt); A0 = p[0], A1 = p[1], m = p[2]",
}

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


####################################### const plus exp model to fit effective mass plateau #######################################

def const_plus_exp(t, p):
    return p[0] * np.exp(-p[1] * t) + p[2]

def const_plus_exp_chi2(t, p, y, W):
    model = p[0] * np.exp(-p[1] * t) + p[2]
    return (model - y) @ W @ (model - y)

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
    def __init__(self, db, fit_method="Nelder-Mead", fit_params={"maxiter":1000, "tol":1e-07}, res_fit_method=None, res_fit_params=None, bootstrap_available=True):
        self.db = db
        self.fit_method = fit_method
        self.fit_params = fit_params
        self.res_fit_method = self.fit_method if res_fit_method is None else res_fit_method
        self.res_fit_params = self.fit_params if res_fit_params is None else res_fit_params
        self.bootstrap_available = bootstrap_available

    def combine_runs(self, sorted_correlator_tags, stream_tags, corr_types, tsrcs):
        pseudo_correlator_tags = {st:{ct:{} for ct in corr_types} for st in stream_tags}
        for st in stream_tags:
            message(f"Combine runs for stream tag {st}")
            for ct in corr_types:
                message(f"--- {ct}")
                tags_to_be_combined = {tsrc: [t for t in sorted_correlator_tags[st][ct] if f"tsrc{tsrc}" in t] for tsrc in tsrcs[st]}
                pseudo_tags = []
                for tsrc in tags_to_be_combined:
                    pseudo_tag = f"{st}/{tags_to_be_combined[tsrc][0].split("/")[-1]}"; pseudo_tags.append(pseudo_tag)
                    self.db.combine_sample(*tags_to_be_combined[tsrc], f=lambda *x: np.concatenate(x, axis=0), dst_tag=pseudo_tag)
                pseudo_correlator_tags[st][ct] = pseudo_tags
        return pseudo_correlator_tags
    
    def correlator_avg_pbc(self, Ct_tag, dst_tag):
        assert isinstance(Ct_tag, str)
        self.db.combine_sample(Ct_tag, f=lambda x: np.mean(x, axis=0), dst_tag=dst_tag)

    def correlator_avg_obc(self, Ct_tags, tbulk, dst_tag, antiperiodic=False):
        message(f"Perform obc tsrc average over all srcs in tbulk = [[{tbulk[0]},{tbulk[-1]}]] with correlator tags: {Ct_tags}")
        tsrcs = [int(re.search(r'tsrc(\d+)', t)[1]) for t in Ct_tags]
        assert len(Ct_tags) == len(tsrcs)
        Ct_tags_in_bulk = []; tsrcs_in_bulk = []
        for Ct_tag, tsrc in zip(Ct_tags, tsrcs):
            if (tsrc >= tbulk[0]) and (tsrc <= tbulk[-1]):
                Ct_tags_in_bulk.append(Ct_tag)
                tsrcs_in_bulk.append(tsrc)
        message(f"tsrcs in bulk: {tsrcs_in_bulk}")
        tmax_fw, tmax_bw = self._get_tmax_fw_bw(tsrcs_in_bulk, tbulk) # these values can be used directly for time slices
        for src_idx, Ct_tag in enumerate(Ct_tags_in_bulk):
            self.db.combine_sample(Ct_tag, f=lambda Ct: self._get_masked_Ct(Ct, tmax_fw[src_idx], tmax_bw[src_idx], antiperiodic), dst_tag=f"{Ct_tag}/masked", verbosity=-1)
        combined_sample = self.db.combine_sample(*[f"{Ct_tag}/masked" for Ct_tag in Ct_tags_in_bulk], f=lambda *Cts_ma: np.ma.concatenate(Cts_ma, axis=0).mean(axis=0).compressed())
        self.db.add_leaf(tag=dst_tag, mean=None, jks=None, sample=combined_sample, misc={"tsrcs":tsrcs_in_bulk, "tbulk":tbulk, "antiperiodic":antiperiodic})
        for Ct_tag in Ct_tags_in_bulk:
            self.db.remove_leaf(f"{Ct_tag}/masked", verbosity=-1)

    # get tmax for each src in forward and backward direction
    def _get_tmax_fw_bw(self, tsrcs, tbulk):
        tmin = tbulk[0]; tmax = tbulk[-1]
        tmax_fw = tmax + 1 - np.array(tsrcs)
        tmax_bw = np.array(tsrcs) - tmin + 1
        return tmax_fw, tmax_bw

    def _get_masked_Ct(self, Cts, tmax_fw, tmax_bw, antiperiodic):
        num_Cts = Cts.shape[0]
        # create masked array and mask all elements
        Cts_ma = np.ma.empty( (2*num_Cts, Cts.shape[1]) )
        Cts_ma.mask = True
        # fill masked array up to tmax_fw and tmax_bw
        for idx in range(num_Cts):
            Ct = Cts[idx]
            Cts_ma[idx, :tmax_fw] = Ct[:tmax_fw]
            Cts_ma[idx+num_Cts, :tmax_bw] = np.roll(np.flip(Ct), 1)[:tmax_bw] 
            if antiperiodic: Cts_ma[idx+num_Cts, 1:tmax_bw] *= -1
        return Cts_ma 

    def fold_correlator(self, Ct_tag, antiperiodic=False):
        message(f"Fold correlator {Ct_tag}.")
        self.db.combine_sample(Ct_tag, f=lambda Ct: self._fold_correlator(Ct, antiperiodic), dst_tag=f"{Ct_tag}/folded")
    
    def _fold_correlator(self, arr, antiperiodic=False):
        half = len(arr) // 2
        arr0 = arr[:half]
        arr1 = np.roll(np.flip(arr[half:]), 1) 
        if antiperiodic: arr1 *= -1.
        arr1[0] = arr0[0]
        return np.mean([arr0, arr1], axis=0)

    def get_tmax_signal_to_noise(self, mean, var, min_stn_val=100, tmin=15, debug=False):
        signal_to_noise = mean / var**.5 #self.db.database[Ct_tag].mean / self.db.jackknife_variance(Ct_tag)**.5
        tmax = next((i for i, x in enumerate(signal_to_noise) if (i > tmin) and ((x < min_stn_val) or np.isnan(x))), -1)
        if tmax == -1:
            tmax = len(mean) #len(self.db.database[Ct_tag].mean)
            message(f"--- Signal to noise ratio never smaller than {min_stn_val} -> return tmax = len(mt) = {tmax}")
        else:
            message(f"--- Signal to noise ratio smaller than {min_stn_val} for tmax = {tmax} (this value is returned) -> can use all time slices up to t={tmax-1}")
        if debug: message(f"--- Signal to noise ratios: {signal_to_noise}")
        return tmax
    
    def get_tmax_from_deviation(self, mt_mean, mt_var, tmax_stn, n, debug=False):
        message(f"--- Check whether m(t+1) in [m(t) - n sigma, m(t) + n sigma] with n = {n} up to t = {tmax_stn}.")
        diff_in_sigma = abs(mt_mean - np.roll(mt_mean,+1)) / (n * mt_var**.5)
        is_in_range = (diff_in_sigma < 1)[:tmax_stn]
        tmax = next((i for i,x in enumerate(is_in_range) if not x and (i > 15)), -1)
        if tmax == -1:
            tmax = tmax_stn
            message(f"--- m(t+1) lies always in [m(t) - n sigma, m(t) + n sigma] with n = {n} up to t = {tmax_stn} -> return tmax = {tmax}")
        else:
            message(f"--- Found tmax = {tmax} -> can use first {tmax} time slices")
        if debug: message(f"--- abs(m(t) - m(t-1)) / (n*sigma) up to t = {tmax_stn}:\n {diff_in_sigma[:tmax_stn]}")
        return tmax

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

    # automatic p0 guess
    def get_p0_guess(self, tag, binsize, fit_model, fit_range):
        assert fit_model in ["double-cosh", "double-sinh", "double-exp"]
        message(f"Get p0 guess(es) for {fit_model} fit model with {tag} and binsize = {binsize}")
        binned_tag = self.db.add_binned_leaf(tag, binsize)     
        Ct_mean = self.db.database[binned_tag].mean; Nt = len(Ct_mean)
        effective_mass = {"double-cosh": effective_mass_acosh1, "double-sinh": effective_mass_acosh1, "double-exp": effective_mass_log1}[fit_model]
        effective_amplitude = {"double-cosh": effective_amplitude_cosh, "double-sinh": effective_amplitude_sinh, "double-exp": effective_amplitude_exp}[fit_model]
        single_model_func = {"double-cosh": cosh_model(Nt), "double-sinh": sinh_model(Nt), "double-exp": exp_model()}[fit_model]
        # ground state parameters
        t0_probe = slice(Nt//4, Nt//4 + Nt//8) # appears to be more stable when using multiple time slices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning) 
            m0_eff = np.mean(effective_mass(Ct_mean)[t0_probe]) 
            A0_eff = np.mean(effective_amplitude(Ct_mean, m0_eff)[t0_probe]) 
            # use single time slice when mean gives nan
            if isnan(m0_eff) or isnan(A0_eff): 
                t0_probe = Nt//4
                m0_eff = np.mean(effective_mass(Ct_mean)[t0_probe]) 
                A0_eff = np.mean(effective_amplitude(Ct_mean, m0_eff)[t0_probe])
        # excited state parameters
        Ct_ground = single_model_func(np.arange(Nt), [A0_eff,m0_eff]) 
        Ct_excited = Ct_mean - Ct_ground
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            #m1_eff = next(m1 for m1 in effective_mass(Ct_excited)[fit_range] if not isnan(m1))
            #A1_eff = next(A1 for A1 in effective_amplitude(Ct_excited, m1_eff)[fit_range] if not isnan(A1))
            m1_eff = effective_mass(Ct_excited)[fit_range[0]]
            A1_eff = effective_amplitude(Ct_excited, m1_eff)[fit_range[0]]
        message(f"guessed p0 = [{A0_eff}, {m0_eff},  {A1_eff}, {m1_eff}]")
        if m1_eff < 1.2 * m0_eff:
            message("guess for excited state mass too small: p0[2] = abs(p0[2]); p0[3] = 2*p0[1]")
            A1_eff = abs(A1_eff); m1_eff = 2. * m0_eff
            message(f"---> [{A0_eff}, {m0_eff},  {A1_eff}, {m1_eff}]")
        return np.array([A0_eff,m0_eff,A1_eff,m1_eff])

    def excited_contributions_fit(self, tag, binsize, initial_fit_ranges, p0, fit_model, verbosity, Nt=None, MIN_TCRIT_LEN=7):
        def _sort_params(p):
            if p[3] <  p[1]: return [p[2], p[3], p[0], p[1]]
            else: return p
        message(f"CORRELATOR: {tag}")
        if p0 is None:
            message(f"P0 is inferred for each initial fit range automatically.")
        else:
            message(f"P0 = {p0}")
        message(f"BINSIZE = {binsize}", verbosity)
        message(f"{fit_model} MODEL = {fit_model_dict[fit_model]}")
        message("---------------------------------------------------------------------------------", verbosity) 
        binned_tag = self.db.add_binned_leaf(tag, binsize)
        cov = self.db.jackknife_covariance(binned_tag); var = np.diag(cov)        
        Nt = len(self.db.database[binned_tag].mean) if Nt is None else Nt
        model_func = {"double-cosh": double_cosh_model(Nt),
                      "double-sinh": double_sinh_model(Nt),
                      "double-exp": double_exp_model()}[fit_model]       
        excited_contribtions_fit_dict = {"tag": None, "mean": None, "jks": None, "sample":None, "misc": None}
        correlated_fit_dict = {"tag": None, "mean": None, "jks": None, "sample":None, "misc": None}
        suggested_fit_ranges = []
        fit_range = initial_fit_ranges[0]
        for t in initial_fit_ranges:
            message(f"INITIAL FIT RANGE: [[{t[0]},{t[-1]}]]", verbosity)
            message("------------------------------- UNCORRELATED FIT --------------------------------", verbosity)
            W = np.linalg.inv(np.diag(var[t]))
            chi2_func = {"double-cosh": lambda t,p,y: double_cosh_chi2(t, p, y, W, Nt),
                         "double-sinh": lambda t,p,y: double_sinh_chi2(t, p, y, W, Nt),
                         "double-exp": lambda t,p,y: double_exp_chi2(t, p, y, W)}[fit_model]
            p0_tmp = self.get_p0_guess(tag, binsize, fit_model, t) if p0 is None else p0
            if np.isnan(p0_tmp).any():
                if excited_contribtions_fit_dict["mean"] is not None:
                    p0_tmp = excited_contribtions_fit_dict["mean"]
                else:
                    p0_tmp[2] = p0_tmp[0]/2; p0_tmp[3] = 2.0 * p0_tmp[0]
                    if np.isnan(p0_tmp).any():
                        p0_tmp = [1.0 if isnan(p) else p for p in p0_tmp]
                message(f"p0 guess contains NaN, use fit result from previous fit range if available, else use available params to estimate NaNs or default to 1: {p0_tmp}")
            try:
                message(f"p0 for fit: {p0_tmp}")
                best_parameter, best_parameter_jks, misc = fit(self.db, t, binned_tag, p0_tmp, chi2_func, self.fit_method, self.fit_params, self.res_fit_method, self.res_fit_params)
                misc["fit_model"] = fit_model
            except ConvergenceError as ce:
                suggested_fit_ranges.append(None)
                message(f"{ce} -> JUMP TO NEXT FIT RANGE")
                message("---------------------------------------------------------------------------------", verbosity) 
                message("---------------------------------------------------------------------------------", verbosity) 
                continue 
            best_parameter = _sort_params(best_parameter); best_parameter_jks = {cfg: _sort_params(best_parameter_jks[cfg]) for cfg in best_parameter_jks}
            best_parameter_cov = jackknife.covariance(self.db.as_array(best_parameter_jks))
            print_fit_results(best_parameter, best_parameter_cov, misc, verbosity)
            message("------------------------------ CORRELATED MEAN FIT ------------------------------", verbosity)
            # check pos.def.
            pos_def = np.all(np.linalg.eigvals(cov[t][:,t]) > 0)
            message(f"Check positive definiteness of binned covariance matrix for fit range [[{t[0]},{t[-1]}]]: {pos_def}")
            cov_correlated = cov
            if not pos_def:
                cov_unbinned = self.db.jackknife_covariance(tag)
                pos_def_unbinned = np.all(np.linalg.eigvals(cov_unbinned[t][:,t]) > 0)
                message(f"Check positive definiteness of unbinned covariance matrix for fit range [[{t[0]},{t[-1]}]]: {pos_def_unbinned}")
                if pos_def_unbinned:
                    message(f"Use unbinned covariance matrix for correlated fit.")
                    cov_correlated = cov_unbinned
            try:
                W_correlated = np.linalg.inv(cov_correlated[t][:,t])
                chi2_func_correlated = {"double-cosh": lambda t,p,y: double_cosh_chi2(t, p, y, W_correlated, Nt),
                                        "double-sinh": lambda t,p,y: double_sinh_chi2(t, p, y, W_correlated, Nt),
                                        "double-exp": lambda t,p,y: double_exp_chi2(t, p, y, W_correlated)}[fit_model]
                message(f"p0 for fit: {p0_tmp}")
                best_parameter_correlated, _, misc_correlated = fit(self.db, t, binned_tag, p0_tmp, chi2_func_correlated, self.fit_method, self.fit_params, self.res_fit_method, self.res_fit_params, perform_jks_fit=False)
                misc_correlated["fit_model"] = fit_model
                best_parameter_correlated = _sort_params(best_parameter_correlated)
                print_fit_results(best_parameter_correlated, None, misc_correlated, verbosity)
                correlated_converged = True
            except ConvergenceError as ce:
                correlated_converged = False
                message(f"{ce} for correlated mean fit") 
                message("---------------------------------------------------------------------------------", verbosity) 
            message("---------------------------------------------------------------------------------", verbosity) 
            #message(f"excited state contributions: {[model_func(i, [0, 0, best_parameter[2], best_parameter[3]]) for i in t]}")
            #message(f"var**5/4 = {(var[t]**.5)/4.}")
            criterion = np.abs([model_func(i, [0, 0, best_parameter[2], best_parameter[3]]) for i in t]) < (var[t]**.5)/4.
            t_crit = t[criterion]; suggested_fit_ranges.append(t_crit)
            if len(t_crit) < MIN_TCRIT_LEN:
                message(f"DETERMINED FIT RANGE {t_crit} HAS FEWER THAN {MIN_TCRIT_LEN} ELEMENTS", verbosity)
                message(f"---> STORED FIT RANGE IS NOT UPDATED", verbosity)
                message("---------------------------------------------------------------------------------", verbosity) 
                message("---------------------------------------------------------------------------------", verbosity) 
                continue
            else:
                message(f"DETERMINED FIT RANGE [[{t_crit[0]},{t_crit[-1]}]]", verbosity)
            if len(t_crit) <= len(fit_range):
                message(f"---> STORED FIT RANGE IS UPDATED", verbosity)
                excited_contribtions_fit_dict["tag"] = f"{binned_tag}/excited_contributions_fit"
                excited_contribtions_fit_dict["mean"] = best_parameter
                excited_contribtions_fit_dict["jks"] = best_parameter_jks
                misc["fit_range_crit"] = t_crit; fit_range = t_crit
                excited_contribtions_fit_dict["misc"] = misc
                # store tag for correlated mean fit here even if it did not converge to avoid crashing of code
                correlated_fit_dict["tag"] = f"{binned_tag}/correlated_excited_contributions_mean_fit"
                if correlated_converged: 
                    correlated_fit_dict["mean"] = best_parameter_correlated
                    correlated_fit_dict["misc"] = misc_correlated
            message("---------------------------------------------------------------------------------", verbosity) 
            message("---------------------------------------------------------------------------------", verbosity) 
        self.db.add_leaf(**correlated_fit_dict)
        if excited_contribtions_fit_dict["misc"] is not None:
            excited_contribtions_fit_dict["misc"]["tested_suggested_fit_ranges"] = (initial_fit_ranges, suggested_fit_ranges)
            self.db.add_leaf(**excited_contribtions_fit_dict)
            return excited_contribtions_fit_dict["misc"]["fit_range_crit"], best_parameter
        else:
            return None, None 

    def correlator_fit(self, tag, binsize, fit_range, p0, fit_model, Nt=None, verbosity=0):
        message(f"CORRELATOR: {tag}")
        message(f"P0 = {p0}")
        message(f"FIT RANGE {fit_range}") 
        message(f"{fit_model} MODEL = {fit_model_dict[fit_model]}")
        for b in range(1, binsize+1):
            message(f"BINSIZE = {b}", verbosity)
            binned_tag = self.db.add_binned_leaf(tag, b)
            message("--------------------------------- JACKKNIFE FIT ---------------------------------", verbosity)
            var = self.db.jackknife_variance(binned_tag)  
            Nt = len(self.db.database[binned_tag].mean) if Nt is None else Nt
            W = np.linalg.inv(np.diag(var[fit_range]))
            chi2_func = {"cosh": lambda t,p,y: cosh_chi2(t, p, y, W, Nt),
                         "sinh": lambda t,p,y: sinh_chi2(t, p, y, W, Nt),
                         "exp": lambda t,p,y: exp_chi2(t, p, y, W)}[fit_model]
            best_parameter, best_parameter_jks, misc = fit(self.db, fit_range, binned_tag, p0, chi2_func, self.fit_method, self.fit_params, self.res_fit_method, self.res_fit_params)
            misc["fit_model"] = fit_model
            best_parameter_cov = jackknife.covariance(self.db.as_array(best_parameter_jks)) 
            print_fit_results(best_parameter, best_parameter_cov, misc, verbosity)
            if b in [1,binsize]:
                message("------------------------------ CORRELATED MEAN FIT ------------------------------", verbosity)
                try:
                    cov = self.db.jackknife_covariance(binned_tag)[fit_range][:,fit_range]
                    # check pos.def. of covariance matrix
                    pos_def = np.all(np.linalg.eigvals(cov) > 0)
                    if not pos_def:
                        message("Covariance matrix is not positive definite -> try to use unbinned covariance matrix for correlated fit.")
                        cov_unbinned = self.db.jackknife_covariance(tag)[fit_range][:,fit_range]
                        pos_def_unbinned = np.all(np.linalg.eigvals(cov_unbinned) > 0)
                        message(f"Check positive definiteness of unbinned covariance matrix for fit range [[{fit_range[0]},{fit_range[-1]}]]: {pos_def_unbinned}")
                        if pos_def_unbinned:
                            message(f"Use unbinned covariance matrix for correlated fit.")
                            cov = cov_unbinned
                    W_correlated = np.linalg.inv(cov)
                    # check pos.def.
                    chi2_func_correlated = {"cosh": lambda t,p,y: cosh_chi2(t, p, y, W_correlated, Nt),
                                            "sinh": lambda t,p,y: sinh_chi2(t, p, y, W_correlated, Nt),
                                            "exp": lambda t,p,y: exp_chi2(t, p, y, W_correlated)}[fit_model]
                    best_parameter_correlated, _, misc_correlated = fit(self.db, fit_range, binned_tag, p0, chi2_func_correlated, self.fit_method, self.fit_params, self.res_fit_method, self.res_fit_params, perform_jks_fit=False)
                    misc_correlated["fit_model"] = fit_model
                    print_fit_results(best_parameter_correlated, None, misc_correlated, verbosity)
                    self.db.add_leaf(tag=f"{binned_tag}/{fit_model}_correlated_mean_fit", mean=best_parameter_correlated, jks=None, sample=None, misc=misc_correlated)
                except ConvergenceError as ce:
                    message(f"{ce} for correlated mean fit") 
                    message("---------------------------------------------------------------------------------", verbosity) 
            if b == 1 and self.bootstrap_available:
                message("--------------------------------- BOOTSTRAP FIT ---------------------------------", verbosity)
                bss = self.db.bss(binned_tag); mean_bss = self.db.database[binned_tag].mean
                W_bss = np.linalg.inv(np.diag(bootstrap.variance(bss)[fit_range]))
                chi2_func_bss = {"cosh": lambda t,p,y: cosh_chi2(t, p, y, W_bss, Nt),
                                 "sinh": lambda t,p,y: sinh_chi2(t, p, y, W_bss, Nt),
                                 "exp": lambda t,p,y: exp_chi2(t, p, y, W_bss)}[fit_model]
                best_parameter_bmean, best_parameter_bss, misc_bss = self._fit_bootstrap(fit_range, mean_bss, bss, best_parameter, chi2_func_bss)
                best_parameter_bcov = bootstrap.covariance(best_parameter_bss)
                print_fit_results(best_parameter_bmean, best_parameter_bcov, misc_bss)
                misc_bss["fit_model"] = fit_model
                misc_bss["bss"] = best_parameter_bss
                self.db.add_leaf(tag=f"{binned_tag}/{fit_model}_bootstrap_fit", mean=best_parameter_bmean, jks=None, sample=None, misc=misc_bss)
            self.db.add_leaf(tag=f"{binned_tag}/{fit_model}_fit", mean=best_parameter, jks=best_parameter_jks, sample=None, misc=misc)
            self.extract_mass_leafs([f"{binned_tag}/{fit_model}_fit"])
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

    def extract_mass_leafs(self, correlator_fit_tags):
        for tag in correlator_fit_tags:
            lf = self.db.database[tag]
            idx_mass = 1 if "combined" not in tag else 2
            self.db.add_leaf(f"{tag}/am", mean=lf.mean[idx_mass], jks={cfg:jk[idx_mass] for cfg,jk in lf.jks.items()}, sample=None, misc=None)
            if "binsize" not in tag and self.bootstrap_available:
                bootstrap_tag = tag.replace("fit", "bootstrap_fit"); lf_bs = self.db.database[bootstrap_tag]  
                self.db.add_leaf(f"{bootstrap_tag}/am", mean=lf_bs.mean[idx_mass], jks=None, sample=None, misc={"bss": lf_bs.misc["bss"][:,idx_mass]})
    
    def correlator_combined_fit(self, tag_PS, tag_A4I, fit_range_PS, fit_range_A4I, binsize, p0, fit_model_combined, Nt=None, verbosity=0):
        message("------------------ COMBINED CORRELATOR FIT PSPS/PSA4I ---------------------") 
        fit_model_PS = fit_model_combined.split("-")[1]
        fit_model_A4I = fit_model_combined.split("-")[2]
        message(f"PSPS correlator: {tag_PS}")
        message(f"PSPS - FIT RANGE {fit_range_PS}") 
        #message(f"PSPS - model: {fit_model_PS}")
        message(f"PSPS {fit_model_PS} MODEL = {fit_model_dict[fit_model_PS]}")
        message(f"PSA4I correlator: {tag_A4I}")
        message(f"PSA4I - FIT RANGE {fit_range_A4I}") 
        #message(f"PSA4I - model: {fit_model_A4I}")
        message(f"PSA4I {fit_model_A4I} MODEL = {fit_model_dict[fit_model_A4I]}")
        #message(f"combined model: {fit_model_combined}")
        message(f"COMBINED - {fit_model_combined} MODEL = {fit_model_dict[fit_model_combined]}")
        message(f"P0 = {p0}")

        Nt = len(self.db.database[tag_PS].mean) if Nt is None else Nt
        fit_range_combined = np.hstack((fit_range_PS, fit_range_A4I))
        combined_tag = f"{tag_PS};{tag_A4I.split('/')[1]}"
        self.db.combine_sample(tag_PS, tag_A4I, f=lambda x,y: np.hstack((x[fit_range_PS],y[fit_range_A4I])), dst_tag=combined_tag)
        for b in range(1, binsize+1):
            message(f"BINSIZE = {b}", verbosity)
            binned_tag = self.db.add_binned_leaf(combined_tag, b)
            message("--------------------------------- JACKKNIFE FIT ---------------------------------", verbosity)
            var = self.db.jackknife_variance(binned_tag)  
            W = np.linalg.inv(np.diag(var))
            chi2_func = {"combined-cosh-sinh": lambda t,p,y: combined_cosh_sinh_chi2(t[:len(fit_range_PS)], t[len(fit_range_PS):], p, y, W, Nt),
                         "combined-exp-exp": lambda t,p,y: combined_exp_exp_model_chi2(t[:len(fit_range_PS)], t[len(fit_range_PS):], p, y, W)}[fit_model_combined]
            best_parameter, best_parameter_jks, misc = fit(self.db, fit_range_combined, binned_tag, p0, chi2_func, self.fit_method, self.fit_params, self.res_fit_method, self.res_fit_params, eval_offset=False)
            misc["fit_model_PSPS"] = fit_model_PS; misc["fit_model_PSA4I"] = fit_model_A4I; misc["fit_model"] = fit_model_combined
            misc["t_PSPS"] = fit_range_PS; misc["t_PSA4I"] = fit_range_A4I
            best_parameter_cov = jackknife.covariance(self.db.as_array(best_parameter_jks)) 
            print_fit_results(best_parameter, best_parameter_cov, misc, verbosity)
            if b in [1,binsize]:
                message("------------------------------ CORRELATED MEAN FIT ------------------------------", verbosity)
                try:
                    W_correlated = np.linalg.inv(self.db.jackknife_covariance(binned_tag))
                    chi2_func_correlated = {"combined-cosh-sinh": lambda t,p,y: combined_cosh_sinh_chi2(t[:len(fit_range_PS)], t[len(fit_range_PS):], p, y, W_correlated, Nt),
                                            "combined-exp-exp": lambda t,p,y: combined_exp_exp_model_chi2(t[:len(fit_range_PS)], t[len(fit_range_PS):], p, y, W_correlated)}[fit_model_combined]
                    best_parameter_correlated, _, misc_correlated = fit(self.db, fit_range_combined, binned_tag, best_parameter, chi2_func_correlated, self.fit_method, self.fit_params, self.res_fit_method, self.res_fit_params, perform_jks_fit=False, eval_offset=False)
                    misc_correlated["fit_model_PSPS"] = fit_model_PS; misc_correlated["fit_model_PSA4I"] = fit_model_A4I; misc_correlated["fit_model"] = fit_model_combined
                    misc_correlated["t_PSPS"] = fit_range_PS; misc_correlated["t_PSA4I"] = fit_range_A4I
                    print_fit_results(best_parameter_correlated, None, misc_correlated, verbosity)
                    self.db.add_leaf(tag=f"{binned_tag}/{fit_model_combined}_correlated_mean_fit", mean=best_parameter_correlated, jks=None, sample=None, misc=misc_correlated)
                except ConvergenceError as ce:
                    message(f"{ce} for correlated mean fit") 
                    message("---------------------------------------------------------------------------------", verbosity) 
            if b == 1 and self.bootstrap_available:
                message("--------------------------------- BOOTSTRAP FIT ---------------------------------", verbosity)
                bss = self.db.bss(binned_tag); mean_bss = self.db.database[binned_tag].mean
                W_bss = np.linalg.inv(np.diag(bootstrap.variance(bss)))
                chi2_func_bss = {"combined-cosh-sinh": lambda t,p,y: combined_cosh_sinh_chi2(t[:len(fit_range_PS)], t[len(fit_range_PS):], p, y, W_bss, Nt),
                                 "combined-exp-exp": lambda t,p,y: combined_exp_exp_model_chi2(t[:len(fit_range_PS)], t[len(fit_range_PS):], p, y, W_bss)}[fit_model_combined]
                best_parameter_bmean, best_parameter_bss, misc_bss = self._fit_bootstrap(fit_range_combined, mean_bss, bss, best_parameter, chi2_func_bss, eval_offset=False)
                misc_bss["fit_model_PSPS"] = fit_model_PS; misc_bss["fit_model_PSA4I"] = fit_model_A4I; misc_bss["fit_model"] = fit_model_combined
                misc_bss["t_PSPS"] = fit_range_PS; misc_bss["t_PSA4I"] = fit_range_A4I
                best_parameter_bcov = bootstrap.covariance(best_parameter_bss)
                print_fit_results(best_parameter_bmean, best_parameter_bcov, misc_bss)
                misc_bss["bss"] = best_parameter_bss
                self.db.add_leaf(tag=f"{binned_tag}/{fit_model_combined}_bootstrap_fit", mean=best_parameter_bmean, jks=None, sample=None, misc=misc_bss)
            self.db.add_leaf(tag=f"{binned_tag}/{fit_model_combined}_fit", mean=best_parameter, jks=best_parameter_jks, sample=None, misc=misc)
            # extract mass leafs
            self.extract_mass_leafs([f"{binned_tag}/{fit_model_combined}_fit"])
            message("------------------------------ BARE DECAY CONSTANT ------------------------------")
            self.db.combine(f"{binned_tag}/{fit_model_combined}_fit", f=bare_decay_constant, dst_tag=f"{binned_tag}/{fit_model_combined}_fit/afbare")
            if b == 1 and self.bootstrap_available:
                bootstrap_tag = f"{binned_tag}/{fit_model_combined}_bootstrap_fit"
                fbare_bss_mean = bare_decay_constant(self.db.database[bootstrap_tag].mean)
                fbare_bss = self.db.combine_bss(self.db.database[bootstrap_tag].misc["bss"], f=bare_decay_constant)
                self.db.add_leaf(tag=f"{bootstrap_tag}/afbare", mean=fbare_bss_mean, jks=None, sample=None, misc={"bss": fbare_bss})
                fbare_bs_str = f"         {fbare_bss_mean:.8f} +- {bootstrap.variance(fbare_bss)**.5:.8f} (bootstrap)"
            message(f"a*fbare = {self.db.database[f'{binned_tag}/{fit_model_combined}_fit/afbare'].mean:.8f} +- {self.db.jackknife_variance(f'{binned_tag}/{fit_model_combined}_fit/afbare')**.5:.8f} (jackknife)")
            if b == 1 and self.bootstrap_available: message(fbare_bs_str)
            message("---------------------------------------------------------------------------------", verbosity) 
            message("---------------------------------------------------------------------------------", verbosity)


    #### BOUNDARY EFFECTS ####
    def boundary_avg(self, Ct_tags, tmin_excited, binsize, antiperiodic=False, cleanup=False):
        message(f"Perform boundary average over all tsrcs with correlator tags: {Ct_tags}")
        message(f"Excited state contributions expected to be removed at t = {tmin_excited}")
        tsrcs = [int(re.search(r'tsrc(\d+)', t)[1]) for t in Ct_tags]
        assert len(Ct_tags) == len(tsrcs)
        # get effective mass estimate for each source first and then average over sources
        mt_tags = []
        for Ct_tag, tsrc in zip(Ct_tags, tsrcs):
            # get masked Ct at each source
            self.db.combine_sample(Ct_tag, f=lambda Ct: _get_masked_Cts_boundary(Ct, tsrc, tmin_excited).mean(axis=0), dst_tag=f"{Ct_tag}/maskedES")
            binned_Ct_tag = self.db.add_binned_leaf(f"{Ct_tag}/maskedES", binsize)
            # compute effective mass on masked Ct for each source
            mt_tag = f"{binned_Ct_tag}/am_t"; mt_tags.append(mt_tag)
            self.db.combine(binned_Ct_tag, f=lambda Ct: np.nan_to_num(_flip_sign_boundary(effective_mass_log2(Ct), tsrc), nan=0.0, posinf=0.0, neginf=0.0), dst_tag=mt_tag) # set invalid values to zero
            if cleanup:
                self.db.remove_leaf(f"{Ct_tag}/maskedES")
                self.db.remove_leaf(binned_Ct_tag)
        # average effective masses over sources
        dst_tag = re.sub(r'(tsrc)\d+', r'\1None', mt_tags[0])
        self.db.combine(*mt_tags, f=lambda *eff_mass: np.ma.filled(np.ma.masked_equal(eff_mass, 0).mean(axis=0), 0), dst_tag=dst_tag) 
        self.db.combine(dst_tag, f=lambda mt: _fold_boundary(mt, antiperiodic), dst_tag=f"{dst_tag}/folded")
        if cleanup:
            for mt_tag in mt_tags: self.db.remove_leaf(mt_tag)
        return dst_tag

    def boundary_fits(self, mt_folded_tag, t0s, MIN_TCRIT_LEN=1):
        ts = np.arange(self.db.database[mt_folded_tag].mean.shape[0])
        mt_cov = self.db.jackknife_covariance(mt_folded_tag); mt_var = np.diag(mt_cov)
        boundary_fit_dict = {"tag": None, "mean": None, "jks": None, "sample":None, "misc": None}
        correlated_fit_dict = {"tag": None, "mean": None, "jks": None, "sample":None, "misc": None}

        zero_idxs = np.where(self.db.database[mt_folded_tag].mean == 0)[0]
        tmax = ts[zero_idxs[2]] if len(zero_idxs) > 2 else ts[-1] + 1
        initial_fit_ranges = [np.arange(t0, tmax) for t0 in t0s]
        
        AIC_arr = []
        suggested_fit_ranges = []
        boundary_range = initial_fit_ranges[0]
        best_parameters = []; best_parameters_jkss = [] # for AIC
        for fit_range in initial_fit_ranges:    
            message(f"Perform const + exp fit of {mt_folded_tag} with fit range: \n\t [[{fit_range[0]},{fit_range[-1]}]]")
            W = np.diag(1/mt_var[fit_range])
            chi2_func = lambda t,p,y: const_plus_exp_chi2(t,p,y,W)
            p0 = [1, 1, self.db.database[mt_folded_tag].mean[fit_range[-1]]]
            try:
                best_parameter, best_parameter_jks, misc = fit(self.db, fit_range, mt_folded_tag, p0, chi2_func, self.fit_method, self.fit_params, jks_fit_method=self.res_fit_method, jks_fit_params=self.res_fit_params)
            except ConvergenceError as ce:
                AIC_arr.append(None)
                suggested_fit_ranges.append(None)
                best_parameters.append(None); best_parameters_jkss.append(None)
                message(f"{ce} -> JUMP TO NEXT FIT RANGE")
                message("---------------------------------------------------------------------------------") 
                message("---------------------------------------------------------------------------------") 
                continue
            best_parameter_cov = jackknife.covariance(self.db.as_array(best_parameter_jks))
            misc["AIC"] = compute_AIC(misc["chi2"], misc["dof"], len(p0)) #; misc["log[P(M)]"] = -misc["AIC"]/2.0
            print_fit_results(best_parameter, best_parameter_cov, misc)
            message(f"P(M) = exp(-AIC / 2) = exp(- [chi2 - dof + k] / 2) = exp(-{misc['AIC']} / 2)")
            message("------------------------------ CORRELATED MEAN FIT ------------------------------")
            W_correlated = np.linalg.inv(mt_cov[fit_range][:,fit_range])
            chi2_func_correlated = lambda t,p,y: const_plus_exp_chi2(t,p,y,W_correlated)
            try:
                p0_correlated = best_parameter
                message(f"p0 for fit: {p0_correlated}")
                best_parameter_correlated, _, misc_correlated =  fit(self.db, fit_range, mt_folded_tag, p0_correlated, chi2_func_correlated, self.fit_method, self.fit_params, jks_fit_method=self.res_fit_method, jks_fit_params=self.res_fit_params, perform_jks_fit=False)
                print_fit_results(best_parameter_correlated, None, misc_correlated)
                correlated_converged = True
            except ConvergenceError as ce:
                correlated_converged = False
                message(f"{ce} for correlated mean fit") 
                message("---------------------------------------------------------------------------------") 
            message("---------------------------------------------------------------------------------") 
            # test that exponential contribution is small compared to statistical error of the data
            criterion = np.abs([const_plus_exp(i, [best_parameter[0], best_parameter[1], 0]) for i in ts]) < (mt_var**.5)/4.
            t_crit = ts[criterion]
            if len(t_crit) < MIN_TCRIT_LEN:
                message(f"SUGGESTED RANGE WITHOUT BOUNDARY EFFECTS {t_crit} IS CONTAINS LESS THAN {MIN_TCRIT_LEN} ELEMENTS")
                message(f"---> SET P(M) = None")
                AIC_arr.append(None)
                suggested_fit_ranges.append(None)
                best_parameters.append(best_parameter); best_parameters_jkss.append(best_parameter_jks) # wont be used for AIC calculation
                message("---------------------------------------------------------------------------------") 
                message("---------------------------------------------------------------------------------") 
                continue
            AIC_arr.append(misc["AIC"])
            suggested_fit_ranges.append(t_crit)       
            best_parameters.append(best_parameter); best_parameters_jkss.append(best_parameter_jks)
            message(f"SUGGESTED BULK RANGE WITHOUT BOUNDARY EFFECTS [[{t_crit[0]},{t_crit[-1]}]]")
            if len(t_crit) < len(boundary_range):
                message(f"---> STORED BULK RANGE IS UPDATED")
                boundary_range = t_crit
                boundary_fit_dict["tag"] = f"{mt_folded_tag}/const_plus_exp_fit"
                boundary_fit_dict["mean"] = best_parameter
                boundary_fit_dict["jks"] = best_parameter_jks
                misc["boundary_range_fit"] = t_crit
                boundary_fit_dict["misc"] = misc
                # store correlated tag already here s.t. correlated dict can be added even if the correlated fit did not converge
                correlated_fit_dict["tag"] = f"{mt_folded_tag}/correlated_const_plus_exp_fit" 
                if correlated_converged: 
                    correlated_fit_dict["mean"] = best_parameter_correlated
                    correlated_fit_dict["misc"] = misc_correlated
            message("---------------------------------------------------------------------------------") 
            message("---------------------------------------------------------------------------------") 
        # compute boundary end with AIC model average of t0s
        message(f"AIC_arr {AIC_arr}")
        # use only valid AIC values for P(M) calculation
        AIC_valid_idxs = np.array([True if AIC_arr[i] is not None else False for i in range(len(AIC_arr))]) # get all non-None AIC idxs
        AIC_valid = [AIC_arr[i] for i in range(len(AIC_arr)) if AIC_valid_idxs[i]] # get all non-None AIC values
        # Numerically more stable calculation of P(M) using AIC
        AIC_min = np.min(AIC_valid)
        delta_AIC = AIC_valid - AIC_min 
        log_P_M = -delta_AIC / 2.0
        max_log_P_M = np.max(log_P_M)
        P_M_valid = np.exp(log_P_M - max_log_P_M)
        P_M_valid = P_M_valid / np.sum(P_M_valid)
        P_M_arr = np.zeros(len(AIC_arr))
        P_M_arr[AIC_valid_idxs] = P_M_valid
        P_M_arr[~AIC_valid_idxs] = None
        message(f"P_M_arr {P_M_arr}")

        # P_M_arr can contain None values
        # best_parameter too
        filtered_P_M_arr = np.array([p for p in P_M_arr if not np.isnan(p)])
        filtered_t0s_crit = np.array([suggested_fit_range[0] for suggested_fit_range, p in zip(suggested_fit_ranges, P_M_arr) if not np.isnan(p)])

        # compute begin of bulk with AIC model average of t0s
        t_crit_AIC_t0 = np.sum(filtered_t0s_crit * filtered_P_M_arr)
        t_crit_AIC_t0_rounded = int(np.round(t_crit_AIC_t0))
        message(f"BEGIN OF BULK DETERMINED BY AIC MODEL AVERAGE OF T0s: {t_crit_AIC_t0} -> rounded to {t_crit_AIC_t0_rounded}")
        boundary_fit_dict["misc"]["boundary_end_AIC_t0"] = t_crit_AIC_t0_rounded

        # compute boundary end with AIC model average of parameters
        filtered_best_parameters = [bp for bp, p in zip(best_parameters, P_M_arr) if not np.isnan(p)]
        filtered_best_parameters_jkss = [self.db.as_array(bp_jks) for bp_jks, p in zip(best_parameters_jkss, P_M_arr) if not np.isnan(p)]
        best_parameter_AIC = np.sum(filtered_best_parameters * filtered_P_M_arr[:,np.newaxis], axis=0)
        best_parameter_AIC_jks = np.sum(filtered_best_parameters_jkss * filtered_P_M_arr[:,np.newaxis,np.newaxis], axis=0)
        best_parameter_AIC_sys_var = np.sum([ p * (bp - best_parameter_AIC)**2 for bp, p in zip(filtered_best_parameters, filtered_P_M_arr)], axis=0)
        criterion_AIC = np.abs([const_plus_exp(i, [best_parameter_AIC[0], best_parameter_AIC[1], 0]) for i in ts]) < (mt_var**.5)/4.
        t_crit_AIC_params = ts[criterion_AIC][0]
        message(f"BEGIN OF BULK DETERMINED BY AIC MODEL AVERAGE OF BEST_PARAMETERs: {t_crit_AIC_params}")

        # store AIC average of boundary fits
        aic_average_dict = {"tag": f"{mt_folded_tag}/const_plus_exp_fit_AIC_avg", 
                            "mean": None, 
                            "jks": None, 
                            "sample": None, 
                            "misc": {"mean": best_parameter_AIC, "jks": best_parameter_AIC_jks, "sys_var": best_parameter_AIC_sys_var,
                                     "t0s": t0s, "tmax": tmax-1, "P_M_arr": P_M_arr, "suggested_fit_ranges": suggested_fit_ranges,
                                     "bulk_begin_AIC_t0": t_crit_AIC_t0_rounded, "bulk_begin_AIC_params": t_crit_AIC_params
                                     }
                            }
        
        self.db.add_leaf(**aic_average_dict)
        self.db.add_leaf(**correlated_fit_dict)
        self.db.add_leaf(**boundary_fit_dict)


     
           
def bare_decay_constant(p):
    # p[0] = A_PSPS, p[1] = A_PSA4I, p[2] = m
    return np.sqrt(2.) * p[1] / np.sqrt(p[0] * p[2])


# These functions are used to perform the boundary average - defined here to avoid slowdown (don't know why at the moment)
def _get_masked_Cts_boundary(Cts, tsrc, tmin_excited): 
    num_Cts = Cts.shape[0]
    Cts_ma = np.ma.empty((num_Cts, Cts.shape[1]) )
    Cts_ma.mask = True
    Cts_aligned = np.roll(Cts, tsrc, axis=1)
    Cts_ma[:,:tsrc-(tmin_excited-1)] = Cts_aligned[:,:tsrc-(tmin_excited-1)]
    Cts_ma[:,tsrc+tmin_excited:] = Cts_aligned[:,tsrc+tmin_excited:]
    return Cts_ma
    
def _fold_boundary(arr, antiperiodic):
    half = len(arr) // 2
    arr0 = arr[:half]
    arr1 = np.flip(arr[half:])
    if antiperiodic: arr1 *= -1.
    return np.mean([arr0, arr1], axis=0)

def _flip_sign_boundary(arr, tsrc):
    arr[:tsrc] = -arr[:tsrc]
    return arr




