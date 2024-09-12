import numpy as np
from statpy.log import message
from statpy.fitting.core import Fitter, ConvergenceError
from statpy.statistics import jackknife, bootstrap
from statpy.fitting.core import fit, print_fit_results, get_pvalue
from numba import njit
from math import isnan
import warnings
from sys import exit

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
def effective_mass_log2(Ct, a=1):
    return np.log(np.roll(Ct, 1) / np.roll(Ct, -1)) / (2*a)

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

    # Wolfgangs hdf5 geometry: 
    # streams -> streams are not averaged at this point, only concatenation
    #   runs
    #     tsrc
    #       ptsrcs
    # need to average over runs, tsrcs and ptsrcs
    def correlator_avg(self, Ct_tags, bc, dst_tag):
        assert bc in ["pbc", "obc"]
        #concat_sample = self.db.combine_sample(Ct_tags, f=lambda *x: np.ma.masked_array(np.concatenate(x, axis=0)))
        self.db.combine_sample(*Ct_tags, f=lambda *x: np.ma.masked_array(np.concatenate(x, axis=0)), 
                               dst_tag=f"{Ct_tags[0].split('/')[0]}/concat")

            
    # Wolfgangs hdf5 geometry 
    def ptsrc_avg(self, Ct_tag, dst_tag):
        self.db.combine_sample(Ct_tag, f=lambda x: np.mean(x, axis=0), dst_tag=dst_tag)

    def tsrc_avg(self, Ctsrc_tags, dst_tag, tbulk=None, antiperiodic=False):
        # extract tsrc positions - don't sort!
        src_positions = [int(k.split("_")[4].split("tsrc")[1]) for k in Ctsrc_tags]
        # determine tbulk
        tbulk = tbulk if tbulk is not None else np.arange(min(src_positions), max(src_positions)+1)
        message(f"Perform tsrc average over all srcs in tbulk = [[{tbulk[0]},{tbulk[-1]}]].")
        assert len(src_positions) == len(Ctsrc_tags)
        max_len, tmax_srcs_fw, valid_mask_fw, tmax_srcs_bw, valid_mask_bw, valid_srcs = self._get_bulk_tsrcs(src_positions, tbulk)
        combined_sample = self.db.combine_sample(*Ctsrc_tags, 
                                                 f=lambda *Cts: self._avg_obc_srcs(max_len, tmax_srcs_fw, valid_mask_fw, tmax_srcs_bw, valid_mask_bw, *Cts, antiperiodic=antiperiodic))
        self.db.add_leaf(tag=dst_tag, mean=None, jks=None, sample=combined_sample, misc={"tsrcs": valid_srcs, "tbulk":tbulk, "antiperiodic":antiperiodic})

    def _get_bulk_tsrcs(self, srcs, tbulk):
        tmin = tbulk[0]; tmax = tbulk[-1]
        max_len = tmax - tmin + 1
        # get tmax for forward and backward average
        tmax_srcs_fw = tmax + 1 - np.array(srcs) 
        tmax_srcs_bw = np.array(srcs) - tmin + 1
        # create masks for positive entries
        valid_mask_fw = (tmax_srcs_fw > 0) & (tmax_srcs_fw <= max_len)
        valid_mask_bw = (tmax_srcs_bw > 0) & (tmax_srcs_bw <= max_len)
        # print averaged tsrcs
        valid_srcs = np.array(srcs)[np.where(valid_mask_fw)[0]]
        message(f"---> {sorted(valid_srcs)}")
        return max_len, tmax_srcs_fw, valid_mask_fw, tmax_srcs_bw, valid_mask_bw, valid_srcs
    
    def _avg_obc_srcs(self, max_len, tmax_srcs_fw, valid_mask_fw, tmax_srcs_bw, valid_mask_bw, *Cts, antiperiodic=False):
        num_Cts = len(Cts)
        # create masked array
        Cts_ma = np.ma.empty((2 * num_Cts, max_len))
        Cts_ma.mask = True
        # fill masked array with relevant time slices for each source position
        for idx in range(num_Cts):
            Ct = Cts[idx]
            if valid_mask_fw[idx]:
                tmax_src_fw = tmax_srcs_fw[idx]
                Cts_ma[idx, :tmax_src_fw] = Ct[:tmax_src_fw]
            if valid_mask_bw[idx]:
                tmax_src_bw = tmax_srcs_bw[idx]
                Cts_ma[idx+num_Cts, :tmax_src_bw] = np.roll(np.flip(Ct), 1)[:tmax_src_bw]
                if antiperiodic: Cts_ma[idx+num_Cts, 1:tmax_src_bw] *= -1.
        return Cts_ma.mean(axis=0)

    def fold_correlator(self, Ct_tag, antiperiodic=False):
        message(f"Fold correlator {Ct_tag}.")
        self.db.combine_sample(Ct_tag, f=lambda Ct: self._fold(Ct, antiperiodic), dst_tag=f"{Ct_tag}/folded")

    def get_tmax_signal_to_noise(self, Ct_tag, min_stn_val=100):
        message(f"Determine tmax by signal to noise ratio < {min_stn_val}.")
        signal_to_noise = self.db.database[Ct_tag].mean / self.db.jackknife_variance(Ct_tag)**.5
        tmax = next((i for i, x in enumerate(signal_to_noise) if (i > 10) and (x < min_stn_val)), -1)
        if tmax == -1:
            message(f"Signal to noise ratio < {min_stn_val} could not be found. return tmax = -1")
            message(f"Signal to noise ratio: {signal_to_noise}")
        message(f"Found tmax = {tmax}.")
        return tmax

    def _fold(self, arr, antiperiodic=False):
        half = len(arr) // 2
        arr0 = arr[:half]
        arr1 = np.roll(np.flip(arr[half:]), 1) 
        if antiperiodic: arr1 *= -1.
        arr1[0] = arr0[0]
        return np.mean([arr0, arr1], axis=0)

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

    def fit_range_fit(self, tag, binsize, initial_fit_ranges, p0, fit_model, verbosity, Nt=None, MIN_TCRIT_LEN=7):
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
        fit_range_dict = {"tag": None, "mean": None, "jks": None, "sample":None, "misc": None}
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
                if fit_range_dict["mean"] is not None:
                    p0_tmp = fit_range_dict["mean"]
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
                fit_range_dict["tag"] = f"{binned_tag}/fit_range_fit"
                fit_range_dict["mean"] = best_parameter
                fit_range_dict["jks"] = best_parameter_jks
                misc["fit_range_crit"] = t_crit; fit_range = t_crit
                fit_range_dict["misc"] = misc
            message("---------------------------------------------------------------------------------", verbosity) 
            message("---------------------------------------------------------------------------------", verbosity) 
        if correlated_converged: self.db.add_leaf(tag=f"{binned_tag}/correlated_fit_range_mean_fit", mean=best_parameter_correlated, jks=None, sample=None, misc=misc_correlated)
        if fit_range_dict["misc"] is not None:
            fit_range_dict["misc"]["tested_suggested_fit_ranges"] = (initial_fit_ranges, suggested_fit_ranges)
            self.db.add_leaf(**fit_range_dict)
            return fit_range_dict["misc"]["fit_range_crit"], best_parameter
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
                    W_correlated = np.linalg.inv(self.db.jackknife_covariance(binned_tag)[fit_range][:,fit_range])
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
            if "binsize" not in tag and self.bootstrap_available:
                bootstrap_tag = tag.replace("fit", "bootstrap_fit"); lf_bs = self.db.database[bootstrap_tag]  
                self.db.add_leaf(f"{bootstrap_tag}/am", mean=lf_bs.mean[1], jks=None, sample=None, misc={"bss": lf_bs.misc["bss"][:,1]})
    
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
            message("------------------------------ BARE DECAY CONSTANT ------------------------------")
            self.db.combine(f"{binned_tag}/{fit_model_combined}_fit", f=bare_decay_constant, dst_tag=f"{binned_tag}/{fit_model_combined}_fit/f_bare")
            if b == 1 and self.bootstrap_available:
                bootstrap_tag = f"{binned_tag}/{fit_model_combined}_bootstrap_fit"
                f_bare_bss_mean = bare_decay_constant(self.db.database[bootstrap_tag].mean)
                f_bare_bss = self.db.combine_bss(self.db.database[bootstrap_tag].misc["bss"], f=bare_decay_constant)
                self.db.add_leaf(tag=f"{bootstrap_tag}/f_bare", mean=f_bare_bss_mean, jks=None, sample=None, misc={"bss": f_bare_bss})
                f_bare_bs_str = f"         {f_bare_bss_mean:.8f} +- {bootstrap.variance(f_bare_bss)**.5:.8f} (bootstrap)"
            message(f"f_bare = {self.db.database[f'{binned_tag}/{fit_model_combined}_fit/f_bare'].mean:.8f} +- {self.db.jackknife_variance(f'{binned_tag}/{fit_model_combined}_fit/f_bare')**.5:.8f} (jackknife)")
            if b == 1 and self.bootstrap_available: message(f_bare_bs_str)
            message("---------------------------------------------------------------------------------", verbosity) 
            message("---------------------------------------------------------------------------------", verbosity) 
           
def bare_decay_constant(p):
    # p[0] = A_PSPS, p[1] = A_PSA4I, p[2] = m
    return np.sqrt(2.) * p[1] / np.sqrt(p[0] * p[2])
