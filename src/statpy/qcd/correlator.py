import numpy as np
from statpy.log import message
from statpy.fitting.core import fitV1, fitMultipleV1
from numba import jit

#########################################################################################################################
################################################# EFFECTIVE MASS CURVES #################################################
#########################################################################################################################

# open boundary conditions
def effective_mass_log(Ct, tmin, tmax):
    return np.array([np.log(Ct[t] / Ct[t+1]) for t in range(tmin,tmax)]) 

# periodic boundary conditions
def effective_mass_acosh(Ct, tmin, tmax):
    return np.array([np.arccosh(0.5 * (Ct[t+1] + Ct[t-1]) / Ct[t]) for t in range(tmin,tmax)])





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

@jit(nopython=True)
def cosh_chi2(t, p, y, W, Nt):
    return ( (p[0] * ( np.exp(-p[1]*t) + np.exp(-p[1]*(Nt-t)) )) - y ) @ W @ ( (p[0] * ( np.exp(-p[1]*t) + np.exp(-p[1]*(Nt-t)) )) - y )


################################ exp model to fit correlator with open boundary conditions ##############################

# f(t) = A * exp(-mt); A = p[0]; m = p[1] 
class exp_model:
    def __init__(self):
        pass   
    def __call__(self, t, p):
        return p[0] * np.exp(-p[1]*t)
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t), p[0] * np.exp(-p[1]*t) * (-t)])
    
@jit(nopython=True)
def exp_chi2(t, p, y, W):
    return ( (p[0] * np.exp(-p[1]*t)) - y ) @ W @ ( (p[0] * np.exp(-p[1]*t)) - y )


####################################### const model to fit effective mass plateau #######################################

class const_model:
        def __init__(self):
            pass  
        def __call__(self, t, p):
            return p[0]
        def parameter_gradient(self, t, p):
            return np.array([np.ones_like(t)])
        
@jit(nopython=True)
def const_chi2(t, p, y, W):
    return (p[0] - y) @ W @ (p[0] - y)


######################################## exp + const model to fit effective mass ########################################


class const_plus_exp_model:
        def __init__(self):
            pass  
        def __call__(self, t, p):
            return p[0] + p[1] * np.exp(-p[2]*t)
        def parameter_gradient(self, t, p):
            return np.array([np.ones_like(t), np.exp(-p[2]*t), p[1] * np.exp(-p[2]*t) * (-t)])

@jit(nopython=True)
def const_plus_exp_chi2(t, p, y, W):
    return ( (p[0] + p[1] * np.exp(-p[2]*t)) - y ) @ W @ ( (p[0] + p[1] * np.exp(-p[2]*t)) - y )



################################################ MASS DETERMINATION #####################################################

def effective_mass_curve_fit(db, tag, t0_min, t0_max, dt, tmax, cov, p0, bc, fit_method, fit_params, jks_fit_method, jks_fit_params, binsize, dst_tag, sys_tags=None, verbosity=0):
    assert bc in ["pbc", "obc"]
    model = {"pbc": cosh_model(len(db.database[tag].mean)), "obc": exp_model()}[bc]

    ts = []
    if (dt is not None) and (tmax is None):
        for t0 in range(t0_min, t0_max+1):
            ts.append(np.arange(dt) + t0)
    elif (dt is None) and (tmax is not None):
        for t0 in range(t0_min, t0_max+1):
            ts.append(np.arange(t0, tmax+1))
    else:
        raise AssertionError

    for t in ts: 
        message(f"effective curve fit window: {t}")
        fit_tag = dst_tag + f"curve_fit_t={t[0]}"
        mass_tag = dst_tag + f"={t[0]}"
        fitV1(db, t, tag, cov[t][:,t], p0, model, fit_method, fit_params, jks_fit_method, jks_fit_params, binsize, fit_tag, sys_tags, verbosity-1)
        misc = {}
        for sys in sys_tags:
            misc[f"MEAN_SHIFTED_{sys}"] = db.database[fit_tag].misc[f"MEAN_SHIFTED_{sys}"][1]
            misc[f"SYS_VAR_{sys}"] = db.database[fit_tag].misc[f"SYS_VAR_{sys}"][1]
        db.add_leaf(tag=mass_tag, mean=db.database[fit_tag].mean[1], jks={cfg:val[1] for cfg, val in db.database[fit_tag].jks.items()}, sample=None, misc=misc)
        db.remove_leaf(fit_tag, verbosity=-1)

def effective_mass_plateau_fit(db, ts, tags, cov, model_type, p0, fit_method, fit_params, jks_fit_method, jks_fit_params, binsize, dst_tag, sys_tags=None, verbosity=0):
    model = {"const": const_model(), "const_plus_exp": const_plus_exp_model()}[model_type]
    # add t Leafs
    for t in ts: db.add_leaf(f"tmp_t{t}", mean=t, jks={}, sample=None, misc=None)
    fitMultipleV1(db, [f"tmp_t{t}" for t in ts], tags, cov, p0, model, fit_method, fit_params, jks_fit_method, jks_fit_params, binsize, dst_tag, sys_tags, verbosity)
    # cleanup Leafs
    for t in ts:
        db.remove_leaf(f"tmp_t{t}", verbosity=-1)

def spectroscopy(db, tag, bc, t0_min, t0_max, dt, tmax, effective_mass_model_type, ts, p0, binsize, fit_method="Nelder-Mead", fit_params={"tol":1e-11, "maxiter":5000}, jks_fit_method="Nelder-Mead", jks_fit_params={"tol":1e-11, "maxiter":5000}, verbosity=-1):
    effective_mass_curve_fit(db, tag, t0_min, t0_max, dt, tmax, np.diag(db.jackknife_variance(tag, binsize)), p0, bc,
                             fit_method, fit_params, jks_fit_method, jks_fit_params, binsize, 
                             dst_tag=f"{tag}/am_t", sys_tags=db.get_sys_tags(tag), verbosity=verbosity)
    p0_am_fit = {"const": [p0[1]], "const_plus_exp": [p0[1], 0.1, 0.1]}[effective_mass_model_type]
    am_fit_tag = {"const": f"{tag}/am_t/const_fit", "const_plus_exp": f"{tag}/am_t/const_plus_exp_fit"}[effective_mass_model_type]
    effective_mass_plateau_fit(db, ts, [f"{tag}/am_t={t}" for t in ts], np.diag([db.jackknife_variance(f"{tag}/am_t={t}", binsize) for t in ts]), effective_mass_model_type, p0_am_fit, fit_method, fit_params, jks_fit_method, jks_fit_params, binsize, dst_tag=am_fit_tag, sys_tags=db.get_sys_tags(*[f"{tag}/am_t={t}" for t in ts]), verbosity=verbosity)