import numpy as np
from statpy.log import message
from statpy.fitting.core import fit, fit_multiple

########################################### EFFECTIVE MASS CURVES ###########################################

# open boundary conditions
def effective_mass_log(Ct, tmin, tmax):
    return np.array([np.log(Ct[t] / Ct[t+1]) for t in range(tmin,tmax)]) 

# periodic boundary conditions
def effective_mass_acosh(Ct, tmin, tmax):
    return np.array([np.arccosh(0.5 * (Ct[t+1] + Ct[t-1]) / Ct[t]) for t in range(tmin,tmax)])

################################################## FITTING #################################################

#################################### FIT MODELS ############################################

# C(t) = A * [exp(-mt) + exp(-m(Nt-t))]; A = p[0]; m = p[1] 
class cosh_model:
    def __init__(self, Nt):
        self.Nt = Nt 
    def __call__(self, t, p):
        return p[0] * ( np.exp(-p[1]*t) + np.exp(-p[1]*(self.Nt-t)) )
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t) + np.exp(-p[1]*(self.Nt-t)), p[0] * (np.exp(-p[1]*t) * (-t) + np.exp(-p[1]*(self.Nt-t)) * (t-self.Nt))], dtype=object)    

# C(t) = A * exp(-mt); A = p[0]; m = p[1] 
class exp_model:
    def __init__(self):
        pass   
    def __call__(self, t, p):
        return p[0] * np.exp(-p[1]*t)
    def parameter_gradient(self, t, p):
        return np.array([np.exp(-p[1]*t), p[0] * np.exp(-p[1]*t) * (-t)], dtype=object)
    
######### const model to fit effective mass plateau #########
class const_model:
        def __init__(self):
            pass  
        def __call__(self, t, p):
            return p[0]
        def parameter_gradient(self, t, p):
            return np.array([1.0], dtype=object)

########## exp + const model to fit effective mass ##########
class const_plus_exp_model:
        def __init__(self):
            pass  
        def __call__(self, t, p):
            return p[0] + p[1] * np.exp(-p[2]*t)
        def parameter_gradient(self, t, p):
            return np.array([1.0, np.exp(-p[2]*t), p[1] * np.exp(-p[2]*t) * (-t)], dtype=object)

def effective_mass_curve_fit(db, tag, t0_min, t0_max, dt, cov, p0, bc, fit_method, fit_params, jks_fit_method, jks_fit_params, binsize, dst_tag, sys_tags=None, verbosity=0):
   assert bc in ["pbc", "obc"]
   model = {"pbc": cosh_model(len(db.database[tag].mean)), "obc": exp_model()}[bc]
   for t0 in range(t0_min, t0_max):
       t = np.arange(dt) + t0
       if verbosity >=0: message(f"fit window: {t}")
       fit(db, t, tag, cov[t][:,t], p0, model, fit_method, fit_params, jks_fit_method, jks_fit_params, binsize, dst_tag + f"={t0}", sys_tags, verbosity)
       db.database[dst_tag + f"={t0}"].mean = db.database[dst_tag + f"={t0}"].mean[1]
       db.database[dst_tag + f"={t0}"].jks = {cfg:val[1] for cfg, val in db.database[dst_tag + f"={t0}"].jks.items()} 
       db.database[dst_tag + f"={t0}"].misc["best_parameter_cov"] = db.database[dst_tag + f"={t0}"].misc["best_parameter_cov"][1][1]
       for sys in sys_tags:
           db.database[dst_tag + f"={t0}"].misc[f"MEAN_SHIFTED_{sys}"] = db.database[dst_tag + f"={t0}"].misc[f"MEAN_SHIFTED_{sys}"][1]
           db.database[dst_tag + f"={t0}"].misc[f"SYS_VAR_{sys}"] = db.database[dst_tag + f"={t0}"].misc[f"SYS_VAR_{sys}"][1]

def effective_mass_fit(db, ts, tags, cov, model_type, p0, fit_method, fit_params, jks_fit_method, jks_fit_params, binsize, dst_tag, sys_tags=None, verbosity=0):
    model = {"const": const_model(), "const_plus_exp": const_plus_exp_model()}[model_type]
    # add t Leafs
    for t in ts: db.add_Leaf(f"tmp_t{t}", mean=t, jks={}, sample=None, misc=None)
    fit_multiple(db, [f"tmp_t{t}" for t in ts], tags, cov, p0, model, fit_method, fit_params, jks_fit_method, jks_fit_params, binsize, dst_tag, sys_tags, verbosity)
    ## cleanup t Leafs
    db.remove(*[f"tmp_t{t}" for t in ts])

def spectroscopy(db, tag, bc, t0_min, t0_max, dt, effective_mass_model_type, ts, p0, binsize, fit_method="Nelder-Mead", fit_params={"tol":1e-7, "maxiter":1000}, jks_fit_method="Migrad", jks_fit_params=None, verbosity=-1):
    effective_mass_curve_fit(db, tag, t0_min, t0_max, dt, np.diag(db.jackknife_variance(tag, binsize=1)), p0, bc,
                             fit_method, fit_params, jks_fit_method, jks_fit_params, binsize, 
                             dst_tag=f"{tag}/am_t", sys_tags=db.get_sys_tags(tag), verbosity=verbosity-1)
    p0_m = {"const": p0[1], "const_plus_exp": [p0[1], 1.0, p0[1]]}[effective_mass_model_type]
    dst_tag = f"{tag}/am"
    fit_tag = {"const": dst_tag, "const_plus_exp": f"{tag}/const_plus_exp_fit"}[effective_mass_model_type]
    effective_mass_fit(db, ts, [f"{tag}/am_t={t}" for t in ts], np.diag([db.jackknife_variance(f"{tag}/am_t={t}", binsize=1) for t in ts]), effective_mass_model_type, p0_m, 
                             fit_method, fit_params, jks_fit_method, jks_fit_params, binsize, dst_tag=fit_tag, sys_tags=db.get_sys_tags(*[f"{tag}/am_t={t}" for t in ts]), verbosity=verbosity)
    db.add_Leaf(tag=dst_tag,
        mean = db.database[fit_tag].mean[0],
        jks = {cfg:val[0] for cfg, val in db.database[fit_tag].jks.items()},
        sample = None,
        misc = {"best_parameter_cov": np.array([[db.database[fit_tag].misc["best_parameter_cov"][0][0]]]),
                "chi2": db.database[fit_tag].misc["chi2"], "dof": db.database[fit_tag].misc["dof"], "pval": db.database[fit_tag].misc["pval"]})
    for sys in db.get_sys_tags(fit_tag):
        db.database[dst_tag].misc[f"MEAN_SHIFTED_{sys}"] = db.database[fit_tag].misc[f"MEAN_SHIFTED_{sys}"][0]
        db.database[dst_tag].misc[f"SYS_VAR_{sys}"] = db.database[fit_tag].misc[f"SYS_VAR_{sys}"][0]