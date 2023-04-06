#!/usr/bin/env python3

import os
from time import time
import numpy as np
from statpy.dbpy import custom_json as json
from statpy.dbpy.leafs import Leaf, SampleLeaf
import statpy as sp

###################################### DATABASE SYSTEM USING LEAFS CONTAINING MEAN AND JKS (SECONDARY OBSERVABLES) ###########################################

class jks_db:
    db_type = "JKS-DB"
    def __init__(self, *args, verbose=True):
        self.t0 = time()
        self.verbose = verbose
        self.database = {}
        for src in args:
            self.add_src(src)

    def add_src(self, src):
        assert os.path.isfile(src)
        self.message(f"load {src}", self.verbose)
        with open(src) as f:
            src_db = json.load(f)
        for t, lf in src_db.items():
            self.database[t] = Leaf(lf.mean, lf.jks)
    
    def message(self, s, verbose=True):
        if verbose: print(f"{self.db_type}:\t\t{time()-self.t0:.6f}s: " + s)
    
    def save(self, dst):
        with open(dst, "w") as f:
            json.dump(self.database, f)

    def print(self):
        s = 'DATABASE CONSISTS OF\n\n\n'
        for tag in self.database.keys():
            s += f'{tag:20s}\n'
        return s

    ################################## FUNCTIONS #######################################

    def apply_f(self, f, tag, dst_tag):
        lf = self.database[tag] 
        lf.mean = f(lf.mean)
        for jks_tag in lf.jks:
            lf.jks[jks_tag] = f(lf.jks[jks_tag])

    def combine(self, f, tag1, tag2, dst_tag):
        lf1 = self.database[tag1]; lf2 = self.database[tag2]
        mean = f(lf1.mean, lf2.mean)
        jks = {}
        for jks_tag in lf1.jks:
            if jks_tag in lf2.jks:
                jks[dst_tag + "-" + jks_tag.split("-")[1]] = f(lf1.jks[jks_tag], lf2.jks[jks_tag])
            else:
                jks[dst_tag + "-" + jks_tag.split("-")[1]] = f(lf1.jks[jks_tag], lf2.mean)
        for jks_tag in lf2.jks:
            if jks_tag not in lf1.jks:
                jks[dst_tag + "-" + jks_tag.split("-")[1]] = f(lf1.mean, lf2.jks[jks_tag])
        self.database[dst_tag] = Leaf(mean, jks)

    ################################## STATISTICS ######################################

    def jackknife_covariance(self):
        # implement delayed m-elimination jackknife covariance
        pass

#    def AMA(self, exact_exact_tag, exact_exact_sample_tag, exact_sloppy_tag, exact_sloppy_sample_tag, sloppy_sloppy_tag, sloppy_sloppy_sample_tag, dst_tag=None, dst_sample_tag=None, store=True):
#        if dst_tag == None: dst_tag = exact_exact_tag
#        if dst_sample_tag == None: dst_sample_tag = exact_exact_sample_tag + "_AMA"
#        def bias(exact, sloppy):
#            return exact - sloppy   
#        exact_exact_jks = self.jackknife_resampling(lambda x: x, exact_exact_tag, exact_exact_sample_tag)
#        exact_sloppy_jks = self.jackknife_resampling(lambda x: x, exact_sloppy_tag, exact_sloppy_sample_tag)
#        sloppy_sloppy_jks = self.jackknife_resampling(lambda x: x, sloppy_sloppy_tag, sloppy_sloppy_sample_tag)
#        b = bias(self.get_data(exact_exact_tag, exact_exact_sample_tag, "mean"), self.get_data(exact_sloppy_tag, exact_sloppy_sample_tag, "mean"))
#        b_jks = {}
#        for cfg in exact_exact_jks:
#            b_jks[cfg] = bias(exact_exact_jks[cfg], exact_sloppy_jks[cfg])
#        b_jkvar = self.jackknife_variance_jks(b, b_jks, dst_tag, dst_sample_tag, dst_cfg_prefix="bias_")
#        m = self.get_data(sloppy_sloppy_tag, sloppy_sloppy_sample_tag, "mean") + b
#        m_jks = {}
#        for cfg in sloppy_sloppy_jks:
#            m_jks[cfg] = sloppy_sloppy_jks[cfg] + b_jks[cfg]
#        m_jkvar = self.jackknife_variance_jks(m, m_jks, dst_tag, dst_sample_tag)
#        if store: 
#            self.add_data(b, dst_tag, dst_sample_tag, "bias")
#            self.add_data(m, dst_tag, dst_sample_tag, "mean")
#        return m, m_jkvar, b, b_jkvar
    

    ############################### SCALE SETTING ###################################

#    def gradient_flow_scale(self, tau_tag, E_tag, sample_tag, jks_tag, verbose=True):
#        tau = self.get_data(tau_tag, sample_tag, "mean")
#        scale = sp.qcd.scale_setting.gradient_flow_scale()
#        # tau0
#        self.apply_f(lambda x: scale.set_sqrt_tau0(tau, x), "E", sample_tag, "mean", "sqrt_tau0")
#        self.apply_f(lambda x: scale.set_sqrt_tau0(tau, x), "E", sample_tag, jks_tag, "sqrt_tau0")
#        sqrt_tau0_var = self.jackknife_variance("sqrt_tau0", sample_tag, jks_tag, dst_data_tag="jkvar"+jks_tag.split("jks")[1])
#        # t0
#        self.apply_f(lambda x: scale.comp_sqrt_t0(x, scale.sqrt_t0_fm), "sqrt_tau0", sample_tag, "mean", "sqrt_t0") 
#        self.apply_f(lambda x: scale.comp_sqrt_t0(x, scale.sqrt_t0_fm), "sqrt_tau0", sample_tag, jks_tag, "sqrt_t0")
#        sqrt_t0_var = self.jackknife_variance("sqrt_t0", sample_tag, jks_tag, dst_data_tag="jkvar"+jks_tag.split("jks")[1])
#        # propagate systematic error of t0
#        self.apply_f(lambda x: scale.comp_sqrt_t0(x, scale.sqrt_t0_fm + scale.sqrt_t0_fm_std), "sqrt_tau0", sample_tag, "mean", "sqrt_t0", "mean+sqrt_t0_fm_std")
#        sqrt_t0_sys_var = (self.database["sqrt_t0"][sample_tag]["mean"] - self.database["sqrt_t0"][sample_tag]["mean+sqrt_t0_fm_std"])**2.0
#        self.add_data(sqrt_t0_sys_var, "sqrt_t0", sample_tag, "sqrt_t0_sys_var")
#        # omega0
#        self.apply_f(lambda x: scale.set_omega0(tau, x), "E", sample_tag, "mean", "omega0")
#        self.apply_f(lambda x: scale.set_omega0(tau, x), "E", sample_tag, jks_tag, "omega0")
#        omega0_var = self.jackknife_variance("omega0", sample_tag, jks_tag, dst_data_tag="jkvar"+jks_tag.split("jks")[1])
#        # w0
#        self.apply_f(lambda x: scale.comp_w0(x, scale.w0_fm), "omega0", sample_tag, "mean", "w0") 
#        self.apply_f(lambda x: scale.comp_w0(x, scale.w0_fm), "omega0", sample_tag, jks_tag, "w0")
#        w0_var = self.jackknife_variance("w0", sample_tag, jks_tag, dst_data_tag="jkvar"+jks_tag.split("jks")[1])
#        # propagate systematic error of w0
#        self.apply_f(lambda x: scale.comp_w0(x, scale.w0_fm + scale.w0_fm_std), "omega0", sample_tag, "mean", "w0", "mean+omega0_fm_std")
#        w0_sys_var = (self.database["w0"][sample_tag]["mean"] - self.database["w0"][sample_tag]["mean+omega0_fm_std"])**2.0
#        self.add_data(w0_sys_var, "w0", sample_tag, "w0_sys_var")
#        if verbose:
#            self.message(f"sqrt(tau0) = {self.get_data('sqrt_tau0', sample_tag, 'mean'):.4f} +- {sqrt_tau0_var**.5:.4f}")
#            self.message(f"omega0 = {self.get_data('omega0', sample_tag, 'mean'):.4f} +- {omega0_var**.5:.4f}")
#            self.message(f"t0/GeV (cutoff) = {self.get_data('sqrt_t0', sample_tag, 'mean'):.4f} +- {sqrt_t0_var**.5:.4f} (STAT) +- {sqrt_t0_sys_var**.5:.4f} (SYS) [{(sqrt_t0_var+sqrt_t0_sys_var)**.5:.4f} (STAT+SYS)]")
#            self.message(f"w0/GeV (cutoff) = {self.get_data('w0', sample_tag, 'mean'):.4f} +- {w0_var**.5:.4f} (STAT) +- {w0_sys_var**.5:.4f} (SYS) [{(w0_var+w0_sys_var)**.5:.4f} (STAT+SYS)]")


################################# DATABASE SYSTEM USING LEAFS CONTAINING SAMPLE, MEAN AND JKS (PRIMARY and SECONDARY OBSERVABLES) #####################################

class sample_db(jks_db):
    db_type = "SAMPLE-DB"

    def add_src(self, src):
        assert os.path.isfile(src)
        self.message(f"load {src}", self.verbose)
        with open(src) as f:
            src_db = json.load(f)
        for t, slf in src_db.items():
            self.database[t] = SampleLeaf(slf.sample, slf.mean, slf.jks)
    
    def remove_cfgs(self, cfgs, tag):
        for cfg in cfgs:
            self.database[tag].sample.pop(str(cfg), None)

    def print(self, verbosity=0):
        self.message(self.__str__(verbosity=verbosity))
    
    def __str__(self, verbosity):
        s = 'DATABASE CONSISTS OF\n\n\n'
        for tag, slf in self.database.items():
            s += f'{tag:20s}\n'
            if verbosity >= 1:
                s += f'└── "sample"\n'
                if slf.mean != None:
                    s += f'└── "mean"\n'
                if slf.jks != None:
                    s += f'└── "jks"\n'
        return s

    ################################## FUNCTIONS #######################################
    
    def apply_f_to_sample(self, f, tag, dst_tag):
        slf = self.database[tag] 
        new_slf = {}
        for cfg in slf.sample:
            new_slf.sample[cfg] = f(slf.sample[cfg])

    ################################## STATISTICS ######################################
   
    def compute_means(self, tags):
        for tag in tags:
            self.database[tag].mean = np.mean(list(self.database[tag].sample.values()), axis=0)

    def compute_jks(self, tags):
        for tag in tags:
            mean = self.database[tag].mean
            sample = self.database[tag].sample
            jks = {}
            for cfg in sample:
                jks[cfg] = mean + (mean - sample[cfg]) / (len(sample) - 1)
            self.database[tag].jks = jks

    def bin(self, binsize, tag):
        sample = np.array(list(self.database[tag].sample)) 
        return {str(b):m for b,m in enumerate(sp.statistics.bin(sample, binsize))}

    def binning_study(self, binsizes, tag):
        self.message("Unbinned sample size: ", len(self.database[tag].sample))
        mean = self.database[tag].mean
        var = {}
        for b in binsizes:
            bsample = self.bin(b, tag)
            jks = sp.statistics.jackknife.samples(lambda x: x, bsample)
            var[b] = sp.statistics.jackknife.variance_jks(mean, jks)
        return var

#################################################################################################################################################
################################################################## TO DO ########################################################################
#################################################################################################################################################




        
###################################### FUNCTIONS #######################################

#    def mean_ratio(self, tag0, sample_tag0, tag1, sample_tag1, dst_tag, dst_sample_tag, jks_tag0=None, jks_tag1=None, store=True):
#        def ratio(nom, denom):
#            return nom / denom
#        if jks_tag0 != None:
#            data0_jks = self.get_data(tag0, sample_tag0, jks_tag0)
#        else:
#            data0_jks = self.jackknife_resampling(lambda x: x, tag0, sample_tag0)
#        if jks_tag1 != None:
#            data1_jks = self.get_data(tag1, sample_tag1, jks_tag1) 
#        else:
#            data1_jks = self.jackknife_resampling(lambda x: x, tag1, sample_tag1)
#        
#        r = ratio(self.get_data(tag0, sample_tag0, "mean"), self.get_data(tag1, sample_tag1, "mean"))
#        r_jks = {}
#        for cfg in data0_jks:
#            r_jks[cfg] = ratio(data0_jks[cfg], data1_jks[cfg])
#        r_jkvar = self.jackknife_variance_jks(r, r_jks, dst_tag, dst_sample_tag)
#        if store:
#            self.add_data(r, dst_tag, dst_sample_tag, "mean")
#        return r, r_jkvar


###################################### FITTING ######################################

#    def multi_mc_fit(self, t):
        

#        return 0

#    def multi_mc_fit(self, t, tags, sample_tags, C, model, p0, estimator, fit_tag="FIT", method="Nelder-Mead", minimizer_params={}, verbose=True, store=False, return_fitter=False):
#
#        mos = []
#        for tag, sample_tag in zip(tags, sample_tags):
#            mos.append(list(map(lambda e: (tag, e), sample_tag)))
#        y = []
#        for mo in mos:
#            for ts in mo:
#                ta, s = ts
#                y.append(self.get_data_arr(ta, s))
#
#        assert isinstance(model, dict)
#        model_func = list(model.values())[0]        
#        assert method in ["Levenberg-Marquardt", "Migrad", "Nelder-Mead"]
#        if method in ["Migrad", "Nelder-Mead"]:
#            fitter = sp.fitting.fit(t, y, C, model_func, p0, estimator, method, minimizer_params)
#        else:
#            fitter = sp.fitting.LM_fit(t, y, C, model_func, p0, estimator, minimizer_params)
#
#        fitter.multi_mc_fit(verbose)
#
#        if store:
#            self.add_data(t, fit_tag, "t", "-")
#            self.add_data(C, fit_tag, "C", "-")
#            self.add_data(list(model.keys())[0], fit_tag, "model", "-")
#            self.add_data(method, fit_tag, "minimizer", "-")
#            self.add_data(minimizer_params, fit_tag, "minimizer_params", "-")
#            self.add_data(p0, fit_tag, "p0", "-")
#            self.add_data(fitter.best_parameter, fit_tag, "best_parameter", "-")
#            self.add_data(fitter.best_parameter_cov, fit_tag, "best_parameter_cov", "-")
#            self.add_data(fitter.jks_parameter, fit_tag, "jks_parameter", "-")
#            self.add_data(fitter.p, fit_tag, "pval", "-")
#            self.add_data(fitter.chi2, fit_tag, "chi2", "-")
#            self.add_data(fitter.dof, fit_tag, "dof", "-")
#            self.add_data(fitter.fit_err, fit_tag, "fit_err", "-")
#        
#        if return_fitter:
#            return fitter

###################################### SPECTROSCOPY ######################################

#    def Ct_binning_study(self, Ct_tag, sample_tag, binsizes, keep_binsizes, t=None, shift=0, var=False, return_vals=False):
#        vals = self.binning_study(Ct_tag, sample_tag, binsizes, keep_binsizes, var)
#        bs = list(vals.keys())
#        if t == None:
#            print(f"Binning study of {Ct_tag} (binsizes={bs}):\n", [np.roll(vals[b], shift) for b in bs])
#        else:
#            print(f"Binning study of {Ct_tag} for t={t} (binsizes={bs}):\n", [np.roll(vals[b], shift)[t] for b in bs])
#        if return_vals:
#            return vals
#
#    def effective_mass_log(self, Ct_tag, sample_tag, dst_tag, tmax, shift=0, store=True):
#        Ct = self.get_data(Ct_tag, sample_tag, "mean")
#        m_eff_log = sp.qcd.spectroscopy.effective_mass_log(Ct, tmax, shift)
#        self.add_data(m_eff_log, dst_tag, sample_tag, "mean")
#        self.jackknife_variance(lambda Ct: sp.qcd.spectroscopy.effective_mass_log(Ct, tmax, shift), Ct_tag, sample_tag, dst_tag=dst_tag, return_var=False, store=store)
#        return self.get_data(dst_tag, sample_tag, "mean"), self.get_data(dst_tag, sample_tag, "jkvar")**0.5
#
#    def effective_mass_acosh(self, Ct_tag, sample_tag, dst_tag, tmax, shift=0, store=True):
#        Ct = self.get_data(Ct_tag, sample_tag, "mean")
#        m_eff_cosh = sp.qcd.spectroscopy.effective_mass_acosh(Ct, tmax, shift)
#        self.add_data(m_eff_cosh, dst_tag, sample_tag, "mean")
#        self.jackknife_variance(lambda Ct: sp.qcd.spectroscopy.effective_mass_acosh(Ct, tmax, shift), Ct_tag, sample_tag, dst_tag=dst_tag, return_var=False, store=store)    
#        return self.get_data(dst_tag, sample_tag, "mean"), self.get_data(dst_tag, sample_tag, "jkvar")**0.5
#
#    def correlator_exp_fit(self, t, Ct_tag, sample_tag, cov, p0, bc="pbc", min_method="Nelder-Mead", min_params={}, shift=0, verbose=True, dst_tag="CORR_FIT", store=True):        
#        Ct_mean = self.get_data(Ct_tag, sample_tag, "mean")
#        Ct_jks = self.get_data(Ct_tag, sample_tag, "jks")
#        Nt = len(Ct_mean)
#        best_parameter, chi2, pval, dof, model = sp.qcd.spectroscopy.correlator_exp_fit(t, Ct_mean[t], cov[t][:,t], p0, bc, Nt, min_method, min_params, shift, verbose=False)
#        best_parameter_jks = {}
#        for cfg in Ct_jks:
#            best_parameter_jks[cfg], _, _, _, _ = sp.qcd.spectroscopy.correlator_exp_fit(t, Ct_jks[cfg][t], cov[t][:,t], best_parameter, bc, Nt, min_method, min_params, shift, verbose=False)
#        best_parameter_cov = sp.statistics.jackknife.covariance_jks(best_parameter, np.array(list(best_parameter_jks.values())))
#        fit_err = lambda x: sp.fitting.fit_std_err(x, best_parameter, model.parameter_gradient, best_parameter_cov)
#        if verbose:
#            print("fit window:", t)
#            for i in range(len(best_parameter)):
#                print(f"parameter[{i}] = {best_parameter[i]} +- {best_parameter_cov[i][i]**0.5}")
#            print(f"chi2 / dof = {chi2} / {dof} = {chi2/dof}, i.e., p = {pval}")
#        if store:
#            model_str = {"pbc": "p[0] * exp(-p[1]t)", "obc": "p[0] * [exp(-p[1]t) + exp(-p[1](T-t))]"}
#            try:
#                self.database[dst_tag]
#            except KeyError:
#                self.database[dst_tag] = {}
#            self.database[dst_tag][sample_tag] = {}
#            self.database[dst_tag][sample_tag]["t"] = t
#            self.database[dst_tag][sample_tag]["model"] = model_str[bc]
#            self.database[dst_tag][sample_tag]["model_func"] = model
#            self.database[dst_tag][sample_tag]["minimizer"] = min_method
#            self.database[dst_tag][sample_tag]["minimizer_params"] = min_params
#            self.database[dst_tag][sample_tag]["p0"] = p0
#            self.database[dst_tag][sample_tag]["best_parameter"] = best_parameter
#            self.database[dst_tag][sample_tag]["best_parameter_cov"] = best_parameter_cov
#            self.database[dst_tag][sample_tag]["best_parameter_jks"] = best_parameter_jks
#            self.database[dst_tag][sample_tag]["pval"] = pval
#            self.database[dst_tag][sample_tag]["chi2"] = chi2
#            self.database[dst_tag][sample_tag]["dof"] = dof
#            self.database[dst_tag][sample_tag]["fit_err"] = fit_err 
#        return best_parameter, best_parameter_cov, best_parameter_jks
#
#    def effective_mass_curve_fit(self, t0min, t0max, nt, Ct_tag, sample_tag, cov, p0_Ct_fit, bc="pbc", min_method="Nelder-Mead", min_params={}, shift=0, verbose=True, dst_tag="M_EFF_CURVE_FIT", dst_tag_Ct="Ct_FIT", store=True):
#        mt = []; mt_var = []; best_parameter_jks_arr = []
#        for t0 in range(t0min, t0max):
#            t = np.arange(t0, t0+nt)
#            best_parameter, best_parameter_cov, best_parameter_jks = self.correlator_exp_fit(t, Ct_tag, sample_tag, cov, p0_Ct_fit, bc, min_method, min_params, shift, verbose=verbose, dst_tag=dst_tag_Ct+f"_{t0}", store=store)
#            mt.append(best_parameter[1])
#            mt_var.append(best_parameter_cov[1][1])
#            best_parameter_jks_arr.append(best_parameter_jks)    
#        # transform jks data appropriately
#        mt = np.array(mt); mt_var = np.array(mt_var)
#        mt_jks = {}
#        for cfg in best_parameter_jks_arr[0]:
#            mt_cfg = []
#            for best_parameter_jks in best_parameter_jks_arr:
#                mt_cfg.append(best_parameter_jks[cfg][1])
#            mt_jks[cfg] = np.array(mt_cfg)
#        if store:
#            self.add_data(mt, dst_tag, sample_tag, "mean")
#            self.add_data(mt_var, dst_tag, sample_tag, "jkvar")
#            self.add_data(mt_jks, dst_tag, sample_tag, "jks")
#        return mt, mt_var, mt_jks
#
#    def effective_mass_const_fit(self, t, mt_tag, sample_tag, dst_tag, p0, method, minimizer_params={}, verbose=True, store=True):    
#        mt = self.get_data(mt_tag, sample_tag, "mean")[t]
#        mt_cov = np.diag(self.get_data(mt_tag, sample_tag, "jkvar"))[t][:,t]
#        m, p, chi2, dof, model = sp.qcd.spectroscopy.const_fit(t, mt, mt_cov, p0, method, minimizer_params, error=False, verbose=False)
#
#        mt_jks = self.get_data(mt_tag, sample_tag, "jks")
#        m_jks = {}
#        for cfg in mt_jks:
#            m_jks[cfg], _, _, _, _ = sp.qcd.spectroscopy.const_fit(t, mt_jks[cfg][t], mt_cov, p0=m, method=method, minimizer_params=minimizer_params, error=False, verbose=False)
#        #for mt_jk in mt_jks.values():
#        #    m_jk, _, _, _, _ = sp.qcd.spectroscopy.const_fit(t, mt_jk[t], mt_cov, p0=m, method=method, minimizer_params=minimizer_params, error=False, verbose=False)
#        #    m_jks.append(m_jk)
#
#        m_cov = sp.statistics.jackknife.covariance_jks(m, np.array(list(m_jks.values())))
#        model = sp.qcd.spectroscopy.const_model()
#        fit_err = lambda t: (model.parameter_gradient(t,m) @ m_cov @ model.parameter_gradient(t,m))**0.5
#
#        if verbose:
#            print("*** constant mass fit ***")
#            print("fit window:", t)
#            print(f"m_eff = {m[0]} +- {m_cov[0][0]**.5}")
#            print(f"chi2 / dof = {chi2} / {dof} = {chi2/dof}, i.e., p = {p}")
#
#        if store:
#            try:
#                self.database[dst_tag]
#            except KeyError:
#                self.database[dst_tag] = {}
#            self.database[dst_tag][sample_tag] = {}
#            self.database[dst_tag][sample_tag]["fit_window"] = t
#            self.database[dst_tag][sample_tag]["mt_cov"] = mt_cov
#            self.database[dst_tag][sample_tag]["model"] = "const."
#            self.database[dst_tag][sample_tag]["model_func"] = model
#            self.database[dst_tag][sample_tag]["minimizer"] = method
#            self.database[dst_tag][sample_tag]["minimizer_params"] = minimizer_params
#            self.database[dst_tag][sample_tag]["p0"] = p0
#            self.database[dst_tag][sample_tag]["m_eff"] = m
#            self.database[dst_tag][sample_tag]["m_eff_cov"] = m_cov
#            self.database[dst_tag][sample_tag]["m_eff_jks"] = m_jks
#            self.database[dst_tag][sample_tag]["pval"] = p
#            self.database[dst_tag][sample_tag]["chi2"] = chi2
#            self.database[dst_tag][sample_tag]["dof"] = dof
#            self.database[dst_tag][sample_tag]["fit_err"] = fit_err
#
#        return m[0], m_cov[0][0]**.5
