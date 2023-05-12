#!/usr/bin/env python3

import os, h5py
from time import time
import numpy as np
import matplotlib.pyplot as plt
from statpy.dbpy import custom_json as json
from statpy.dbpy.leafs import Leaf 
import statpy as sp

###################################### DATABASE SYSTEM USING LEAFS CONTAINING MEAN AND JKS (SECONDARY OBSERVABLES) ###########################################

class JKS_DB:
    db_type = "JKS-DB"
    def __init__(self, *args, verbosity=0):
        self.t0 = time()
        self.verbosity = verbosity
        self.database = {}
        for src in args:
            if isinstance(src, str):
                self.add_src(src)
            if isinstance(src, tuple):
                self.add_Leaf(*src)
            if isinstance(src, dict):
                for t, lf in src:
                    self.database[t] = Leaf(lf.mean, lf.jks, None, None, lf.info)

    def add_src(self, src):
        assert os.path.isfile(src)
        self.message(f"load {src}")
        with open(src) as f:
            src_db = json.load(f)
        for t, lf in src_db.items():
            self.database[t] = Leaf(lf.mean, lf.jks, lf.sample, lf.nrwf, lf.info)

    def add_Leaf(self, tag, mean, jks, sample, nrwf, info):
        assert (isinstance(sample, dict) or sample==None)
        assert (isinstance(jks, dict) or jks==None)
        assert (isinstance(nrwf, dict) or nrwf==None)
        assert (isinstance(info, dict) or info==None)
        self.database[tag] = Leaf(mean, jks, sample, nrwf, info)

    def message(self, s, verbosity=None):
        if verbosity == None: verbosity = self.verbosity
        if verbosity >= 0: print(f"{self.db_type}:\t\t{time()-self.t0:.6f}s: " + s)
    
    def save(self, dst):
        with open(dst, "w") as f:
            json.dump(self.database, f)

    def print(self, verbosity=0):
        self.message(self.__str__(verbosity=verbosity))    

    def __str__(self, verbosity):
        s = '\n\n\tDATABASE CONSISTS OF\n\n'
        for tag, lf in self.database.items():
            s += f'\t{tag:20s}\n'
            if verbosity >= 1:
                if np.array(lf.mean).any() != None:
                    s += f'\t└── mean\n'
                if np.array(lf.jks).any() != None:
                    s += f'\t└── jks\n'
                if np.array(lf.sample).any() != None:
                    s += f'\t└── sample\n' 
                if np.array(lf.nrwf).any() != None:
                    s += f'\t└── nrwf\n' 
                if lf.info != None:
                    s += f'\t└── info\n'
        return s

    def print_info(self, tag):
        self.message(self.__info_str__(tag))

    def __info_str__(self, tag):
        s = f'\n\n\tINFO DICT OF {tag}\n\n'
        for k, i in self.database[tag].info.items():
            s += f'\t{k:20s}: {i}\n'
        return s
    
    def remove(self, *tags):
        for tag in tags:
            try:
                del self.database[tag]
            except KeyError:
                print(f"{tag} not in database")

    # helper function
    def as_array(self, obj):
        if isinstance(obj, np.ndarray):
            return obj
        else:
            return np.array(list(obj.values()))
    
    ################################## FUNCTIONS #######################################

    def combine_mean(self, *tags, f=lambda x: x, dst_tag=None):
        lfs = [self.database[tag] for tag in tags]
        mean = f(*[lf.mean for lf in lfs])
        if dst_tag == None:
            return mean
        try:
            self.database[dst_tag].mean = mean
        except KeyError:
            self.database[dst_tag] = Leaf(mean, None, None)

    def combine_jks(self, *tags, f=lambda x: x, dst_tag=None):
        lfs = [self.database[tag] for tag in tags]
        jks = {}
        cfgs = np.unique(np.concatenate([list(lf.jks.keys()) for lf in lfs]))
        for cfg in cfgs:
            x = [lf.jks[cfg] if cfg in lf.jks else lf.mean for lf in lfs]
            jks[cfg] = f(*x)
        if dst_tag == None:
            return jks
        try:
            self.database[dst_tag].jks = jks
        except KeyError:
            self.database[dst_tag] = Leaf(None, jks, None)

    def combine(self, *tags, f=lambda x: x, dst_tag=None):
        mean = self.combine_mean(*tags, f=f)
        jks = self.combine_jks(*tags, f=f)
        if dst_tag == None:
            return Leaf(mean, jks, None)
        self.database[dst_tag] = Leaf(mean, jks, None)

    ################################## STATISTICS ######################################

    def compute_jks(self, tag, binsize, shift=0, verbose=False):
        lf = self.database[tag]
        if binsize == 1:
            return lf.jks
        jks_tags = sorted(list(lf.jks.keys()), key=lambda x: (x.split("-")[0], int(x.split("-")[-1])))
        if verbose:
            print(jks_tags)
        N = len(jks_tags)
        Nb = N // binsize 
        jks_tags = jks_tags[:Nb*binsize] # cut off excess data
        jks_bin = {}
        for i in range(Nb):
            s = sum([lf.jks[np.roll(jks_tags, shift)[idx]] for idx in np.arange(i*binsize, (i+1)*binsize)]) - binsize * lf.mean
            jks_bin[i] = lf.mean + s * (N-1) / (N-binsize)
        return jks_bin

    def jackknife_variance(self, tag, binsize, pavg=False):
        permutations = np.arange(1)
        if pavg:
            permutations = np.arange(binsize)
        var = []
        for p in permutations:
            jks = self.as_array(self.compute_jks(tag, binsize, p))
            var.append(sp.statistics.jackknife.variance_jks(np.mean(jks, axis=0), jks))
        return np.mean(var, axis=0)
    
    def jackknife_covariance(self, tag, binsize, pavg=False):
        permutations = np.arange(1)
        if pavg:
            permutations = np.arange(binsize)
        cov = []
        for p in permutations:
            jks = self.as_array(self.compute_jks(tag, binsize, p))
            cov.append(sp.statistics.jackknife.covariance_jks(np.mean(jks, axis=0), jks))
        return np.mean(cov, axis=0)

    def binning_study(self, tag, binsizes, pavg=False):
        self.message(f"Unbinned sample size: {len(self.database[tag].sample)}")
        var = {}
        for b in binsizes:
            var[b] = self.jackknife_variance(tag, b, pavg)
        return var
    
    def AMA(self, exact_exact_tag, exact_sloppy_tag, sloppy_sloppy_tag, dst_tag):
        self.combine(exact_exact_tag, exact_sloppy_tag, f=lambda x,y: x-y, dst_tag=dst_tag+"_bias")
        self.combine(sloppy_sloppy_tag, dst_tag+"_bias", f=lambda x,y: x+y, dst_tag=dst_tag)

    ############################### SCALE SETTING ###################################

    def gradient_flow_scale(self, ensemble_label, binsize, verbose=True):
        tau = self.database[ensemble_label + "/tau"].mean
        scale = sp.qcd.scale_setting.gradient_flow_scale()
        # tau0
        self.combine(ensemble_label + "/E", f=lambda x: scale.set_sqrt_tau0(tau, x), dst_tag=ensemble_label + "/sqrt_tau0")
        sqrt_tau0_var = self.jackknife_variance(ensemble_label + "/sqrt_tau0", binsize)
        # t0
        self.combine(ensemble_label + "/sqrt_tau0", f=lambda x: scale.comp_sqrt_t0(x, scale.sqrt_t0_fm), dst_tag=ensemble_label + "/sqrt_t0") 
        sqrt_t0_stat_var = self.jackknife_variance(ensemble_label + "/sqrt_t0", binsize)
        # propagate systematic error of t0
        sqrt_t0_mean_shifted = scale.comp_sqrt_t0(self.database[ensemble_label + "/sqrt_tau0"].mean, scale.sqrt_t0_fm + scale.sqrt_t0_fm_std)
        sqrt_t0_sys_var = (self.database[ensemble_label + "/sqrt_t0"].mean - sqrt_t0_mean_shifted)**2.0
        if self.database[ensemble_label + "/sqrt_t0"].info == None: self.database[ensemble_label + "/sqrt_t0"].info = {} 
        self.database[ensemble_label + "/sqrt_t0"].info = {"sqrt_t0_stat_var": sqrt_t0_stat_var,
            "sqrt_t0_mean_shifted": sqrt_t0_mean_shifted, "sqrt_t0_sys_var": sqrt_t0_sys_var, "sqrt_t0_var": sqrt_t0_stat_var + sqrt_t0_sys_var}
        print("sqrt_t0_mean_shifted = ", sqrt_t0_mean_shifted)
        print("sqrt_t0_sys_var = ", sqrt_t0_sys_var)
        # omega0
        self.combine(ensemble_label + "/E", f=lambda x: scale.set_omega0(tau, x), dst_tag=ensemble_label + "/omega0")
        omega0_var = self.jackknife_variance(ensemble_label + "/omega0", binsize)
        # w0
        self.combine(ensemble_label + "/omega0", f=lambda x: scale.comp_w0(x, scale.w0_fm), dst_tag=ensemble_label + "/w0") 
        w0_stat_var = self.jackknife_variance(ensemble_label + "/w0", binsize)
        # propagate systematic error of w0
        w0_mean_shifted = scale.comp_w0(self.database[ensemble_label + "/omega0"].mean, scale.w0_fm + scale.w0_fm_std)
        w0_sys_var = (self.database[ensemble_label + "/w0"].mean - w0_mean_shifted)**2.0
        if self.database[ensemble_label + "/w0"].info == None: self.database[ensemble_label + "/w0"].info = {} 
        self.database[ensemble_label + "/w0"].info = {"w0_stat_var": w0_stat_var, 
            "w0_mean_shifted": w0_mean_shifted, "w0_sys_var": w0_sys_var, "w0_var": w0_stat_var + w0_sys_var}
        if verbose:
            self.message(f"sqrt(tau0) = {self.database[ensemble_label + '/sqrt_tau0'].mean:.4f} +- {sqrt_tau0_var**.5:.4f}")
            self.message(f"omega0 = {self.database[ensemble_label + '/omega0'].mean:.4f} +- {omega0_var**.5:.4f}")
            self.message(f"t0/GeV (cutoff) = {self.database[ensemble_label + '/sqrt_t0'].mean:.4f} +- {sqrt_t0_stat_var**.5:.4f} (STAT) +- {sqrt_t0_sys_var**.5:.4f} (SYS) [{(sqrt_t0_stat_var+sqrt_t0_sys_var)**.5:.4f} (STAT+SYS)]")
            self.message(f"w0/GeV (cutoff) = {self.database[ensemble_label + '/w0'].mean:.4f} +- {w0_stat_var**.5:.4f} (STAT) +- {w0_sys_var**.5:.4f} (SYS) [{(w0_stat_var+w0_sys_var)**.5:.4f} (STAT+SYS)]")

    ################################## FITTING ######################################

    def fit(self, t, tags, cov, p0, model, method, minimizer_params, binsize, dst_tag, verbosity=0):
        if isinstance(tags, str):
            return self.fit_single(t, tags, cov, p0, model, method, minimizer_params, binsize, dst_tag, verbosity)
        if isinstance(tags, list) or isinstance(tags, np.ndarray):
            return self.fit_multiple(t, tags, cov, p0, model, method, minimizer_params, binsize, dst_tag, verbosity)
 
    def fit_single(self, t, tag, cov, p0, model, method, minimizer_params, binsize, dst_tag, verbosity=0):
        fitter = sp.fitting.Fitter(t, cov, model, lambda x: x, method, minimizer_params)
        self.combine_mean(tag, f=lambda y: fitter.estimate_parameters(fitter.chi_squared, y[t], p0)[0], dst_tag=dst_tag) 
        best_parameter = self.database[dst_tag].mean
        self.combine_jks(tag, f=lambda y: fitter.estimate_parameters(fitter.chi_squared, y[t], best_parameter)[0], dst_tag=dst_tag)  
        best_parameter_cov = self.jackknife_covariance(dst_tag, binsize, pavg=True)
        if verbosity >=1: 
            print(f"jackknife parameter covariance is ", best_parameter_cov) 
        chi2 = fitter.chi_squared(best_parameter, self.database[tag].mean[t])
        dof = len(t) - len(best_parameter)
        pval = fitter.get_pvalue(chi2, dof)
        self.database[dst_tag].info = {"t": t, "best_parameter_cov": best_parameter_cov, "chi2": chi2, "dof": dof, "pvalue": pval}
        if verbosity >= 0:
            for i in range(len(best_parameter)):
                print(f"parameter[{i}] = {best_parameter[i]} +- {best_parameter_cov[i][i]**0.5}")
            print(f"chi2 / dof = {chi2} / {dof} = {chi2/dof}, i.e., p = {pval}") 
    
    def fit_multiple(self, t, tags, cov, p0, model, method, minimizer_params, binsize, dst_tag, verbosity=0):
        fitter = sp.fitting.Fitter(t, cov, model, lambda x: x, method, minimizer_params)
        self.combine_mean(*tags, f=lambda *y: fitter.estimate_parameters(fitter.chi_squared, y, p0)[0], dst_tag=dst_tag) 
        best_parameter = self.database[dst_tag].mean
        self.combine_jks(*tags, f=lambda *y: fitter.estimate_parameters(fitter.chi_squared, y, best_parameter)[0], dst_tag=dst_tag)
        best_parameter_cov = self.jackknife_covariance(dst_tag, binsize, pavg=True)
        if verbosity >=1: 
            print(f"jackknife parameter covariance is ", best_parameter_cov) 
        chi2 = fitter.chi_squared(best_parameter, np.array([self.database[tag].mean for tag in tags]))
        dof = len(t) - len(best_parameter)
        pval = fitter.get_pvalue(chi2, dof)
        self.database[dst_tag].info = {"t": t, "best_parameter_cov": best_parameter_cov, "chi2": chi2, "dof": dof, "pvalue": pval}
        if verbosity >= 0:
            for i in range(len(best_parameter)):
                print(f"parameter[{i}] = {best_parameter[i]} +- {best_parameter_cov[i][i]**0.5}")
            print(f"chi2 / dof = {chi2} / {dof} = {chi2/dof}, i.e., p = {pval}")  

    def _fit_indep(self, t, tags, binsizes, cov, p0, model, method, minimizer_params, verbosity=0):
        y = np.array([self.database[tag].mean for tag in tags])
        y_jks = np.array([self.compute_jks(tag, binsize) for tag, binsize in zip(tags, binsizes)]) 
        fitter = sp.fitting.Fitter(t, cov, model, lambda x: x, method, minimizer_params)
        best_parameter, chi2, _ = fitter.estimate_parameters(fitter.chi_squared, y, p0)
        best_parameter_jks = np.zeros_like(y_jks)
        best_parameter_cov = np.zeros((len(best_parameter), len(best_parameter)))         
        for i in range(len(t)):
            yt_jks = {}
            for cfg in y_jks[i]:
                yt_jks[cfg], _, _ = fitter.estimate_parameters(fitter.chi_squared, np.array(list(y[:i]) + [y_jks[i][cfg]] + list(y[i+1:])), best_parameter)
            best_parameter_jks[i] = yt_jks
            best_parameter_t_cov = sp.statistics.jackknife.covariance_jks(best_parameter, self.as_array(yt_jks))
            if verbosity >=1: 
                print(f"jackknife parameter covariance from t[{i}] is ", best_parameter_t_cov)
            best_parameter_cov += best_parameter_t_cov
        dof = len(t) - len(best_parameter)
        pval = fitter.get_pvalue(chi2, dof)
        if verbosity >= 0:
            for i in range(len(best_parameter)):
                print(f"parameter[{i}] = {best_parameter[i]} +- {best_parameter_cov[i][i]**0.5}")
            print(f"chi2 / dof = {chi2} / {dof} = {chi2/dof}, i.e., p = {pval}")
        return best_parameter, best_parameter_cov
    
    def model_prediction_var(self, t, best_parameter, best_parameter_cov, model_parameter_gradient):
        return model_parameter_gradient(t, best_parameter) @ best_parameter_cov @ model_parameter_gradient(t, best_parameter)

    ################################## SPECTROSCOPY ######################################

    def effective_mass_curve_fit(self, t0_min, t0_max, nt, tag, cov, p0, bc, method, minimizer_params, binsize, dst_tag, verbosity=0):
        assert bc in ["pbc", "obc"]
        model = {"pbc": sp.qcd.correlator.cosh_model(len(self.database[tag].mean)), "obc": sp.qcd.correlator.exp_model()}[bc]
        for t0 in range(t0_min, t0_max):
            t = np.arange(nt) + t0
            if verbosity >=0: self.message(f"fit window: {t}")
            self.fit(t, tag, cov[t][:,t], p0, model, method, minimizer_params, binsize, dst_tag=dst_tag + f"={t0}", verbosity=verbosity)
            self.database[dst_tag + f"={t0}"].mean = self.database[dst_tag + f"={t0}"].mean[1]
            self.database[dst_tag + f"={t0}"].jks = {cfg:val[1] for cfg, val in self.database[dst_tag + f"={t0}"].jks.items()} 
            self.database[dst_tag + f"={t0}"].info["best_parameter_cov"] = self.database[dst_tag + f"={t0}"].info["best_parameter_cov"][1]

################################# DATABASE SYSTEM USING LEAFS CONTAINING SAMPLE, RWF, MEAN AND JKS (PRIMARY and SECONDARY OBSERVABLES) #####################################

class Sample_DB(JKS_DB):
    db_type = "SAMPLE-DB"
    def remove_cfgs(self, cfgs, tag):
        for cfg in cfgs:
            self.database[tag].sample.pop(str(cfg), None)

    def cfgs(self, tag):
        return [int(x.split("-")[-1]) for x in self.database[tag].sample.keys()]
    
    def merge_samples(self, *tags, dst_tag=None, dst_cfgs=None):
        assert dst_cfgs != None
        lfs = [self.database[tag] for tag in tags]
        sample = np.concatenate([self.as_array(lf.sample) for lf in lfs], axis=0)
        dst_sample = {}
        for cfg, val in zip(dst_cfgs, sample):
            dst_sample[cfg] = val
        if dst_tag == None:
            return Leaf(None, None, sample)
        else:
            self.database[dst_tag] = Leaf(None, None, dst_sample)

    def return_JKS_DB(self):
        return JKS_DB(self.database)

    ################################## FUNCTIONS #######################################

    def combine_sample(self, *tags, f=lambda x: x, dst_tag=None):
        lfs = [self.database[tag] for tag in tags]
        f_sample = {}
        cfgs = np.unique([list(lf.sample.keys()) for lf in lfs]) 
        for cfg in cfgs:
            x = [lf.sample[cfg] if cfg in lf.sample else lf.mean for lf in lfs]
            f_sample[cfg] = f(*x) 
        if dst_tag == None:
            return Leaf(None, None, f_sample)
        self.database[dst_tag] = Leaf(None, None, f_sample)
 
    ################################## STATISTICS ######################################
 
    def init_sample_means(self, *tags):
        if len(tags) == 0:
            tags = self.database.keys()
        for tag in tags:
            lf = self.database[tag]
            if lf.sample != None:
                if lf.nrwf == None:
                    lf.mean = np.mean(self.as_array(lf.sample), axis=0)
                else:
                    lf.mean = np.mean(self.as_array(lf.nrwf)[:,None] * self.as_array(lf.sample), axis=0)

    def init_sample_jks(self, *tags):
        if len(tags) == 0:
            tags = self.database.keys()
        for tag in tags:
            lf = self.database[tag]
            if lf.nrwf == None:
                jks = {}
                for cfg in lf.sample:
                    jks[cfg] = lf.mean + (lf.mean - lf.sample[cfg]) / (len(lf.sample) - 1)
            else:
                jks = {}
                for cfg in lf.sample:
                    jks[cfg] = lf.mean + (lf.mean - lf.sample[cfg]) * lf.nrwf[cfg] / (len(lf.sample) - lf.nrwf[cfg])
            lf.jks = jks

    def compute_sample_jks(self, tag, binsize, f=lambda x: x):
        lf = self.database[tag]
        if binsize == 1:
            if lf.jks == None:
                self.init_sample_jks(tag)
            jks = self.as_array(lf.jks)
        else:
            if lf.nrwf == None:
                bsample = sp.statistics.bin(self.as_array(lf.sample), binsize)
                jks = sp.statistics.jackknife.samples(f, bsample)
            else:
                bsample = sp.statistics.bin(self.as_array(lf.sample), binsize, self.as_array(lf.nrwf)); bnrwf = sp.statistics.bin(self.as_array(lf.nrwf), binsize)
                jks = sp.statistics.jackknife.samples(f, bsample, bnrwf[:, None])
        return jks
    
    def sample_jackknife_variance(self, tag, binsize, f=lambda x: x):
        jks = self.compute_sample_jks(tag, binsize, f)
        return sp.statistics.jackknife.variance_jks(np.mean(jks, axis=0), jks)

    def sample_jackknife_covariance(self, tag, binsize, f=lambda x: x):
        jks = self.compute_sample_jks(tag, binsize, f)
        return sp.statistics.jackknife.covariance_jks(np.mean(jks, axis=0), jks)
    
    def sample_binning_study(self, tag, binsizes):
        self.message(f"Unbinned sample size: {len(self.database[tag].sample)}")
        var = {}
        for b in binsizes:
            var[b] = self.sample_jackknife_variance(tag, b)
        return var
    
    ################################## SPECTROSCOPY ######################################

    def lattice_charm_fit_range(self, tag, binsize, ds, p0, model_type, fit_method="Levenberg-Marquardt", fit_params=None, jks_fit_method=None, jks_fit_params=None, verbosity=0):
        """
        * perform uncorrelated fits at binsize b=20 using double symmetric double exponential fit model for different fit ranges $[d,N_t -d]$
        * ground state captured in one exponential and excited states in the other
        * determine size of excited state contribution using fit model for all t
        * cut off time values from fit range for which excited state contribution is smaller than $\frac{\text{std}\left(C(t)\right)}{4}$
        * final fit range is the smallest of the reduced fit windows
        """
        if fit_params == None:
            fit_params = {"maxiter":5000, "tol":1e-8, "eps1":1e-11, "eps2":1e-11, "eps3":1e-9}
        if jks_fit_method == None:
            jks_fit_method = fit_method; jks_fit_params = fit_params
        jks = self.compute_sample_jks(tag, binsize)
        mean = np.mean(jks, axis=0)
        var = sp.statistics.jackknife.variance_jks(mean, jks)
        Nt = len(mean)
        fit_range = np.arange(Nt)
        for d in ds:
            t = np.arange(d, Nt-d)
            self.message(f"fit range: {t}", verbosity)
            y = mean[t]; y_jks = {cfg:Ct[t] for cfg,Ct in enumerate(jks)}; cov = np.diag(var)[t][:,t]
            model = {"cosh": sp.qcd.correlator.double_cosh_model(Nt), "sinh": sp.qcd.correlator.double_sinh_model(Nt)}[model_type]
            best_parameter, best_parameter_cov = sp.qcd.correlator.fit(t, y, y_jks, cov, p0, model, 
                                                    fit_method, fit_params, jks_fit_method, jks_fit_params, verbosity=verbosity)
            # sort parameters
            if best_parameter[1] <  best_parameter[3]: sorted_bp = [best_parameter[2], best_parameter[3], 0, 0]
            else: sorted_bp = [best_parameter[0], best_parameter[1], 0, 0]
            criterion = np.array([model(t, sorted_bp) for t in range(Nt)]) < var**.5/4
            t_reduced = np.arange(Nt)[criterion]
            if len(t_reduced) < len(fit_range):
                fit_range = t_reduced
            self.message(f"reduced fit range {t_reduced}", verbosity)
            if verbosity >= 0:     
                print("----------------------------------------------------------------------------------------------------------------------------------")
                print("----------------------------------------------------------------------------------------------------------------------------------")
        self.message(f"FINAL REDUCED FIT RANGE: {fit_range}", verbosity)
        return fit_range
    
    def lattice_charm_spectroscopy(self, tag, B, fit_range, p0, model_type="cosh", fit_method="Migrad", fit_params=None, make_plot=True, verbosity=0):
        """
        * perform correlated fits for binsizes up to $b_c = 20 \approx N/100$ using symmetric exponential fit form
        * covariance of correlators is estimated on unbinned dataset
        """
        A_dict = {}; A_var_dict = {}
        m_dict = {}; m_var_dict = {}
        # estimate cov using unbinned sample
        cov = self.sample_jackknife_covariance(tag, binsize=1)
        for b in range(1,B+1):
            self.message(f"BINSIZE = {b}\n", verbosity)
            jks = self.compute_sample_jks(tag, binsize=b)
            mean = np.mean(jks, axis=0)
            t = fit_range
            y = mean[t]; y_jks = {cfg:Ct[t] for cfg,Ct in enumerate(jks)}
            model = {"cosh": sp.qcd.correlator.cosh_model(len(mean)), "sinh": sp.qcd.correlator.sinh_model(len(mean))}[model_type]
            best_parameter, best_parameter_cov, best_parameter_jks = sp.qcd.correlator.fit(t, y, y_jks, cov[t][:,t], 
                                                    p0, model, fit_method, fit_params, verbosity=verbosity)
            A_dict[b] = best_parameter[0]; A_var_dict[b] = best_parameter_cov[0][0]
            m_dict[b] = best_parameter[1]; m_var_dict[b] = best_parameter_cov[1][1]
            if verbosity >=0:
                print("\n-----------------------------------------------------------------------------------------")
                print("-----------------------------------------------------------------------------------------\n")
            if b == B:
                if make_plot:
                    fig, ax0 = plt.subplots(figsize=(12,8))
                    # correlator
                    color = "C0"
                    ax0.set_xlabel(r"source-sink separation $t$")
                    ax0.set_ylabel(r"$C(t)$", color=color)     
                    ax0.errorbar(np.arange(len(mean)), mean, sp.statistics.jackknife.variance_jks(mean, jks)**0.5, linestyle="", capsize=3, color=color)
                    ax0.tick_params(axis='y', labelcolor=color)
                    ax0.grid(axis="x")
                    ax0.set_title(f"{tag} - binsize = {B}")
                    # correlator fit
                    trange = np.arange(t[0], t[-1], 0.1)
                    color = "C2"
                    #model = sp.qcd.correlator.cosh_model(len(mean))
                    fy = np.array([model(t, best_parameter) for t in trange])
                    fy_err = np.array([self.model_prediction_var(t, best_parameter, best_parameter_cov, model.parameter_gradient) for t in trange])**.5
                    model_label = {"cosh": r"$A_0 (e^{-m_0 t} + e^{-m_0 (T-t)})$ - fit", "sinh": r"$A_0 (e^{-m_0 t} - e^{-m_0 (T-t)})$ - fit"}[model_type]
                    ax0.plot(trange, fy, color=color, lw=.5, label=model_label)
                    ax0.fill_between(trange, fy-fy_err, fy+fy_err, alpha=0.5, color=color)
                    #ax0.set_ylim(0.0, mean[fit_range[0]]*2.)
                    ax0.legend(loc="upper left")
                    # fit range marker
                    ax0.axvline(t[0], color="gray", linestyle="--")
                    ax0.axvline(t[-1], color="gray", linestyle="--")
                    # effective mass curve
                    d = 5
                    mt = sp.qcd.correlator.effective_mass_acosh(mean, tmax=len(mean)-d, tmin=d)
                    mt_var = self.sample_jackknife_variance(tag, B, lambda Ct: sp.qcd.correlator.effective_mass_acosh(Ct, tmax=len(mean)-d, tmin=d))
                    ax1 = ax0.twinx()
                    color = "C3"
                    ax1.set_ylabel("$m(t)$", color=color)
                    ax1.errorbar(np.arange(d, d+len(mt)), mt, mt_var**.5, linestyle="", capsize=3, color=color)
                    ax1.tick_params(axis='y', labelcolor=color)
                    ax1.grid(axis="y")
                    # effective mass from fit
                    color = "C4"
                    meff_arr = np.array([best_parameter[1] for t in trange])
                    ax1.plot(trange, meff_arr, color=color, lw=.5, label=r"$m_{eff} = $" + f"{best_parameter[1]:.4f} +- {best_parameter_cov[1][1]**.5:.4f}")
                    ax1.fill_between(trange, meff_arr-best_parameter_cov[1][1]**.5, meff_arr+best_parameter_cov[1][1]**.5, alpha=0.5, color=color)
                    ax1.set_ylim(0.0, best_parameter[1]*3.)
                    ax1.legend(loc="upper right")
                    plt.tight_layout()
                    plt.plot()
        return A_dict, A_var_dict, m_dict, m_var_dict
    

    def lattice_charm_combined_cosh_sinh_fit(self, tag0, tag1, B, fit_range, p0, correlated=False, fit_method="Migrad", fit_params=None, jks_fit_method=None, jks_fit_params=None, 
                                             verbosity=0, make_plot=False):
        """
        * perform combined fits for binsizes up to $b_c = 20 \approx N/100$ using symmetric exponential fit form
        * covariance of correlators is estimated on unbinned dataset
        """
        A_PS_dict = {}; A_PS_var_dict = {}
        A_A4_dict = {}; A_A4_var_dict = {}
        m_dict = {}; m_var_dict = {}
        t = fit_range
        # estimate cov using unbinned sample
        jks0_ub = self.compute_sample_jks(tag0, 1); jks1_ub = self.compute_sample_jks(tag1, 1)
        jks_ub = np.array([np.hstack((jks0_ub[cfg][t],jks1_ub[cfg][t])) for cfg in range(len(jks0_ub))])
        cov = sp.statistics.jackknife.covariance_jks(np.mean(jks_ub, axis=0), jks_ub)
        if not correlated:
            cov = np.diag(np.diag(cov))
        for b in range(1, B+1):
            #self.message(f"BINSIZE = {b}\n", verbosity)
            print(f"BINSIZE = {b}\n")
            jks0 = self.compute_sample_jks(tag0, binsize=b); jks1 = self.compute_sample_jks(tag1, binsize=b)
            mean0 = np.mean(jks0, axis=0); mean1 = np.mean(jks1, axis=0)
            jks_arr = np.array([np.hstack((jks0[cfg][t],jks1[cfg][t])) for cfg in range(len(jks0))])
            jks = {cfg:jks_arr[cfg] for cfg in range(len(jks_arr))}
            mean = np.mean(jks_arr, axis=0)
            model = sp.qcd.correlator.combined_cosh_sinh_model(len(jks0[0]))
            best_parameter, best_parameter_cov, best_parameter_jks = sp.qcd.correlator.fit(t, mean, jks, cov, p0, model, fit_method, fit_params, jks_fit_method, jks_fit_params, verbosity)
            A_PS_dict[b] = best_parameter[0]; A_PS_var_dict[b] = best_parameter_cov[0][0]
            A_A4_dict[b] = best_parameter[1]; A_A4_var_dict[b] = best_parameter_cov[1][1]
            m_dict[b] = best_parameter[2]; m_var_dict[b] = best_parameter_cov[2][2]
            if verbosity >=0:
                print("\n-----------------------------------------------------------------------------------------")
                print("-----------------------------------------------------------------------------------------\n")
            if b == B:
                if make_plot:
                    fig, ax0 = plt.subplots(figsize=(12,8))
                    # PSPS correlator
                    color = "C0"
                    ax0.set_xlabel(r"source-sink separation $t$")
                    ax0.set_ylabel(r"$C_{PSPS}(t)$", color=color)     
                    ax0.errorbar(np.arange(len(mean0)), mean0, sp.statistics.jackknife.variance_jks(mean0, jks0)**0.5, linestyle="", capsize=3, 
                                 color=color, label=f"PSPS data")
                    ax0.tick_params(axis='y', labelcolor=color)
                    ax0.grid()
                    ax0.set_title(f"combined fit {tag0} + {tag1}  - binsize = {B}")            
                    # PSA4 correlator
                    color = "C1"
                    ax1 = ax0.twinx()
                    ax1.set_ylabel(r"$C_{PSA4}(t)$", color=color) 
                    ax1.errorbar(np.arange(len(mean1)), mean1, sp.statistics.jackknife.variance_jks(mean1, jks1)**0.5, linestyle="", capsize=3, 
                                 color=color, label=f"PSA4 data")
                    ax1.tick_params(axis='y', labelcolor=color)
                    # PSPS fit
                    trange = np.arange(t[0], t[-1], 0.1)
                    color = "C2"
                    fy_PS = np.array([model(t, best_parameter)[0] for t in trange])
                    fy_err_PS = np.array([self.model_prediction_var(t, best_parameter, best_parameter_cov, lambda x,y: model.parameter_gradient(x,y)[0]) for t in trange])**.5
                    model_label = {"cosh": r"$A_{PS} (e^{-m t} + e^{-m (T-t)})$ - fit", "sinh": r"$A_{A4} (e^{-m t} - e^{-m (T-t)})$ - fit"}
                    ax0.plot(trange, fy_PS, color=color, lw=.5, label=model_label["cosh"])
                    ax0.fill_between(trange, fy_PS-fy_err_PS, fy_PS+fy_err_PS, alpha=0.5, color=color)
                    ax0.legend(loc="upper left")
                    # PSA4 fit
                    color = "C3"
                    fy_A4 = np.array([model(t, best_parameter)[1] for t in trange])
                    fy_err_A4 = np.array([self.model_prediction_var(t, best_parameter, best_parameter_cov, lambda x,y: model.parameter_gradient(x,y)[1]) for t in trange])**.5
                    ax1.plot(trange, fy_A4, color=color, lw=.5, label=model_label["sinh"])
                    ax1.fill_between(trange, fy_A4-fy_err_A4, fy_A4+fy_err_A4, alpha=0.5, color=color)
                    ax1.set_ylim(mean1[2], mean1[-2])
                    ax1.legend(loc="upper right")
                    # fit range marker
                    ax0.axvline(t[0], color="gray", linestyle="--")
                    ax0.axvline(t[-1], color="gray", linestyle="--")
                    plt.tight_layout()
                    plt.plot()
        return A_PS_dict, A_PS_var_dict, A_A4_dict, A_A4_var_dict, m_dict, m_var_dict 