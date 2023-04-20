#!/usr/bin/env python3

import os, h5py
from time import time
import numpy as np
from statpy.dbpy import custom_json as json
from statpy.dbpy.leafs import Leaf #, SampleLeaf
import statpy as sp

###################################### DATABASE SYSTEM USING LEAFS CONTAINING MEAN AND JKS (SECONDARY OBSERVABLES) ###########################################

class JKS_DB:
    db_type = "JKS-DB"
    def __init__(self, *args, verbose=True):
        self.t0 = time()
        self.verbose = verbose
        self.database = {}
        for src in args:
            self.add_src(src)

    def add_src(self, src):
        assert os.path.isfile(src)
        self.message(f"load {src}")
        with open(src) as f:
            src_db = json.load(f)
        for t, lf in src_db.items():
            self.database[t] = Leaf(lf.mean, lf.jks)

    def add_Leaf(self, tag, mean, jks, sample=None, nrwf=None, info=None):
        assert (isinstance(sample, dict) or sample==None)
        assert (isinstance(jks, dict) or jks==None)
        assert (isinstance(nrwf, dict) or nrwf==None)
        assert (isinstance(info, dict) or info==None)
        self.database[tag] = Leaf(mean, jks, sample, nrwf, info)

    def message(self, s):
        if self.verbose: print(f"{self.db_type}:\t\t{time()-self.t0:.6f}s: " + s)
    
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
                if len(lf.info) != 0:
                    s += f'\t└── info\n'
        return s

    def print_info(self, tag):
        self.message(self.__info_str__(tag))

    def __info_str__(self, tag):
        s = f'\n\n\tINFO DICT OF {tag}\n\n'
        for k, i in self.database[tag].info.items():
            s += f'\t{k:20s}: {i}\n'
        return s

    # helper function
    def as_array(self, obj):
        if isinstance(obj, np.ndarray):
            return obj
        else:
            return np.array(list(obj.values()))
    
    ################################## FUNCTIONS #######################################

    def apply_f(self, f, tag, dst_tag):
        lf = self.database[tag]
        mean = f(lf.mean)
        jks = {}
        for jks_tag in lf.jks:
            jks[jks_tag] = f(lf.jks[jks_tag])
        self.database[dst_tag] = Leaf(mean, jks, None)

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
        self.database[dst_tag] = Leaf(mean, jks, None)

    ################################## STATISTICS ######################################

    def compute_jks(self, tag, binsize, permutation_avg=False):
        lf = self.database[tag]
        if binsize == 1:
            return lf.jks
        # get sorted jks-tags 
        jks_tags = sorted(list(lf.jks.keys()), key=lambda x: int(x.split("-")[-1]))
        N = len(jks_tags)
        Nb = N // binsize 
        jks_tags = jks_tags[:Nb*binsize] # cut off excess data
        jks_bin = {}
        for i in range(Nb):
            jks_avg = []
            # average over permutations
            if permutation_avg == True: permutations = np.arange(binsize)
            else: permutations = np.arange(1)
            for p in permutations:
                s = sum([lf.jks[np.roll(jks_tags, p)[idx]] for idx in np.arange(i*binsize, (i+1)*binsize)]) - binsize * lf.mean
                jks_avg.append(lf.mean + s * (N-1) / (N-binsize))
            jks_bin[i] = np.mean(jks_avg, axis=0)
        return jks_bin

    def jackknife_variance(self, tag, binsize, permutation_avg=False):
        jks = self.as_array(self.compute_jks(tag, binsize, permutation_avg))
        return sp.statistics.jackknife.variance_jks(np.mean(jks, axis=0), jks)
    
    def jackknife_covariance(self, tag, binsize, permutation_avg=False):
        jks = self.as_array(self.compute_jks(tag, binsize, permutation_avg))
        return sp.statistics.jackknife.covariance_jks(np.mean(jks, axis=0), jks)

    def binning_study(self, tag, binsizes, permutation_avg=False):
        self.message(f"Unbinned sample size: {len(self.database[tag].sample)}")
        var = {}
        for b in binsizes:
            var[b] = self.jackknife_variance(tag, b, permutation_avg)
        return var

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

    def gradient_flow_scale(self, ensemble_label, binsize, verbose=True):
        tau = self.database[ensemble_label + "/tau"].mean
        scale = sp.qcd.scale_setting.gradient_flow_scale()
        # tau0
        self.apply_f(lambda x: scale.set_sqrt_tau0(tau, x), ensemble_label + "/E", ensemble_label + "/sqrt_tau0")
        sqrt_tau0_var = self.jackknife_variance(ensemble_label + "/sqrt_tau0", binsize)
        # t0
        self.apply_f(lambda x: scale.comp_sqrt_t0(x, scale.sqrt_t0_fm), ensemble_label + "/sqrt_tau0", ensemble_label + "/sqrt_t0") 
        sqrt_t0_stat_var = self.jackknife_variance(ensemble_label + "/sqrt_t0", binsize)
        # propagate systematic error of t0
        sqrt_t0_mean_shifted = scale.comp_sqrt_t0(self.database[ensemble_label + "/sqrt_tau0"].mean, scale.sqrt_t0_fm + scale.sqrt_t0_fm_std)
        sqrt_t0_sys_var = (self.database[ensemble_label + "/sqrt_t0"].mean - sqrt_t0_mean_shifted)**2.0
        self.database[ensemble_label + "/sqrt_t0"].info["sqrt_t0_stat_var"] = sqrt_t0_stat_var
        self.database[ensemble_label + "/sqrt_t0"].info["sqrt_t0_mean_shifted"] = sqrt_t0_mean_shifted
        self.database[ensemble_label + "/sqrt_t0"].info["sqrt_t0_sys_var"] = sqrt_t0_sys_var
        self.database[ensemble_label + "/sqrt_t0"].info["sqrt_t0_var"] = sqrt_t0_stat_var + sqrt_t0_sys_var
        # omega0
        self.apply_f(lambda x: scale.set_omega0(tau, x), ensemble_label + "/E", ensemble_label + "/omega0")
        omega0_var = self.jackknife_variance(ensemble_label + "/omega0", binsize)
        # w0
        self.apply_f(lambda x: scale.comp_w0(x, scale.w0_fm), ensemble_label + "/omega0", ensemble_label + "/w0") 
        w0_stat_var = self.jackknife_variance(ensemble_label + "/w0", binsize)
        # propagate systematic error of w0
        w0_mean_shifted = scale.comp_w0(self.database[ensemble_label + "/omega0"].mean, scale.w0_fm + scale.w0_fm_std)
        w0_sys_var = (self.database[ensemble_label + "/w0"].mean - w0_mean_shifted)**2.0
        self.database[ensemble_label + "/w0"].info["w0_stat_var"] = w0_stat_var
        self.database[ensemble_label + "/w0"].info["w0_mean_shifted"] = w0_mean_shifted
        self.database[ensemble_label + "/w0"].info["w0_sys_var"] = w0_sys_var
        self.database[ensemble_label + "/w0"].info["w0_var"] = w0_stat_var + w0_sys_var
        if verbose:
            self.message(f"sqrt(tau0) = {self.database[ensemble_label + '/sqrt_tau0'].mean:.4f} +- {sqrt_tau0_var**.5:.4f}")
            self.message(f"omega0 = {self.database[ensemble_label + '/omega0'].mean:.4f} +- {omega0_var**.5:.4f}")
            self.message(f"t0/GeV (cutoff) = {self.database[ensemble_label + '/sqrt_t0'].mean:.4f} +- {sqrt_t0_stat_var**.5:.4f} (STAT) +- {sqrt_t0_sys_var**.5:.4f} (SYS) [{(sqrt_t0_stat_var+sqrt_t0_sys_var)**.5:.4f} (STAT+SYS)]")
            self.message(f"w0/GeV (cutoff) = {self.database[ensemble_label + '/w0'].mean:.4f} +- {w0_stat_var**.5:.4f} (STAT) +- {w0_sys_var**.5:.4f} (SYS) [{(w0_stat_var+w0_sys_var)**.5:.4f} (STAT+SYS)]")

    ################################## FITTING ######################################
 
    def fit(self, t, tag, binsize, cov, p0, model, method, minimizer_params, dst_tag, verbosity=0):
        y = self.database[tag].mean
        y_jks = self.compute_jks(tag, binsize) 
        fitter = sp.fitting.Fitter(t, cov, model, lambda x: x, method, minimizer_params)
        best_parameter, chi2, _ = fitter.estimate_parameters(fitter.chi_squared, y, p0)
        best_parameter_jks = {}
        for cfg in y_jks:
            best_parameter_jks[cfg], _, _ = fitter.estimate_parameters(fitter.chi_squared, y_jks[cfg], best_parameter)
        best_parameter_cov = sp.statistics.jackknife.covariance_jks(best_parameter, best_parameter_jks)
        if verbosity >=1: 
            print(f"jackknife parameter covariance is ", best_parameter_cov) 
        dof = len(t) - len(best_parameter)
        pval = fitter.get_pvalue(chi2, dof)
        if verbosity >= 0:
            for i in range(len(best_parameter)):
                print(f"parameter[{i}] = {best_parameter[i]} +- {best_parameter_cov[i][i]**0.5}")
            print(f"chi2 / dof = {chi2} / {dof} = {chi2/dof}, i.e., p = {pval}")
        self.add_Leaf(dst_tag, best_parameter, best_parameter_jks, None, None, 
                      info={"best_parameter_cov": best_parameter_cov, "chi2": chi2, "dof": dof, "pvalue": pval})
        return best_parameter, best_parameter_cov 

    def fit_indep(self, t, tags, binsizes, cov, model, p0, method, minimizer_params, verbosity=0):
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
    
################################# DATABASE SYSTEM USING LEAFS CONTAINING SAMPLE, MEAN AND JKS (PRIMARY and SECONDARY OBSERVABLES) #####################################

class Sample_DB(JKS_DB):
    db_type = "SAMPLE-DB"
    def add_src(self, src):
        assert os.path.isfile(src)
        self.message(f"load {src}")
        with open(src) as f:
            src_db = json.load(f)
        for t, lf in src_db.items():
            self.database[t] = Leaf(lf.mean, lf.jks, lf.sample)

    def remove_cfgs(self, cfgs, tag):
        for cfg in cfgs:
            self.database[tag].sample.pop(str(cfg), None)

    def cfgs(self, tag):
        return sorted([int(x.split("-")[-1]) for x in self.database[tag].sample.keys()])

    ################################## FUNCTIONS #######################################
    
    def apply_f_to_sample(self, f, tag, dst_tag):
        lf = self.database[tag] 
        sample = {}
        for cfg in lf.sample:
            sample[cfg] = f(lf.sample[cfg])
        if tag == dst_tag:
            self.database[dst_tag].sample = sample
        else:
            self.database[dst_tag] = Leaf(None, None, sample)

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

############################## DATABASE SYSTEM USING LEAFS CONTAINING MEAN, JKS, SAMPLE and RWF (PRIMARY and SECONDARY OBSERVABLES) ##################################

class SampleRWF_DB(Sample_DB):
    db_type = "SAMPLE-RWF-DB" 
    def add_src(self, src):
        assert isinstance(src, tuple)
        sample_path, rwf_path, ensemble_label, observable_tag = src
        assert os.path.isfile(sample_path)
        assert os.path.isfile(rwf_path)
        self.message(f"load {sample_path} with {rwf_path}")
        sample = np.array(h5py.File(sample_path, "r").get(observable_tag)[:]); sample = {f"{ensemble_label}-{cfg}":val for cfg, val in enumerate(sample)}
        rwf = np.loadtxt(rwf_path)[:,1]; nrwf = rwf / np.mean(rwf); nrwf = {f"{ensemble_label}-{cfg}":val for cfg, val in enumerate(nrwf)}
        self.add_Leaf(ensemble_label + "/" + observable_tag, None, None, sample, nrwf)
    
#class sample_db_from_data(sample_db):
#    def __init__(self, *args, verbose=True):
#        self.t0 = time()
#        self.verbose = verbose
#        self.database = {}
#        for src in args:
#            assert isinstance(src, tuple)
#            self.add_Leaf(*src)



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
