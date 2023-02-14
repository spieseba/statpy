#!/usr/bin/env python3

import os, sys
import numpy as np
import statpy.dbpy.np_json as json
import statpy as sp
import matplotlib.pyplot as plt

STRIP_TAGS = ["mean", "jkvar", "jks"]

class DBpy:
    def __init__(self, file, safe_mode=False):
        self.file = file
        if os.path.isfile(self.file):
            with open(self.file) as f:
                self.database = json.load(f)
        else:
            self.database = {}
        
        self.safe_mode = safe_mode


    def _query_yes_no(self, question, default="yes"):
        if self.safe_mode:
            query_yes_no(question, default)
        else: 
            return True

    def __str__(self, verbosity=0):
        s = 'DATABASE CONSISTS OF\n\n\n'
        for tag, tag_dict in self.database.items():
            s += f'{tag:20s}\n'
            if verbosity >= 1:
                for sample_tag, sample_dict in tag_dict.items():
                    s += f'└── {sample_tag:20s}\n'
                    if verbosity >= 2:
                        for cfg_tag, val in sample_dict.items():
                            s += f'\t└── {cfg_tag}\n'
                            if verbosity >= 3:
                                s += '\t\t' + f'{val.__str__()}'.replace('\n', '\n\t\t')
                            s += '\n'
        return s

    def print(self, verbosity):
        print(self.__str__(verbosity=verbosity))  

    def print_sample(self, tag, sample_tag, verbosity=0):
        for cfg_tag, val in self.database[tag][sample_tag].items():
            s += f'\t└── {cfg_tag}\n'
            if verbosity >= 1:
                s += '\t\t' + f'{val.__str__()}'.replace('\n', '\n\t\t')
            s += '\n'

    def _add_data(self, data, tag, sample_tag, cfg_tag):
        if tag in self.database:
            if sample_tag in self.database[tag]:
                self.database[tag][sample_tag][cfg_tag] = data
            else:
                self.database[tag][sample_tag] = {cfg_tag: data}
        else:
            self.database[tag] = {sample_tag: {cfg_tag: data}} 

    def add_data(self, data, tag, sample_tag, cfg_tag):
        try: 
            self.database[tag][sample_tag][cfg_tag]
            if self._query_yes_no(f"{cfg_tag} is already in database for {tag}/{sample_tag} Overwrite?"):
                self._add_data(data, tag, sample_tag, cfg_tag)
        except KeyError:
            self._add_data(data, tag, sample_tag, cfg_tag)

    def _add_data_arr(self, data, tag, sample_tag, idxs, cfg_prefix, overwrite=True):
        try:
            self.database[tag]
        except KeyError:
            self.database[tag] = {}
        if overwrite:
            self.database[tag][sample_tag] = {}
        for idx in range(len(data)):
            self._add_data(data[idx], tag, sample_tag, cfg_prefix + str(idxs[idx]))

    def add_data_arr(self, data, tag, sample_tag, idxs=None, cfg_prefix=""):
        if idxs == None:
            idxs = range(len(data))
        try:
            self.database[tag][sample_tag]
            if self._query_yes_no(f"{sample_tag} is already in database for {tag}. Overwrite?"):
                self._add_data_arr(data, tag, sample_tag, idxs, cfg_prefix)
        except KeyError:
            self._add_data_arr(data, tag, sample_tag, idxs, cfg_prefix)

    def add_tag_entry(self, dst_tag, src, src_tag):
        with open(src) as f:
            src_db = json.load(f)
        try:
            self.database[dst_tag]
            if self._query_yes_no(f"{dst_tag} is already in database. Overwrite?"):
                self.database[dst_tag] = src_db[src_tag]
        except KeyError:
            self.database[dst_tag] = src_db[src_tag]

    def add_sample_tag_entry(self, dst_tag, dst_sample_tag, src, src_tag):
        with open(src) as f:
            src_db = json.load(f)
        try:
            self.database[dst_tag]
            try: 
                self.database[dst_tag][dst_sample_tag]
                if self._query_yes_no(f"{dst_sample_tag} is already in database for {dst_tag}. Overwrite?"):
                    self.database[dst_tag][dst_sample_tag] = src_db[src_tag]
            except KeyError:
                self.database[dst_tag][dst_sample_tag] = src_db[src_tag]
        except KeyError:
            self.database[dst_tag] = {dst_sample_tag: src_db[src_tag]}

    def apply_f(self, f, tag, sample_tag, dst_tag, dst_sample_tag, overwrite=False):
        f_dict = {}
        for cfg_idx, cfg_val in self.database[tag][sample_tag].items():  
            if cfg_idx not in STRIP_TAGS:
                f_dict[cfg_idx] = f(cfg_val)
        try:
            self.database[dst_tag]
            try: 
                self.database[dst_tag][dst_sample_tag]
                if overwrite:
                    self.database[dst_tag][dst_sample_tag] = f_dict
                elif self._query_yes_no(f"{dst_tag} is already in database. Overwrite?"):
                    self.database[dst_tag][dst_sample_tag] = f_dict
            except KeyError:
                self.database[dst_tag][dst_sample_tag] = f_dict 
        except KeyError:
            self.database[dst_tag] = {dst_sample_tag: f_dict}

    def delete(self, tag):
        self._delete(tag, self.database)

    def _delete(self, tag, db):
        if tag in db:
            del db[tag]
        for sub_db in db.values():
            if isinstance(sub_db, dict):
                self._delete(tag, sub_db)

    def get_tag(self, tag):
        return self._get_tag(tag, self.database)

    def _get_tag(self, tag, db):
        if tag in db:
            return db[tag]
        for sub_db in db.values():
            if isinstance(sub_db, dict):
                return self._get_tag(tag, sub_db)

    def get_data(self, tag, sample_tag, cfg_tag):
        try:
            return self.database[tag][sample_tag][cfg_tag]
        except KeyError:
            print(f"requested data not in database. tag = {tag}, sample_tag = {sample_tag}, cfg_tag = {cfg_tag}")
            return None
        
    def get_data_dict(self, tag, sample_tag, strip=True):
        try:
            d = dict(self.database[tag][sample_tag]).copy()
            if strip  == True:
                for key in STRIP_TAGS:
                    d.pop(key, None)
            return d
        except KeyError:
            print(f"requested data not in database. tag = {tag}, sample_tag = {sample_tag}")
            return None

    def get_data_arr(self, tag, sample_tag):
        try:
            return np.array(list(self.get_data_dict(tag, sample_tag).values()))
        except KeyError:
            print(f"requested data not in database. tag = {tag}, sample_tag = {sample_tag}")
            return None

    def save(self):
        with open(self.file, "w") as f:
            json.dump(self.database, f)

###################################### STATISTICS ######################################

    def sample_mean(self, tag, sample_tag, data_axis=0, store=True):
        mean = np.mean(self.get_data_arr(tag, sample_tag), axis=data_axis)
        if store:
            self.add_data(mean, tag, sample_tag, "mean")
        return mean

    def jackknife_resampling(self, f, tag, sample_tag, eps=1.0, store=True, dst_tag=None, dst_sample_tag=None):
        if dst_tag==None: dst_tag = tag
        data = self.get_data_dict(tag, sample_tag)
        mean = self.sample_mean(tag, sample_tag)
        jks = {}
        for cfg_tag in data.keys():
            jks[cfg_tag] = f( mean + eps*(mean - data[cfg_tag]) / (len(data) - 1) ) 
        if store:
            if dst_sample_tag == None:
                dst_sample_tag = sample_tag 
            self.add_data(jks, dst_tag, dst_sample_tag, "jks")
        return jks

    def jackknife_variance(self, f, tag, sample_tag, eps=1.0, dst_tag=None, dst_sample_tag=None, dst_jks_sample_tag=None, return_var=True, store=True):
        if dst_tag==None: dst_tag = tag
        if dst_sample_tag==None: dst_sample_tag = sample_tag
        if dst_jks_sample_tag==None: dst_jks_sample_tag = sample_tag
        self.jackknife_resampling(f, tag, sample_tag, eps, store, dst_tag, dst_jks_sample_tag)
        f_mean = f( self.database[tag][sample_tag]["mean"] )
        jks_data = np.array(list(self.get_data(dst_tag, dst_jks_sample_tag, "jks").values()))
        N = len(jks_data)
        var = np.mean([ (jks_data[k] - f_mean)**2 for k in range(N) ], axis=0) * (N - 1)
        if store:
            self.add_data(var, dst_tag, dst_sample_tag, "jkvar")
        if return_var:
            return var

    def jackknife_variance_jks(self, mean, jks, dst_tag, dst_sample_tag, dst_cfg_prefix="", store=True):
        var = sp.statistics.jackknife.variance_jks(mean, np.array(list(jks.values())))
        if store:
            self.add_data(var, dst_tag, dst_sample_tag, dst_cfg_prefix+"jkvar")
            self.add_data(jks, dst_tag, dst_sample_tag, dst_cfg_prefix+"jks")
        return var

    def jackknife_covariance_jks(self, mean, jks, dst_tag, dst_sample_tag, dst_cfg_prexix="", store=True):
        cov = sp.statistics.jackknife.covariance_jks(mean, np.array(list(jks.values())))
        if store:
            self.add_data(cov, dst_tag, dst_sample_tag, dst_cfg_prexix+"jkcov")
            self.add_data(jks, dst_tag, dst_sample_tag, dst_cfg_prexix+"jks")
        return cov

    def bin(self, binsize, tag, sample_tag, cfg_prefix="", store=True):
        data_arr = self.get_data_arr(tag, sample_tag)
        binned_data = {}
        for binned_cfg, binned_value in enumerate(sp.statistics.bin(data_arr, binsize)):
            binned_data[cfg_prefix + str(binned_cfg)] = binned_value
        if store:
            self.database[tag][f"{sample_tag}_binned{binsize}"] = binned_data
        return binned_data

    def binning_study(self, tag, sample_tag, binsizes=[1,2,4,8], keep_binsizes=[]):
        stds = {}
        for binsize in binsizes:
            self.bin(binsize, tag, sample_tag)
            stds[binsize] = np.array([y**0.5 for y in self.jackknife_variance(lambda x: x, tag, sample_tag + f"_binned{binsize}")])
        # clean up
        for binsize in [binsize for binsize in binsizes if binsize not in keep_binsizes]:
            del self.database[tag][sample_tag + f"_binned{binsize}"]
        return stds   

    def AMA(self, exact_exact_tag, exact_exact_sample_tag, exact_sloppy_tag, exact_sloppy_sample_tag, sloppy_sloppy_tag, sloppy_sloppy_sample_tag, dst_tag=None, dst_sample_tag=None, store=True):
        if dst_tag == None: dst_tag = exact_exact_tag
        if dst_sample_tag == None: dst_sample_tag = exact_exact_sample_tag + "_AMA"
        def bias(exact, sloppy):
            return exact - sloppy   
        exact_exact_jks = self.jackknife_resampling(lambda x: x, exact_exact_tag, exact_exact_sample_tag)
        exact_sloppy_jks = self.jackknife_resampling(lambda x: x, exact_sloppy_tag, exact_sloppy_sample_tag)
        sloppy_sloppy_jks = self.jackknife_resampling(lambda x: x, sloppy_sloppy_tag, sloppy_sloppy_sample_tag)
        b = bias(self.get_data(exact_exact_tag, exact_exact_sample_tag, "mean"), self.get_data(exact_sloppy_tag, exact_sloppy_sample_tag, "mean"))
        b_jks = {}
        for cfg in exact_exact_jks:
            b_jks[cfg] = bias(exact_exact_jks[cfg], exact_sloppy_jks[cfg])
        b_jkvar = self.jackknife_variance_jks(b, b_jks, dst_tag, dst_sample_tag, dst_cfg_prefix="bias_")
        m = self.get_data(sloppy_sloppy_tag, sloppy_sloppy_sample_tag, "mean") + b
        m_jks = {}
        for cfg in sloppy_sloppy_jks:
            m_jks[cfg] = sloppy_sloppy_jks[cfg] + b_jks[cfg]
        m_jkvar = self.jackknife_variance_jks(m, m_jks, dst_tag, dst_sample_tag)
        if store: 
            self.add_data(b, dst_tag, dst_sample_tag, "bias")
            self.add_data(m, dst_tag, dst_sample_tag, "mean")
        return m, m_jkvar, b, b_jkvar

###################################### FUNCTIONS #######################################

    def mean_ratio(self, tag0, sample_tag0, tag1, sample_tag1, dst_tag, dst_sample_tag, jks_tag0=None, jks_tag1=None, store=True):
        def ratio(nom, denom):
            return nom / denom
        if jks_tag0 != None:
            data0_jks = self.get_data(tag0, sample_tag0, jks_tag0)
        else:
            data0_jks = self.jackknife_resampling(lambda x: x, tag0, sample_tag0)
        if jks_tag1 != None:
            data1_jks = self.get_data(tag1, sample_tag1, jks_tag1) 
        else:
            data1_jks = self.jackknife_resampling(lambda x: x, tag1, sample_tag1)
        
        r = ratio(self.get_data(tag0, sample_tag0, "mean"), self.get_data(tag1, sample_tag1, "mean"))
        r_jks = {}
        for cfg in data0_jks:
            r_jks[cfg] = ratio(data0_jks[cfg], data1_jks[cfg])
        r_jkvar = self.jackknife_variance_jks(r, r_jks, dst_tag, dst_sample_tag)
        if store:
            self.add_data(r, dst_tag, dst_sample_tag, "mean")
        return r, r_jkvar

################################### SCALE SETTING ###################################

    def energy_density(self, Et_tag, sample_tag, tlo, thi, return_Edens=True):
        Edens = np.mean(self.get_data_arr(Et_tag, sample_tag)[:,:,tlo:thi], axis=2).real
        self.add_data_arr(Edens, "Edens" + Et_tag.split("Et")[1], sample_tag)
        if return_Edens:
            return Edens

    def set_scale(self, Edens_tag, sample_tag, binsize, tau, nskip=0, scales=["t0", "w0"], dst_suffix="", store=True):
        Edens_binned = sp.statistics.bin(self.get_data_arr(Edens_tag, sample_tag)[nskip:], binsize)
        if binsize != 1:
            sample_tag = sample_tag + f"_binned{binsize}"
            self.add_data_arr(Edens_binned, Edens_tag, sample_tag)
        print("effective number of measurements: ", len(Edens_binned))
        scale = sp.qcd.scale_setting.scale(tau, Edens_binned)
        if "t0" in scales:
            sqrt_tau0, sqrt_tau0_std, t2E, t2E_std, aGeV_inv_t0, aGeV_inv_std_t0 = scale.lattice_spacing("t0")
            if store:
                self.add_data(sqrt_tau0, "flow_scale_t0" + dst_suffix, sample_tag, "sqrt_tau0")
                self.add_data(sqrt_tau0_std, "flow_scale_t0" + dst_suffix, sample_tag, "sqrt_tau0_std")
                self.add_data(t2E, "flow_scale_t0" + dst_suffix, sample_tag, "t2E")
                self.add_data(t2E_std, "flow_scale_t0" + dst_suffix, sample_tag, "t2E_std")
                self.add_data(aGeV_inv_t0, "flow_scale_t0" + dst_suffix, sample_tag, "aGeV_inv")
                self.add_data(aGeV_inv_std_t0, "flow_scale_t0" + dst_suffix, sample_tag, "aGeV_inv_std")
        if "w0" in scales:
            wau0, wau0_std, tdt2E, tdt2E_std, aGeV_inv_w0, aGeV_inv_std_w0 = scale.lattice_spacing("w0")
            if store:
                self.add_data(wau0, "flow_scale_w0" + dst_suffix, sample_tag, "wau0")
                self.add_data(wau0_std, "flow_scale_w0" + dst_suffix, sample_tag, "wau0_std")
                self.add_data(tdt2E, "flow_scale_w0" + dst_suffix, sample_tag, "tdt2E")
                self.add_data(tdt2E_std, "flow_scale_w0" + dst_suffix, sample_tag, "tdt2E_std")
                self.add_data(aGeV_inv_w0, "flow_scale_w0" + dst_suffix, sample_tag, "aGeV_inv")
                self.add_data(aGeV_inv_std_w0, "flow_scale_w0" + dst_suffix, sample_tag, "aGeV_inv_std")

    

    

###################################### FITTING ######################################

    def multi_mc_fit(self, t, tags, sample_tags, C, model, p0, estimator, fit_tag="FIT", method="Nelder-Mead", minimizer_params={}, verbose=True, store=False, return_fitter=False):

        mos = []
        for tag, sample_tag in zip(tags, sample_tags):
            mos.append(list(map(lambda e: (tag, e), sample_tag)))
        y = []
        for mo in mos:
            for ts in mo:
                ta, s = ts
                y.append(self.get_data_arr(ta, s))

        assert isinstance(model, dict)
        model_func = list(model.values())[0]        
        assert method in ["Levenberg-Marquardt", "Migrad", "Nelder-Mead"]
        if method in ["Migrad", "Nelder-Mead"]:
            fitter = sp.fitting.fit(t, y, C, model_func, p0, estimator, method, minimizer_params)
        else:
            fitter = sp.fitting.LM_fit(t, y, C, model_func, p0, estimator, minimizer_params)

        fitter.multi_mc_fit(verbose)

        if store:
            self.add_data(t, fit_tag, "t", "-")
            self.add_data(C, fit_tag, "C", "-")
            self.add_data(list(model.keys())[0], fit_tag, "model", "-")
            self.add_data(method, fit_tag, "minimizer", "-")
            self.add_data(minimizer_params, fit_tag, "minimizer_params", "-")
            self.add_data(p0, fit_tag, "p0", "-")
            self.add_data(fitter.best_parameter, fit_tag, "best_parameter", "-")
            self.add_data(fitter.best_parameter_cov, fit_tag, "best_parameter_cov", "-")
            self.add_data(fitter.jks_parameter, fit_tag, "jks_parameter", "-")
            self.add_data(fitter.p, fit_tag, "pval", "-")
            self.add_data(fitter.chi2, fit_tag, "chi2", "-")
            self.add_data(fitter.dof, fit_tag, "dof", "-")
            self.add_data(fitter.fit_err, fit_tag, "fit_err", "-")
        
        if return_fitter:
            return fitter

###################################### SPECTROSCOPY ######################################

    def Ct_binning_study(self, Ct_tag, sample_tag, binsizes, keep_binsizes, t=None, shift=0):
        stds = self.binning_study(Ct_tag, sample_tag, binsizes, keep_binsizes)
        nbins = list(stds.keys())
        if t == None:
            print(f"Binning study of {Ct_tag} (nbin={nbins}):\n", [np.roll(stds[n], shift) for n in nbins])
        else:
            print(f"Binning study of {Ct_tag} for t={t} (nbin={nbins}):\n", [np.roll(stds[n], shift)[t] for n in nbins])

    def effective_mass_log(self, Ct_tag, sample_tag, dst_tag, tmax, shift=0, store=True):
        Ct = self.get_data(Ct_tag, sample_tag, "mean")
        m_eff_log = sp.qcd.spectroscopy.effective_mass_log(Ct, tmax, shift)
        self.add_data(m_eff_log, dst_tag, sample_tag, "mean")
        self.jackknife_variance(lambda Ct: sp.qcd.spectroscopy.effective_mass_log(Ct, tmax, shift), Ct_tag, sample_tag, dst_tag=dst_tag, return_var=False, store=store)
        return self.get_data(dst_tag, sample_tag, "mean"), self.get_data(dst_tag, sample_tag, "jkvar")**0.5

    def effective_mass_acosh(self, Ct_tag, sample_tag, dst_tag, tmax, shift=0, store=True):
        Ct = self.get_data(Ct_tag, sample_tag, "mean")
        m_eff_cosh = sp.qcd.spectroscopy.effective_mass_acosh(Ct, tmax, shift)
        self.add_data(m_eff_cosh, dst_tag, sample_tag, "mean")
        self.jackknife_variance(lambda Ct: sp.qcd.spectroscopy.effective_mass_acosh(Ct, tmax, shift), Ct_tag, sample_tag, dst_tag=dst_tag, return_var=False, store=store)    
        return self.get_data(dst_tag, sample_tag, "mean"), self.get_data(dst_tag, sample_tag, "jkvar")**0.5

    def correlator_exp_fit(self, t, Ct_tag, sample_tag, cov, p0, bc="pbc", min_method="Nelder-Mead", min_params={}, shift=0, verbose=True, dst_tag="CORR_FIT", store=True):        
       
        Ct_mean = self.get_data(Ct_tag, sample_tag, "mean")
        Ct_jks = self.get_data(Ct_tag, sample_tag, "jks")
        Nt = len(Ct_mean)
        best_parameter, chi2, pval, dof, model = sp.qcd.spectroscopy.correlator_exp_fit(t, Ct_mean[t], cov[t][:,t], p0, bc, Nt, min_method, min_params, shift, verbose)
        best_parameter_jks = {}
        for cfg in Ct_jks:
            best_parameter_jks[cfg], _, _, _, _ = sp.qcd.spectroscopy.correlator_exp_fit(t, Ct_jks[cfg][t], cov[t][:,t], p0, Nt, min_method, min_params, shift, verbose=False)
        best_parameter_cov = sp.statistics.jackknife.covariance_jks(best_parameter, np.array(list(best_parameter_jks.values())))
        fit_err = lambda x: sp.fitting.fit_std_err(x, parameter, model.parameter_gradient, parameter_cov)
        if store:
            model_str = {"pbc": "p[0] * exp(-p[1]t)", "obc": "p[0] * [exp(-p[1]t) + exp(-p[1](T-t))]"}
            try:
                self.database[dst_tag]
            except KeyError:
                self.database[dst_tag] = {}
            self.database[dst_tag][dst_sample_tag] = {}
            self.database[dst_tag][dst_sample_tag]["t"] = t
            self.database[dst_tag][dst_sample_tag]["model"] = model_str[bc]
            self.database[dst_tag][dst_sample_tag]["model_func"] = model
            self.database[dst_tag][dst_sample_tag]["minimizer"] = min_method
            self.database[dst_tag][dst_sample_tag]["minimizer_params"] = min_params
            self.database[dst_tag][dst_sample_tag]["p0"] = p0
            self.database[dst_tag][dst_sample_tag]["best_parameter"] = best_parameter
            self.database[dst_tag][dst_sample_tag]["best_parameter_cov"] = best_parameter_cov
            self.database[dst_tag][dst_sample_tag]["best_parameter_jks"] = best_parameter_jks
            self.database[dst_tag][dst_sample_tag]["pval"] = pval
            self.database[dst_tag][dst_sample_tag]["chi2"] = chi2
            self.database[dst_tag][dst_sample_tag]["dof"] = dof
            self.database[dst_tag][dst_sample_tag]["fit_err"] = fit_err 
    return best_parameter, best_parameter_cov, best_parameter_jks

    def effective_mass_curve_fit(self, t0min, t0max, nt, Ct_tag, sample_tag, cov, p0, bc="pbc", method="Nelder-Mead", min_params={}, shift=0, verbose=True, dst_tag="M_EFF_CURVE_FIT", dst_tag_Ct="Ct_FIT", store=True):
        mt = []; mt_var = []; best_parameter_jks_arr = []
        for t0 in range(t0min, t0max):
            t = np.arange(t0, t0+nt)
            best_parameter, best_parameter_cov, best_parameter_jks = self.correlator_exp_fit(t, Ct_tag, sample_tag, cov, p0, bc, min_method, min_params, shift, verbose=verbose, dst_tag=dst_tag_Ct, store)
            mt.append(best_parameter[1])
            mt_var.append(best_parameter_cov[1][1])
            best_parameter_jks_arr.append(best_parameter_jks)    
    
        mt = np.array(mt); mt_var = np.array(mt_var)
        mt_jks = {}
        for cfg in best_parameter_jks_arr[0]:
            best_parameter_jks_arr    
        
        self.add_data(mt, dst_tag, dst_sample_tag, "mean")
        self.add_data(mt_var, dst_tag, dst_sample_tag, "jkvar")
        self.add_data(mt_jks_dict, dst_tag, dst_sample_tag, "jks")

        return mt, mt_var


    def effective_mass_const_fit(self, t, mt_tag, sample_tag, dst_tag, p0, method, minimizer_params={}, verbose=True, store=True):    
        mt = self.get_data(mt_tag, sample_tag, "mean")[t]
        mt_cov = np.diag(self.get_data(mt_tag, sample_tag, "jkvar"))[t[0]:t[-1]+1,t[0]:t[-1]+1]
        m, p, chi2, dof, model = sp.qcd.spectroscopy.const_fit(t, np.array([mt]), mt_cov, p0, method, minimizer_params, error=False, verbose=False)

        mt_jks = self.get_data(mt_tag, sample_tag, "jks")
        m_jks = []
        for mt_jk in mt_jks.values():
            m_jk, _, _, _, _ = sp.qcd.spectroscopy.const_fit(t, np.array([mt_jk[t]]), mt_cov, p0=m, method=method, minimizer_params=minimizer_params, error=False, verbose=False)
            m_jks.append(m_jk)

        m_cov = sp.statistics.jackknife.covariance_jks(m, m_jks)
        model = sp.qcd.spectroscopy.const_model()
        fit_err = lambda t: (model.parameter_gradient(t,m) @ m_cov @ model.parameter_gradient(t,m))**0.5

        if verbose:
            print("*** constant mass fit ***")
            print("fit window:", t)
            print(f"m_eff = {m[0]} +- {m_cov[0][0]**.5}")
            print(f"chi2 / dof = {chi2} / {dof} = {chi2/dof}, i.e., p = {p}")

        if store:
            try:
                self.database[dst_tag]
            except KeyError:
                self.database[dst_tag] = {}
            self.database[dst_tag][sample_tag] = {}
            self.database[dst_tag][sample_tag]["fit_window"] = t
            self.database[dst_tag][sample_tag]["mt_cov"] = mt_cov
            self.database[dst_tag][sample_tag]["model"] = "const."
            self.database[dst_tag][sample_tag]["model_func"] = model
            self.database[dst_tag][sample_tag]["minimizer"] = method
            self.database[dst_tag][sample_tag]["minimizer_params"] = minimizer_params
            self.database[dst_tag][sample_tag]["p0"] = p0
            self.database[dst_tag][sample_tag]["m_eff"] = m
            self.database[dst_tag][sample_tag]["m_eff_cov"] = m_cov
            self.database[dst_tag][sample_tag]["m_eff_jks"] = m_jks
            self.database[dst_tag][sample_tag]["pval"] = p
            self.database[dst_tag][sample_tag]["chi2"] = chi2
            self.database[dst_tag][sample_tag]["dof"] = dof
            self.database[dst_tag][sample_tag]["fit_err"] = fit_err

        return m[0], m_cov[0][0]**.5

################################################################################################################################################
################################################################################################################################################
################################################################################################################################################

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")
