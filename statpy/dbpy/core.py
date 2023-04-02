#!/usr/bin/env python3

import os, sys, copy
import numpy as np
from statpy.dbpy import np_json as json
import statpy as sp

class DBpy:
    def __init__(self, src, safe_mode=False):
        self.safe_mode = safe_mode
        assert type(src) == list 
        if os.path.isfile(src[0]):
            with open(src[0]) as f:
                self.database = json.load(f)
        else:
            self.database = {}
        self.update_tags()
        self.compute_means() 
        self.compute_jks()
        for s in src[1:]:
            self.merge(s)
    
    def save(self, dst):
        with open(dst, "w") as f:
            json.dump(self.database, f)
    
    def _query_yes_no(self, question, default="yes"):
        if self.safe_mode:
            query_yes_no(question, default)
        else: 
            return True
    
    def __str__(self, verbosity):
        s = 'DATABASE CONSISTS OF\n\n\n'
        for tag, tag_dict in self.database.items():
            s += f'{tag:20s}\n'
            if verbosity >= 1:
                for sample_tag, sample_dict in tag_dict.items():
                    s += f'└── {sample_tag:20s}\n'
                    if verbosity >= 2:
                        if tag != "cfgs":
                            for cfg_tag in sample_dict:
                                s += f'\t└── {cfg_tag}\n'
                            #if verbosity >= 3:
                            #    s += '\t\t' + f'{val.__str__()}'.replace('\n', '\n\t\t')
                            #s += '\n'
        return s
    
    def print(self, verbosity=1):
        print(self.__str__(verbosity=verbosity))  

    def _add_data(self, data, dst_tag, dst_sample_tag, dst_data_tag):
        if dst_tag in self.database:
            if dst_sample_tag in self.database[dst_tag]:
                self.database[dst_tag][dst_sample_tag][dst_data_tag] = data
            else:
                self.database[dst_tag][dst_sample_tag] = {dst_data_tag: data}
        else:
            self.database[dst_tag] = {dst_sample_tag: {dst_data_tag: data}}

    def add_data(self, data, dst_tag, dst_sample_tag, dst_data_tag):
        try: 
            self.database[dst_tag][dst_sample_tag][dst_data_tag]
            if self._query_yes_no(f"{dst_data_tag} is already in database for {dst_sample_tag} in {dst_tag}. Overwrite?"):
                self._add_data(data, dst_tag, dst_sample_tag, dst_data_tag)
        except KeyError:
            self._add_data(data, dst_tag, dst_sample_tag, dst_data_tag)
    
    def _get_data(self, tag, sample_tag, data_tag):
        try:
            return copy.deepcopy(self.database[tag][sample_tag][data_tag])
        except KeyError:
            print(f"requested data not in database. tag = {tag}, sample_tag = {sample_tag}, data_tag = {data_tag}")

    def get_data(self, tag, sample_tag, data_tag, array=False): 
        if array:
            return np.array(list(self._get_data(tag, sample_tag, data_tag).values()))
        else:
            return self._get_data(tag, sample_tag, data_tag)

    def merge(self, src):
        assert os.path.isfile(src)
        with open(src) as f:
            src_db = json.load(f)
        src_cfgs = src_db.pop("cfgs")
        # add src_data to db
        for tag in src_db:
            for sample_tag in src_db[tag]:
                for data_tag in src_db[tag][sample_tag]:
                    self._add_data(src_db[tag][sample_tag][data_tag], tag, sample_tag, data_tag)
        # ensure that all samples of same ensemble have same size and fill up missing cfgs with nans
        for sample_tag in src_cfgs:
            try:
                self.database["cfgs"][sample_tag] = list(set(self.database["cfgs"][sample_tag] + src_cfgs[sample_tag]))
            except KeyError:
                self.database["cfgs"][sample_tag] = src_cfgs[sample_tag]
        self.update_tags()
        self.compute_means()
        self.compute_jks()
        self.fill_db_with_nan()
        self.replace_nan_with_mean()

    def compute_means(self, tags=None):
        if tags == None:
            tags = self.database_tags
        for tag in tags:
            for sample_tag in self.database[tag]:
                if "mean" not in self.database[tag][sample_tag]:
                    self.sample_mean(tag, sample_tag)

    def compute_jks(self, tags=None):
        if tags == None:
            tags = self.database_tags
        for tag in tags:    
            for sample_tag in self.database[tag]: 
                if "jks" not in self.database[tag][sample_tag]:
                    self.jackknife_resampling(tag, sample_tag)

    def update_tags(self):
        self.database_tags = list(self.database.keys())
        self.database_tags.remove("cfgs")
 
    def fill_db_with_nan(self):
        for tag in self.database_tags:
            for sample_tag in self.database[tag]:
                for cfg in self.database["cfgs"][sample_tag]:
                    if str(cfg) not in self.database[tag][sample_tag]["sample"]:
                        self.database[tag][sample_tag]["sample"][str(cfg)] = np.nan

    def replace_nan_with_mean(self):
        for tag in self.database_tags:
            for sample_tag in self.database[tag]:
                for cfg in self.database[tag][sample_tag]["sample"]:
                    if np.isnan(self.database[tag][sample_tag]["sample"][cfg]).any(): 
                        self.database[tag][sample_tag]["sample"][cfg] = self.get_data(tag, sample_tag, "mean")

    def remove_cfgs(self, cfgs, sample_tag):
        self.database["cfgs"][sample_tag] = [cfg for cfg in self.database["cfgs"][sample_tag] if cfg not in cfgs]
        for tag in self.database_tags:
            if sample_tag in self.database[tag]:
                for data_tag in self.database[tag][sample_tag]:
                    if type(self.database[tag][sample_tag][data_tag]) == dict:
                        for cfg in cfgs:
                            self.database[tag][sample_tag][data_tag].pop(str(cfg), None)
        
###################################### FUNCTIONS #######################################

    def apply_f(self, f, tag, sample_tag, data_tag, dst_tag):
        obj = self.database[tag][sample_tag][data_tag]
        if type(obj) == dict:
            f_obj = {}
            for key, val in obj.items():  
                f_obj[key] = f(val)
        else:
            f_obj = f(obj)
        self.add_data(f_obj, dst_tag, sample_tag, data_tag)

    def apply_f2(self, f, tag0, tag1, sample_tag, data_tag, dst_tag):
        obj0 = self.get_data(tag0, sample_tag, data_tag); obj1 = self.get_data(tag1, sample_tag, data_tag)
        assert type(obj0) == type(obj1)
        if type(obj0) == dict:
            f_obj = {}
            for (key0, val0), (key1, val1) in zip(obj0.items(), obj1.items()):
                f_obj[key] = f(val0, val1)
        else:
            f_obj = f(obj0, obj1)
        self.add_data(f_obj, dst_tag, sample_tag, data_tag)

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

###################################### STATISTICS ######################################

    def sample_mean(self, tag, sample_tag, data_tag="sample", data_axis=0, nanmean=False):
        if nanmean:
            mean = np.nanmean(self.get_data(tag, sample_tag, data_tag, array=True), axis=data_axis)
        else:
            mean = np.mean(self.get_data(tag, sample_tag, data_tag, array=True), axis=data_axis)
        self.add_data(mean, tag, sample_tag, "mean")
        return mean

    def jackknife_resampling(self, tag, sample_tag, data_tag="sample", jks_tag="jks"):
        data = self.get_data(tag, sample_tag, data_tag)
        mean = self.get_data(tag, sample_tag, "mean")
        jks = {}
        for cfg in data:
            jks[cfg] = mean + (mean - data[cfg]) / (len(data) - 1) 
        self.add_data(jks, tag, sample_tag, jks_tag)
        return jks

    def jackknife_variance(self, tag, sample_tag, jks_tag="jks", dst_data_tag="jkvar"):
        mean = self.get_data(tag, sample_tag, "mean")
        jks = self.get_data(tag, sample_tag, jks_tag, array=True)
        var = sp.statistics.jackknife.variance_jks(mean, jks)
        self.add_data(var, tag, sample_tag, dst_data_tag)
        return var

    def jackknife_covariance(self, tag, sample_tag, jks_tag="jks", dst_data_tag="jkcov"):
        mean = self.get_data(tag, sample_tag, "mean")
        jks = self.get_data(tag, sample_tag, jks_tag, array=True)
        cov = sp.statistics.jackknife.covariance_jks(mean, jks)
        self.add_data(cov, tag, sample_tag, dst_data_tag)
        return cov

    def bin(self, binsize, tag, sample_tag):
        data = self.get_data(tag, sample_tag, "sample", array=True) 
        bata = {}
        for block, block_mean in enumerate(sp.statistics.bin(data, binsize)):
            bata[str(block)] = block_mean
        self.add_data(bata, tag, sample_tag, f"sample_b{binsize}")
        return bata

    def binning_study(self, binsizes, tag, sample_tag, rm_binsizes=[]):
        print("Original sample size: ", len(self.database["cfgs"][sample_tag]))
        var = {}
        for b in binsizes:
            if b == 1: 
                postfix = ""
            else: 
                postfix = f"_b{b}"
                self.bin(b, tag, sample_tag)
            self.jackknife_resampling(tag, sample_tag, data_tag="sample"+postfix, jks_tag="jks"+postfix)
            var[b] = self.jackknife_variance(tag, sample_tag, jks_tag="jks"+postfix, dst_data_tag="jkvar"+postfix)
        for b in rm_binsizes:
            if b == 1: 
                postfix = ""
            else: 
                postfix = f"_b{b}"
                del self.database[tag][sample_tag]["sample"+postfix]
            del self.database[tag][sample_tag]["jks"+postfix]
            del self.database[tag][sample_tag]["jkvar"+postfix]
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


################################### SCALE SETTING ###################################

    def flow_scale(self, tau_tag, E_tag, sample_tag, jks_tag):
        tau = self.get_data(tau_tag, sample_tag, "mean")
        scale = sp.qcd.scale_setting.gradient_flow_scale()
        self.apply_f(lambda x: scale.set_sqrt_tau0(tau, x), "E", sample_tag, "mean", "sqrt_tau0")
        self.apply_f(lambda x: scale.set_sqrt_tau0(tau, x), "E", sample_tag, jks_tag, "sqrt_tau0")
        sqrt_tau0_var = self.jackknife_variance("sqrt_tau0", sample_tag, jks_tag, dst_data_tag="jkvar"+jks_tag.split("jks")[1])
        self.apply_f(lambda x: scale.set_omega0(tau, x), "E", sample_tag, "mean", "omega0")
        self.apply_f(lambda x: scale.set_omega0(tau, x), "E", sample_tag, jks_tag, "omega0")
        omega0_var = self.jackknife_variance("omega0", sample_tag, jks_tag, dst_data_tag="jkvar"+jks_tag.split("jks")[1])

        print(f"sqrt(tau0) = {self.get_data('sqrt_tau0', sample_tag, 'mean'):.4f} +- {sqrt_tau0_var**.5:.4f}")
        print(f"wau0 = {self.get_data('omega0', sample_tag, 'mean'):.4f} +- {omega0_var**.5:.4f}")

#    def set_scale(self, Edens_tag, sample_tag, binsize, tau, nskip=0, scales=["t0", "w0"], dst_suffix="", store=True):
#        Edens_binned = sp.statistics.bin(self.get_data_arr(Edens_tag, sample_tag)[nskip:], binsize)
#        if binsize != 1:
#            sample_tag = sample_tag + f"_binned{binsize}"
#            self.add_data_arr(Edens_binned, Edens_tag, sample_tag)
#        print("effective number of measurements: ", len(Edens_binned))
#        scale = sp.qcd.scale_setting.scale(tau, Edens_binned)
#        if "t0" in scales:
#            sqrt_tau0, sqrt_tau0_std, t2E, t2E_std, aGeV_inv_t0, aGeV_inv_t0_std, aGeV_inv_t0_jks = scale.lattice_spacing("t0")
#            if store:
#                self.add_data(sqrt_tau0, "flow_scale_t0" + dst_suffix, sample_tag, "sqrt_tau0")
#                self.add_data(sqrt_tau0_std, "flow_scale_t0" + dst_suffix, sample_tag, "sqrt_tau0_std")
#                self.add_data(t2E, "flow_scale_t0" + dst_suffix, sample_tag, "t2E")
#                self.add_data(t2E_std, "flow_scale_t0" + dst_suffix, sample_tag, "t2E_std")
#                self.add_data(aGeV_inv_t0, "flow_scale_t0" + dst_suffix, sample_tag, "aGeV_inv")
#                self.add_data(aGeV_inv_t0_std, "flow_scale_t0" + dst_suffix, sample_tag, "aGeV_inv_std")
#                self.add_data(aGeV_inv_t0_jks, "flow_scale_t0" + dst_suffix, sample_tag, "aGeV_inv_jks")
#        if "w0" in scales:
#            wau0, wau0_std, tdt2E, tdt2E_std, aGeV_inv_w0, aGeV_inv_w0_std, aGeV_inv_w0_jks = scale.lattice_spacing("w0")
#            if store:
#                self.add_data(wau0, "flow_scale_w0" + dst_suffix, sample_tag, "wau0")
#                self.add_data(wau0_std, "flow_scale_w0" + dst_suffix, sample_tag, "wau0_std")
#                self.add_data(tdt2E, "flow_scale_w0" + dst_suffix, sample_tag, "tdt2E")
#                self.add_data(tdt2E_std, "flow_scale_w0" + dst_suffix, sample_tag, "tdt2E_std")
#                self.add_data(aGeV_inv_w0, "flow_scale_w0" + dst_suffix, sample_tag, "aGeV_inv")
#                self.add_data(aGeV_inv_w0_std, "flow_scale_w0" + dst_suffix, sample_tag, "aGeV_inv_std")
#                self.add_data(aGeV_inv_w0_jks, "flow_scale_w0" + dst_suffix, sample_tag, "aGeV_inv_jks")    

###################################### FITTING ######################################

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
