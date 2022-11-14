#!/usr/bin/env python3

import os, sys
import numpy as np
import statpy.dbpy.np_json as json
import statpy as sp

class DBpy:
    def __init__(self, file):
        self.file = file
        if os.path.isfile(self.file):
            with open(self.file) as f:
                self.database = json.load(f)
        else:
            self.database = {}

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
            if query_yes_no(f"{cfg_tag} is already in database for {tag}/{sample_tag} Overwrite?"):
                self._add_data(data, tag, sample_tag, cfg_tag)
        except KeyError:
            self._add_data(data, tag, sample_tag, cfg_tag)

    def _add_data_arr(self, data, tag, sample_tag, idxs, cfg_prefix):
        for idx in range(len(data)):
            self._add_data(data[idx], tag, sample_tag, cfg_prefix + str(idxs[idx]))

    def add_data_arr(self, data, tag, sample_tag, idxs=None, cfg_prefix=""):
        if idxs == None:
            idxs = range(len(data))
        try:
            self.database[tag][sample_tag]
            if query_yes_no(f"{sample_tag} is already in database for {tag}. Overwrite?"):
                self._add_data_arr(data, tag, sample_tag, idxs, cfg_prefix)
        except KeyError:
            self._add_data_arr(data, tag, sample_tag, idxs, cfg_prefix)

    def add_tag_entry(self, dst_tag, src, src_tag):
        with open(src) as f:
            src_db = json.load(f)
        try:
            self.database[dst_tag]
            if query_yes_no(f"{dst_tag} is already in database. Overwrite?"):
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
                if query_yes_no(f"{dst_sample_tag} is already in database for {dst_tag}. Overwrite?"):
                    self.database[dst_tag][dst_sample_tag] = src_db[src_tag]
            except KeyError:
                self.database[dst_tag][dst_sample_tag] = src_db[src_tag]
        except KeyError:
            self.database[dst_tag] = {dst_sample_tag: src_db[src_tag]}

    def apply_f(self, f, tag, sample_tag, dst_tag, dst_sample_tag, overwrite=False):
        f_dict = {}
        for cfg_idx, cfg_val in self.database[tag][sample_tag].items():  
            f_dict[cfg_idx] = f(cfg_val)
        try:
            self.database[dst_tag]
            try: 
                self.database[dst_tag][dst_sample_tag]
                if overwrite:
                    self.database[dst_tag][dst_sample_tag] = f_dict
                elif query_yes_no(f"{dst_tag} is already in database. Overwrite?"):
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
        
    def get_data_dict(self, tag, sample_tag):
        try:
            return dict(self.database[tag][sample_tag])
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

    def sample_mean(self, tag, sample_tag, store=True):
        mean = np.mean(self.get_data_arr(tag, sample_tag), axis=0)
        if store:
            self._add_data(mean, tag, sample_tag + "_mean", "-")
        return mean

    def jackknife_resampling(self, f, tag, sample_tag, jk_suffix="_jackknife", eps=1.0, store=True, dst_tag=None):
        if dst_tag==None: dst_tag = tag
        data = self.get_data_dict(tag, sample_tag)
        mean = self.sample_mean(tag, sample_tag.split("_binned")[0])
        jk_tag = sample_tag + jk_suffix
        jk_sample = {}
        for cfg_tag in data.keys():
            jk_sample[cfg_tag] = f( mean + eps*(mean - data[cfg_tag]) / (len(data) - 1) )
        if store:
            self.database[dst_tag][jk_tag] = jk_sample
        return jk_sample

    def jackknife_variance(self, f, tag, sample_tag, jk_suffix="_jackknife", eps=1.0, store=True, dst_tag=None):
        if dst_tag==None: dst_tag = tag
        self.jackknife_resampling(f, tag, sample_tag, jk_suffix, eps, store, dst_tag)
        f_mean = f( self.database[tag][sample_tag.split("_binned")[0] + "_mean"]["-"] )
        jk_data = self.get_data_arr(dst_tag, sample_tag + jk_suffix)
        N = len(jk_data)
        jk_var = np.mean([ (jk_data[k] - f_mean)**2 for k in range(N) ], axis=0) * (N - 1)
        if store:
            self.add_data(jk_var, dst_tag, sample_tag + "_jkvar", "-") 
        return jk_var
        
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
            del self.database[tag][sample_tag + f"_binned{binsize}_jackknife"]
            del self.database[tag][sample_tag + f"_binned{binsize}_jkvar"]
        return stds   

###################################### FITTING ######################################

    def multi_mc_fit(self, t, tag, sample_tag, C, model, p0, estimator, fit_tag="FIT", method="Nelder-Mead", minimizer_params={}, verbose=True, store=False, return_fitter=False):

        y = np.array([self.get_data_arr(tag, sample_tag + f"{ti}") for ti in t])
        assert isinstance(model, dict)
        model_func = list(model.values())[0]        
        assert method in ["Levenberg-Marquardt", "Migrad", "Nelder-Mead"]
        if method in ["Migrad", "Nelder-Mead"]:
            fitter = sp.fitting.fit(t, y, C, model_func, p0, estimator, method, minimizer_params)
        else:
            fitter = sp.fitting.LM_fit(t, y, C, model_func, p0, estimator, minimizer_params)

        fitter.multi_mc_fit(verbose)

        if store:
            self.database[tag][fit_tag] = {}
            self.database[tag][fit_tag]["t"] = t
            self.database[tag][fit_tag]["C"] = C
            self.database[tag][fit_tag]["model"] = list(model.keys())[0]
            self.database[tag][fit_tag]["minimizer"] = method
            self.database[tag][fit_tag]["minimizer_params"] = minimizer_params
            self.database[tag][fit_tag]["p0"] = p0
            self.database[tag][fit_tag]["best_parameter"] = fitter.best_parameter
            self.database[tag][fit_tag]["best_parameter_cov"] = fitter.best_parameter_cov
            self.database[tag][fit_tag]["jk_parameter"] = fitter.jk_parameter
            self.database[tag][fit_tag]["pval"] = fitter.p
            self.database[tag][fit_tag]["dof"] = fitter.dof
            self.database[tag][fit_tag]["fit_err"] = fitter.fit_err
        
        if return_fitter:
            return fitter


##################################################################################################################

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