"""FitConfig dataclass and high-level correlator fit / averaging routines.

All functions take a database handle as first argument. Fits additionally take
a FitConfig describing fit method + parameters.
"""
import re
import warnings
from dataclasses import dataclass, field
from math import isnan

import numpy as np

from statpy.log import message
from statpy.fitting.core import Fitter, ConvergenceError
from statpy.fitting.core import fit, print_fit_results, get_pvalue
from statpy.statistics import jackknife, bootstrap

from statpy.qcd.correlator.primitives import (
    effective_mass_acosh1, effective_mass_log1, effective_mass_log2,
    effective_amplitude_cosh, effective_amplitude_sinh, effective_amplitude_exp,
)
from statpy.qcd.correlator.models import (
    fit_model_dict,
    cosh_model, cosh_chi2,
    sinh_model, sinh_chi2,
    exp_model, exp_chi2,
    double_cosh_model, double_cosh_chi2,
    double_sinh_model, double_sinh_chi2,
    double_exp_model, double_exp_chi2,
    combined_cosh_sinh_chi2, combined_exp_exp_model_chi2,
)
from statpy.qcd.correlator._masking import (
    bare_decay_constant,
    _get_masked_Ct, _get_tmax_fw_bw, _get_masked_Cts_boundary,
    _fold_boundary, _flip_sign_boundary,
)


@dataclass
class FitConfig:
    fit_method: str = "Nelder-Mead"
    fit_params: dict = field(default_factory=lambda: {"maxiter": 5000, "tol": 1e-07})
    bootstrap_available: bool = True


# ---------------------------------------------------------------------------
# correlator averaging / folding
# ---------------------------------------------------------------------------

def correlator_avg_pbc(db, Ct_tag, dst_tag):
    assert isinstance(Ct_tag, str)
    db.combine_sample(Ct_tag, f=lambda x: np.mean(x, axis=0), dst_tag=dst_tag)


def correlator_avg_obc(db, Ct_tags, tbulk, dst_tag, tmax_from_tsrc=None, antiperiodic=False):
    message(f"Perform obc tsrc average over all srcs in tbulk = [[{tbulk[0]},{tbulk[-1]}]] with correlator tags: {Ct_tags}")
    message(f"tmax_from_tsrc: {tmax_from_tsrc}")
    # Get src positions in bulk
    tsrcs = [int(re.search(r'tsrc(\d+)', t)[1]) for t in Ct_tags]
    assert len(Ct_tags) == len(tsrcs)
    Ct_tags_in_bulk = []
    tsrcs_in_bulk = []
    for Ct_tag, tsrc in zip(Ct_tags, tsrcs):
        if (tsrc >= tbulk[0]) and (tsrc <= tbulk[-1]):
            Ct_tags_in_bulk.append(Ct_tag)
            tsrcs_in_bulk.append(tsrc)
    message(f"tsrcs in bulk: {tsrcs_in_bulk}")
    tmax_fw, tmax_bw = _get_tmax_fw_bw(tsrcs_in_bulk, tbulk)
    if tmax_from_tsrc is not None:
        tmax_fw = np.minimum(tmax_fw, tmax_from_tsrc+1)
        tmax_bw = np.minimum(tmax_bw, tmax_from_tsrc+1)
    for src_idx, Ct_tag in enumerate(Ct_tags_in_bulk):
        db.combine_sample(Ct_tag, f=lambda Ct: _get_masked_Ct(Ct, tmax_fw[src_idx], tmax_bw[src_idx], antiperiodic), dst_tag=f"{Ct_tag}/masked", verbosity=-1)
    combined_sample = db.combine_sample(*[f"{Ct_tag}/masked" for Ct_tag in Ct_tags_in_bulk], f=lambda *Cts_ma: np.ma.concatenate(Cts_ma, axis=0).mean(axis=0).compressed())
    db.add_leaf(tag=dst_tag, mean=None, jks=None, sample=combined_sample, misc={"tsrcs":tsrcs_in_bulk, "tbulk":tbulk, "antiperiodic":antiperiodic})
    for Ct_tag in Ct_tags_in_bulk:
        db.remove_leaf(f"{Ct_tag}/masked", verbosity=-1)


def am_t_avg_obc(db, Ct_tags, tbulk, dst_tag, binsize, tmax_from_tsrc=None, keep_am_t_per_src=False):
    message(f"Perform obc tsrc average for effective mass over all srcs in tbulk = [[{tbulk[0]},{tbulk[-1]}]] with effective mass tags: {Ct_tags}")
    message("WARNING: This method uses effective_mass_log2 and is only tested for PSPS_SMSMS")
    message(f"tmax_from_tsrc: {tmax_from_tsrc}")
    tsrcs = [int(re.search(r'tsrc(\d+)', t)[1]) for t in Ct_tags]
    assert len(Ct_tags) == len(tsrcs)
    Ct_tags_in_bulk = []
    tsrcs_in_bulk = []
    mt_tags_in_bulk = []
    Ct_tags_binned = []
    for Ct_tag, tsrc in zip(Ct_tags, tsrcs):
        if (tsrc >= tbulk[0]) and (tsrc <= tbulk[-1]):
            Ct_tags_in_bulk.append(Ct_tag)
            tsrcs_in_bulk.append(tsrc)
            Ct_binned = db.add_binned_leaf(Ct_tag, binsize)
            Ct_tags_binned.append(Ct_binned)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                db.combine(Ct_binned, f=lambda Ct: effective_mass_log2(Ct, ax=1), dst_tag=f"{Ct_binned}/am_t")
            mt_tags_in_bulk.append(f"{Ct_binned}/am_t")
    message(f"tsrcs in bulk: {tsrcs_in_bulk}")
    tmax_fw, tmax_bw = _get_tmax_fw_bw(tsrcs_in_bulk, tbulk)
    if tmax_from_tsrc is not None:
        tmax_fw = np.minimum(tmax_fw, tmax_from_tsrc+1)
        tmax_bw = np.minimum(tmax_bw, tmax_from_tsrc+1)
    for src_idx, mt_tag in enumerate(mt_tags_in_bulk):
        db.combine(mt_tag, f=lambda mt: _get_masked_Ct(mt, tmax_fw[src_idx], tmax_bw[src_idx], antiperiodic=True), dst_tag=f"{mt_tag}/masked")
    db.combine(*[f"{mt_tag}/masked" for mt_tag in mt_tags_in_bulk], f=lambda *mts_ma: np.ma.concatenate(mts_ma, axis=0).mean(axis=0).compressed(), dst_tag=dst_tag)
    for mt_tag in mt_tags_in_bulk:
        if not keep_am_t_per_src:
            db.remove_leaf(mt_tag)
        db.remove_leaf(f"{mt_tag}/masked")
    for Ct_tag_binned in Ct_tags_binned:
        db.remove_leaf(Ct_tag_binned)


def fold_correlator_leaf(db, Ct_tag, antiperiodic=False):
    from statpy.qcd.correlator.primitives import fold_correlator
    message(f"Fold correlator {Ct_tag}.")
    db.combine_sample(Ct_tag, f=lambda Ct: fold_correlator(Ct, antiperiodic), dst_tag=f"{Ct_tag}/folded")


# ---------------------------------------------------------------------------
# PSA4 improvement 
# ---------------------------------------------------------------------------

def determine_PSA4I(db, tag_PSPS_sml, tag_PSA4_sml, beta):
    """Improved PSA4 correlator per https://arxiv.org/pdf/1502.04999.pdf"""
    def compute_cA(beta):
        p0 = 9.2056
        p1 = -13.9847
        return - 0.006033 * 6./beta * (1 + np.exp(p0 + p1*beta/6.))
    def derivative(f):
        return 0.5 * (np.roll(f, -1) - np.roll(f, 1))
    def compute_PSA4I(PS_A4, PS_PS, beta):
        PS_A4I = PS_A4 - compute_cA(beta) * derivative(PS_PS)
        PS_A4I[0] = 0.
        PS_A4I[-1] = 0
        return PS_A4I
    tag_PSA4I = tag_PSA4_sml.replace("PSA4", "PSA4I")
    db.combine_sample(tag_PSA4_sml, tag_PSPS_sml, f=lambda x,y: compute_PSA4I(x, y, beta), dst_tag=tag_PSA4I)


# ---------------------------------------------------------------------------
# fit-parameter heuristics
# ---------------------------------------------------------------------------

def get_p0_guess(db, tag, binsize, fit_model, fit_range):
    assert fit_model in ["double-cosh", "double-sinh", "double-exp"]
    message(f"Get p0 guess(es) for {fit_model} fit model with {tag} and binsize = {binsize}")
    binned_tag = db.add_binned_leaf(tag, binsize)
    Ct_mean = db.database[binned_tag].mean
    Nt = len(Ct_mean)
    effective_mass = {"double-cosh": effective_mass_acosh1, "double-sinh": effective_mass_acosh1, "double-exp": effective_mass_log1}[fit_model]
    effective_amplitude = {"double-cosh": effective_amplitude_cosh, "double-sinh": effective_amplitude_sinh, "double-exp": effective_amplitude_exp}[fit_model]
    single_model_func = {"double-cosh": cosh_model(Nt), "double-sinh": sinh_model(Nt), "double-exp": exp_model()}[fit_model]
    # ground state parameters
    t0_probe = slice(Nt//4, Nt//4 + Nt//8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        m0_eff = np.mean(effective_mass(Ct_mean)[t0_probe])
        A0_eff = np.mean(effective_amplitude(Ct_mean, m0_eff)[t0_probe])
        if isnan(m0_eff) or isnan(A0_eff):
            t0_probe = Nt//4
            m0_eff = np.mean(effective_mass(Ct_mean)[t0_probe])
            A0_eff = np.mean(effective_amplitude(Ct_mean, m0_eff)[t0_probe])
    # excited state parameters
    Ct_ground = single_model_func(np.arange(Nt), [A0_eff,m0_eff])
    Ct_excited = Ct_mean - Ct_ground
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        m1_eff = effective_mass(Ct_excited)[fit_range[0]]
        A1_eff = effective_amplitude(Ct_excited, m1_eff)[fit_range[0]]
    message(f"guessed p0 = [{A0_eff}, {m0_eff},  {A1_eff}, {m1_eff}]")
    if m1_eff < 1.2 * m0_eff:
        message("guess for excited state mass too small: p0[2] = abs(p0[2]); p0[3] = 2*p0[1]")
        A1_eff = abs(A1_eff)
        m1_eff = 2. * m0_eff
        message(f"---> [{A0_eff}, {m0_eff},  {A1_eff}, {m1_eff}]")
    return np.array([A0_eff,m0_eff,A1_eff,m1_eff])


# ---------------------------------------------------------------------------
# excited-state / ground-state / combined fits
# ---------------------------------------------------------------------------

def excited_contributions_fit(db, tag, binsize, initial_fit_ranges, p0, fit_model, config: FitConfig, verbosity, Nt=None, MIN_TCRIT_LEN=7, folded=False):
    def _sort_params(p):
        if p[3] <  p[1]: 
            return [p[2], p[3], p[0], p[1]]
        else: 
            return p
    message(f"CORRELATOR: {tag}")
    if p0 is None:
        message("P0 is inferred for each initial fit range automatically.")
    else:
        message(f"P0 = {p0}")
    message(f"BINSIZE = {binsize}", verbosity)
    message(f"{fit_model} MODEL = {fit_model_dict[fit_model]}")
    message("---------------------------------------------------------------------------------", verbosity)
    binned_tag = db.add_binned_leaf(tag, binsize)
    cov = db.jackknife_covariance(binned_tag)
    var = np.diag(cov)
    Nt = len(db.database[binned_tag].mean) if Nt is None else Nt
    model_func = {"double-cosh": double_cosh_model(Nt),
                  "double-sinh": double_sinh_model(Nt),
                  "double-exp": double_exp_model()}[fit_model]
    bc = "open" if fit_model == "double-exp" else "periodic"
    excited_contribtions_fit_dict = {"tag": None, "mean": None, "jks": None, "sample":None, "misc": None}
    binned_correlated_fit_dict = {"tag": None, "mean": None, "jks": None, "sample":None, "misc": None}
    unbinned_correlated_fit_dict = {"tag": None, "mean": None, "jks": None, "sample":None, "misc": None}
    suggested_fit_ranges = []
    fit_range = initial_fit_ranges[0]
    for t in initial_fit_ranges:
        message(f"INITIAL FIT RANGE: [[{t[0]},{t[-1]}]]", verbosity)
        message("------------------------------- UNCORRELATED FIT --------------------------------", verbosity)
        W = np.linalg.inv(np.diag(var[t]))
        chi2_func = {"double-cosh": lambda t,p,y: double_cosh_chi2(t, p, y, W, Nt),
                     "double-sinh": lambda t,p,y: double_sinh_chi2(t, p, y, W, Nt),
                     "double-exp": lambda t,p,y: double_exp_chi2(t, p, y, W)}[fit_model]
        p0_tmp = get_p0_guess(db, tag, binsize, fit_model, t) if p0 is None else p0
        if np.isnan(p0_tmp).any():
            if excited_contribtions_fit_dict["mean"] is not None:
                p0_tmp = excited_contribtions_fit_dict["mean"]
            else:
                p0_tmp[2] = p0_tmp[0]/2
                p0_tmp[3] = 2.0 * p0_tmp[0]
                if np.isnan(p0_tmp).any():
                    p0_tmp = [1.0 if isnan(p) else p for p in p0_tmp]
            message(f"p0 guess contains NaN, use fit result from previous fit range if available, else use available params to estimate NaNs or default to 1: {p0_tmp}")
        try:
            message(f"p0 for fit: {p0_tmp}")
            best_parameter, best_parameter_jks, misc = fit(db, t, binned_tag, p0_tmp, chi2_func, config.fit_method, config.fit_params)
            misc["fit_model"] = fit_model
        except ConvergenceError as ce:
            suggested_fit_ranges.append(None)
            message(f"{ce} -> JUMP TO NEXT FIT RANGE")
            message("---------------------------------------------------------------------------------", verbosity)
            message("---------------------------------------------------------------------------------", verbosity)
            continue
        best_parameter = _sort_params(best_parameter)
        best_parameter_jks = {cfg: _sort_params(best_parameter_jks[cfg]) for cfg in best_parameter_jks}
        best_parameter_cov = jackknife.covariance(db.as_array(best_parameter_jks))
        print_fit_results(best_parameter, best_parameter_cov, misc, verbosity)
        message("------------------------------ CORRELATED MEAN FIT ------------------------------", verbosity)
        message("Try correlated fit with binned covariance matrix")
        binned_correlated_converged = False
        message(f"Check positive definiteness of binned covariance matrix for fit range [[{t[0]},{t[-1]}]].")
        pos_def = np.all(np.linalg.eigvals(cov[t][:,t]) > 0)
        if pos_def:
            message("--> binned covariance matrix positive definite. Try correlated fit.")
            cov_correlated = cov
            try:
                W_correlated = np.linalg.inv(cov_correlated[t][:,t])
                chi2_func_correlated = {"double-cosh": lambda t,p,y: double_cosh_chi2(t, p, y, W_correlated, Nt),
                                        "double-sinh": lambda t,p,y: double_sinh_chi2(t, p, y, W_correlated, Nt),
                                        "double-exp": lambda t,p,y: double_exp_chi2(t, p, y, W_correlated)}[fit_model]
                message(f"p0 for fit: {p0_tmp}")
                binned_best_parameter_correlated, _, binned_misc_correlated = fit(db, t, binned_tag, p0_tmp, chi2_func_correlated, config.fit_method, config.fit_params, perform_jks_fit=False)
                binned_misc_correlated["fit_model"] = fit_model
                binned_best_parameter_correlated = _sort_params(binned_best_parameter_correlated)
                print_fit_results(binned_best_parameter_correlated, None, binned_misc_correlated, verbosity)
                binned_correlated_converged = True
            except ConvergenceError as ce:
                message(f"{ce} for correlated mean fit with binned covariance matrix")
                message("---------------------------------------------------------------------------------", verbosity)
        else:
            message("--> binned covariance matrix NOT positive definite.")

        unbinned_correlated_converged = False
        message("Try correlated fit with unbinned covariance matrix.")
        cov_unbinned = db.jackknife_covariance(tag)
        message(f"Check positive definiteness of unbinned covariance matrix for fit range [[{t[0]},{t[-1]}]].")
        pos_def_unbinned = np.all(np.linalg.eigvals(cov_unbinned[t][:,t]) > 0)
        if pos_def_unbinned:
            message("--> unbinned covariance matrix positive definite. Try correlated fit.")
            cov_correlated = cov_unbinned
            try:
                W_correlated = np.linalg.inv(cov_correlated[t][:,t])
                chi2_func_correlated = {"double-cosh": lambda t,p,y: double_cosh_chi2(t, p, y, W_correlated, Nt),
                                        "double-sinh": lambda t,p,y: double_sinh_chi2(t, p, y, W_correlated, Nt),
                                        "double-exp": lambda t,p,y: double_exp_chi2(t, p, y, W_correlated)}[fit_model]
                message(f"p0 for fit: {p0_tmp}")
                unbinned_best_parameter_correlated, _, unbinned_misc_correlated = fit(db, t, binned_tag, p0_tmp, chi2_func_correlated, config.fit_method, config.fit_params, perform_jks_fit=False)
                unbinned_misc_correlated["fit_model"] = fit_model
                unbinned_best_parameter_correlated = _sort_params(unbinned_best_parameter_correlated)
                print_fit_results(unbinned_best_parameter_correlated, None, unbinned_misc_correlated, verbosity)
                unbinned_correlated_converged = True
            except ConvergenceError as ce:
                message(f"{ce} for correlated mean fit with unbinned covariance matrix")
                message("---------------------------------------------------------------------------------", verbosity)
        else:
            message("--> unbinned covariance matrix NOT positive definite.")
        message("---------------------------------------------------------------------------------", verbosity)
        excited_state_contribution = np.abs([model_func(i, [0, 0, best_parameter[2], best_parameter[3]]) for i in t])
        std_over_four = (var[t]**.5)/4.
        if bc == "periodic" and not folded:
            symmetrized_std_over_four = (std_over_four + std_over_four[::-1]) / 2
            criterion = excited_state_contribution < symmetrized_std_over_four
        else:
            criterion = excited_state_contribution < std_over_four
        t_crit = t[criterion]
        suggested_fit_ranges.append(t_crit)
        if len(t_crit) < MIN_TCRIT_LEN:
            message(f"DETERMINED FIT RANGE {t_crit} HAS FEWER THAN {MIN_TCRIT_LEN} ELEMENTS", verbosity)
            message("---> STORED FIT RANGE IS NOT UPDATED", verbosity)
            message("---------------------------------------------------------------------------------", verbosity)
            message("---------------------------------------------------------------------------------", verbosity)
            continue
        else:
            message(f"DETERMINED FIT RANGE [[{t_crit[0]},{t_crit[-1]}]]", verbosity)
        if len(t_crit) <= len(fit_range):
            message("---> STORED FIT RANGE IS UPDATED", verbosity)
            excited_contribtions_fit_dict["tag"] = f"{binned_tag}/excited_contributions_fit"
            excited_contribtions_fit_dict["mean"] = best_parameter
            excited_contribtions_fit_dict["jks"] = best_parameter_jks
            misc["fit_range_crit"] = t_crit
            fit_range = t_crit
            excited_contribtions_fit_dict["misc"] = misc
            binned_correlated_fit_dict["tag"] = f"{binned_tag}/binned_correlated_excited_contributions_mean_fit"
            if binned_correlated_converged:
                binned_correlated_fit_dict["mean"] = binned_best_parameter_correlated
                binned_correlated_fit_dict["misc"] = binned_misc_correlated
            unbinned_correlated_fit_dict["tag"] = f"{binned_tag}/unbinned_correlated_excited_contributions_mean_fit"
            if unbinned_correlated_converged:
                unbinned_correlated_fit_dict["mean"] = unbinned_best_parameter_correlated
                unbinned_correlated_fit_dict["misc"] = unbinned_misc_correlated
        message("---------------------------------------------------------------------------------", verbosity)
        message("---------------------------------------------------------------------------------", verbosity)
    db.add_leaf(**binned_correlated_fit_dict)
    db.add_leaf(**unbinned_correlated_fit_dict)
    if excited_contribtions_fit_dict["misc"] is not None:
        excited_contribtions_fit_dict["misc"]["tested_suggested_fit_ranges"] = (initial_fit_ranges, suggested_fit_ranges)
        db.add_leaf(**excited_contribtions_fit_dict)
        return excited_contribtions_fit_dict["misc"]["fit_range_crit"], best_parameter
    else:
        return None, None


def ground_state_fit(db, tag, binsize, fit_range, p0, fit_model, config: FitConfig, Nt=None, verbosity=0):
    message(f"CORRELATOR: {tag}")
    message(f"P0 = {p0}")
    message(f"FIT RANGE {fit_range}")
    message(f"{fit_model} MODEL = {fit_model_dict[fit_model]}")
    for b in range(1, binsize+1):
        message(f"BINSIZE = {b}", verbosity)
        binned_tag = db.add_binned_leaf(tag, b)
        message("--------------------------------- JACKKNIFE FIT ---------------------------------", verbosity)
        var = db.jackknife_variance(binned_tag)
        Nt = len(db.database[binned_tag].mean) if Nt is None else Nt
        W = np.linalg.inv(np.diag(var[fit_range]))
        chi2_func = {"cosh": lambda t,p,y: cosh_chi2(t, p, y, W, Nt),
                     "sinh": lambda t,p,y: sinh_chi2(t, p, y, W, Nt),
                     "exp": lambda t,p,y: exp_chi2(t, p, y, W)}[fit_model]
        best_parameter, best_parameter_jks, misc = fit(db, fit_range, binned_tag, p0, chi2_func, config.fit_method, config.fit_params)
        misc["fit_model"] = fit_model
        best_parameter_cov = jackknife.covariance(db.as_array(best_parameter_jks))
        print_fit_results(best_parameter, best_parameter_cov, misc, verbosity)
        if b in [1,binsize]:
            message("------------------------------ CORRELATED MEAN FIT ------------------------------", verbosity)
            message("Try fit with covariance matrix")
            cov = db.jackknife_covariance(binned_tag)[fit_range][:,fit_range]
            message(f"--- Check positive definiteness of covariance matrix at binsize = {b} for fit range [[{fit_range[0]},{fit_range[-1]}]].")
            pos_def = np.all(np.linalg.eigvals(cov) > 0)
            if pos_def:
                message("  --> covariance matrix positive definite. Try fit.")
                try:
                    W_correlated = np.linalg.inv(cov)
                    chi2_func_correlated = {"cosh": lambda t,p,y: cosh_chi2(t, p, y, W_correlated, Nt),
                                        "sinh": lambda t,p,y: sinh_chi2(t, p, y, W_correlated, Nt),
                                        "exp": lambda t,p,y: exp_chi2(t, p, y, W_correlated)}[fit_model]
                    best_parameter_correlated, _, misc_correlated = fit(db, fit_range, binned_tag, p0, chi2_func_correlated, config.fit_method, config.fit_params, perform_jks_fit=False)
                    misc_correlated["fit_model"] = fit_model
                    print_fit_results(best_parameter_correlated, None, misc_correlated, verbosity)
                    db.add_leaf(tag=f"{binned_tag}/{fit_model}_binned_correlated_mean_fit", mean=best_parameter_correlated, jks=None, sample=None, misc=misc_correlated)
                except ConvergenceError as ce:
                    message(f"{ce} for correlated mean fit with covariance matrix")
                    message("---------------------------------------------------------------------------------", verbosity)
            else:
                message("  --> covariance matrix NOT positive definite.")
            if b != 1:
                message("----------------")
                message("Try fit with unbinned covariance matrix")
                cov_unbinned = db.jackknife_covariance(tag)[fit_range][:,fit_range]
                message(f"--- Check positive definiteness of unbinned covariance matrix for fit range [[{fit_range[0]},{fit_range[-1]}]].")
                pos_def_unbinned = np.all(np.linalg.eigvals(cov_unbinned) > 0)
                if pos_def_unbinned:
                    message("  --> unbinned covariance matrix positive definite. Try fit.")
                    cov_correlated = cov_unbinned
                    try:
                        W_correlated = np.linalg.inv(cov_correlated)
                        chi2_func_correlated = {"cosh": lambda t,p,y: cosh_chi2(t, p, y, W_correlated, Nt),
                                        "sinh": lambda t,p,y: sinh_chi2(t, p, y, W_correlated, Nt),
                                        "exp": lambda t,p,y: exp_chi2(t, p, y, W_correlated)}[fit_model]
                        best_parameter_correlated, _, misc_correlated = fit(db, fit_range, binned_tag, p0, chi2_func_correlated, config.fit_method, config.fit_params, perform_jks_fit=False)
                        misc_correlated["fit_model"] = fit_model
                        print_fit_results(best_parameter_correlated, None, misc_correlated, verbosity)
                        db.add_leaf(tag=f"{binned_tag}/{fit_model}_unbinned_correlated_mean_fit", mean=best_parameter_correlated, jks=None, sample=None, misc=misc_correlated)
                    except ConvergenceError as ce:
                        message(f"{ce} for correlated mean fit with unbinned covariance matrix")
                        message("---------------------------------------------------------------------------------", verbosity)
                else:
                    message("--> unbinned covariance matrix NOT positive definite.")
            message("---------------------------------------------------------------------------------", verbosity)
        if b == 1 and config.bootstrap_available:
            message("--------------------------------- BOOTSTRAP FIT ---------------------------------", verbosity)
            bss = db.bss(binned_tag)
            mean_bss = db.database[binned_tag].mean
            W_bss = np.linalg.inv(np.diag(bootstrap.variance(bss)[fit_range]))
            chi2_func_bss = {"cosh": lambda t,p,y: cosh_chi2(t, p, y, W_bss, Nt),
                             "sinh": lambda t,p,y: sinh_chi2(t, p, y, W_bss, Nt),
                             "exp": lambda t,p,y: exp_chi2(t, p, y, W_bss)}[fit_model]
            best_parameter_bmean, best_parameter_bss, misc_bss = _fit_bootstrap(db, fit_range, mean_bss, bss, best_parameter, chi2_func_bss, config)
            best_parameter_bcov = bootstrap.covariance(best_parameter_bss)
            print_fit_results(best_parameter_bmean, best_parameter_bcov, misc_bss)
            misc_bss["fit_model"] = fit_model
            misc_bss["bss"] = best_parameter_bss
            db.add_leaf(tag=f"{binned_tag}/{fit_model}_bootstrap_fit", mean=best_parameter_bmean, jks=None, sample=None, misc=misc_bss)
        db.add_leaf(tag=f"{binned_tag}/{fit_model}_fit", mean=best_parameter, jks=best_parameter_jks, sample=None, misc=misc)
        message("---------------------------------------------------------------------------------", verbosity)
        message("---------------------------------------------------------------------------------", verbosity)


def _fit_bootstrap(db, t, mean, bss, p0, chi2_func, config: FitConfig, eval_offset=True):
    t_eval = t if eval_offset else np.arange(len(t))
    if not eval_offset: 
        assert len(t) == len(mean)
    fitter = Fitter(config.fit_method, config.fit_params)
    def fit_func(y):
        return fitter.estimate_parameters(t, chi2_func, y[t_eval], p0)[0]
    best_parameter = fit_func(mean)
    def fit_func_bss(y):
        return fitter.estimate_parameters(t, chi2_func, y[t_eval], best_parameter)[0]
    best_parameter_bss = db.combine_bss(bss, f=fit_func_bss)
    chi2 = chi2_func(t, best_parameter, mean[t_eval])
    dof = len(t) - len(best_parameter)
    pval = get_pvalue(chi2, dof)
    misc = {"t": t, "chi2": chi2, "dof": dof, "pval": pval}
    return best_parameter, best_parameter_bss, misc


def correlator_combined_fit(db, tag_PS, tag_A4I, fit_range_PS, fit_range_A4I, binsize, p0, fit_model_combined, config: FitConfig, Nt=None, verbosity=0):
    message("------------------ COMBINED CORRELATOR FIT PSPS/PSA4I ---------------------")
    fit_model_PS = fit_model_combined.split("-")[1]
    fit_model_A4I = fit_model_combined.split("-")[2]
    message(f"PSPS correlator: {tag_PS}")
    message(f"PSPS - FIT RANGE {fit_range_PS}")
    message(f"PSPS {fit_model_PS} MODEL = {fit_model_dict[fit_model_PS]}")
    message(f"PSA4I correlator: {tag_A4I}")
    message(f"PSA4I - FIT RANGE {fit_range_A4I}")
    message(f"PSA4I {fit_model_A4I} MODEL = {fit_model_dict[fit_model_A4I]}")
    message(f"COMBINED - {fit_model_combined} MODEL = {fit_model_dict[fit_model_combined]}")
    message(f"P0 = {p0}")

    Nt = len(db.database[tag_PS].mean) if Nt is None else Nt
    fit_range_combined = np.hstack((fit_range_PS, fit_range_A4I))
    combined_tag = f"{tag_PS};{tag_A4I.split('/')[1]}"
    db.combine_sample(tag_PS, tag_A4I, f=lambda x,y: np.hstack((x[fit_range_PS],y[fit_range_A4I])), dst_tag=combined_tag)
    for b in range(1, binsize+1):
        message(f"BINSIZE = {b}", verbosity)
        binned_tag = db.add_binned_leaf(combined_tag, b)
        message("--------------------------------- JACKKNIFE FIT ---------------------------------", verbosity)
        var = db.jackknife_variance(binned_tag)
        W = np.linalg.inv(np.diag(var))
        chi2_func = {"combined-cosh-sinh": lambda t,p,y: combined_cosh_sinh_chi2(t[:len(fit_range_PS)], t[len(fit_range_PS):], p, y, W, Nt),
                     "combined-exp-exp": lambda t,p,y: combined_exp_exp_model_chi2(t[:len(fit_range_PS)], t[len(fit_range_PS):], p, y, W)}[fit_model_combined]
        best_parameter, best_parameter_jks, misc = fit(db, fit_range_combined, binned_tag, p0, chi2_func, config.fit_method, config.fit_params, eval_offset=False)
        misc["fit_model_PSPS"] = fit_model_PS
        misc["fit_model_PSA4I"] = fit_model_A4I
        misc["fit_model"] = fit_model_combined
        misc["t_PSPS"] = fit_range_PS
        misc["t_PSA4I"] = fit_range_A4I
        best_parameter_cov = jackknife.covariance(db.as_array(best_parameter_jks))
        print_fit_results(best_parameter, best_parameter_cov, misc, verbosity)
        if b in [1,binsize]:
            message("------------------------------ CORRELATED MEAN FIT ------------------------------", verbosity)
            try:
                W_correlated = np.linalg.inv(db.jackknife_covariance(binned_tag))
                chi2_func_correlated = {"combined-cosh-sinh": lambda t,p,y: combined_cosh_sinh_chi2(t[:len(fit_range_PS)], t[len(fit_range_PS):], p, y, W_correlated, Nt),
                                        "combined-exp-exp": lambda t,p,y: combined_exp_exp_model_chi2(t[:len(fit_range_PS)], t[len(fit_range_PS):], p, y, W_correlated)}[fit_model_combined]
                best_parameter_correlated, _, misc_correlated = fit(db, fit_range_combined, binned_tag, best_parameter, chi2_func_correlated, config.fit_method, config.fit_params, perform_jks_fit=False, eval_offset=False)
                misc_correlated["fit_model_PSPS"] = fit_model_PS
                misc_correlated["fit_model_PSA4I"] = fit_model_A4I
                misc_correlated["fit_model"] = fit_model_combined
                misc_correlated["t_PSPS"] = fit_range_PS
                misc_correlated["t_PSA4I"] = fit_range_A4I
                print_fit_results(best_parameter_correlated, None, misc_correlated, verbosity)
                db.add_leaf(tag=f"{binned_tag}/{fit_model_combined}_correlated_mean_fit", mean=best_parameter_correlated, jks=None, sample=None, misc=misc_correlated)
            except ConvergenceError as ce:
                message(f"{ce} for correlated mean fit")
                message("---------------------------------------------------------------------------------", verbosity)
        if b == 1 and config.bootstrap_available:
            message("--------------------------------- BOOTSTRAP FIT ---------------------------------", verbosity)
            bss = db.bss(binned_tag)
            mean_bss = db.database[binned_tag].mean
            W_bss = np.linalg.inv(np.diag(bootstrap.variance(bss)))
            chi2_func_bss = {"combined-cosh-sinh": lambda t,p,y: combined_cosh_sinh_chi2(t[:len(fit_range_PS)], t[len(fit_range_PS):], p, y, W_bss, Nt),
                             "combined-exp-exp": lambda t,p,y: combined_exp_exp_model_chi2(t[:len(fit_range_PS)], t[len(fit_range_PS):], p, y, W_bss)}[fit_model_combined]
            best_parameter_bmean, best_parameter_bss, misc_bss = _fit_bootstrap(db, fit_range_combined, mean_bss, bss, best_parameter, chi2_func_bss, config, eval_offset=False)
            misc_bss["fit_model_PSPS"] = fit_model_PS
            misc_bss["fit_model_PSA4I"] = fit_model_A4I
            misc_bss["fit_model"] = fit_model_combined
            misc_bss["t_PSPS"] = fit_range_PS
            misc_bss["t_PSA4I"] = fit_range_A4I
            best_parameter_bcov = bootstrap.covariance(best_parameter_bss)
            print_fit_results(best_parameter_bmean, best_parameter_bcov, misc_bss)
            misc_bss["bss"] = best_parameter_bss
            db.add_leaf(tag=f"{binned_tag}/{fit_model_combined}_bootstrap_fit", mean=best_parameter_bmean, jks=None, sample=None, misc=misc_bss)
        db.add_leaf(tag=f"{binned_tag}/{fit_model_combined}_fit", mean=best_parameter, jks=best_parameter_jks, sample=None, misc=misc)
        message("------------------------------ BARE DECAY CONSTANT ------------------------------")
        db.combine(f"{binned_tag}/{fit_model_combined}_fit", f=bare_decay_constant, dst_tag=f"{binned_tag}/{fit_model_combined}_fit/afbare")
        if b == 1 and config.bootstrap_available:
            bootstrap_tag = f"{binned_tag}/{fit_model_combined}_bootstrap_fit"
            fbare_bss_mean = bare_decay_constant(db.database[bootstrap_tag].mean)
            fbare_bss = db.combine_bss(db.database[bootstrap_tag].misc["bss"], f=bare_decay_constant)
            db.add_leaf(tag=f"{bootstrap_tag}/afbare", mean=fbare_bss_mean, jks=None, sample=None, misc={"bss": fbare_bss})
            fbare_bs_str = f"         {fbare_bss_mean:.8f} +- {bootstrap.variance(fbare_bss)**.5:.8f} (bootstrap)"
        message(f"a*fbare = {db.database[f'{binned_tag}/{fit_model_combined}_fit/afbare'].mean:.8f} +- {db.jackknife_variance(f'{binned_tag}/{fit_model_combined}_fit/afbare')**.5:.8f} (jackknife)")
        if b == 1 and config.bootstrap_available: 
            message(fbare_bs_str)
        message("---------------------------------------------------------------------------------", verbosity)
        message("---------------------------------------------------------------------------------", verbosity)


# ---------------------------------------------------------------------------
# boundary averaging
# ---------------------------------------------------------------------------

def boundary_avg(db, Ct_tags, tmin_excited, binsize, tmax_from_tsrc=None, antiperiodic=False, cleanup=False, excluded_tsrcs=[]):
    message(f"Perform boundary average over all tsrcs with correlator tags: {Ct_tags}")
    message(f"Excited state contributions expected to be removed at t = {tmin_excited}")
    message(f"tmax_from_tsrc = {tmax_from_tsrc}")
    tsrcs = [int(re.search(r'tsrc(\d+)', t)[1]) for t in Ct_tags]
    message(f"Exclude the following srcs: {excluded_tsrcs}")
    for tsrc in excluded_tsrcs:
        if tsrc not in tsrcs:
            message(f"tsrc = {tsrc} not in tags anyway -> continue")
            continue
        tsrc_str = re.search(r'tsrc(\d+)', Ct_tags[0]).group()
        tag_to_be_removed = Ct_tags[0].replace(tsrc_str, f"tsrc{tsrc}")
        tsrcs.remove(tsrc)
        Ct_tags.remove(tag_to_be_removed)
        message(f"---> filtered tags: {Ct_tags}")
        message(f"---> filtered tsrcs: {tsrcs}")
    assert len(Ct_tags) == len(tsrcs)
    mt_tags = []
    for Ct_tag, tsrc in zip(Ct_tags, tsrcs):
        db.combine_sample(Ct_tag, f=lambda Ct: _get_masked_Cts_boundary(Ct, tsrc, tmin_excited, tmax_from_tsrc).mean(axis=0), dst_tag=f"{Ct_tag}/maskedES")
        binned_Ct_tag = db.add_binned_leaf(f"{Ct_tag}/maskedES", binsize)
        mt_tag = f"{binned_Ct_tag}/am_t"
        mt_tags.append(mt_tag)
        db.combine(binned_Ct_tag, f=lambda Ct: np.nan_to_num(_flip_sign_boundary(effective_mass_log2(Ct), tsrc), nan=0.0, posinf=0.0, neginf=0.0), dst_tag=mt_tag)
        if cleanup:
            db.remove_leaf(f"{Ct_tag}/maskedES")
            db.remove_leaf(binned_Ct_tag)
    dst_tag = re.sub(r'(tsrc)\d+', r'\1None', mt_tags[0])
    db.combine(*mt_tags, f=lambda *eff_mass: np.ma.filled(np.ma.masked_equal(eff_mass, 0).mean(axis=0), 0), dst_tag=dst_tag)
    db.combine(dst_tag, f=lambda mt: _fold_boundary(mt, antiperiodic), dst_tag=f"{dst_tag}/folded")
    if cleanup:
        for mt_tag in mt_tags: 
            db.remove_leaf(mt_tag)
    return dst_tag
