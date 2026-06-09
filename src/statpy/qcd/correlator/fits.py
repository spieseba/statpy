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
from statpy.fitting.core import print_fit_results, get_pvalue
from statpy.statistics import jackknife, bootstrap

from statpy.qcd.correlator.primitives import (
    meff_cosh, meff_exp_forward, meff_exp_symmetric,
    Aeff_cosh, Aeff_sinh, Aeff_exp,
    fold_correlator,
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


def binned_tag(tag, binsize):
    """Conventional tag of the binsize-``binsize`` variant of ``tag``.

    ``binsize == 1`` returns ``tag`` unchanged (no binning needed). This is
    the single home of the ``<tag>/binsize<N>`` naming convention: producers
    (the fit routines below) and consumers must both route tag construction
    through it so they agree.
    """
    return tag if binsize == 1 else f"{tag}/binsize{binsize}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FitConfig:
    """Optimizer choice + parameters; ``bootstrap_available`` gates bootstrap fits."""
    fit_method: str = "Nelder-Mead"
    fit_params: dict = field(default_factory=lambda: {"maxiter": 5000, "tol": 1e-07})
    bootstrap_available: bool = True


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

_LOG_DIVIDER_WIDTH = 81

def _log_divider(title=None, fill="-"):
    """Fixed-width log section divider, with optional centered title."""
    if title is None:
        return fill * _LOG_DIVIDER_WIDTH
    pad = _LOG_DIVIDER_WIDTH - len(title) - 2
    left = pad // 2
    return f"{fill * left} {title} {fill * (pad - left)}"


def _make_slicer(t, slice_data):
    """Return ``y -> y[t]`` if ``slice_data`` else ``y -> y`` (data already sliced)."""
    return (lambda y: y[t]) if slice_data else (lambda y: y)


def _make_chi2(fit_model, W, Nt):
    """Return a chi^2 lambda(t, p, y) for the given fit model + weight matrix."""
    if fit_model == "double-cosh":
        return lambda t, p, y: double_cosh_chi2(t, p, y, W, Nt)
    if fit_model == "double-sinh":
        return lambda t, p, y: double_sinh_chi2(t, p, y, W, Nt)
    if fit_model == "double-exp":
        return lambda t, p, y: double_exp_chi2(t, p, y, W)
    if fit_model == "cosh":
        return lambda t, p, y: cosh_chi2(t, p, y, W, Nt)
    if fit_model == "sinh":
        return lambda t, p, y: sinh_chi2(t, p, y, W, Nt)
    if fit_model == "exp":
        return lambda t, p, y: exp_chi2(t, p, y, W)
    raise ValueError(f"Unknown fit_model: {fit_model!r}")


# ---------------------------------------------------------------------------
# Core fit primitives
# ---------------------------------------------------------------------------

def fit_mean(db, t, tag, p0, chi2_func, config: FitConfig, slice_data=True):
    """Fit the mean of the entry at ``tag``.

    Returns ``(best_parameter, misc)``. ``misc`` carries
    ``{"t", "chi2", "dof", "pval"}``.

    ``slice_data=True`` (default): ``t`` is an index array into the data entry;
    the data is sliced as ``y[t]`` before being passed to ``chi2_func``.

    ``slice_data=False``: the data entry is *already* pre-sliced (length matches
    ``len(t)``) and ``t`` is a structured key consumed by ``chi2_func`` directly
    rather than used to index the data. Used by combined fits where the data is
    a concatenation of two sub-ranges and ``t = np.hstack((range_A, range_B))``.

    Raises:
        ValueError: ``p0`` not 1-D / empty / non-numeric, ``len(t)`` mismatch
            when ``slice_data=False``, or non-positive degrees of freedom.
        KeyError: ``tag`` not in ``db.database``.
        ConvergenceError: fit did not converge or produced non-finite
            parameters / chi^2.
    """
    p0 = np.asarray(p0, dtype=float)
    if p0.ndim != 1 or p0.size == 0:
        raise ValueError(f"'p0' must be a non-empty 1-D array, got shape {p0.shape}")
    if not slice_data and len(t) != len(db.database[tag].mean):
        raise ValueError(
            f"with slice_data=False, len(t)={len(t)} must equal data length "
            f"{len(db.database[tag].mean)} for tag {tag!r}"
        )
    dof = len(t) - p0.size
    if dof <= 0:
        raise ValueError(
            f"non-positive degrees of freedom: len(t)={len(t)}, n_params={p0.size}, dof={dof}"
        )

    sl = _make_slicer(t, slice_data)
    fitter = Fitter(config.fit_method, config.fit_params)
    try:
        best = fitter.estimate_parameters(t, chi2_func, sl(db.database[tag].mean), p0)[0]
    except ConvergenceError as e:
        raise ConvergenceError(f"mean fit for tag {tag!r} did not converge: {e}") from e
    if not np.isfinite(best).all():
        raise ConvergenceError(f"mean fit for tag {tag!r} produced non-finite parameters: {best}")
    chi2 = chi2_func(t, best, sl(db.database[tag].mean))
    if not np.isfinite(chi2):
        raise ConvergenceError(f"non-finite chi^2 = {chi2} for tag {tag!r}")
    return best, {"t": t, "chi2": chi2, "dof": dof, "pval": get_pvalue(chi2, dof)}


def _fit_resamples(transform, label, t, tag, seed, chi2_func, config, slice_data, **transform_kwargs):
    """Run ``transform(tag, f=..., **transform_kwargs)`` to fit each resample.

    ``transform`` is :meth:`DB.transform_jks` or :meth:`DB.transform_bss`;
    ``label`` ("jackknife" / "bootstrap") is used only in the error message.
    ``transform_kwargs`` are forwarded to ``transform`` (e.g. ``bootstraps=``
    for :meth:`DB.transform_bss` on raw-data entries).
    """
    sl = _make_slicer(t, slice_data)
    fitter = Fitter(config.fit_method, config.fit_params)
    try:
        return transform(tag, f=lambda y: fitter.estimate_parameters(t, chi2_func, sl(y), seed)[0], **transform_kwargs)
    except ConvergenceError as e:
        raise ConvergenceError(f"{label} fit for tag {tag!r} did not converge: {e}") from e


def fit_jks(db, t, tag, p0, chi2_func, config: FitConfig, slice_data=True):
    """Fit the mean and the jackknife resamples for the entry at ``tag``.

    Mean is fit first (via :func:`fit_mean`); each jackknife sample is then
    fit seeded from the mean's best parameter. See :func:`fit_mean` for the
    meaning of ``slice_data``.

    Returns ``(best_parameter, best_parameter_jks, misc)``.
    """
    best, misc = fit_mean(db, t, tag, p0, chi2_func, config, slice_data=slice_data)
    best_jks = _fit_resamples(db.transform_jks, "jackknife", t, tag, best, chi2_func, config, slice_data)
    return best, best_jks, misc


def fit_bss(db, t, tag, p0, chi2_func, config: FitConfig, slice_data=True, bootstraps=None):
    """Fit the mean and the bootstrap resamples for the entry at ``tag``.

    Mean is fit first (via :func:`fit_mean`); each bootstrap sample is then
    fit seeded from the mean's best parameter. See :func:`fit_mean` for the
    meaning of ``slice_data``. ``bootstraps`` is the index matrix passed
    through to :meth:`DB.transform_bss` for raw-data entries; ignored if the
    entry already carries ``entry.bss``.

    Returns ``(best_parameter, best_parameter_bss, misc)``.
    """
    best, misc = fit_mean(db, t, tag, p0, chi2_func, config, slice_data=slice_data)
    best_bss = _fit_resamples(db.transform_bss, "bootstrap", t, tag, best, chi2_func, config, slice_data, bootstraps=bootstraps)
    return best, best_bss, misc


# ---------------------------------------------------------------------------
# Correlator averaging / folding
# ---------------------------------------------------------------------------

def correlator_avg_pbc(db, Ct_tag, dst_tag):
    """Average a PBC correlator over its source axis; write to ``dst_tag``."""
    assert isinstance(Ct_tag, str)
    entry = db.database[Ct_tag]
    new_sample = entry.sample.mean(axis=1)
    db.add_entry(dst_tag, sample=new_sample, weights=entry.weights, cfgs=entry.cfgs, misc=entry.misc)


def correlator_avg_obc(db, Ct_tags, tbulk, dst_tag, tmax_from_tsrc=None, antiperiodic=False):
    """OBC tsrc average: per-src mask to bulk fw/bw tmax, then concatenate-and-mean across sources."""
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
        entry = db.database[Ct_tag]
        masked = np.ma.array([_get_masked_Ct(Ct, tmax_fw[src_idx], tmax_bw[src_idx], antiperiodic) for Ct in entry.sample])
        db.add_entry(f"{Ct_tag}/masked", sample=masked, weights=entry.weights, cfgs=entry.cfgs)
    ref_lf = db.database[Ct_tags_in_bulk[0]]
    masked_samples = [db.database[f"{Ct_tag}/masked"].sample for Ct_tag in Ct_tags_in_bulk]
    combined_sample = np.array([
        np.ma.concatenate([ms[i] for ms in masked_samples], axis=0).mean(axis=0).compressed()
        for i in range(len(ref_lf.cfgs))
    ])
    db.add_entry(
        dst_tag, sample=combined_sample, weights=ref_lf.weights, cfgs=ref_lf.cfgs,
        misc={"tsrcs": tsrcs_in_bulk, "tbulk": tbulk, "antiperiodic": antiperiodic},
    )
    for Ct_tag in Ct_tags_in_bulk:
        db.remove_entry(f"{Ct_tag}/masked")


def fold_correlator_leaf(db, Ct_tag, antiperiodic=False):
    """Fold correlator around the symmetric center; result stored at ``{Ct_tag}/folded``."""
    message(f"Fold correlator {Ct_tag}.")
    entry = db.database[Ct_tag]
    folded = np.array([fold_correlator(Ct, antiperiodic) for Ct in entry.sample])
    db.add_entry(f"{Ct_tag}/folded", sample=folded, weights=entry.weights, cfgs=entry.cfgs, misc=entry.misc)


def boundary_avg(db, Ct_tags, tmin_excited, binsize, tmax_from_tsrc=None, antiperiodic=False, cleanup=False, excluded_tsrcs=[]):
    """Per-tsrc effective mass with excited-state region masked, averaged across sources, then folded.

    Returns the dst tag (``tsrc<None>/am_t``); ``{dst}/folded`` is also written.
    """
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
        entry = db.database[Ct_tag]
        maskedES_sample = np.array([
            _get_masked_Cts_boundary(Ct, tsrc, tmin_excited, tmax_from_tsrc).mean(axis=0)
            for Ct in entry.sample
        ])
        masked_tag = f"{Ct_tag}/maskedES"
        db.add_entry(masked_tag, sample=maskedES_sample, weights=entry.weights, cfgs=entry.cfgs)
        binned_Ct_tag = binned_tag(masked_tag, binsize)
        if binned_Ct_tag != masked_tag and binned_Ct_tag not in db.database:
            db.add_entry(binned_Ct_tag, **db.bin_entry(masked_tag, binsize))
        mt_tag = f"{binned_Ct_tag}/am_t"
        mt_tags.append(mt_tag)
        db.transform(binned_Ct_tag, f=lambda Ct: np.nan_to_num(_flip_sign_boundary(meff_exp_symmetric(Ct), tsrc), nan=0.0, posinf=0.0, neginf=0.0), dst_tag=mt_tag)
        if cleanup:
            db.remove_entry(masked_tag)
            db.remove_entry(binned_Ct_tag)
    dst_tag = re.sub(r'(tsrc)\d+', r'\1None', mt_tags[0])
    db.combine(*mt_tags, f=lambda *eff_mass: np.ma.filled(np.ma.masked_equal(eff_mass, 0).mean(axis=0), 0), dst_tag=dst_tag)
    db.transform(dst_tag, f=lambda mt: _fold_boundary(mt, antiperiodic), dst_tag=f"{dst_tag}/folded")
    if cleanup:
        for mt_tag in mt_tags:
            db.remove_entry(mt_tag)
    return dst_tag


# ---------------------------------------------------------------------------
# Two-state private helpers and heuristics
# ---------------------------------------------------------------------------

@dataclass
class _EntrySpec:
    """A pending ``db.add_entry`` call: expand with ``**spec.__dict__`` to commit."""
    tag: str | None = None
    mean: object = None
    jks: object = None
    cfgs: object = None
    misc: dict | None = None


def _sort_two_state_params(p):
    """Order the two states of a double-* fit so the lower mass comes first."""
    if p[3] < p[1]:
        return [p[2], p[3], p[0], p[1]]
    return p


def _resolve_initial_p0(p0_guess, prev_mean):
    """Replace NaNs in ``p0_guess``: prefer ``prev_mean`` if available, else
    derive ``p[2]`` and ``p[3]`` from ``p[0]``, then fall back to 1.0 for any
    leftover NaNs. Returns the resolved p0; assumes the input already has the
    expected length."""
    if not np.isnan(p0_guess).any():
        return p0_guess
    if prev_mean is not None:
        return prev_mean
    p0_guess[2] = p0_guess[0] / 2
    p0_guess[3] = 2.0 * p0_guess[0]
    if np.isnan(p0_guess).any():
        p0_guess = [1.0 if isnan(p) else p for p in p0_guess]
    return p0_guess


def _select_plateau_range(t, var_t, best_parameter, model_func, bc, folded):
    """Indices in ``t`` where the excited-state contribution drops below sigma/4
    (symmetrized for periodic + unfolded BC) — i.e., the ground-state plateau."""
    excited = np.abs([model_func(i, [0, 0, best_parameter[2], best_parameter[3]]) for i in t])
    std_over_four = (var_t ** 0.5) / 4.0
    if bc == "periodic" and not folded:
        std_over_four = (std_over_four + std_over_four[::-1]) / 2
    return t[excited < std_over_four]


def get_p0_guess(db, binned_corr_tag, fit_model, fit_range):
    """Heuristic two-state ``[A0, m0, A1, m1]`` initial guess for double-{cosh,sinh,exp} fits.

    ``A0, m0`` come from the effective-mass / -amplitude averaged over a central
    window ``[Nt/4, 3*Nt/8)``; ``A1, m1`` come from the residual ``Ct - C_ground``
    evaluated at ``fit_range[0]``. Any element may be NaN when the effective
    primitives encounter non-positive arguments — the caller (e.g.
    ``_resolve_initial_p0``) is responsible for filling NaNs.
    """
    if fit_model not in ("double-cosh", "double-sinh", "double-exp"):
        raise ValueError(f"Unknown fit_model: {fit_model!r}")
    message(f"Get p0 guess(es) for {fit_model} fit model with {binned_corr_tag}")
    Ct_mean = db.database[binned_corr_tag].mean
    Nt = len(Ct_mean)
    effective_mass = {"double-cosh": meff_cosh, "double-sinh": meff_cosh, "double-exp": meff_exp_forward}[fit_model]
    effective_amplitude = {"double-cosh": Aeff_cosh, "double-sinh": Aeff_sinh, "double-exp": Aeff_exp}[fit_model]
    single_model_func = {"double-cosh": cosh_model(Nt), "double-sinh": sinh_model(Nt), "double-exp": exp_model()}[fit_model]
    # ground state parameters
    window = slice(Nt//4, Nt//4 + Nt//8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        m0_eff = np.nanmean(effective_mass(Ct_mean)[window])
        A0_eff = np.nanmean(effective_amplitude(Ct_mean, m0_eff)[window])
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


def _try_correlated_fit(db, binned_corr_tag, t, cov_t, p0, fit_model, Nt, config, label):
    """Run a correlated mean fit on already-sliced ``cov_t``.

    Returns ``(best_parameter, misc)`` on success or ``(None, None)`` if the
    covariance matrix is not positive definite or the fit fails to converge.
    Caller is responsible for any post-processing (parameter sorting, printing,
    persisting to db).
    """
    message(f"Check positive definiteness of {label} covariance matrix for fit range [[{t[0]},{t[-1]}]].")
    if not np.all(np.linalg.eigvals(cov_t) > 0):
        message(f"--> {label} covariance matrix not positive definite.")
        return None, None
    message(f"--> {label} covariance matrix positive definite. Try correlated fit.")
    try:
        W = np.linalg.inv(cov_t)
        chi2 = _make_chi2(fit_model, W, Nt)
        best, misc = fit_mean(db, t, binned_corr_tag, p0, chi2, config)
        misc["fit_model"] = fit_model
        return best, misc
    except ConvergenceError as ce:
        message(f"{ce} for correlated mean fit with {label} covariance matrix")
        return None, None


def _fit_one_excited_range(db, tag, binned_corr_tag, t, p0_input, prev_excited_mean, fit_model, Nt, var, cov_binned, model_func, bc, folded, binsize, config, silent):
    """One iteration of the excited-state-contribution fit loop.

    Returns ``None`` on convergence error, otherwise
    ``(t_plateau, excited_spec, binned_corr_spec, unbinned_corr_spec, best_parameter)``.
    The caller decides whether the candidate specs supersede the running best
    based on ``len(t_plateau)`` vs the current accepted fit range.
    """
    message(f"Excited fit range: [[{t[0]},{t[-1]}]]", silent)
    message(_log_divider("uncorrelated fit"), silent)
    chi2_func = _make_chi2(fit_model, np.linalg.inv(np.diag(var[t])), Nt)
    p0_guess = get_p0_guess(db, binned_corr_tag, fit_model, t) if p0_input is None else p0_input
    had_nan = np.isnan(p0_guess).any()
    p0_tmp = _resolve_initial_p0(p0_guess, prev_excited_mean)
    if had_nan:
        message(f"p0 guess contains NaN, use fit result from previous fit range if available, else use available params to estimate NaNs or default to 1: {p0_tmp}")
    try:
        message(f"p0 for fit: {p0_tmp}")
        best_parameter, best_parameter_jks, misc = fit_jks(db, t, binned_corr_tag, p0_tmp, chi2_func, config)
        misc["fit_model"] = fit_model
    except ConvergenceError as ce:
        message(f"{ce} -> jump to next fit range")
        message(_log_divider(), silent)
        message(_log_divider(), silent)
        return None
    best_parameter = _sort_two_state_params(best_parameter)
    best_parameter_jks = np.array([_sort_two_state_params(jk) for jk in best_parameter_jks])
    best_parameter_cov = jackknife.covariance(best_parameter_jks)
    print_fit_results(best_parameter, best_parameter_cov, misc, silent)

    message(_log_divider("correlated mean fit"), silent)
    message("Try correlated fit with binned covariance matrix")
    binned_best, binned_misc = _try_correlated_fit(db, binned_corr_tag, t, cov_binned[t][:, t], p0_tmp, fit_model, Nt, config, "binned")
    if binned_best is not None:
        binned_best = _sort_two_state_params(binned_best)
        print_fit_results(binned_best, None, binned_misc, silent)

    message("Try correlated fit with unbinned covariance matrix.")
    cov_unbinned = db.jackknife_covariance(tag)
    unbinned_best, unbinned_misc = _try_correlated_fit(db, binned_corr_tag, t, cov_unbinned[t][:, t], p0_tmp, fit_model, Nt, config, "unbinned")
    if unbinned_best is not None:
        unbinned_best = _sort_two_state_params(unbinned_best)
        print_fit_results(unbinned_best, None, unbinned_misc, silent)
    message(_log_divider(), silent)

    t_plateau = _select_plateau_range(t, var[t], best_parameter, model_func, bc, folded)

    excited_spec = _EntrySpec(
        tag=f"{binned_corr_tag}/excited_contributions_fit",
        mean=best_parameter, jks=best_parameter_jks,
        cfgs=db.database[binned_corr_tag].cfgs, misc=misc,
    )
    binned_corr_spec = _EntrySpec(tag=f"{binned_corr_tag}/binned_correlated_excited_contributions_mean_fit")
    if binned_best is not None:
        binned_corr_spec.mean = binned_best
        binned_corr_spec.misc = binned_misc
    unbinned_corr_spec = _EntrySpec(tag=f"{binned_corr_tag}/unbinned_correlated_excited_contributions_mean_fit")
    if unbinned_best is not None:
        unbinned_corr_spec.mean = unbinned_best
        unbinned_corr_spec.misc = unbinned_misc
    return t_plateau, excited_spec, binned_corr_spec, unbinned_corr_spec, best_parameter


# ---------------------------------------------------------------------------
# Excited-state / ground-state fits
# ---------------------------------------------------------------------------

class PlateauTooShortError(ValueError):
    """No candidate fit range reached ``min_plateau_len`` slices."""


def excited_contributions_fit(db, tag, binsize, excited_fit_ranges, p0, fit_model, config: FitConfig, silent=False, Nt=None, min_plateau_len=5, folded=False):
    """Two-state fits across candidate ranges; pick the one whose plateau (where excited
    contributions drop below sigma/4) is shortest but at least ``min_plateau_len`` long.

    Returns ``(plateau_fit_range, last_best_parameter)``. Raises
    ``PlateauTooShortError`` if no candidate range reaches ``min_plateau_len``.
    """
    message(f"Correlator: {tag}")
    if p0 is None:
        message("P0 is inferred for each initial fit range automatically.")
    else:
        message(f"P0 = {p0}")
    message(f"Binsize = {binsize}", silent)
    message(f"{fit_model} model = {fit_model_dict[fit_model]}")
    message(_log_divider(), silent)
    binned_corr_tag = binned_tag(tag, binsize)
    if binned_corr_tag != tag and binned_corr_tag not in db.database:
        db.add_entry(binned_corr_tag, **db.bin_entry(tag, binsize))
    cov = db.jackknife_covariance(binned_corr_tag)
    var = np.diag(cov)
    Nt = len(db.database[binned_corr_tag].mean) if Nt is None else Nt
    model_func = {"double-cosh": double_cosh_model(Nt),
                  "double-sinh": double_sinh_model(Nt),
                  "double-exp": double_exp_model()}[fit_model]
    bc = "open" if fit_model == "double-exp" else "periodic"
    excited_spec = _EntrySpec()
    binned_corr_spec = _EntrySpec()
    unbinned_corr_spec = _EntrySpec()
    suggested_fit_ranges = []
    fit_range = excited_fit_ranges[0]
    last_best_parameter = None
    for t in excited_fit_ranges:
        result = _fit_one_excited_range(
            db, tag, binned_corr_tag, t, p0, excited_spec.mean,
            fit_model, Nt, var, cov, model_func, bc, folded, binsize, config, silent,
        )
        if result is None:
            suggested_fit_ranges.append(None)
            continue
        t_plateau, excited_cand, binned_cand, unbinned_cand, best_parameter = result
        last_best_parameter = best_parameter
        suggested_fit_ranges.append(t_plateau)
        if len(t_plateau) < min_plateau_len:
            message(f"Determined fit range {t_plateau} has fewer than {min_plateau_len} elements", silent)
            message("---> Stored fit range is not updated", silent)
            message(_log_divider(), silent)
            message(_log_divider(), silent)
            continue
        message(f"Determined fit range [[{t_plateau[0]},{t_plateau[-1]}]]", silent)
        if len(t_plateau) <= len(fit_range):
            message("---> Stored fit range is updated", silent)
            excited_cand.misc["plateau_fit_range"] = t_plateau
            fit_range = t_plateau
            excited_spec, binned_corr_spec, unbinned_corr_spec = excited_cand, binned_cand, unbinned_cand
        message(_log_divider(), silent)
        message(_log_divider(), silent)
    if excited_spec.misc is None:
        best = max((len(s) for s in suggested_fit_ranges if s is not None), default=0)
        raise PlateauTooShortError(
            f"excited_contributions_fit({tag!r}): no candidate range reached "
            f"min_plateau_len={min_plateau_len} (longest plateau found: {best}); "
            f"lower min_plateau_len or loosen the S/N cut"
        )
    db.add_entry(**binned_corr_spec.__dict__)
    db.add_entry(**unbinned_corr_spec.__dict__)
    excited_spec.misc["tested_suggested_fit_ranges"] = (excited_fit_ranges, suggested_fit_ranges)
    db.add_entry(**excited_spec.__dict__)
    return excited_spec.misc["plateau_fit_range"], last_best_parameter


def ground_state_fit(db, tag, binsize, fit_range, p0, fit_model, config: FitConfig, Nt=None, silent=False, bootstraps=None):
    """Ground-state fit at every binsize 1..``binsize``: jackknife always; correlated mean
    (binned + unbinned) at b=1 and b=``binsize``; bootstrap at b=1. Persists each fit as a entry.

    ``bootstraps`` is the index matrix forwarded to :meth:`DB.bss` and
    :func:`fit_bss` for the b=1 bootstrap branch; required when
    ``config.bootstrap_available``.

    Returns the list of fit-result tags (one per binsize 1..``binsize``).
    """
    message(f"Correlator: {tag}")
    message(f"P0 = {p0}")
    message(f"Fit range {fit_range}")
    message(f"{fit_model} model = {fit_model_dict[fit_model]}")
    fit_tags = []
    for b in range(1, binsize+1):
        message(f"Binsize = {b}", silent)
        binned_corr_tag = binned_tag(tag, b)
        if binned_corr_tag != tag and binned_corr_tag not in db.database:
            db.add_entry(binned_corr_tag, **db.bin_entry(tag, b))
        message(_log_divider("jackknife fit"), silent)
        var = db.jackknife_variance(binned_corr_tag)
        Nt = len(db.database[binned_corr_tag].mean) if Nt is None else Nt
        W = np.linalg.inv(np.diag(var[fit_range]))
        chi2_func = _make_chi2(fit_model, W, Nt)
        best_parameter, best_parameter_jks, misc = fit_jks(db, fit_range, binned_corr_tag, p0, chi2_func, config)
        misc["fit_model"] = fit_model
        best_parameter_cov = jackknife.covariance(best_parameter_jks)
        print_fit_results(best_parameter, best_parameter_cov, misc, silent)
        if b in [1,binsize]:
            message(_log_divider("correlated mean fit"), silent)
            message("Try fit with covariance matrix")
            cov_t = db.jackknife_covariance(binned_corr_tag)[fit_range][:,fit_range]
            best, misc_corr = _try_correlated_fit(db, binned_corr_tag, fit_range, cov_t, p0, fit_model, Nt, config, "binned")
            if best is not None:
                print_fit_results(best, None, misc_corr, silent)
                db.add_entry(f"{binned_corr_tag}/{fit_model}_binned_correlated_mean_fit", mean=best, misc=misc_corr)
            if b != 1:
                message("Try fit with unbinned covariance matrix")
                cov_t_unbinned = db.jackknife_covariance(tag)[fit_range][:,fit_range]
                best, misc_corr = _try_correlated_fit(db, binned_corr_tag, fit_range, cov_t_unbinned, p0, fit_model, Nt, config, "unbinned")
                if best is not None:
                    print_fit_results(best, None, misc_corr, silent)
                    db.add_entry(f"{binned_corr_tag}/{fit_model}_unbinned_correlated_mean_fit", mean=best, misc=misc_corr)
            message(_log_divider(), silent)
        if b == 1 and config.bootstrap_available:
            message(_log_divider("bootstrap fit"), silent)
            assert bootstraps is not None, "ground_state_fit needs bootstraps= when config.bootstrap_available"
            bss = db.bss(binned_corr_tag, bootstraps)
            W_bss = np.linalg.inv(np.diag(bootstrap.variance(bss)[fit_range]))
            chi2_func_bss = _make_chi2(fit_model, W_bss, Nt)
            best_parameter_bmean, best_parameter_bss, misc_bss = fit_bss(db, fit_range, binned_corr_tag, best_parameter, chi2_func_bss, config, bootstraps=bootstraps)
            best_parameter_bcov = bootstrap.covariance(best_parameter_bss)
            print_fit_results(best_parameter_bmean, best_parameter_bcov, misc_bss)
            misc_bss["fit_model"] = fit_model
            db.add_entry(f"{binned_corr_tag}/{fit_model}_bootstrap_fit", mean=best_parameter_bmean, bss=best_parameter_bss, misc=misc_bss)
        fit_tag = f"{binned_corr_tag}/{fit_model}_fit"
        db.add_entry(
            fit_tag,
            mean=best_parameter, jks=best_parameter_jks,
            cfgs=db.database[binned_corr_tag].cfgs, misc=misc,
        )
        fit_tags.append(fit_tag)
        message(_log_divider(), silent)
        message(_log_divider(), silent)
    return fit_tags


# ---------------------------------------------------------------------------
# Decay-constant / combined PSPS+PSA4I fit machinery.
#
# Currently unused. Kept here pending a planned refactor for decay constant
# project;
# Touches: ``determine_PSA4I`` (PSA4 improvement) and ``correlator_combined_fit``
# (joint PSPS/PSA4I jackknife + correlated + bootstrap fit, plus the bare decay
# constant ``afbare`` derived from each).
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
    lf_a = db.database[tag_PSA4_sml]
    lf_b = db.database[tag_PSPS_sml]
    new_sample = np.array([compute_PSA4I(a, b, beta) for a, b in zip(lf_a.sample, lf_b.sample)])
    db.add_entry(tag_PSA4I, sample=new_sample, weights=lf_a.weights, cfgs=lf_a.cfgs)


def correlator_combined_fit(db, tag_PS, tag_A4I, fit_range_PS, fit_range_A4I, binsize, p0, fit_model_combined, config: FitConfig, Nt=None, silent=False, bootstraps=None):
    """Joint PSPS / PSA4I fit on the concatenated ``(PS ++ A4I)`` data entry.

    Same per-binsize structure as :func:`ground_state_fit` (jackknife, correlated mean,
    bootstrap), plus the bare decay constant ``afbare`` derived from each fit.
    """
    message(_log_divider("combined correlator fit PSPS/PSA4I"))
    fit_model_PS = fit_model_combined.split("-")[1]
    fit_model_A4I = fit_model_combined.split("-")[2]
    message(f"PSPS correlator: {tag_PS}")
    message(f"PSPS - fit range {fit_range_PS}")
    message(f"PSPS {fit_model_PS} model = {fit_model_dict[fit_model_PS]}")
    message(f"PSA4I correlator: {tag_A4I}")
    message(f"PSA4I - fit range {fit_range_A4I}")
    message(f"PSA4I {fit_model_A4I} model = {fit_model_dict[fit_model_A4I]}")
    message(f"Combined - {fit_model_combined} model = {fit_model_dict[fit_model_combined]}")
    message(f"P0 = {p0}")

    Nt = len(db.database[tag_PS].mean) if Nt is None else Nt
    fit_range_combined = np.hstack((fit_range_PS, fit_range_A4I))
    combined_tag = f"{tag_PS};{tag_A4I.split('/')[1]}"
    lf_ps = db.database[tag_PS]
    lf_a4 = db.database[tag_A4I]
    combined_sample = np.array([np.hstack((x[fit_range_PS], y[fit_range_A4I])) for x, y in zip(lf_ps.sample, lf_a4.sample)])
    db.add_entry(combined_tag, sample=combined_sample, weights=lf_ps.weights, cfgs=lf_ps.cfgs)
    for b in range(1, binsize+1):
        message(f"Binsize = {b}", silent)
        binned_corr_tag = binned_tag(combined_tag, b)
        if binned_corr_tag != combined_tag and binned_corr_tag not in db.database:
            db.add_entry(binned_corr_tag, **db.bin_entry(combined_tag, b))
        message(_log_divider("jackknife fit"), silent)
        var = db.jackknife_variance(binned_corr_tag)
        W = np.linalg.inv(np.diag(var))
        chi2_func = {"combined-cosh-sinh": lambda t,p,y: combined_cosh_sinh_chi2(t[:len(fit_range_PS)], t[len(fit_range_PS):], p, y, W, Nt),
                     "combined-exp-exp": lambda t,p,y: combined_exp_exp_model_chi2(t[:len(fit_range_PS)], t[len(fit_range_PS):], p, y, W)}[fit_model_combined]
        # combined entry is pre-sliced (PS ++ A4I); t is the structured PS/A4I index used inside chi2_func
        best_parameter, best_parameter_jks, misc = fit_jks(db, fit_range_combined, binned_corr_tag, p0, chi2_func, config, slice_data=False)
        misc["fit_model_PSPS"] = fit_model_PS
        misc["fit_model_PSA4I"] = fit_model_A4I
        misc["fit_model"] = fit_model_combined
        misc["t_PSPS"] = fit_range_PS
        misc["t_PSA4I"] = fit_range_A4I
        best_parameter_cov = jackknife.covariance(best_parameter_jks)
        print_fit_results(best_parameter, best_parameter_cov, misc, silent)
        if b in [1,binsize]:
            message(_log_divider("correlated mean fit"), silent)
            try:
                W_correlated = np.linalg.inv(db.jackknife_covariance(binned_corr_tag))
                chi2_func_correlated = {"combined-cosh-sinh": lambda t,p,y: combined_cosh_sinh_chi2(t[:len(fit_range_PS)], t[len(fit_range_PS):], p, y, W_correlated, Nt),
                                        "combined-exp-exp": lambda t,p,y: combined_exp_exp_model_chi2(t[:len(fit_range_PS)], t[len(fit_range_PS):], p, y, W_correlated)}[fit_model_combined]
                # combined entry is pre-sliced; t is the structured PS/A4I index used inside chi2_func_correlated
                best_parameter_correlated, misc_correlated = fit_mean(db, fit_range_combined, binned_corr_tag, best_parameter, chi2_func_correlated, config, slice_data=False)
                misc_correlated["fit_model_PSPS"] = fit_model_PS
                misc_correlated["fit_model_PSA4I"] = fit_model_A4I
                misc_correlated["fit_model"] = fit_model_combined
                misc_correlated["t_PSPS"] = fit_range_PS
                misc_correlated["t_PSA4I"] = fit_range_A4I
                print_fit_results(best_parameter_correlated, None, misc_correlated, silent)
                db.add_entry(f"{binned_corr_tag}/{fit_model_combined}_correlated_mean_fit", mean=best_parameter_correlated, misc=misc_correlated)
            except ConvergenceError as ce:
                message(f"{ce} for correlated mean fit")
                message(_log_divider(), silent)
        if b == 1 and config.bootstrap_available:
            message(_log_divider("bootstrap fit"), silent)
            assert bootstraps is not None, "correlator_combined_fit needs bootstraps= when config.bootstrap_available"
            bss = db.bss(binned_corr_tag, bootstraps)
            W_bss = np.linalg.inv(np.diag(bootstrap.variance(bss)))
            chi2_func_bss = {"combined-cosh-sinh": lambda t,p,y: combined_cosh_sinh_chi2(t[:len(fit_range_PS)], t[len(fit_range_PS):], p, y, W_bss, Nt),
                             "combined-exp-exp": lambda t,p,y: combined_exp_exp_model_chi2(t[:len(fit_range_PS)], t[len(fit_range_PS):], p, y, W_bss)}[fit_model_combined]
            # combined entry is pre-sliced; t is the structured PS/A4I index used inside chi2_func_bss
            best_parameter_bmean, best_parameter_bss, misc_bss = fit_bss(db, fit_range_combined, binned_corr_tag, best_parameter, chi2_func_bss, config, slice_data=False, bootstraps=bootstraps)
            misc_bss["fit_model_PSPS"] = fit_model_PS
            misc_bss["fit_model_PSA4I"] = fit_model_A4I
            misc_bss["fit_model"] = fit_model_combined
            misc_bss["t_PSPS"] = fit_range_PS
            misc_bss["t_PSA4I"] = fit_range_A4I
            best_parameter_bcov = bootstrap.covariance(best_parameter_bss)
            print_fit_results(best_parameter_bmean, best_parameter_bcov, misc_bss)
            db.add_entry(f"{binned_corr_tag}/{fit_model_combined}_bootstrap_fit", mean=best_parameter_bmean, bss=best_parameter_bss, misc=misc_bss)
        db.add_entry(
            f"{binned_corr_tag}/{fit_model_combined}_fit",
            mean=best_parameter, jks=best_parameter_jks,
            cfgs=db.database[binned_corr_tag].cfgs, misc=misc,
        )
        message(_log_divider("bare decay constant"))
        db.transform(f"{binned_corr_tag}/{fit_model_combined}_fit", f=bare_decay_constant, dst_tag=f"{binned_corr_tag}/{fit_model_combined}_fit/afbare")
        if b == 1 and config.bootstrap_available:
            bootstrap_tag = f"{binned_corr_tag}/{fit_model_combined}_bootstrap_fit"
            fbare_bss_mean = bare_decay_constant(db.database[bootstrap_tag].mean)
            fbare_bss = db.transform_bss(bootstrap_tag, f=bare_decay_constant)
            db.add_entry(f"{bootstrap_tag}/afbare", mean=fbare_bss_mean, bss=fbare_bss)
            fbare_bs_str = f"         {fbare_bss_mean:.8f} +- {bootstrap.variance(fbare_bss)**.5:.8f} (bootstrap)"
        message(f"a*fbare = {db.database[f'{binned_corr_tag}/{fit_model_combined}_fit/afbare'].mean:.8f} +- {db.jackknife_variance(f'{binned_corr_tag}/{fit_model_combined}_fit/afbare')**.5:.8f} (jackknife)")
        if b == 1 and config.bootstrap_available:
            message(fbare_bs_str)
        message(_log_divider(), silent)
        message(_log_divider(), silent)
