"""FitConfig dataclass and high-level correlator fit routines.

All functions take a database handle as first argument and a FitConfig
describing fit method + parameters. (Source averaging / folding lives in
``averaging.py``.)
"""
from dataclasses import dataclass, field

import numpy as np

from statpy.fitting.core import ConvergenceError, Fitter, get_pvalue, print_fit_results
from statpy.log import message
from statpy.qcd.correlator.models import (
    combined_corr_chi2,
    cosh_chi2,
    cosh_model,
    double_cosh_chi2,
    double_cosh_model,
    double_exp_chi2,
    double_exp_model,
    double_sinh_chi2,
    double_sinh_model,
    exp_chi2,
    exp_model,
    fit_model_dict,
    sinh_chi2,
    sinh_model,
)
from statpy.qcd.correlator.primitives import (
    binned_tag,
    meff_cosh,
    meff_exp_forward,
)
from statpy.statistics import bootstrap, jackknife

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FitConfig:
    """Optimizer choice + parameters; ``bootstrap_available`` gates bootstrap fits."""
    fit_method: str = "Nelder-Mead"
    fit_params: dict = field(default_factory=lambda: {"maxiter": 5000, "tol": 1e-07})
    bootstrap_available: bool = True


@dataclass(frozen=True)
class FitTags:
    """Database references to a primary jackknife fit and its optional bootstrap fit."""
    jackknife: str
    bootstrap: str | None = None


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


def _ensure_binned(db, tag, binsize):
    """Return the binned tag for ``tag``, creating the binned entry if missing."""
    binned = binned_tag(tag, binsize)
    if binned != tag and binned not in db.database:
        db.add_entry(binned, **db.bin_entry(tag, binsize))
    return binned


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


# Backward-propagator sign per block for :func:`combined_corr_chi2`.
_BLOCK_SIGN = {"cosh": 1.0, "sinh": -1.0, "exp": 0.0}


def _combined_model_name(fit_models):
    """Return the combined model name for two validated block models."""
    if len(fit_models) != 2:
        raise ValueError(f"fit_models must contain two models, got {len(fit_models)}")
    unknown = [model for model in fit_models if model not in _BLOCK_SIGN]
    if unknown:
        raise ValueError(f"Unknown combined block model(s): {unknown}")
    return "combined-" + "-".join(fit_models)


def _make_combined_chi2(fit_models, W, block_lengths, Nt):
    """Return a chi^2 lambda for two concatenated correlator blocks."""
    _combined_model_name(fit_models)
    sign = np.repeat([_BLOCK_SIGN[model] for model in fit_models], block_lengths)
    amp_idx = np.repeat(np.arange(2), block_lengths)
    return lambda t, p, y: combined_corr_chi2(t, p, y, W, Nt, amp_idx, sign)


# ---------------------------------------------------------------------------
# Core fit primitives
# ---------------------------------------------------------------------------

def fit_mean(db, t, tag, p0, chi2_func, config: FitConfig, slice_data=True):
    """Fit the mean of the entry at ``tag``; returns ``(best_parameter, misc)``
    with ``misc = {"t", "chi2", "dof", "pval"}``.

    ``slice_data=True``: ``t`` indexes the data, the fit sees ``y[t]``.
    ``slice_data=False``: the entry is already pre-sliced to length ``len(t)``
    and ``t`` is only passed through to ``chi2_func`` (combined fits, where
    ``t`` is the concatenation of two fit ranges).
    """
    p0 = np.asarray(p0, dtype=float)
    if p0.ndim != 1 or p0.size == 0:
        raise ValueError(f"'p0' must be a non-empty 1-D array, got shape {p0.shape}")
    if not slice_data and len(t) != len(db.database[tag].central_value):
        raise ValueError(
            f"with slice_data=False, len(t)={len(t)} must equal data length "
            f"{len(db.database[tag].central_value)} for tag {tag!r}"
        )
    dof = len(t) - p0.size
    if dof <= 0:
        raise ValueError(
            f"non-positive degrees of freedom: len(t)={len(t)}, n_params={p0.size}, dof={dof}"
        )

    sl = _make_slicer(t, slice_data)
    fitter = Fitter(config.fit_method, config.fit_params)
    try:
        best = fitter.estimate_parameters(t, chi2_func, sl(db.database[tag].central_value), p0)[0]
    except ConvergenceError as e:
        raise ConvergenceError(f"mean fit for tag {tag!r} did not converge: {e}") from e
    if not np.isfinite(best).all():
        raise ConvergenceError(f"mean fit for tag {tag!r} produced non-finite parameters: {best}")
    chi2 = chi2_func(t, best, sl(db.database[tag].central_value))
    if not np.isfinite(chi2):
        raise ConvergenceError(f"non-finite chi^2 = {chi2} for tag {tag!r}")
    return best, {"t": t, "chi2": chi2, "dof": dof, "pval": get_pvalue(chi2, dof)}


def _fit_resamples(transform, label, t, tag, seed, chi2_func, config, slice_data, **transform_kwargs):
    """Fit each resample via ``transform`` (:meth:`DB.transform_jks` or
    :meth:`DB.transform_bss`); ``label`` is used only in the error message."""
    sl = _make_slicer(t, slice_data)
    fitter = Fitter(config.fit_method, config.fit_params)
    try:
        return transform(tag, f=lambda y: fitter.estimate_parameters(t, chi2_func, sl(y), seed)[0], **transform_kwargs)
    except ConvergenceError as e:
        raise ConvergenceError(f"{label} fit for tag {tag!r} did not converge: {e}") from e


def fit_jks(db, t, tag, p0, chi2_func, config: FitConfig, slice_data=True):
    """Fit the mean, then every jackknife sample seeded from the mean fit.
    Returns ``(best_parameter, best_parameter_jks, misc)``."""
    best, misc = fit_mean(db, t, tag, p0, chi2_func, config, slice_data=slice_data)
    best_jks = _fit_resamples(db.transform_jks, "jackknife", t, tag, best, chi2_func, config, slice_data)
    return best, best_jks, misc


def fit_bss(db, t, tag, p0, chi2_func, config: FitConfig, slice_data=True, bootstraps=None):
    """Fit the mean, then every bootstrap sample seeded from the mean fit.
    Returns ``(best_parameter, best_parameter_bss, misc)``. ``bootstraps`` is
    ignored if the entry already carries ``entry.bss``."""
    best, misc = fit_mean(db, t, tag, p0, chi2_func, config, slice_data=slice_data)
    best_bss = _fit_resamples(db.transform_bss, "bootstrap", t, tag, best, chi2_func, config, slice_data, bootstraps=bootstraps)
    return best, best_bss, misc


# ---------------------------------------------------------------------------
# Two-state private helpers and heuristics
# ---------------------------------------------------------------------------

@dataclass
class _EntrySpec:
    """A pending ``db.add_entry`` call: expand with ``**spec.__dict__`` to commit."""
    tag: str | None = None
    central_value: object = None
    jks: object = None
    cfgs: object = None
    misc: dict | None = None


@dataclass
class _Candidate:
    """An accepted excited-fit range, pending the deferred jackknife fits."""
    idx: int            # position in excited_fit_ranges
    t: object           # candidate fit range
    t_plateau: object   # plateau determined from the mean fit
    seed: object        # unsorted mean-fit parameters, seed the jackknife fits
    excited: _EntrySpec
    binned: _EntrySpec
    unbinned: _EntrySpec


def _sort_two_state_params(p):
    """Order the two states of a double-* fit so the lower mass comes first."""
    if p[3] < p[1]:
        return [p[2], p[3], p[0], p[1]]
    return p


def _select_plateau_range(t, var_t, best_parameter, model_func, bc, folded):
    """Indices in ``t`` where the excited-state contribution drops below sigma/4
    (symmetrized for periodic + unfolded BC) — i.e., the ground-state plateau."""
    excited = np.abs([model_func(i, [0, 0, best_parameter[2], best_parameter[3]]) for i in t])
    std_over_four = (var_t ** 0.5) / 4.0
    if bc == "periodic" and not folded:
        std_over_four = (std_over_four + std_over_four[::-1]) / 2
    return t[excited < std_over_four]


def get_p0_guesses(t, y, variance, fit_model, m0, mass_gaps, *, Nt=None):
    """Return one [A0, m0, A1, m0 + gap] seed per supplied positive mass gap.

    ``t``, ``y`` and diagonal ``variance`` contain only the fit-window data.
    For each fixed mass pair, solve for signed amplitudes by minimizing
    sum((model - y)**2 / variance). Gap order is preserved; no nonlinear fit,
    previous-candidate seed or fallback is used. Periodic models require the
    full temporal extent ``Nt``, not the length of the fit window.

    Returns an array of shape (len(mass_gaps), 4). Invalid inputs or a mass
    pair whose amplitudes cannot be resolved numerically raise ValueError.
    """
    t, y, variance, gaps = [np.asarray(x, dtype=float) for x in (t, y, variance, mass_gaps)]
    if t.ndim != 1 or t.size < 2 or y.shape != t.shape or variance.shape != t.shape:
        raise ValueError("t, y and variance must be matching 1D arrays with at least two points")
    if not all(np.isfinite(x).all() for x in (t, y, variance)) or np.any(variance <= 0):
        raise ValueError("Fit data must be finite and variances strictly positive")
    if not np.isfinite(m0) or m0 <= 0:
        raise ValueError("m0 must be finite and positive")
    if gaps.ndim != 1 or gaps.size == 0 or not np.isfinite(gaps).all() or np.any(gaps <= 0):
        raise ValueError("mass_gaps must be a nonempty 1D array of finite positive gaps")
    if fit_model == "double-exp":
        model = exp_model()
    elif fit_model in ("double-cosh", "double-sinh"):
        if Nt is None or not np.isfinite(Nt) or Nt <= 0:
            raise ValueError("Periodic models require a finite positive Nt")
        model = cosh_model(Nt) if fit_model == "double-cosh" else sinh_model(Nt)
    else:
        raise ValueError(f"Unknown fit_model: {fit_model!r}")

    sigma = np.sqrt(variance)
    seeds = []
    for gap in gaps:
        m1 = m0 + gap
        if not np.isfinite(m1) or m1 <= m0:
            raise ValueError(f"Mass gap {gap} does not produce a finite mass above m0")
        design = np.column_stack((model(t, [1.0, m0]), model(t, [1.0, m1]))) / sigma[:, None] 
        scale = np.max(np.abs(design), axis=0) # scale columns to avoid treating the smaller excited kernel as zero
        if not np.isfinite(design).all() or np.any(scale == 0):
            raise ValueError(f"Nonfinite or vanishing model column for mass gap {gap}")
        amplitudes, _, rank, _ = np.linalg.lstsq(design / scale, y / sigma, rcond=None)
        amplitudes = amplitudes / scale
        if rank < 2 or not np.isfinite(amplitudes).all():
            raise ValueError(f"Cannot resolve two amplitudes for mass gap {gap}")
        seeds.append([amplitudes[0], m0, amplitudes[1], m1])
    return np.asarray(seeds)


def _try_correlated_fit(db, tag, t, cov_t, p0, make_chi2, config, label, slice_data=True):
    """Correlated mean fit on already-sliced ``cov_t``; ``make_chi2`` maps the
    inverted covariance to a chi^2 function. Returns ``(best_parameter, misc)``,
    or ``(None, None)`` if ``cov_t`` is not positive definite or the fit does
    not converge. Post-processing (misc fields, sorting, persisting) is the
    caller's job."""
    message(f"Check positive definiteness of {label} covariance matrix for fit range [[{t[0]},{t[-1]}]].")
    if not np.all(np.linalg.eigvals(cov_t) > 0):
        message(f"--> {label} covariance matrix not positive definite.")
        return None, None
    message(f"--> {label} covariance matrix positive definite. Try correlated fit.")
    try:
        chi2 = make_chi2(np.linalg.inv(cov_t))
        return fit_mean(db, t, tag, p0, chi2, config, slice_data=slice_data)
    except ConvergenceError as ce:
        message(f"{ce} for correlated mean fit with {label} covariance matrix")
        return None, None


def _fit_one_excited_range(db, binned_corr_tag, t, m0, mass_gaps, fit_model, Nt, var, cov_binned, cov_unbinned, model_func, bc, folded, config, silent):
    """One candidate range of the excited-fit loop (mean fits only). Returns
    ``None`` if all seeds fail to converge, otherwise ``(t_plateau, excited_spec,
    binned_corr_spec, unbinned_corr_spec, best_parameter, seed)`` where ``seed``
    is the unsorted mean-fit parameter for the deferred jackknife fits."""
    message(f"Excited fit range: [[{t[0]},{t[-1]}]]", silent)
    message(_log_divider("uncorrelated fit"), silent)
    def make_chi2(W):
        return _make_chi2(fit_model, W, Nt)
    chi2_func = make_chi2(np.diag(1.0 / var[t]))
    y = db.database[binned_corr_tag].central_value[t]
    p0s = get_p0_guesses(t, y, var[t], fit_model, m0, mass_gaps, Nt=Nt)
    attempts = []
    best = None
    for p0 in p0s:
        attempt = {"p0": p0.copy()}
        message(f"p0 for fit: {p0}", silent)
        try:
            parameters, diagnostics = fit_mean(
                db, t, binned_corr_tag, p0, chi2_func, config,
            )
        except ConvergenceError as exc:
            attempt["error"] = str(exc)
            message(str(exc), silent)
        else:
            attempt.update(parameters=parameters, diagnostics=diagnostics)
            if best is None or diagnostics["chi2"] < best[1]["chi2"]:
                best = parameters, diagnostics
        attempts.append(attempt)

    if best is None:
        message("All initial guesses failed for this window", silent)
        return None

    seed, diagnostics = best
    misc = {
        **diagnostics,
        "fit_model": fit_model,
        "initial_guess_attempts": attempts,
    }
    best_parameter = _sort_two_state_params(seed)
    print_fit_results(best_parameter, None, misc, silent)

    # correlated mean fits (cross-checks only; the uncorrelated result above
    # drives the plateau selection)
    message(_log_divider("correlated mean fit"), silent)
    message("Try correlated fit with binned covariance matrix")
    binned_best, binned_misc = _try_correlated_fit(db, binned_corr_tag, t, cov_binned[t][:, t], seed, make_chi2, config, "binned")
    if binned_best is not None:
        binned_misc["fit_model"] = fit_model
        binned_best = _sort_two_state_params(binned_best)
        print_fit_results(binned_best, None, binned_misc, silent)

    message("Try correlated fit with unbinned covariance matrix.")
    unbinned_best, unbinned_misc = _try_correlated_fit(db, binned_corr_tag, t, cov_unbinned[t][:, t], seed, make_chi2, config, "unbinned")
    if unbinned_best is not None:
        unbinned_misc["fit_model"] = fit_model
        unbinned_best = _sort_two_state_params(unbinned_best)
        print_fit_results(unbinned_best, None, unbinned_misc, silent)
    message(_log_divider(), silent)

    t_plateau = _select_plateau_range(t, var[t], best_parameter, model_func, bc, folded)
    misc["plateau_fit_range"] = t_plateau

    # entry specs are pending: the caller commits them (add_entry) only for
    # the range that wins the plateau selection
    excited_spec = _EntrySpec(
        tag=f"{binned_corr_tag}/excited_contributions_fit",
        central_value=best_parameter,
        cfgs=db.database[binned_corr_tag].cfgs, misc=misc,
    )
    binned_corr_spec = _EntrySpec(tag=f"{binned_corr_tag}/binned_correlated_excited_contributions_mean_fit")
    if binned_best is not None:
        binned_corr_spec.central_value = binned_best
        binned_corr_spec.misc = binned_misc
    unbinned_corr_spec = _EntrySpec(tag=f"{binned_corr_tag}/unbinned_correlated_excited_contributions_mean_fit")
    if unbinned_best is not None:
        unbinned_corr_spec.central_value = unbinned_best
        unbinned_corr_spec.misc = unbinned_misc
    return t_plateau, excited_spec, binned_corr_spec, unbinned_corr_spec, best_parameter, seed


# ---------------------------------------------------------------------------
# Excited-state / ground-state fits
# ---------------------------------------------------------------------------

class PlateauTooShortError(ValueError):
    """No candidate fit range reached ``min_plateau_len`` slices."""


def excited_contributions_fit(db, tag, binsize, excited_fit_ranges, fit_model, config: FitConfig, silent=False, Nt=None, min_plateau_len=5, folded=False, *, m0=None, mass_gaps=None):
    """Two-state fits across candidate ranges; pick the one whose plateau (where
    excited contributions drop below sigma/4) is shortest but at least
    ``min_plateau_len`` long. Returns ``(plateau_fit_range, last_best_parameter)``;
    raises ``PlateauTooShortError`` if no candidate qualifies.

    Mass seeds are shared across windows, in lattice units. By default, ``m0``
    is the NaN-ignoring mean effective mass over ``[n//4:n//4+n//8]`` of the
    stored correlator (length ``n``), and ``mass_gaps`` is ``m0 * [0.25, 0.5, 1]``.
    Both can be overridden. Each window generates its own amplitude guesses
    and selects its lowest-chi-square converged central fit independently.
    """
    message(f"Correlator: {tag}")
    message(f"Binsize = {binsize}", silent)
    message(f"{fit_model} model = {fit_model_dict[fit_model]}")
    message(_log_divider(), silent)
    binned_corr_tag = _ensure_binned(db, tag, binsize)
    if m0 is None:
        y = db.database[binned_corr_tag].central_value
        n = len(y)
        meff = meff_exp_forward if fit_model == "double-exp" else meff_cosh
        m0 = np.nanmean(meff(y)[n // 4:n // 4 + n // 8])
    if mass_gaps is None:
        mass_gaps = m0 * np.array([0.25, 0.5, 1.0])
    message(f"Initial mass seed = {m0}, mass gaps = {mass_gaps}", silent)
    cov = db.jackknife_covariance(binned_corr_tag)
    cov_unbinned = cov if binned_corr_tag == tag else db.jackknife_covariance(tag)
    var = np.diag(cov)
    Nt = len(db.database[binned_corr_tag].central_value) if Nt is None else Nt
    model_func = {"double-cosh": double_cosh_model(Nt),
                  "double-sinh": double_sinh_model(Nt),
                  "double-exp": double_exp_model()}[fit_model]
    bc = "open" if fit_model == "double-exp" else "periodic"
    # Pass 1 — mean fits only. Every candidate range yields a suggested
    # plateau; a candidate qualifies if its plateau has at least
    # min_plateau_len slices and is no longer than the first (widest) fit
    # range. Pass 2 tries qualifiers by shortest plateau, later ranges winning
    # ties. Initial guesses do not depend on results from other windows.
    candidates = []
    suggested_fit_ranges = []
    last_best_parameter = None
    for idx, t in enumerate(excited_fit_ranges):
        result = _fit_one_excited_range(
            db, binned_corr_tag, t, m0, mass_gaps,
            fit_model, Nt, var, cov, cov_unbinned, model_func, bc, folded, config, silent,
        )
        if result is None:
            suggested_fit_ranges.append(None)
            continue
        t_plateau, excited_cand, binned_cand, unbinned_cand, best_parameter, seed = result
        last_best_parameter = best_parameter
        suggested_fit_ranges.append(t_plateau)
        if len(t_plateau) < min_plateau_len:
            message(f"Determined fit range {t_plateau} has fewer than {min_plateau_len} elements", silent)
            message("---> Stored fit range is not updated", silent)
            message(_log_divider(), silent)
            message(_log_divider(), silent)
            continue
        message(f"Determined fit range [[{t_plateau[0]},{t_plateau[-1]}]]", silent)
        if len(t_plateau) <= len(excited_fit_ranges[0]):
            candidates.append(_Candidate(idx, t, t_plateau, seed, excited_cand, binned_cand, unbinned_cand))
        message(_log_divider(), silent)
        message(_log_divider(), silent)

    # Pass 2 — deferred jackknife fits: shortest plateau first, later range
    # wins ties. If a candidate's resample fits fail, the next-best range takes over.
    # Only the winner's entries are committed to the db.
    candidates.sort(key=lambda c: (len(c.t_plateau), -c.idx))
    winner = None
    for cand in candidates:
        message(_log_divider(f"jackknife fits for fit range [[{cand.t[0]},{cand.t[-1]}]]"), silent)
        chi2_func = _make_chi2(fit_model, np.diag(1.0 / var[cand.t]), Nt)
        try:
            jks = _fit_resamples(db.transform_jks, "jackknife", cand.t, binned_corr_tag, cand.seed, chi2_func, config, slice_data=True)
        except ConvergenceError as ce:
            message(f"{ce} -> fall back to next-best fit range")
            suggested_fit_ranges[cand.idx] = None
            continue
        cand.excited.jks = np.array([_sort_two_state_params(jk) for jk in jks])
        print_fit_results(cand.excited.central_value, jackknife.covariance(cand.excited.jks), cand.excited.misc, silent)
        winner = cand
        break
    if winner is None:
        best = max((len(s) for s in suggested_fit_ranges if s is not None), default=0)
        raise PlateauTooShortError(
            f"excited_contributions_fit({tag!r}): no candidate range reached "
            f"min_plateau_len={min_plateau_len} (longest plateau found: {best}); "
            f"lower min_plateau_len or loosen the S/N cut"
        )
    db.add_entry(**winner.binned.__dict__)
    db.add_entry(**winner.unbinned.__dict__)
    winner.excited.misc["tested_suggested_fit_ranges"] = (excited_fit_ranges, suggested_fit_ranges)
    db.add_entry(**winner.excited.__dict__)
    return winner.excited.misc["plateau_fit_range"], last_best_parameter


def ground_state_fit(db, tag, binsize, fit_range, p0, fit_model, config: FitConfig, Nt=None, silent=False, bootstraps=None):
    """Ground-state fit of ``tag`` at every binsize b = 1..``binsize``: jackknife
    fit (the primary result), correlated mean fits as cross-checks at the
    endpoint binsizes, and a bootstrap fit at b = 1 seeded from the jackknife
    result. Returns a list of ``FitTags`` in binsize order (1..``binsize``).
    Bootstrap tags are present only at binsize 1 when enabled; correlated
    mean cross-checks are not included."""
    if config.bootstrap_available and bootstraps is None:
        raise ValueError("ground_state_fit needs bootstraps= when config.bootstrap_available")
    message(f"Correlator: {tag}")
    message(f"P0 = {p0}")
    message(f"Fit range {fit_range}")
    message(f"{fit_model} model = {fit_model_dict[fit_model]}")
    Nt = len(db.database[tag].central_value) if Nt is None else Nt
    def make_chi2(W):
        return _make_chi2(fit_model, W, Nt)
    fit_tags = []
    for b in range(1, binsize + 1):
        bootstrap_fit_tag = None
        message(f"Binsize = {b}", silent)
        binned_corr_tag = _ensure_binned(db, tag, b)

        # 1. uncorrelated jackknife fit (primary result)
        message(_log_divider("jackknife fit"), silent)
        var = db.jackknife_variance(binned_corr_tag)
        chi2_func = make_chi2(np.diag(1.0 / var[fit_range]))
        best_parameter, best_parameter_jks, misc = fit_jks(db, fit_range, binned_corr_tag, p0, chi2_func, config)
        misc["fit_model"] = fit_model
        best_parameter_cov = jackknife.covariance(best_parameter_jks)
        print_fit_results(best_parameter, best_parameter_cov, misc, silent)

        # 2. correlated mean fits (cross-checks) at the endpoint binsizes;
        #    the unbinned covariance adds nothing at b == 1
        if b in [1, binsize]:
            message(_log_divider("correlated mean fit"), silent)
            correlated_fits = [("binned", binned_corr_tag, "Try fit with covariance matrix")]
            if b != 1:
                correlated_fits.append(("unbinned", tag, "Try fit with unbinned covariance matrix"))
            for label, cov_tag, note in correlated_fits:
                message(note)
                cov_t = db.jackknife_covariance(cov_tag)[fit_range][:, fit_range]
                best, misc_corr = _try_correlated_fit(db, binned_corr_tag, fit_range, cov_t, p0, make_chi2, config, label)
                if best is not None:
                    misc_corr["fit_model"] = fit_model
                    print_fit_results(best, None, misc_corr, silent)
                    db.add_entry(f"{binned_corr_tag}/{fit_model}_{label}_correlated_mean_fit", central_value=best, misc=misc_corr)
            message(_log_divider(), silent)

        # 3. bootstrap fit, seeded from the jackknife result; b == 1 only
        #    because the bootstrap indices refer to unbinned configurations
        if b == 1 and config.bootstrap_available:
            message(_log_divider("bootstrap fit"), silent)
            bss = db.bss(binned_corr_tag, bootstraps)
            W_bss = np.diag(1.0 / bootstrap.variance(bss)[fit_range])
            chi2_func_bss = _make_chi2(fit_model, W_bss, Nt)
            best_parameter_bcentral, best_parameter_bss, misc_bss = fit_bss(db, fit_range, binned_corr_tag, best_parameter, chi2_func_bss, config, bootstraps=bootstraps)
            best_parameter_bcov = bootstrap.covariance(best_parameter_bss)
            print_fit_results(best_parameter_bcentral, best_parameter_bcov, misc_bss)
            misc_bss["fit_model"] = fit_model
            bootstrap_fit_tag = f"{binned_corr_tag}/{fit_model}_bootstrap_fit"
            db.add_entry(bootstrap_fit_tag, central_value=best_parameter_bcentral, bss=best_parameter_bss, misc=misc_bss)

        # persist the jackknife fit as this binsize's primary result
        fit_tag = f"{binned_corr_tag}/{fit_model}_fit"
        db.add_entry(
            fit_tag,
            central_value=best_parameter, jks=best_parameter_jks,
            cfgs=db.database[binned_corr_tag].cfgs, misc=misc,
        )
        fit_tags.append(FitTags(jackknife=fit_tag, bootstrap=bootstrap_fit_tag))
        message(_log_divider(), silent)
        message(_log_divider(), silent)
    return fit_tags


# ---------------------------------------------------------------------------
# Two-correlator combined fit (shared mass)
# ---------------------------------------------------------------------------

def correlator_combined_fit(db, tags, combined_tag, fit_ranges, binsize, p0, fit_models,
                            config: FitConfig, Nt=None, silent=False, bootstraps=None):
    """Joint fit of two concatenated correlator blocks with a shared mass.

    Blocks 0 and 1 have independent amplitudes ``p[0]`` and ``p[1]`` and a
    shared ground-state mass ``p[2]``. ``p0 = [A0, A1, m]``. Each block uses
    a ``cosh``, ``sinh``, or ``exp`` model; no particular smearing is required.
    Both fit ranges must lie where excited-state contributions are
    sufficiently suppressed for these single-state models to apply; this
    function does not check that assumption. Returns a list of ``FitTags`` in
    binsize order (1..``binsize``). Bootstrap tags are present only at binsize
    1 when enabled; correlated mean cross-checks are not included.
    """
    if len(tags) != 2 or len(fit_ranges) != 2:
        raise ValueError("tags and fit_ranges must each contain two entries")
    if len(p0) != 3:
        raise ValueError(f"p0 must contain [A0, A1, m], got {len(p0)} entries")
    fit_model_combined = _combined_model_name(fit_models)
    if config.bootstrap_available and bootstraps is None:
        raise ValueError("correlator_combined_fit needs bootstraps= when config.bootstrap_available")
    entries = [db.database[tag] for tag in tags]
    if not np.array_equal(entries[0].cfgs, entries[1].cfgs):
        raise ValueError(f"correlator_combined_fit: cfgs of {tags[0]!r} and {tags[1]!r} differ; cannot pair configs")
    if not np.array_equal(entries[0].weights, entries[1].weights):
        raise ValueError(f"correlator_combined_fit: weights of {tags[0]!r} and {tags[1]!r} differ; cannot pair configs")
    if len(entries[0].central_value) != len(entries[1].central_value):
        raise ValueError(f"correlator_combined_fit: data lengths of {tags[0]!r} and {tags[1]!r} differ")

    message(_log_divider("combined correlator fit"), silent)
    for i, (tag, fit_range, fit_model) in enumerate(zip(tags, fit_ranges, fit_models)):
        message(f"block {i}: {tag}, fit range {fit_range}, {fit_model} model = {fit_model_dict[fit_model]}", silent)
    message("shared mass m = p[2]", silent)
    message(f"P0 = {p0}")

    Nt = len(entries[0].central_value) if Nt is None else Nt
    fit_range_combined = np.hstack(fit_ranges)
    def make_chi2(W):
        return _make_combined_chi2(fit_models, W, tuple(map(len, fit_ranges)), Nt)
    combined_misc = {
        "fit_model": fit_model_combined,
        "fit_models": fit_models,
        "t_blocks": fit_ranges,
        "tags": tags,
    }

    # combined data entry: per config, blocks pre-sliced to their fit ranges
    # and concatenated — all fits below use slice_data=False
    samples = [entry.sample for entry in entries]
    combined_sample = np.array([
        np.hstack([sample[fit_range] for sample, fit_range in zip(config_samples, fit_ranges)])
        for config_samples in zip(*samples)
    ])
    db.add_entry(combined_tag, sample=combined_sample, weights=entries[0].weights, cfgs=entries[0].cfgs)

    fit_tags = []
    for b in range(1, binsize + 1):
        bootstrap_fit_tag = None
        message(f"Binsize = {b}", silent)
        binned_corr_tag = _ensure_binned(db, combined_tag, b)

        # 1. uncorrelated jackknife fit (primary result)
        message(_log_divider("jackknife fit"), silent)
        chi2_func = make_chi2(np.diag(1.0 / db.jackknife_variance(binned_corr_tag)))
        best_parameter, best_parameter_jks, misc = fit_jks(db, fit_range_combined, binned_corr_tag, p0, chi2_func, config, slice_data=False)
        misc.update(combined_misc)
        print_fit_results(best_parameter, jackknife.covariance(best_parameter_jks), misc, silent)

        # 2. correlated mean fit (cross-check) at the endpoint binsizes,
        #    seeded from the jackknife result
        if b in [1, binsize]:
            message(_log_divider("correlated mean fit"), silent)
            cov = db.jackknife_covariance(binned_corr_tag)
            best, misc_corr = _try_correlated_fit(db, binned_corr_tag, fit_range_combined, cov, best_parameter, make_chi2, config, "binned", slice_data=False)
            if best is not None:
                misc_corr.update(combined_misc)
                print_fit_results(best, None, misc_corr, silent)
                db.add_entry(f"{binned_corr_tag}/{fit_model_combined}_correlated_mean_fit", central_value=best, misc=misc_corr)
            message(_log_divider(), silent)

        # 3. bootstrap fit, seeded from the jackknife result; b == 1 only
        #    because the bootstrap indices refer to unbinned configurations
        if b == 1 and config.bootstrap_available:
            message(_log_divider("bootstrap fit"), silent)
            W_bss = np.diag(1.0 / bootstrap.variance(db.bss(binned_corr_tag, bootstraps)))
            best_parameter_bcentral, best_parameter_bss, misc_bss = fit_bss(db, fit_range_combined, binned_corr_tag, best_parameter, make_chi2(W_bss), config, slice_data=False, bootstraps=bootstraps)
            misc_bss.update(combined_misc)
            print_fit_results(best_parameter_bcentral, bootstrap.covariance(best_parameter_bss), misc_bss)
            bootstrap_fit_tag = f"{binned_corr_tag}/{fit_model_combined}_bootstrap_fit"
            db.add_entry(bootstrap_fit_tag, central_value=best_parameter_bcentral, bss=best_parameter_bss, misc=misc_bss)

        # persist the jackknife fit as this binsize's primary result
        fit_tag = f"{binned_corr_tag}/{fit_model_combined}_fit"
        db.add_entry(
            fit_tag,
            central_value=best_parameter, jks=best_parameter_jks,
            cfgs=db.database[binned_corr_tag].cfgs, misc=misc,
        )
        fit_tags.append(FitTags(jackknife=fit_tag, bootstrap=bootstrap_fit_tag))

        message(_log_divider(), silent)
        message(_log_divider(), silent)
    return fit_tags
