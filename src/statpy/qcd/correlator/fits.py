"""FitConfig dataclass and high-level correlator fit routines.

All functions take a database handle as first argument and a FitConfig
describing fit method + parameters. (Source averaging / folding lives in
``averaging.py``.)
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from statpy.fitting.core import ConvergenceError, Fitter, get_pvalue, print_fit_results
from statpy.log import format_paren, message
from statpy.qcd.correlator.models import (
    FIT_MODEL_FORMULAS,
    CoshModel,
    DoubleCoshModel,
    DoubleExpModel,
    DoubleSinhModel,
    ExpModel,
    SinhModel,
    combined_corr_chi2,
    cosh_chi2,
    double_cosh_chi2,
    double_exp_chi2,
    double_sinh_chi2,
    exp_chi2,
    sinh_chi2,
)
from statpy.qcd.correlator.primitives import (
    binned_tag,
    effective_mass,
)
from statpy.statistics import bootstrap, jackknife

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FitConfig:
    """Optimizer choice and parameters."""
    fit_method: str = "Nelder-Mead"
    fit_params: dict = field(default_factory=lambda: {"maxiter": 5000, "tol": 1e-07})


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


def _make_slicer(fit_range, slice_data):
    """Return ``y -> y[fit_range]`` if ``slice_data`` else ``y -> y`` (data already sliced)."""
    return (lambda y: y[fit_range]) if slice_data else (lambda y: y)


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
_BACKWARD_SIGN = {"cosh": 1.0, "sinh": -1.0, "exp": 0.0}


def _combined_model_name(fit_models):
    """Return the combined model name for two validated block models."""
    if len(fit_models) != 2:
        raise ValueError(f"fit_models must contain two models, got {len(fit_models)}")
    unknown = [model for model in fit_models if model not in _BACKWARD_SIGN]
    if unknown:
        raise ValueError(f"Unknown combined block model(s): {unknown}")
    return "combined-" + "-".join(fit_models)


def _make_combined_chi2(fit_models, W, block_lengths, Nt):
    """Return a chi^2 lambda for two concatenated correlator blocks."""
    _combined_model_name(fit_models)
    backward_sign = np.repeat([_BACKWARD_SIGN[model] for model in fit_models], block_lengths)
    amplitude_index = np.repeat(np.arange(2), block_lengths)
    return lambda t, p, y: combined_corr_chi2(t, p, y, W, Nt, amplitude_index, backward_sign)


# ---------------------------------------------------------------------------
# Core fit primitives
# ---------------------------------------------------------------------------

def fit_mean(db, fit_range, tag, p0, chi2_func, config: FitConfig, slice_data=True):
    """Fit the mean of the entry at ``tag``; returns ``(best_parameter, misc)``
    with ``misc = {"fit_range", "chi2", "dof", "pval"}``.

    ``slice_data=True``: ``fit_range`` indexes the data, the fit sees ``y[fit_range]``.
    ``slice_data=False``: the entry is already pre-sliced to length ``len(fit_range)``
    and ``fit_range`` is only passed through to ``chi2_func`` (combined fits, where
    ``fit_range`` is the concatenation of two fit ranges).
    """
    p0 = np.asarray(p0, dtype=float)
    if p0.ndim != 1 or p0.size == 0:
        raise ValueError(f"'p0' must be a non-empty 1-D array, got shape {p0.shape}")
    if not slice_data and len(fit_range) != len(db.database[tag].central_value):
        raise ValueError(
            f"with slice_data=False, len(fit_range)={len(fit_range)} must equal data length "
            f"{len(db.database[tag].central_value)} for tag {tag!r}"
        )
    dof = len(fit_range) - p0.size
    if dof <= 0:
        raise ValueError(
            f"non-positive degrees of freedom: len(fit_range)={len(fit_range)}, n_params={p0.size}, dof={dof}"
        )

    sl = _make_slicer(fit_range, slice_data)
    fitter = Fitter(config.fit_method, config.fit_params)
    try:
        best = fitter.estimate_parameters(fit_range, chi2_func, sl(db.database[tag].central_value), p0)[0]
    except ConvergenceError as e:
        raise ConvergenceError(f"mean fit for tag {tag!r} did not converge: {e}") from e
    if not np.isfinite(best).all():
        raise ConvergenceError(f"mean fit for tag {tag!r} produced non-finite parameters: {best}")
    chi2 = chi2_func(fit_range, best, sl(db.database[tag].central_value))
    if not np.isfinite(chi2):
        raise ConvergenceError(f"non-finite chi^2 = {chi2} for tag {tag!r}")
    return best, {"fit_range": fit_range, "chi2": chi2, "dof": dof, "pval": get_pvalue(chi2, dof)}


def _fit_resamples(transform, label, fit_range, tag, p0, chi2_func, config, slice_data, **transform_kwargs):
    """Fit each resample via ``transform`` (:meth:`DB.transform_jks` or
    :meth:`DB.transform_bss`); ``label`` is used only in the error message."""
    sl = _make_slicer(fit_range, slice_data)
    fitter = Fitter(config.fit_method, config.fit_params)
    try:
        return transform(tag, f=lambda y: fitter.estimate_parameters(fit_range, chi2_func, sl(y), p0)[0], **transform_kwargs)
    except ConvergenceError as e:
        raise ConvergenceError(f"{label} fit for tag {tag!r} did not converge: {e}") from e


def fit_jks(db, fit_range, tag, p0, chi2_func, config: FitConfig, slice_data=True):
    """Fit the mean, then every jackknife sample starting from the mean fit.
    Returns ``(best_parameter, best_parameter_jks, misc)``."""
    best, misc = fit_mean(db, fit_range, tag, p0, chi2_func, config, slice_data=slice_data)
    best_jks = _fit_resamples(db.transform_jks, "jackknife", fit_range, tag, best, chi2_func, config, slice_data)
    return best, best_jks, misc


def fit_bss(db, fit_range, tag, p0, chi2_func, config: FitConfig, slice_data=True, bootstraps=None):
    """Fit the mean, then every bootstrap sample starting from the mean fit.
    Returns ``(best_parameter, best_parameter_bss, misc)``. ``bootstraps`` is
    ignored if the entry already carries ``entry.bss``."""
    best, misc = fit_mean(db, fit_range, tag, p0, chi2_func, config, slice_data=slice_data)
    best_bss = _fit_resamples(db.transform_bss, "bootstrap", fit_range, tag, best, chi2_func, config, slice_data, bootstraps=bootstraps)
    return best, best_bss, misc


# ---------------------------------------------------------------------------
# Two-state fit results and helpers
# ---------------------------------------------------------------------------

@dataclass
class ExcitedFitResult:
    """One window's fit results, jackknives and optional failure reason."""
    fit_range: np.ndarray
    central_value: np.ndarray | None = None
    jks: np.ndarray | None = None
    misc: dict = field(default_factory=dict)
    binned_correlated: tuple = (None, None)
    unbinned_correlated: tuple = (None, None)
    failure_reason: str | None = None


def _sort_two_state_params(p):
    """Order the two states of a double-* fit so the lower mass comes first."""
    if p[3] < p[1]:
        return [p[2], p[3], p[0], p[1]]
    return p


def _select_ground_state_range(fit_range, var_fit_range, best_parameter, model_func, boundary_condition, folded):
    """Propose ground-state slices using the sigma/4 excited-contribution cut."""
    excited = np.abs([model_func(i, [0, 0, best_parameter[2], best_parameter[3]]) for i in fit_range])
    std_over_four = (var_fit_range ** 0.5) / 4.0
    if boundary_condition == "periodic" and not folded:
        std_over_four = (std_over_four + std_over_four[::-1]) / 2
    return fit_range[excited < std_over_four]


def get_p0_guesses(t, y, var_fit_range, fit_model, m0, mass_gaps, *, Nt=None):
    """Return one ``[A0, m0, A1, m0 + gap]`` guess per mass gap, shape ``(len(mass_gaps), 4)``.

    ``t``, ``y`` and ``var_fit_range`` hold the fit-window data. For each mass pair
    the amplitudes are solved by weighted least squares. Periodic models need the
    full ``Nt``. Raises ``ValueError`` for invalid inputs or unresolvable amplitudes.
    """
    t, y, var_fit_range, gaps = [np.asarray(x, dtype=float) for x in (t, y, var_fit_range, mass_gaps)]
    if t.ndim != 1 or t.size < 2 or y.shape != t.shape or var_fit_range.shape != t.shape:
        raise ValueError("t, y and var_fit_range must be matching 1D arrays with at least two points")
    if not all(np.isfinite(x).all() for x in (t, y, var_fit_range)) or np.any(var_fit_range <= 0):
        raise ValueError("Fit data must be finite and variances strictly positive")
    if not np.isfinite(m0) or m0 <= 0:
        raise ValueError("m0 must be finite and positive")
    if gaps.ndim != 1 or gaps.size == 0 or not np.isfinite(gaps).all() or np.any(gaps <= 0):
        raise ValueError("mass_gaps must be a nonempty 1D array of finite positive gaps")
    if fit_model == "double-exp":
        model = ExpModel()
    elif fit_model in ("double-cosh", "double-sinh"):
        if Nt is None or not np.isfinite(Nt) or Nt <= 0:
            raise ValueError("Periodic models require a finite positive Nt")
        model = CoshModel(Nt) if fit_model == "double-cosh" else SinhModel(Nt)
    else:
        raise ValueError(f"Unknown fit_model: {fit_model!r}")

    sigma = np.sqrt(var_fit_range)
    guesses = []
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
        guesses.append([amplitudes[0], m0, amplitudes[1], m1])
    return np.asarray(guesses)


def _inverse_covariance(cov, n_samples):
    """Return ``(inverse, None)``, or ``(None, reason)`` if ``cov`` is not invertible."""
    if len(cov) >= n_samples:
        return None, f"singular: {len(cov)} points from {n_samples} samples"
    sigma = np.sqrt(np.diag(cov))
    if not np.all(np.isfinite(sigma) & (sigma > 0)):
        return None, "not positive definite: non-positive or non-finite variance"
    scale = np.outer(sigma, sigma)
    try:
        factor = cho_factor(cov / scale, lower=True)
    except np.linalg.LinAlgError:
        return None, "not positive definite"
    return cho_solve(factor, np.eye(len(cov))) / scale, None


def _try_correlated_fit(db, tag, fit_range, cov_fit_range, n_samples, p0, make_chi2, config, label, slice_data=True, silent=False):
    """Correlated mean fit on already-sliced ``cov_fit_range``; ``make_chi2`` maps the
    inverted covariance to a chi^2 function. Returns ``(best_parameter, misc)``,
    or ``(None, {"failure_reason": ...})`` if ``cov_fit_range`` is not positive definite
    or the fit does not converge. Post-processing (misc fields, sorting, persisting) is the
    caller's job."""
    message(f"Check positive definiteness of {label} covariance matrix for fit range [[{fit_range[0]},{fit_range[-1]}]].", silent)
    inverse, reason = _inverse_covariance(cov_fit_range, n_samples)
    if inverse is None:
        message(f"--> {label} covariance matrix {reason}.", silent)
        return None, {"failure_reason": f"covariance matrix {reason}"}
    message(f"--> {label} covariance matrix positive definite. Try correlated fit.", silent)
    try:
        chi2 = make_chi2(inverse)
        return fit_mean(db, fit_range, tag, p0, chi2, config, slice_data=slice_data)
    except ConvergenceError as ce:
        message(f"{ce} for correlated mean fit with {label} covariance matrix", silent)
        return None, {"failure_reason": "fit did not converge"}


_PARAMETER_NAMES = ("A0", "m0", "A1", "m1")


def _labelled(label, rows):
    """Prefix the first row with ``label`` and align the others under it."""
    return [f"  {label if i == 0 else '':<20} {row}" for i, row in enumerate(rows)]


def _format_parameters(parameters, errors=None, names=_PARAMETER_NAMES):
    if errors is None:
        return "   ".join(f"{n} {v:.6g}" for n, v in zip(names, parameters))
    return "   ".join(f"{n} {format_paren(v, e)}" for n, v, e in zip(names, parameters, errors))


def _parameter_names(fit_model):
    """Return the parameter names of fit_model, e.g. ("A", "m") for "cosh"."""
    parameters = FIT_MODEL_FORMULAS[fit_model].split("; ")[1]
    return tuple(p.split(" = ")[0] for p in parameters.split(", "))


def _fit_table_row(binsize, fit, cells):
    """Format one row of the per-binsize fit table."""
    return f"  {binsize:>7}  {fit:<21}" + "".join(f"{cell:<18}" for cell in cells[:-2]) + f"{cells[-2]:<10}{cells[-1]}"


def _fit_table_rows(binsize, fit, parameters, misc, errors=None):
    """Return the table row of one fit, or a row with the failure reason."""
    if parameters is None:
        return [f"  {binsize:>7}  {fit:<21}{misc['failure_reason']}"]
    if errors is None:
        cells = [f"{v:.6g}" for v in parameters]
    else:
        cells = [format_paren(v, e) for v, e in zip(parameters, errors)]
    return [_fit_table_row(binsize, fit, cells + [f"{misc['chi2'] / misc['dof']:.3g}", f"{misc['pval']:.2f}"])]


def log_fit_header(title, tag, fit_model, fit_range, p0, silent=False):
    """Log the correlator, model, fit range, start values and the fit-table columns."""
    formula, parameters = FIT_MODEL_FORMULAS[fit_model].split("; ")
    names = _parameter_names(fit_model)
    dof = len(fit_range) - len(names)
    message("\n".join([
        title,
        *_labelled("Correlator", [tag]),
        *_labelled("Model", [f"{fit_model}: {formula}", parameters]),
        *_labelled("Fit range", [f"{_format_range(fit_range)} ({len(fit_range)} points, {dof} dof)"]),
        *_labelled("Start", [_format_parameters(p0, names=names)]),
        "",
        _fit_table_row("Binsize", "Fit", [*names, "chi2/dof", "p"]),
    ]), silent)


def _format_range(t):
    return f"[{t[0]}, {t[-1]}]" if len(t) else "none"


def _format_chi2(misc):
    return f"chi2/dof {misc['chi2']:.3g}/{misc['dof']} = {misc['chi2'] / misc['dof']:#.3g}, p = {misc['pval']:.2f}"


def _fit_one_excited_range(db, *, binned_corr_tag, fit_range, m0, mass_gaps, fit_model, Nt, var, cov_binned, cov_unbinned, n_binned, n_unbinned, model_func, boundary_condition, folded, config, silent):
    """Fit one window's mean and jackknives, retaining convergence failures; log one block."""
    def make_chi2(W):
        return _make_chi2(fit_model, W, Nt)
    chi2_func = make_chi2(np.diag(1.0 / var[fit_range]))
    y = db.database[binned_corr_tag].central_value[fit_range]
    p0s = get_p0_guesses(fit_range, y, var[fit_range], fit_model, m0, mass_gaps, Nt=Nt)
    attempts = []
    result = ExcitedFitResult(fit_range=fit_range, misc={
        "fit_range": fit_range, "fit_model": fit_model, "initial_guess_attempts": attempts,
    })
    best = None
    for i, p0 in enumerate(p0s, 1):
        attempt = {"p0": p0.copy()}
        try:
            parameters, diagnostics = fit_mean(
                db, fit_range, binned_corr_tag, p0, chi2_func, config,
            )
        except ConvergenceError as exc:
            attempt["failure_reason"] = str(exc)
        else:
            attempt.update(parameters=parameters, diagnostics=diagnostics)
            if best is None or diagnostics["chi2"] < best[2]["chi2"]:
                best = i, parameters, diagnostics
        attempts.append(attempt)

    lines = [f"Window [{fit_range[0]}, {fit_range[-1]}] ({len(fit_range)} points)"]
    rows = [f"{'#':<10}" + "".join(f"{n:<13}" for n in _PARAMETER_NAMES) + "chi2/dof"]
    for i, attempt in enumerate(attempts, 1):
        rows.append((f"{i:<3}{'start':<7}" + "".join(f"{v:<13.6g}" for v in attempt["p0"])).rstrip())
        if "failure_reason" in attempt:
            rows.append(f"{'':<3}{'fit':<7}failed: {attempt['failure_reason']}")
            continue
        diagnostics = attempt["diagnostics"]
        status = f"{diagnostics['chi2'] / diagnostics['dof']:#.6g}"
        if best is not None and i == best[0]:
            status += "  used"
        fitted = _sort_two_state_params(attempt["parameters"])
        rows.append(f"{'':<3}{'fit':<7}" + "".join(f"{v:<13.6g}" for v in fitted) + status)
    lines += _labelled("Guesses", rows)

    if best is None:
        result.failure_reason = "All initial guesses failed for this window"
        lines += _labelled("Result", ["failed: all initial guesses failed"])
        message("\n".join(lines), silent)
        return result

    used_attempt, best_parameter, diagnostics = best
    result.misc.update(diagnostics)
    result.misc["used_attempt"] = used_attempt
    result.central_value = np.asarray(_sort_two_state_params(best_parameter))

    # correlated mean fits (cross-checks only; the uncorrelated result above
    # determines the proposed ground-state range)
    correlated_rows = []
    for label, cov, n_samples in (("unbinned", cov_unbinned, n_unbinned), ("binned", cov_binned, n_binned)):
        parameters, misc = _try_correlated_fit(db, binned_corr_tag, fit_range, cov[fit_range][:, fit_range], n_samples, best_parameter, make_chi2, config, label, silent=True)
        if parameters is None:
            correlated_rows.append(f"{label:<10}{misc['failure_reason']}")
        else:
            misc["fit_model"] = fit_model
            parameters = _sort_two_state_params(parameters)
            correlated_rows += [f"{label:<10}{_format_parameters(parameters)}", f"{'':<10}{_format_chi2(misc)}"]
        setattr(result, f"{label}_correlated", (parameters, misc))

    result.misc["ground_state_fit_range"] = _select_ground_state_range(
        fit_range, var[fit_range], result.central_value, model_func, boundary_condition, folded,
    )
    try:
        jks = _fit_resamples(db.transform_jks, "jackknife", fit_range, binned_corr_tag, best_parameter, chi2_func, config, slice_data=True)
        if not np.isfinite(jks).all():
            raise ConvergenceError("Jackknife fits produced non-finite parameters")
    except ConvergenceError as exc:
        result.failure_reason = str(exc)
        lines += _labelled("Result", [_format_parameters(result.central_value), _format_chi2(result.misc)])
        lines += _labelled("Jackknives", [f"failed: {exc}"])
    else:
        result.jks = np.array([_sort_two_state_params(jk) for jk in jks])
        errors = np.sqrt(np.diag(jackknife.covariance(result.jks)))
        lines += _labelled("Result", [_format_parameters(result.central_value, errors), _format_chi2(result.misc)])
    lines += _labelled("Correlated", correlated_rows)
    lines += _labelled("Ground-state range", [f"{_format_range(result.misc['ground_state_fit_range'])} (sigma/4)"])
    message("\n".join(lines), silent)
    return result


def _excited_fits_summary(results):
    """Return summary rows: window, used guess, masses, chi2/dof, ground range."""
    rows = [f"  {'Window':<12}{'Guess':<7}{'m0':<16}{'m1':<16}{'chi2/dof':<10}Ground-state range"]
    for result in results:
        window = _format_range(result.fit_range)
        if result.central_value is None:
            rows.append(f"  {window:<12}failed: {result.failure_reason}")
            continue
        misc = result.misc
        if result.jks is None:
            m0, m1 = (f"{result.central_value[i]:.6g}" for i in (1, 3))
            suffix = "  (jackknives failed)"
        else:
            errors = np.sqrt(np.diag(jackknife.covariance(result.jks)))
            m0, m1 = (format_paren(result.central_value[i], errors[i]) for i in (1, 3))
            suffix = ""
        rows.append(
            f"  {window:<12}{misc['used_attempt']:<7}{m0:<16}{m1:<16}"
            f"{misc['chi2'] / misc['dof']:<#10.3g}{_format_range(misc['ground_state_fit_range'])}{suffix}"
        )
    return rows


# ---------------------------------------------------------------------------
# Excited-state / ground-state fits
# ---------------------------------------------------------------------------

def excited_contribution_fits(db, tag, binsize, excited_fit_ranges, fit_model, config: FitConfig, silent=False, Nt=None, folded=False, *, m0=None, mass_gaps=None):
    """Fit each window's central value and jackknives.

    Return ExcitedFitResult objects in input order, retaining convergence failures.
    Ground-state range selection and result storage belong to the caller.
    """
    binned_corr_tag = _ensure_binned(db, tag, binsize)
    if m0 is None:
        y = db.database[binned_corr_tag].central_value
        n = len(y)
        estimator = "log" if fit_model == "double-exp" else "arccosh"
        m0 = np.nanmean(effective_mass(y, estimator=estimator)[n // 4:n // 4 + n // 8])
    if mass_gaps is None:
        mass_gaps = m0 * np.array([0.25, 0.5, 1.0])
    n_binned = len(db.database[binned_corr_tag].jks)
    formula, parameters = FIT_MODEL_FORMULAS[fit_model].split("; ")
    message("\n".join([
        "Excited-state fits",
        f"  {'Correlator':<20} {tag}",
        *_labelled("Model", [f"{fit_model}: {formula}", parameters]),
        f"  {'Binsize':<20} {binsize} ({n_binned} jackknives)",
        f"  {'Mass guess':<20} {m0:.6g}; gaps " + ", ".join(f"{gap:.6g}" for gap in mass_gaps),
    ]), silent)
    cov = db.jackknife_covariance(binned_corr_tag)
    cov_unbinned = cov if binned_corr_tag == tag else db.jackknife_covariance(tag)
    var = np.diag(cov)
    Nt = len(db.database[binned_corr_tag].central_value) if Nt is None else Nt
    model_func = {"double-cosh": DoubleCoshModel(Nt),
                  "double-sinh": DoubleSinhModel(Nt),
                  "double-exp": DoubleExpModel()}[fit_model]
    boundary_condition = "open" if fit_model == "double-exp" else "periodic"
    results = []
    for fit_range in excited_fit_ranges:
        results.append(_fit_one_excited_range(
            db,
            binned_corr_tag=binned_corr_tag,
            fit_range=np.asarray(fit_range),
            m0=m0,
            mass_gaps=mass_gaps,
            fit_model=fit_model,
            Nt=Nt,
            var=var,
            cov_binned=cov,
            cov_unbinned=cov_unbinned,
            n_binned=n_binned,
            n_unbinned=len(db.database[tag].jks),
            model_func=model_func,
            boundary_condition=boundary_condition,
            folded=folded,
            config=config,
            silent=silent,
        ))
    message("\n".join(["Summary"] + _excited_fits_summary(results)), silent)
    return results


def _jackknife_fit(db, binned_tag, fit_range, p0, make_chi2, config, misc_extra):
    """Diagonal jackknife fit (primary result); return (best_parameter, jks, misc)."""
    chi2_func = make_chi2(np.diag(1.0 / db.jackknife_variance(binned_tag)[fit_range]))
    best, jks, misc = fit_jks(db, fit_range, binned_tag, p0, chi2_func, config)
    misc.update(misc_extra)
    return best, jks, misc


def _correlated_mean_fit(db, label, cov_tag, binned_tag, fit_range, p0, make_chi2, config, fit_model, misc_extra):
    """Correlated mean fit with the covariance of cov_tag (cross-check); store it if it converges.

    Return (best_parameter, misc), or (None, {"failure_reason": ...}).
    """
    cov_fit_range = db.jackknife_covariance(cov_tag)[fit_range][:, fit_range]
    best, misc = _try_correlated_fit(db, binned_tag, fit_range, cov_fit_range, len(db.database[cov_tag].jks), p0, make_chi2, config, label, silent=True)
    if best is not None:
        misc.update(misc_extra)
        db.add_entry(f"{binned_tag}/{fit_model}_{label}_correlated_mean_fit", central_value=best, misc=misc)
    return best, misc


def _bootstrap_fit(db, binned_tag, fit_range, start, make_chi2, config, fit_model, misc_extra, bootstraps):
    """Bootstrap fit starting from start; store it and return (tag, best_parameter, bss, misc)."""
    chi2_func = make_chi2(np.diag(1.0 / bootstrap.variance(db.bss(binned_tag, bootstraps))[fit_range]))
    best, bss, misc = fit_bss(db, fit_range, binned_tag, start, chi2_func, config, bootstraps=bootstraps)
    misc.update(misc_extra)
    fit_tag = f"{binned_tag}/{fit_model}_bootstrap_fit"
    db.add_entry(fit_tag, central_value=best, bss=bss, misc=misc)
    return fit_tag, best, bss, misc


def _store_jackknife_fit(db, binned_tag, fit_model, best, jks, misc):
    """Store the jackknife fit as the binsize's primary result; return its tag."""
    fit_tag = f"{binned_tag}/{fit_model}_fit"
    db.add_entry(fit_tag, central_value=best, jks=jks, cfgs=db.database[binned_tag].cfgs, misc=misc)
    return fit_tag


def ground_state_fit(db, tag, binsize, fit_range, p0, fit_model, config: FitConfig, *, Nt=None, correlated_fits=(), bootstraps=None, silent=False) -> FitTags:
    """Fit one binsize with jackknives and the requested cross-checks and bootstraps; log its table rows."""
    for source in correlated_fits:
        if source not in ("binned", "unbinned"):
            raise ValueError(f"Unknown correlated covariance source: {source!r}")
    if bootstraps is not None and binsize != 1:
        raise ValueError("Bootstrap fits require binsize == 1")
    Nt = len(db.database[tag].central_value) if Nt is None else Nt
    def make_chi2(W):
        return _make_chi2(fit_model, W, Nt)
    misc_extra = {"fit_model": fit_model}
    binned_tag = _ensure_binned(db, tag, binsize)

    best, jks, misc = _jackknife_fit(db, binned_tag, fit_range, p0, make_chi2, config, misc_extra)
    rows = _fit_table_rows(binsize, "jackknife", best, misc, np.sqrt(np.diag(jackknife.covariance(jks))))
    for source in correlated_fits:
        cov_tag = binned_tag if source == "binned" else tag
        best_c, misc_c = _correlated_mean_fit(db, source, cov_tag, binned_tag, fit_range, p0, make_chi2, config, fit_model, misc_extra)
        rows += _fit_table_rows("", f"correlated {source}", best_c, misc_c)
    bootstrap_tag = None
    if bootstraps is not None:
        bootstrap_tag, best_b, bss, misc_b = _bootstrap_fit(db, binned_tag, fit_range, best, make_chi2, config, fit_model, misc_extra, bootstraps)
        rows += _fit_table_rows("", "bootstrap", best_b, misc_b, np.sqrt(np.diag(bootstrap.covariance(bss))))
    message("\n".join(rows), silent, continuation=True)
    return FitTags(jackknife=_store_jackknife_fit(db, binned_tag, fit_model, best, jks, misc), bootstrap=bootstrap_tag)


# ---------------------------------------------------------------------------
# Two-correlator combined fit (shared mass)
# ---------------------------------------------------------------------------

def combined_correlator_fit(db, tags, combined_tag, fit_ranges, binsize, p0, fit_models,
                            config: FitConfig, Nt=None, silent=False, bootstraps=None):
    """Fit two correlators with separate amplitudes and a shared mass, ``p = [A0, A1, m]``.

    Each block uses a ``cosh``, ``sinh`` or ``exp`` model. Returns one ``FitTags``
    per binsize 1..``binsize``; the bootstrap tag is set only at binsize 1.
    """
    if len(tags) != 2 or len(fit_ranges) != 2:
        raise ValueError("tags and fit_ranges must each contain two entries")
    if len(p0) != 3:
        raise ValueError(f"p0 must contain [A0, A1, m], got {len(p0)} entries")
    fit_model_combined = _combined_model_name(fit_models)
    entries = [db.database[tag] for tag in tags]
    if not np.array_equal(entries[0].cfgs, entries[1].cfgs):
        raise ValueError(f"combined_correlator_fit: cfgs of {tags[0]!r} and {tags[1]!r} differ; cannot pair configs")
    if not np.array_equal(entries[0].weights, entries[1].weights):
        raise ValueError(f"combined_correlator_fit: weights of {tags[0]!r} and {tags[1]!r} differ; cannot pair configs")
    if len(entries[0].central_value) != len(entries[1].central_value):
        raise ValueError(f"combined_correlator_fit: data lengths of {tags[0]!r} and {tags[1]!r} differ")

    message(_log_divider("combined correlator fit"), silent)
    for i, (tag, fit_range, fit_model) in enumerate(zip(tags, fit_ranges, fit_models)):
        message(f"block {i}: {tag}, fit range {fit_range}, {fit_model} model = {FIT_MODEL_FORMULAS[fit_model]}", silent)
    message("shared mass m = p[2]", silent)
    message(f"P0 = {p0}")

    Nt = len(entries[0].central_value) if Nt is None else Nt
    fit_range_combined = np.hstack(fit_ranges)
    def make_chi2(W):
        return _make_combined_chi2(fit_models, W, tuple(map(len, fit_ranges)), Nt)
    combined_misc = {
        "fit_model": fit_model_combined,
        "fit_models": fit_models,
        "fit_ranges": fit_ranges,
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
        #    starting from the jackknife result
        if b in [1, binsize]:
            message(_log_divider("correlated mean fit"), silent)
            cov = db.jackknife_covariance(binned_corr_tag)
            best, misc_corr = _try_correlated_fit(db, binned_corr_tag, fit_range_combined, cov, len(db.database[binned_corr_tag].jks), best_parameter, make_chi2, config, "binned", slice_data=False, silent=silent)
            if best is not None:
                misc_corr.update(combined_misc)
                print_fit_results(best, None, misc_corr, silent)
                db.add_entry(f"{binned_corr_tag}/{fit_model_combined}_correlated_mean_fit", central_value=best, misc=misc_corr)
            message(_log_divider(), silent)

        # 3. bootstrap fit, starting from the jackknife result; b == 1 only
        #    because the bootstrap indices refer to unbinned configurations
        if b == 1 and bootstraps is not None:
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
