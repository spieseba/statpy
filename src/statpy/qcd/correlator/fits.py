"""Variable-based correlator fits and separate result logging."""
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from statpy.fitting.core import ConvergenceError, Fitter, get_pvalue
from statpy.log import format_paren, message
from statpy.qcd.correlator.models import (
    FIT_MODEL_FORMULAS,
    CoshModel,
    DoubleCoshModel,
    DoubleExpModel,
    DoubleSinhModel,
    ExpModel,
    SinhModel,
    combined_correlator_chi2,
    cosh_chi2,
    double_cosh_chi2,
    double_exp_chi2,
    double_sinh_chi2,
    exp_chi2,
    sinh_chi2,
)
from statpy.qcd.correlator.primitives import effective_mass
from statpy.statistics import jackknife
from statpy.variable import Variable

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FitConfig:
    """Optimizer choice and parameters."""
    fit_method: str = "Nelder-Mead"
    fit_params: dict = field(default_factory=lambda: {"maxiter": 5000, "tol": 1e-07})


@dataclass
class FitResult:
    """Fit parameters, diagnostics, correlated cross-checks and failure reason."""
    parameters: Variable | None
    diagnostics: dict
    correlated: dict = field(default_factory=dict)
    failure_reason: str | None = None


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


# Backward-propagator sign s of each block model in combined_correlator_chi2.
_BACKWARD_SIGN = {"cosh": 1.0, "sinh": -1.0, "exp": 0.0}


def _make_combined_chi2(fit_model, W, Nt, L):
    """Return a chi^2 lambda for a "combined-<model0>-<model1>" fit of a joined two-block variable."""
    block_models = fit_model.split("-")[1:]
    if len(block_models) != 2 or any(m not in _BACKWARD_SIGN for m in block_models):
        raise ValueError(f"Unknown fit_model: {fit_model!r}")
    s0, s1 = (_BACKWARD_SIGN[m] for m in block_models)
    return lambda t, p, y: combined_correlator_chi2(t, p, y, W, Nt, L, s0, s1)


# ---------------------------------------------------------------------------
# Core fit primitives
# ---------------------------------------------------------------------------

def fit_mean(variable, fit_range, p0, chi2_func, config: FitConfig):
    """Fit the central value over ``fit_range`` and return parameters and metadata."""
    p0 = np.asarray(p0, dtype=float)
    if p0.ndim != 1 or p0.size == 0:
        raise ValueError(f"'p0' must be a non-empty 1-D array, got shape {p0.shape}")
    dof = len(fit_range) - p0.size
    if dof <= 0:
        raise ValueError(
            f"non-positive degrees of freedom: len(fit_range)={len(fit_range)}, num_params={p0.size}, dof={dof}"
        )

    y = variable.central_value[fit_range]
    fitter = Fitter(config.fit_method, config.fit_params)
    try:
        best = fitter.estimate_parameters(fit_range, chi2_func, y, p0)[0]
    except ConvergenceError as e:
        raise ConvergenceError(f"mean fit did not converge: {e}") from e
    if not np.isfinite(best).all():
        raise ConvergenceError(f"mean fit produced non-finite parameters: {best}")
    chi2 = chi2_func(fit_range, best, y)
    if not np.isfinite(chi2):
        raise ConvergenceError(f"non-finite chi^2 = {chi2}")
    return best, {"fit_range": fit_range, "chi2": chi2, "dof": dof, "pval": get_pvalue(chi2, dof)}


def _fit_resamples(samples, label, fit_range, p0, chi2_func, config):
    """Fit each replicate in order; label identifies convergence errors."""
    fitter = Fitter(config.fit_method, config.fit_params)
    try:
        return np.array([fitter.estimate_parameters(fit_range, chi2_func, y[fit_range], p0)[0] for y in samples])
    except ConvergenceError as e:
        raise ConvergenceError(f"{label} fit did not converge: {e}") from e


def fit_jackknife(variable, fit_range, p0, chi2_func, config: FitConfig) -> FitResult:
    """Fit the central value, then each jackknife replicate starting from that fit."""
    best, misc = fit_mean(variable, fit_range, p0, chi2_func, config)
    best_jks = _fit_resamples(variable.jackknife_replicates, "jackknife", fit_range, best, chi2_func, config)
    return FitResult(Variable.from_results(central_value=best, jackknife_replicates=best_jks, layout=variable.layout), misc)


def fit_bootstrap(variable, fit_range, p0, chi2_func, config: FitConfig) -> FitResult:
    """Fit the central value, then stored bootstrap replicates."""
    samples = variable.bootstrap_replicates
    if samples is None:
        raise ValueError("variable has no stored bootstrap_replicates")
    best, misc = fit_mean(variable, fit_range, p0, chi2_func, config)
    best_bss = _fit_resamples(samples, "bootstrap", fit_range, best, chi2_func, config)
    return FitResult(Variable.from_results(central_value=best, bootstrap_replicates=best_bss), misc)


def _sort_two_state_params(p):
    """Order the two states of a double-* fit so the lower mass comes first."""
    if p[3] < p[1]:
        return [p[2], p[3], p[0], p[1]]
    return p


def propose_ground_state_range(fit_range, variance, parameters, fit_model, *, Nt, folded=False):
    """Propose ground-state slices using the sigma/4 excited-contribution cut."""
    model_func = {"double-cosh": DoubleCoshModel(Nt),
                  "double-sinh": DoubleSinhModel(Nt),
                  "double-exp": DoubleExpModel()}[fit_model]
    boundary_condition = "open" if fit_model == "double-exp" else "periodic"
    excited = np.abs([model_func(i, [0, 0, parameters[2], parameters[3]]) for i in fit_range])
    std_over_four = (variance ** 0.5) / 4.0
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


def _inverse_covariance(cov, num_samples):
    """Return ``(inverse, None)``, or ``(None, reason)`` if ``cov`` is not invertible."""
    if len(cov) >= num_samples:
        return None, f"singular: {len(cov)} points from {num_samples} samples"
    sigma = np.sqrt(np.diag(cov))
    if not np.all(np.isfinite(sigma) & (sigma > 0)):
        return None, "not positive definite: non-positive or non-finite variance"
    scale = np.outer(sigma, sigma)
    try:
        factor = cho_factor(cov / scale, lower=True)
    except np.linalg.LinAlgError:
        return None, "not positive definite"
    return cho_solve(factor, np.eye(len(cov))) / scale, None


def _try_correlated_fit(variable, fit_range, cov_fit_range, num_samples, p0, make_chi2, config):
    """Return a correlated mean fit or its failure diagnostics."""
    inverse, reason = _inverse_covariance(cov_fit_range, num_samples)
    if inverse is None:
        return None, {"failure_reason": f"covariance matrix {reason}"}
    try:
        chi2 = make_chi2(inverse)
        return fit_mean(variable, fit_range, p0, chi2, config)
    except ConvergenceError:
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


def _fit_table_row(bin_size, fit, cells):
    """Format one row of the per-bin-size fit table."""
    return f"  {bin_size:>8}  {fit:<21}" + "".join(f"{cell:<18}" for cell in cells[:-2]) + f"{cells[-2]:<10}{cells[-1]}"


def _fit_table_rows(bin_size, fit, parameters, misc, errors=None):
    """Return the table row of one fit, or a row with the failure reason."""
    if parameters is None:
        return [f"  {bin_size:>8}  {fit:<21}{misc['failure_reason']}"]
    if errors is None:
        cells = [f"{v:.6g}" for v in parameters]
    else:
        cells = [format_paren(v, e) for v, e in zip(parameters, errors)]
    return [_fit_table_row(bin_size, fit, cells + [f"{misc['chi2'] / misc['dof']:.3g}", f"{misc['pval']:.2f}"])]


def log_fit_header(title, tag, fit_model, fit_range, p0, silent=False, *, L=None):
    """Log the correlator, model, fit range, start values and the fit-table columns.

    For combined models, L is the length of one joined correlator; the fit range is shown per block.
    """
    formula, parameters = FIT_MODEL_FORMULAS[fit_model].split("; ")
    names = _parameter_names(fit_model)
    dof = len(fit_range) - len(names)
    if fit_model.startswith("combined-"):
        fit_range = np.asarray(fit_range)
        ranges = f"{_format_range(fit_range[fit_range < L])} + {_format_range(fit_range[fit_range >= L] - L)}"
    else:
        ranges = _format_range(fit_range)
    message("\n".join([
        title,
        *_labelled("Correlator", [tag]),
        *_labelled("Model", [f"{fit_model}: {formula}", parameters]),
        *_labelled("Fit range", [f"{ranges} ({len(fit_range)} points, {dof} dof)"]),
        *_labelled("Start", [_format_parameters(p0, names=names)]),
        "",
        _fit_table_row("Bin size", "Fit", [*names, "chi2/dof", "p"]),
    ]), silent)


def _format_range(t):
    return f"[{t[0]}, {t[-1]}]" if len(t) else "none"


def _format_chi2(misc):
    return f"chi2/dof {misc['chi2']:.3g}/{misc['dof']} = {misc['chi2'] / misc['dof']:#.3g}, p = {misc['pval']:.2f}"


def _fit_one_excited_range(variable, *, fit_range, m0, mass_gaps, fit_model, Nt, var, cov_binned, cov_unbinned, num_binned, num_unbinned, folded, config):
    """Fit one window's mean and jackknives, retaining convergence failures."""
    def make_chi2(W):
        return _make_chi2(fit_model, W, Nt)
    chi2_func = make_chi2(np.diag(1.0 / var[fit_range]))
    y = variable.central_value[fit_range]
    p0s = get_p0_guesses(fit_range, y, var[fit_range], fit_model, m0, mass_gaps, Nt=Nt)
    attempts = []
    result = FitResult(parameters=None, diagnostics={
        "fit_range": fit_range, "fit_model": fit_model, "initial_guess_attempts": attempts,
    })
    best = None
    for i, p0 in enumerate(p0s, 1):
        attempt = {"p0": p0.copy()}
        try:
            parameters, diagnostics = fit_mean(
                variable, fit_range, p0, chi2_func, config,
            )
        except ConvergenceError as exc:
            attempt["failure_reason"] = str(exc)
        else:
            attempt.update(parameters=parameters, diagnostics=diagnostics)
            if best is None or diagnostics["chi2"] < best[2]["chi2"]:
                best = i, parameters, diagnostics
        attempts.append(attempt)

    if best is None:
        result.failure_reason = "All initial guesses failed for this window"
        result.correlated = {"unbinned": (None, None), "binned": (None, None)}
        return result

    used_attempt, best_parameter, diagnostics = best
    result.diagnostics.update(diagnostics)
    result.diagnostics["used_attempt"] = used_attempt
    central_value = np.asarray(_sort_two_state_params(best_parameter))

    # correlated mean fits (cross-checks only)
    for label, cov, num_samples in (("unbinned", cov_unbinned, num_unbinned), ("binned", cov_binned, num_binned)):
        parameters, misc = _try_correlated_fit(variable, fit_range, cov[fit_range][:, fit_range], num_samples, best_parameter, make_chi2, config)
        if parameters is not None:
            misc["fit_model"] = fit_model
            parameters = _sort_two_state_params(parameters)
        result.correlated[label] = (parameters, misc)

    result.diagnostics["ground_state_fit_range"] = propose_ground_state_range(
        fit_range, var[fit_range], central_value, fit_model, Nt=Nt, folded=folded,
    )
    sorted_jks = None
    try:
        jks = _fit_resamples(variable.jackknife_replicates, "jackknife", fit_range, best_parameter, chi2_func, config)
        if not np.isfinite(jks).all():
            raise ConvergenceError("Jackknife fits produced non-finite parameters")
    except ConvergenceError as exc:
        result.failure_reason = str(exc)
    else:
        sorted_jks = np.array([_sort_two_state_params(jk) for jk in jks])
    result.parameters = Variable.from_results(central_value=central_value, jackknife_replicates=sorted_jks, layout=variable.layout)
    return result


def _excited_window_lines(result):
    """Return log lines for one excited-state window."""
    fit_range = result.diagnostics["fit_range"]
    attempts = result.diagnostics["initial_guess_attempts"]
    lines = [f"Window [{fit_range[0]}, {fit_range[-1]}] ({len(fit_range)} points)"]
    rows = [f"{'#':<10}" + "".join(f"{n:<13}" for n in _PARAMETER_NAMES) + "chi2/dof"]
    for i, attempt in enumerate(attempts, 1):
        rows.append((f"{i:<3}{'start':<7}" + "".join(f"{v:<13.6g}" for v in attempt["p0"])).rstrip())
        if "failure_reason" in attempt:
            rows.append(f"{'':<3}{'fit':<7}failed: {attempt['failure_reason']}")
            continue
        diagnostics = attempt["diagnostics"]
        status = f"{diagnostics['chi2'] / diagnostics['dof']:#.6g}"
        if i == result.diagnostics.get("used_attempt"):
            status += "  used"
        fitted = _sort_two_state_params(attempt["parameters"])
        rows.append(f"{'':<3}{'fit':<7}" + "".join(f"{v:<13.6g}" for v in fitted) + status)
    lines += _labelled("Guesses", rows)

    if result.parameters is None:
        lines += _labelled("Result", ["failed: all initial guesses failed"])
        return lines
    parameters = result.parameters
    if parameters.jackknife_replicates is None:
        lines += _labelled("Result", [_format_parameters(parameters.central_value), _format_chi2(result.diagnostics)])
        lines += _labelled("Jackknives", [f"failed: {result.failure_reason}"])
    else:
        errors = np.sqrt(np.diag(jackknife.covariance(parameters.jackknife_replicates)))
        lines += _labelled("Result", [_format_parameters(parameters.central_value, errors), _format_chi2(result.diagnostics)])
    correlated_rows = []
    for label, (parameters, misc) in result.correlated.items():
        if parameters is None:
            correlated_rows.append(f"{label:<10}{misc['failure_reason']}")
        else:
            correlated_rows += [f"{label:<10}{_format_parameters(parameters)}", f"{'':<10}{_format_chi2(misc)}"]
    lines += _labelled("Correlated", correlated_rows)
    lines += _labelled("Ground-state range", [f"{_format_range(result.diagnostics['ground_state_fit_range'])} (sigma/4)"])
    return lines


def _excited_fits_summary(results):
    """Return summary rows: window, used guess, masses, chi2/dof, ground range."""
    rows = [f"  {'Window':<12}{'Guess':<7}{'m0':<16}{'m1':<16}{'chi2/dof':<10}Ground-state range"]
    for result in results:
        window = _format_range(result.diagnostics["fit_range"])
        if result.parameters is None:
            rows.append(f"  {window:<12}failed: {result.failure_reason}")
            continue
        misc = result.diagnostics
        if result.parameters.jackknife_replicates is None:
            m0, m1 = (f"{result.parameters.central_value[i]:.6g}" for i in (1, 3))
            suffix = "  (jackknives failed)"
        else:
            errors = np.sqrt(np.diag(jackknife.covariance(result.parameters.jackknife_replicates)))
            m0, m1 = (format_paren(result.parameters.central_value[i], errors[i]) for i in (1, 3))
            suffix = ""
        rows.append(
            f"  {window:<12}{misc['used_attempt']:<7}{m0:<16}{m1:<16}"
            f"{misc['chi2'] / misc['dof']:<#10.3g}{_format_range(misc['ground_state_fit_range'])}{suffix}"
        )
    return rows


# ---------------------------------------------------------------------------
# Excited-state fits
# ---------------------------------------------------------------------------

def default_mass_guess(central_value, fit_model):
    """Return the effective-mass guess and default mass gaps."""
    y = central_value
    n = len(y)
    estimator = "log" if fit_model == "double-exp" else "arccosh"
    m0 = np.nanmean(effective_mass(y, estimator=estimator)[n // 4:n // 4 + n // 8])
    return m0, m0 * np.array([0.25, 0.5, 1.0])


def excited_state_fits(variable, windows, fit_model, config: FitConfig, *, unbinned, m0, mass_gaps, Nt, folded=False) -> list[FitResult]:
    """Fit each window's central value and jackknives, retaining failures."""
    num_binned = len(variable.jackknife_replicates)
    cov = variable.jackknife_covariance()
    cov_unbinned = cov if variable is unbinned else unbinned.jackknife_covariance()
    var = np.diag(cov)
    results = []
    for fit_range in windows:
        results.append(_fit_one_excited_range(
            variable,
            fit_range=np.asarray(fit_range),
            m0=m0,
            mass_gaps=mass_gaps,
            fit_model=fit_model,
            Nt=Nt,
            var=var,
            cov_binned=cov,
            cov_unbinned=cov_unbinned,
            num_binned=num_binned,
            num_unbinned=len(unbinned.jackknife_replicates),
            folded=folded,
            config=config,
        ))
    return results


def log_excited_state_fits(name, bin_size, variable, fit_model, m0, mass_gaps, results, silent=False):
    """Log the excited-state header, windows and summary."""
    num_binned = len(variable.jackknife_replicates)
    formula, parameters = FIT_MODEL_FORMULAS[fit_model].split("; ")
    message("\n".join([
        "Excited-state fits",
        f"  {'Correlator':<20} {name}",
        *_labelled("Model", [f"{fit_model}: {formula}", parameters]),
        f"  {'Bin size':<20} {bin_size} ({num_binned} jackknives)",
        f"  {'Mass guess':<20} {m0:.6g}; gaps " + ", ".join(f"{gap:.6g}" for gap in mass_gaps),
    ]), silent)
    for result in results:
        message("\n".join(_excited_window_lines(result)), silent)
    message("\n".join(["Summary"] + _excited_fits_summary(results)), silent)


def ground_state_fit(variable, fit_range, p0, fit_model, config: FitConfig, *, Nt, L=None, correlated=None) -> tuple[FitResult, FitResult | None]:
    """Return the jackknife fit, correlated cross-checks and optional bootstrap fit."""
    if fit_model.startswith("combined-"):
        if L is None:
            raise ValueError("combined fits require L")
        def make_chi2(W):
            return _make_combined_chi2(fit_model, W, Nt, L)
    else:
        def make_chi2(W):
            return _make_chi2(fit_model, W, Nt)
    misc_extra = {"fit_model": fit_model}
    chi2_func = make_chi2(np.diag(1.0 / variable.jackknife_variance()[fit_range]))
    result = fit_jackknife(variable, fit_range, p0, chi2_func, config)
    result.diagnostics.update(misc_extra)
    for label, covariance_variable in (correlated or {}).items():
        cov_fit_range = covariance_variable.jackknife_covariance()[fit_range][:, fit_range]
        best, misc = _try_correlated_fit(variable, fit_range, cov_fit_range, len(covariance_variable.jackknife_replicates), p0, make_chi2, config)
        if best is not None:
            misc.update(misc_extra)
        result.correlated[label] = (best, misc)
    bootstrap_result = None
    if variable.bootstrap_replicates is not None:
        chi2_func = make_chi2(np.diag(1.0 / variable.bootstrap_variance()[fit_range]))
        bootstrap_result = fit_bootstrap(variable, fit_range, result.parameters.central_value, chi2_func, config)
        bootstrap_result.diagnostics.update(misc_extra)
    return result, bootstrap_result


def log_ground_state_fit(bin_size, jackknife, bootstrap, silent=False):
    """Log jackknife, correlated and bootstrap fit table rows."""
    parameters = jackknife.parameters
    rows = _fit_table_rows(bin_size, "jackknife", parameters.central_value, jackknife.diagnostics, np.sqrt(np.diag(parameters.jackknife_covariance())))
    for label, (best, misc) in jackknife.correlated.items():
        rows += _fit_table_rows("", f"correlated {label}", best, misc)
    if bootstrap is not None:
        parameters = bootstrap.parameters
        rows += _fit_table_rows("", "bootstrap", parameters.central_value, bootstrap.diagnostics, np.sqrt(np.diag(parameters.bootstrap_covariance())))
    message("\n".join(rows), silent, continuation=True)
