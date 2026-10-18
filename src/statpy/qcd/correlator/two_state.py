"""Start values and ground-state range proposal for two-state correlator fits."""

import numpy as np

from statpy.qcd.correlator import models
from statpy.qcd.correlator.primitives import effective_mass

# ---------------------------------------------------------------------------
# Start values
# ---------------------------------------------------------------------------

def default_mass_guess(central_value, fit_model):
    """Return the effective-mass guess and default mass gaps."""
    y = central_value
    n = len(y)
    estimator = "log" if fit_model == "double-exp" else "arccosh"
    m0 = np.nanmean(effective_mass(y, estimator=estimator)[n // 4:n // 4 + n // 8])
    return m0, m0 * np.array([0.25, 0.5, 1.0])


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
        model = models.exp()
    elif fit_model in ("double-cosh", "double-sinh"):
        if Nt is None or not np.isfinite(Nt) or Nt <= 0:
            raise ValueError("Periodic models require a finite positive Nt")
        model = models.cosh(Nt) if fit_model == "double-cosh" else models.sinh(Nt)
    else:
        raise ValueError(f"Unknown fit_model: {fit_model!r}")

    sigma = np.sqrt(var_fit_range)
    guesses = []
    for gap in gaps:
        m1 = m0 + gap
        if not np.isfinite(m1) or m1 <= m0:
            raise ValueError(f"Mass gap {gap} does not produce a finite mass above m0")
        design = np.column_stack((model(t, np.array([1.0, m0])), model(t, np.array([1.0, m1])))) / sigma[:, None] 
        scale = np.max(np.abs(design), axis=0) # scale columns to avoid treating the smaller excited kernel as zero
        if not np.isfinite(design).all() or np.any(scale == 0):
            raise ValueError(f"Nonfinite or vanishing model column for mass gap {gap}")
        amplitudes, _, rank, _ = np.linalg.lstsq(design / scale, y / sigma, rcond=None)
        amplitudes = amplitudes / scale
        if rank < 2 or not np.isfinite(amplitudes).all():
            raise ValueError(f"Cannot resolve two amplitudes for mass gap {gap}")
        guesses.append([amplitudes[0], m0, amplitudes[1], m1])
    return np.asarray(guesses)


# ---------------------------------------------------------------------------
# Ground-state range
# ---------------------------------------------------------------------------

def propose_ground_state_range(fit_range, variance, parameters, fit_model, *, Nt, folded=False):
    """Propose ground-state slices using the sigma/4 excited-contribution cut."""
    model_func = {"double-cosh": models.double_cosh(Nt),
                  "double-sinh": models.double_sinh(Nt),
                  "double-exp": models.double_exp()}[fit_model]
    boundary_condition = "open" if fit_model == "double-exp" else "periodic"
    excited = np.abs(model_func(fit_range, np.array([0.0, 0.0, parameters[2], parameters[3]])))
    std_over_four = (variance ** 0.5) / 4.0
    if boundary_condition == "periodic" and not folded:
        std_over_four = (std_over_four + std_over_four[::-1]) / 2
    return fit_range[excited < std_over_four]
