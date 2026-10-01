"""Pure correlator / effective-mass primitives. No database state."""
from numbers import Real

import numpy as np
from scipy.optimize import elementwise

from statpy.log import message


def _validate_time_parity(time_parity):
    """Return the sign for even (+1) or odd (-1) time parity; reject booleans."""
    if (
        isinstance(time_parity, (bool, np.bool_))
        or not isinstance(time_parity, Real)
        or time_parity not in (-1, 1)
    ):
        raise ValueError("time_parity must be +1 (even) or -1 (odd), not a boolean")
    return int(time_parity)


@np.errstate(divide='ignore', invalid='ignore')
def effective_mass(corr, *, estimator, Nt=None):
    """Return effective masses of a 1D correlator.

    estimator selects the formula:
      "log": forward log ratio, log(C(t)/C(t+1)).
      "log_symmetric": centered log ratio, log(C(t-1)/C(t+1))/2.
      "arccosh": three-point recurrence, valid for cosh and sinh.
      "cosh_midpoint": abs(acosh(C(t-1)/Cmid) - acosh(C(t+1)/Cmid))/2.
      "cosh_solve": solves cosh(m(t-Nt/2))/cosh(m(t+1-Nt/2)) = C(t)/C(t+1) for m.
      "sinh_solve": solves sinh(m(t-Nt/2))/sinh(m(t+1-Nt/2)) = C(t)/C(t+1) for m.
      (*_solve: Gattringer & Lang, Lect. Notes Phys. 788 (2010), p. 145.)
    Nt is required for cosh_solve and sinh_solve (also for folded input);
    cosh_midpoint uses the correlator length and needs unfolded data.
    Times start at zero; neighbor formulas wrap endpoints as with np.roll.
    Undefined points (e.g. arccosh argument < 1) return NaN or inf without warning.
    """
    corr = np.asarray(corr)
    if estimator == "cosh_midpoint":
        Nt = len(corr) if Nt is None else Nt
        midpoint = corr[Nt // 2]
        return np.abs(
            np.arccosh(np.roll(corr, 1) / midpoint)
            - np.arccosh(np.roll(corr, -1) / midpoint)
        ) / 2
    if estimator == "log":
        return np.log(corr / np.roll(corr, -1))
    if estimator == "log_symmetric":
        return np.log(np.roll(corr, 1) / np.roll(corr, -1)) / 2
    if estimator == "arccosh":
        return np.arccosh(0.5 * (np.roll(corr, -1) + np.roll(corr, 1)) / corr)
    if estimator in ("cosh_solve", "sinh_solve"):
        if Nt is None:
            raise ValueError(f"{estimator} needs Nt")
        return _solve_effective_mass(corr, kernel=estimator.removesuffix("_solve"), Nt=Nt)

    raise ValueError(f"Unknown effective-mass estimator: {estimator}")


def _solve_effective_mass(corr, *, kernel, Nt):
    """Solve kernel(m|t-Nt/2|)/kernel(m|t+1-Nt/2|) = C(t)/C(t+1) per t; kernel is "cosh" or "sinh"."""
    t = np.arange(len(corr))
    a = np.abs(t - Nt / 2)
    b = np.abs(t + 1 - Nt / 2)
    log_ratio = np.log(corr / np.roll(corr, -1))
    if kernel == "cosh":
        # log(cosh[m(t - Nt/2)] / cosh[m(t + 1 - Nt/2)]) - log(C(t) / C(t+1))
        def objective_function(m, a, b, log_ratio):
            return (
                m * (a - b)
                + np.log1p(np.exp(-2 * m * a))
                - np.log1p(np.exp(-2 * m * b))
                - log_ratio
            )
        lower = np.abs(log_ratio)
        upper = lower + np.log(2)
        valid = a != b
    else:
        # log(sinh[m(t - Nt/2)] / sinh[m(t + 1 - Nt/2)]) - log(C(t)/C(t+1))
        def objective_function(m, a, b, log_ratio):
            return (
                m * (a - b)
                + np.log(-np.expm1(-2 * m * a))
                - np.log(-np.expm1(-2 * m * b))
                - log_ratio
            )
        upper = np.abs(log_ratio)
        lower = upper - np.abs(np.log(a/b))
        valid = (a != b) & (a != 0) & (b != 0) & (lower > 0)
        lower = np.where(valid, lower, np.nan)
    result = elementwise.find_root(
        objective_function, (lower, upper),
        args=(a, b, log_ratio)
    )
    mass = np.where(result.success & valid, result.x, np.nan)
    return mass


def effective_amplitude(corr, mass, *, kernel, Nt=None):
    """Divide a 1D correlator by a single-state kernel, preserving its sign.

    kernel selects the time dependence:
      "exp": exp(-mass*t).
      "cosh"/"sinh": exp(-mass*t) plus/minus exp(-mass*(Nt-t)).
    Nt is required for cosh and sinh (also for folded data).
    Times start at zero; exponential sums set the normalization.
    """
    corr = np.asarray(corr)
    if kernel in ("cosh", "sinh") and Nt is None:
        raise ValueError(f"{kernel} kernel needs Nt")
    t = np.arange(len(corr))
    forward = np.exp(-mass * t)
    if kernel == "exp":
        return corr / forward
    if kernel == "cosh":
        return corr / (forward + np.exp(-mass * (Nt - t)))
    if kernel == "sinh":
        return corr / (forward - np.exp(-mass * (Nt - t)))
    raise ValueError(f"Unknown effective-amplitude kernel: {kernel}")


def meson_fold_correlator(corr, time_parity=1):
    """Fold a meson correlator around T/2; time_parity is +1 (even) or -1 (odd)."""
    time_parity = _validate_time_parity(time_parity)
    half = len(corr) // 2
    first_half = corr[:half]
    second_half = np.roll(np.flip(corr[half:]), 1) * time_parity
    second_half[0] = first_half[0]
    return np.mean([first_half, second_half], axis=0)


def get_tmax_signal_to_noise(mean, var, min_signal_to_noise, tmin=None, debug=False):
    """Return the first failing slice (exclusive endpoint), or the data length.

    A failure is S/N below the required threshold or NaN. Search all slices
    unless tmin is supplied, in which case only t > tmin is considered.
    """
    signal_to_noise = mean / var**.5
    times = np.arange(len(signal_to_noise))
    invalid = (signal_to_noise < min_signal_to_noise) | np.isnan(signal_to_noise)
    if tmin is not None:
        invalid &= times > tmin
    candidates = times[invalid]

    tmax = int(candidates[0]) if candidates.size else len(mean)
    if not candidates.size:
        reason = "End of data; no failing slice in search region"
    else:
        reason = "First NaN" if np.isnan(signal_to_noise[tmax]) else "First value below threshold"
    search_region = "all time slices" if tmin is None else f"t > {tmin}"
    message(
        "Signal-to-noise cutoff\n"
        f"  {'Threshold':<14} {min_signal_to_noise}\n"
        f"  {'Search region':<14} {search_region}\n"
        f"  {'tmax':<14} {tmax} (exclusive)\n"
        f"  {'Reason':<14} {reason}"
    )
    if debug:
        message(f"Signal-to-noise values ({len(signal_to_noise)} slices):\n{signal_to_noise}")
    return tmax


def binned_tag(tag, binsize):
    """Conventional tag of the binsize-``binsize`` variant of ``tag``.

    ``binsize == 1`` returns ``tag`` unchanged (no binning needed). This is
    the single home of the ``<tag>/binsize<N>`` naming convention: producers
    and consumers must both route tag construction through it so they agree.
    """
    return tag if binsize == 1 else f"{tag}/binsize{binsize}"
