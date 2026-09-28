"""Pure correlator / effective-mass primitives. No database state."""
from numbers import Real

import numpy as np

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
    Nt (cosh_midpoint only) defaults to the correlator length; midpoint data must be present.
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
    raise ValueError(f"Unknown effective-mass estimator: {estimator}")


def effective_amplitude(corr, mass, *, kernel, Nt=None):
    """Divide a 1D correlator by a single-state kernel, preserving its sign.

    kernel selects the time dependence:
      "exp": exp(-mass*t).
      "cosh"/"sinh": exp(-mass*t) plus/minus exp(-mass*(Nt-t)).
    Nt defaults to the correlator length; pass the original Nt for folded data.
    Times start at zero; exponential sums set the normalization.
    """
    corr = np.asarray(corr)
    Nt = len(corr) if Nt is None else Nt
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


def get_tmax_signal_to_noise(mean, var, min_signal_to_noise=100, tmin=15, debug=False):
    signal_to_noise = mean / var**.5
    tmax = next((i for i, x in enumerate(signal_to_noise) if (i > tmin) and ((x < min_signal_to_noise) or np.isnan(x))), -1)
    if tmax == -1:
        tmax = len(mean)
        message(f"--- Signal to noise ratio never smaller than {min_signal_to_noise} -> return tmax = len(mean) = {tmax}")
    else:
        message(f"--- Signal to noise ratio smaller than {min_signal_to_noise} for tmax = {tmax} -> return tmax = {tmax}")
    if debug:
        message(f"--- Signal to noise ratios: {signal_to_noise}")
        message(f"--- len(stn) = {len(signal_to_noise)}")
    return tmax


def binned_tag(tag, binsize):
    """Conventional tag of the binsize-``binsize`` variant of ``tag``.

    ``binsize == 1`` returns ``tag`` unchanged (no binning needed). This is
    the single home of the ``<tag>/binsize<N>`` naming convention: producers
    and consumers must both route tag construction through it so they agree.
    """
    return tag if binsize == 1 else f"{tag}/binsize{binsize}"
