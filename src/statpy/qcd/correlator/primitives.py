"""Pure correlator / effective-mass primitives. No database, no state."""
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


def effective_mass(corr, *, estimator, Nt=None, axis=0):
    """Return effective masses along the time axis.

    log: forward log ratio; log_symmetric: centered log ratio.
    arccosh: three-point recurrence, valid for cosh and sinh.
    cosh_midpoint: abs(acosh(C(t-1)/Cmid) - acosh(C(t+1)/Cmid))/2.
    Nt defaults to the time-axis length; midpoint data must be present.
    Times start at zero; neighbor formulas wrap endpoints as with np.roll.
    """
    corr = np.asarray(corr)
    if estimator == "cosh_midpoint":
        Nt = corr.shape[axis] if Nt is None else Nt
        midpoint = np.take(corr, [Nt // 2], axis=axis)
        with np.errstate(invalid='ignore'):
            return np.abs(
                np.arccosh(np.roll(corr, 1, axis=axis) / midpoint)
                - np.arccosh(np.roll(corr, -1, axis=axis) / midpoint)
            ) / 2.
    with np.errstate(divide='ignore', invalid='ignore'):
        if estimator == "log":
            return np.log(corr / np.roll(corr, -1, axis=axis))
        if estimator == "log_symmetric":
            return np.log(np.roll(corr, 1, axis=axis) / np.roll(corr, -1, axis=axis)) / 2
        if estimator == "arccosh":
            return np.arccosh(0.5 * (np.roll(corr, -1, axis=axis) + np.roll(corr, 1, axis=axis)) / corr)
    raise ValueError(f"Unknown effective-mass estimator: {estimator}")


def effective_amplitude(corr, mass, *, kernel, Nt=None, axis=0):
    """Divide corr by a single-state kernel, preserving its sign.

    exp: exp(-mass*t); cosh/sinh: add/subtract exp(-mass*(Nt-t)).
    Nt defaults to the time-axis length; pass the original Nt for folded data.
    Times start at zero along axis; exponential sums set the normalization.
    """
    corr = np.asarray(corr)
    Nt = corr.shape[axis] if Nt is None else Nt
    shape = [1] * corr.ndim
    shape[axis] = corr.shape[axis]
    t = np.arange(corr.shape[axis]).reshape(shape)
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
