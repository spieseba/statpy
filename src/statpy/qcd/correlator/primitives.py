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


# ---------------------------------------------------------------------------
# periodic boundary conditions
# ---------------------------------------------------------------------------

def meff_cosh(Ct, ax=0):
    with np.errstate(invalid='ignore'):
        return np.arccosh(0.5 * (np.roll(Ct, -1, axis=ax) + np.roll(Ct, 1, axis=ax)) / Ct)

# spectrum paper
def meff_cosh_midpoint(Ct, a=1):
    Nt = len(Ct)
    with np.errstate(invalid='ignore'):
        return np.abs(np.arccosh(np.roll(Ct,a)/Ct[Nt//2]) - np.arccosh(np.roll(Ct,-a)/Ct[Nt//2])) / (2. * a)

def meff_sinh(Ct):
    Nt = len(Ct)
    eff_m = np.arcsinh(Ct/Ct[Nt-1])
    return np.abs(np.roll(eff_m, -1) - eff_m)

# ---------------------------------------------------------------------------
# open boundary conditions
# ---------------------------------------------------------------------------

def meff_exp_forward(Ct, ax=0):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(Ct / np.roll(Ct, -1, axis=ax))

# spectrum paper
def meff_exp_symmetric(Ct, ax=0):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(np.roll(Ct, 1, axis=ax) / np.roll(Ct, -1, axis=ax)) / 2

# cosh
def Aeff_cosh(Ct, m):
    Nt = len(Ct)
    t = np.arange(Nt)
    return Ct / (np.exp(-m*t) + np.exp(-m*(Nt-t)))

# sinh
def Aeff_sinh(Ct, m):
    Nt = len(Ct)
    t = np.arange(Nt)
    return Ct / (np.exp(-m*t) - np.exp(-m*(Nt-t)))

# exp
def Aeff_exp(Ct, m):
    Nt = len(Ct)
    return Ct / np.exp(-m*np.arange(Nt))


def meson_fold_correlator(arr, time_parity=1):
    """Fold a meson correlator around T/2; time_parity is +1 (even) or -1 (odd)."""
    time_parity = _validate_time_parity(time_parity)
    half = len(arr) // 2
    arr0 = arr[:half]
    arr1 = np.roll(np.flip(arr[half:]), 1) * time_parity
    arr1[0] = arr0[0]
    return np.mean([arr0, arr1], axis=0)


def get_tmax_signal_to_noise(mean, var, min_stn_val=100, tmin=15, debug=False):
    signal_to_noise = mean / var**.5
    tmax = next((i for i, x in enumerate(signal_to_noise) if (i > tmin) and ((x < min_stn_val) or np.isnan(x))), -1)
    if tmax == -1:
        tmax = len(mean)
        message(f"--- Signal to noise ratio never smaller than {min_stn_val} -> return tmax = len(mt) = {tmax}")
    else:
        message(f"--- Signal to noise ratio smaller than {min_stn_val} for tmax = {tmax} -> return tmax = {tmax}")
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
