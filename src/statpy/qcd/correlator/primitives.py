"""Pure correlator / effective-mass primitives. No database, no state."""
import numpy as np
from statpy.log import message


### periodic boundary conditions ###
def meff_cosh(Ct, ax=0):
    with np.errstate(invalid='ignore'):
        return np.arccosh(0.5 * (np.roll(Ct, -1, axis=ax) + np.roll(Ct, 1, axis=ax)) / Ct)

# spectrum paper
def meff_cosh_midpoint(Ct, a=1):
    Nt = len(Ct)
    with np.errstate(invalid='ignore'):
        return np.abs((np.arccosh(np.roll(Ct,a)/Ct[Nt//2]) - np.arccosh(np.roll(Ct,-a)/Ct[Nt//2]))) / (2. * a)

def meff_sinh(Ct):
    Nt = len(Ct)
    eff_m = np.arcsinh(Ct/Ct[Nt-1])
    return np.abs(np.roll(eff_m, -1) - eff_m)

### open boundary conditions ###
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
    return Ct / np.array([(np.exp(-m*t)) + np.exp(-m*(Nt-t)) for t in range(Nt)])

# sinh
def Aeff_sinh(Ct, m):
    Nt = len(Ct)
    return Ct / np.array([(np.exp(-m*t)) - np.exp(-m*(Nt-t)) for t in range(Nt)])

# exp
def Aeff_exp(Ct, m):
    Nt = len(Ct)
    return Ct / np.array([(np.exp(-m*t)) for t in range(Nt)])


def fold_correlator(arr, antiperiodic=False):
    half = len(arr) // 2
    arr0 = arr[:half]
    arr1 = np.roll(np.flip(arr[half:]), 1)
    if antiperiodic: 
        arr1 *= -1.
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
