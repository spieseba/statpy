"""Private helpers: masking for OBC averaging + boundary folding."""
import numpy as np


def bare_decay_constant(p):
    # p[0] = A_PSPS, p[1] = A_PSA4I, p[2] = m
    return np.sqrt(2.) * p[1] / np.sqrt(p[0] * p[2])


# Get masked Cts for obc averaging
def _get_masked_Ct(Cts, tmax_fw, tmax_bw, antiperiodic):
    num_Cts = Cts.shape[0]
    # create masked array and mask all elements
    Cts_ma = np.ma.empty( (2*num_Cts, Cts.shape[1]) )
    Cts_ma.mask = True
    # fill masked array up to tmax_fw and tmax_bw
    for idx in range(num_Cts):
        Ct = Cts[idx]
        Cts_ma[idx, :tmax_fw] = Ct[:tmax_fw]
        Cts_ma[idx+num_Cts, :tmax_bw] = np.roll(np.flip(Ct), 1)[:tmax_bw]
        if antiperiodic: 
            Cts_ma[idx+num_Cts, 1:tmax_bw] *= -1
    return Cts_ma


# get tmax for each src in forward and backward direction
def _get_tmax_fw_bw(tsrcs, tbulk):
    tmin = tbulk[0]
    tmax = tbulk[-1]
    tmax_fw = tmax + 1 - np.array(tsrcs)
    tmax_bw = np.array(tsrcs) - tmin + 1
    return tmax_fw, tmax_bw


# mask excited states and time slices further away than tmax from source position
def _get_masked_Cts_boundary(Cts, tsrc, tmin_excited, tmax_from_tsrc=None):
    # get Cts in terms of original lattice
    Cts_aligned = np.roll(Cts, tsrc, axis=1)
    num_Cts = Cts.shape[0]
    # 1: create masked array
    Cts_ma = np.ma.empty((num_Cts, Cts.shape[1]) )
    Cts_ma.mask = True
    # 2: fill in all elements that are not excited states
    Cts_ma[:,:tsrc-(tmin_excited-1)] = Cts_aligned[:,:tsrc-(tmin_excited-1)]
    Cts_ma[:,tsrc+tmin_excited:] = Cts_aligned[:,tsrc+tmin_excited:]
    # 3: make sure all time slices that are further away than tmax from source position are masked
    if tmax_from_tsrc is not None:
        times = np.arange(Cts.shape[1])
        tmax_mask = np.abs(times - tsrc) > tmax_from_tsrc
        Cts_ma.mask = np.logical_or(Cts_ma.mask, tmax_mask[np.newaxis,:])
    return Cts_ma


def _fold_boundary(arr, antiperiodic):
    half = len(arr) // 2
    arr0 = arr[:half]
    arr1 = np.flip(arr[half:])
    if antiperiodic: 
        arr1 *= -1.
    return np.mean([arr0, arr1], axis=0)


def _flip_sign_boundary(arr, tsrc):
    arr[:tsrc] = -arr[:tsrc]
    return arr
