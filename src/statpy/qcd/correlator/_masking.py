"""Private helpers: masking for OBC averaging + boundary folding."""
import numpy as np

from statpy.qcd.correlator.primitives import _validate_time_parity

# Masked sample for OBC averaging (vectorized over configs).
# sample (N_cfg, num_Cts, T) -> masked (N_cfg, 2*num_Cts, T): forward Cts in the
# first num_Cts rows, time-reversed backward Cts in the second. Mask is config-independent.
# Mesons only: the backward half is folded as the same state (time_parity=-1 -> sinh, else cosh).
def _get_masked_meson_sample(sample, tmax_fw, tmax_bw, time_parity):
    time_parity = _validate_time_parity(time_parity)
    N, num_Cts, T = sample.shape
    # backward = time-reverse each Ct (flip then roll by 1)
    bw = np.roll(np.flip(sample, axis=2), 1, axis=2)
    bw[:, :, 1:] *= time_parity
    out = np.ma.empty((N, 2*num_Cts, T))
    out.mask = True
    out[:, :num_Cts, :tmax_fw] = sample[:, :, :tmax_fw]
    out.mask[:, :num_Cts, :tmax_fw] = False
    out[:, num_Cts:, :tmax_bw] = bw[:, :, :tmax_bw]
    out.mask[:, num_Cts:, :tmax_bw] = False
    return out


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


def _fold_meson_boundary(arr, time_parity):
    time_parity = _validate_time_parity(time_parity)
    half = len(arr) // 2
    arr0 = arr[:half]
    arr1 = np.flip(arr[half:]) * time_parity
    return np.mean([arr0, arr1], axis=0)


def _flip_sign_boundary(arr, tsrc):
    arr[:tsrc] = -arr[:tsrc]
    return arr
