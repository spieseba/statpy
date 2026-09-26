"""Private helpers: masking for OBC averaging + boundary folding."""
import numpy as np

from statpy.qcd.correlator.primitives import _validate_time_parity


# Masked sample for OBC averaging (vectorized over configs).
# sample (N_cfg, n_corrs, T) -> masked (N_cfg, 2*n_corrs, T): forward corrs in the
# first n_corrs rows, time-reversed backward corrs in the second. Mask is config-independent.
# Mesons only: the backward half is folded as the same state (time_parity=-1 -> sinh, else cosh).
def _get_masked_meson_sample(sample, tmax_fw, tmax_bw, time_parity):
    time_parity = _validate_time_parity(time_parity)
    N, n_corrs, T = sample.shape
    # backward = time-reverse each corr (flip then roll by 1)
    bw = np.roll(np.flip(sample, axis=2), 1, axis=2)
    bw[:, :, 1:] *= time_parity
    out = np.ma.empty((N, 2*n_corrs, T))
    out.mask = True
    out[:, :n_corrs, :tmax_fw] = sample[:, :, :tmax_fw]
    out.mask[:, :n_corrs, :tmax_fw] = False
    out[:, n_corrs:, :tmax_bw] = bw[:, :, :tmax_bw]
    out.mask[:, n_corrs:, :tmax_bw] = False
    return out


# get tmax for each src in forward and backward direction
def _get_tmax_fw_bw(tsrcs, bulk_range):
    tmin = bulk_range[0]
    tmax = bulk_range[-1]
    tmax_fw = tmax + 1 - np.array(tsrcs)
    tmax_bw = np.array(tsrcs) - tmin + 1
    return tmax_fw, tmax_bw


# mask excited states and time slices further away than tmax from source position
def _get_masked_corrs_boundary(corrs, tsrc, tmin_excited, tmax_from_tsrc=None):
    # get corrs in terms of original lattice
    corrs_aligned = np.roll(corrs, tsrc, axis=1)
    n_corrs = corrs.shape[0]
    # 1: create masked array
    corrs_ma = np.ma.empty((n_corrs, corrs.shape[1]) )
    corrs_ma.mask = True
    # 2: fill in all elements that are not excited states
    corrs_ma[:,:tsrc-(tmin_excited-1)] = corrs_aligned[:,:tsrc-(tmin_excited-1)]
    corrs_ma[:,tsrc+tmin_excited:] = corrs_aligned[:,tsrc+tmin_excited:]
    # 3: make sure all time slices that are further away than tmax from source position are masked
    if tmax_from_tsrc is not None:
        times = np.arange(corrs.shape[1])
        tmax_mask = np.abs(times - tsrc) > tmax_from_tsrc
        corrs_ma.mask = np.logical_or(corrs_ma.mask, tmax_mask[np.newaxis,:])
    return corrs_ma


def _fold_meson_boundary(eff_mass, time_parity):
    time_parity = _validate_time_parity(time_parity)
    half = len(eff_mass) // 2
    first_half = eff_mass[:half]
    second_half = np.flip(eff_mass[half:]) * time_parity
    return np.mean([first_half, second_half], axis=0)


def _flip_sign_boundary(eff_mass, tsrc):
    eff_mass[:tsrc] = -eff_mass[:tsrc]
    return eff_mass
