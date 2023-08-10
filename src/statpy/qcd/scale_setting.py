#!/usr/bin/env python3

import numpy as np

class gradient_flow_scale:
    def __init__(self, ct0=0.3, cw0=0.3):
        # t0
        self.ct0 = ct0
        self.sqrt_t0_fm = 0.1638; self.sqrt_t0_fm_std = 0.0010 # taken from arXiv:1401.3270
        # w0
        self.cw0 = cw0
        self.w0_fm = 0.1670; self.w0_fm_std = 0.0010 # taken from arXiv:1401.3270

    def linear_interpolation(self, x, f, c):
        xb, xa = 0.0, 0.0
        fb, fa = 0.0, 0.0
        for i in range(len(x)):
            if f[i] < c:
                xb = x[i]
                fb = f[i]
            else:
                xa = x[i]
                fa = f[i]
                break
        assert xa != 0.0, f"c = {c} could not be reached"
        m = (c - fb) / (fa - fb)
        x0 = xa * m + xb * (1.0 - m)   
        return x0

    def derivative(self, f, dx, accuracy=2):
        assert accuracy in [2,4]
        if accuracy==2:
            return (np.roll(f, -1) - np.roll(f, 1)) / (2*dx)
        if accuracy==4:
            return ( 0.25 * np.roll(f, 2) - 2.0 * np.roll(f, 1) + 2.0 * np.roll(f, -1) - 0.25 * np.roll(f, -2) ) / (3.0 * dx)

    ########################################## t0 ###############################################
    
    def comp_t2E(self, tau, E):
        return tau**2. * E
    
    def comp_sqrt_tau0(self, tau, t2E):
        return np.sqrt(self.linear_interpolation(tau, t2E, self.ct0))
    
    def comp_sqrt_t0_GeV(self, sqrt_tau0, sqrt_t0_fm):
        return sqrt_tau0 * 0.1973 / sqrt_t0_fm
    
    def set_sqrt_tau0(self, tau, E):
        t2E = self.comp_t2E(tau, E)
        sqrt_tau0 = self.comp_sqrt_tau0(tau, t2E)
        return sqrt_tau0

    def set_sqrt_t0_GeV(self, tau, E, sqrt_t0_fm=None):
        if sqrt_t0_fm == None:
            sqrt_t0_fm = self.sqrt_t0_fm
        t2E = self.comp_t2E(tau, E)
        sqrt_tau0 = self.comp_sqrt_tau0(tau, t2E)
        return self.comp_sqrt_t0_GeV(sqrt_tau0, sqrt_t0_fm)

    ########################################## w0 ###############################################
    
    def comp_tdt2E(self, tau, E):
        t2E = self.comp_t2E(tau, E)
        return tau * self.derivative(f=t2E, dx=(tau[1]-tau[0]))

    def comp_omega0(self, tau, tdt2E):
        return np.sqrt(self.linear_interpolation(tau[1:-1], tdt2E[1:-1], self.cw0))
    
    def comp_w0_GeV(self, omega0, w0_fm):
        return omega0 * 0.1973 / w0_fm

    def set_omega0(self, tau, E):
        tdt2E = self.comp_tdt2E(tau, E)
        return self.comp_omega0(tau, tdt2E)

    def set_w0_GeV(self, tau, E, w0_fm=None):
        if w0_fm == None:
            w0_fm = self.w0_fm
        tdt2E = self.comp_tdt2E(tau, E)
        omega0 = self.comp_omega0(tau, tdt2E)
        return omega0 * 0.1973 / w0_fm
    
def db_local_gradient_flow_scale(db, leaf_prefix):
    scale = gradient_flow_scale()
    db.combine_sample(leaf_prefix + "/tau", leaf_prefix + "/E", f=scale.set_sqrt_tau0, dst_tag=leaf_prefix + "/local_sqrt_tau0")
    db.combine_sample(leaf_prefix + "/tau", leaf_prefix + "/E", f=scale.set_omega0, dst_tag=leaf_prefix + "/local_omega0")

def db_gradient_flow_scale(db, leaf_prefix, binsize, verbose=True):
    tau = db.database[leaf_prefix + "/tau"].mean
    scale = gradient_flow_scale()
    # tau0
    db.combine(leaf_prefix + "/E", f=lambda x: scale.set_sqrt_tau0(tau, x), dst_tag=leaf_prefix + "/sqrt_tau0")
    sqrt_tau0_var = db.jackknife_variance(leaf_prefix + "/sqrt_tau0", binsize)
    # t0
    db.combine(leaf_prefix + "/sqrt_tau0", f=lambda x: scale.comp_sqrt_t0_GeV(x, scale.sqrt_t0_fm), dst_tag=leaf_prefix + "/sqrt_t0_GeV") 
    sqrt_t0_GeV_stat_var = db.jackknife_variance(leaf_prefix + "/sqrt_t0_GeV", binsize)
    # propagate systematic error of t0
    sqrt_t0_GeV_mean_shifted = scale.comp_sqrt_t0_GeV(db.database[leaf_prefix + "/sqrt_tau0"].mean, scale.sqrt_t0_fm + scale.sqrt_t0_fm_std)
    db.propagate_sys_var(sqrt_t0_GeV_mean_shifted, dst_tag=leaf_prefix + "/sqrt_t0_GeV", sys_tag="sqrt_t0_GeV")
    sqrt_t0_GeV_sys_var = db.database[leaf_prefix + "/sqrt_t0_GeV"].misc["SYS_VAR_sqrt_t0_GeV"]
    # omega0
    db.combine(leaf_prefix + "/E", f=lambda x: scale.set_omega0(tau, x), dst_tag=leaf_prefix + "/omega0")
    omega0_var = db.jackknife_variance(leaf_prefix + "/omega0", binsize)
    # w0
    db.combine(leaf_prefix + "/omega0", f=lambda x: scale.comp_w0_GeV(x, scale.w0_fm), dst_tag=leaf_prefix + "/w0_GeV") 
    w0_GeV_stat_var = db.jackknife_variance(leaf_prefix + "/w0_GeV", binsize)
    # propagate systematic error of w0
    w0_GeV_mean_shifted = scale.comp_w0_GeV(db.database[leaf_prefix + "/omega0"].mean, scale.w0_fm + scale.w0_fm_std)
    db.propagate_sys_var(w0_GeV_mean_shifted, dst_tag=leaf_prefix + "/w0_GeV", sys_tag="w0_GeV")
    w0_GeV_sys_var = db.database[leaf_prefix + "/w0_GeV"].misc["SYS_VAR_w0_GeV"]
    if verbose:
        db.message(f"sqrt(tau0) = {db.database[leaf_prefix + '/sqrt_tau0'].mean:.4f} +- {sqrt_tau0_var**.5:.4f}")
        db.message(f"omega0 = {db.database[leaf_prefix + '/omega0'].mean:.4f} +- {omega0_var**.5:.4f}")
        db.message(f"sqrt(t0)/GeV (cutoff) = {db.database[leaf_prefix + '/sqrt_t0_GeV'].mean:.4f} +- {sqrt_t0_GeV_stat_var**.5:.4f} (STAT) +- {sqrt_t0_GeV_sys_var**.5:.4f} (SYS) [{(sqrt_t0_GeV_stat_var+sqrt_t0_GeV_sys_var)**.5:.4f} (STAT+SYS)]")
        db.message(f"w0/GeV (cutoff) = {db.database[leaf_prefix + '/w0_GeV'].mean:.4f} +- {w0_GeV_stat_var**.5:.4f} (STAT) +- {w0_GeV_sys_var**.5:.4f} (SYS) [{(w0_GeV_stat_var+w0_GeV_sys_var)**.5:.4f} (STAT+SYS)]")