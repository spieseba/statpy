import numpy as np 
from statpy.log import message


class gradient_flow_scale:
    def __init__(self, ct0=0.3, cw0=0.3):
        # t0
        self.ct0 = ct0
        self.sqrt_t0_fm = 0.1638; self.sqrt_t0_fm_std = 0.0010 # quenched QCD value - taken from arXiv:1401.3270
        # w0
        self.cw0 = cw0
        self.w0_fm = 0.1670; self.w0_fm_std = 0.0010 # quenched QCD value - taken from arXiv:1401.3270

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

    def compute_t2E(self, tau, E):
        return tau**2. * E
    
    def compute_sqrt_tau0(self, tau, E):
        t2E = self.compute_t2E(tau, E)
        return np.sqrt(self.linear_interpolation(tau, t2E, self.ct0))
    
    # sqrt_t0 = 0.1638 fm, sqrt_tau0 = sqrt_t0 / a ==> a[fm] = sqrt_t0 / sqrt_tau0
    def compute_a_fm_from_sqrt_tau0(self, sqrt_tau0, sqrt_t0_fm=None):
        if sqrt_t0_fm is None:
            sqrt_t0_fm = self.sqrt_t0_fm
        return sqrt_t0_fm / sqrt_tau0

    # 0.1973 GeV fm = 1, a = X fm ==> a^-1 = 1/ (X fm) = 0.1973 / (X 0.1973 fm) = 0.1973 GeV / X  
    def compute_a_inv_GeV_from_sqrt_tau0(self, sqrt_tau0, sqrt_t0_fm=None):
        return 0.1973 / self.compute_a_fm_from_sqrt_tau0(sqrt_tau0, sqrt_t0_fm)
    
    ########################################## w0 ###############################################
    
    def compute_tdt2E(self, tau, E):
        t2E = self.compute_t2E(tau, E)
        return tau * self.derivative(f=t2E, dx=(tau[1]-tau[0]))

    def compute_omega0(self, tau, E):
        tdt2E = self.compute_tdt2E(tau, E)
        return np.sqrt(self.linear_interpolation(tau[1:-1], tdt2E[1:-1], self.cw0))
     
    # w0 = 0.1670 fm, omega0 = w0 / a ==> a[fm] = w0/omega0
    def compute_a_fm_from_omega0(self, omega0, w0_fm=None):
        if w0_fm is None:
            w0_fm = self.w0_fm
        return w0_fm / omega0

    # 0.1973 GeV fm = 1, a = X fm ==> a^-1 = 1/ (X fm) = 0.1973 / (X 0.1973 fm) = 0.1973 GeV / X 
    def compute_a_inv_GeV_from_omega0(self, omega0, w0_fm=None):
        return 0.1973 / self.compute_a_fm_from_omega0(omega0, w0_fm)
    

def db_local_gradient_flow_scale(db, leaf_prefix, scale_type="both"):
    assert scale_type in ["sqrt_tau0", "omega0", "both"]
    scale = gradient_flow_scale()
    if scale_type in ["sqrt_tau0", "both"]: 
        db.combine_sample(leaf_prefix + "/tau", leaf_prefix + "/E", f=scale.compute_sqrt_tau0, dst_tag=leaf_prefix + "/local_sqrt_tau0")
    if scale_type in ["omega0", "both"]:     
        db.combine_sample(leaf_prefix + "/tau", leaf_prefix + "/E", f=scale.compute_omega0, dst_tag=leaf_prefix + "/local_omega0")

def db_gradient_flow_scale(db, leaf_prefix, binsize, verbose=True):
    tau = db.database[leaf_prefix + "/tau"].mean
    scale = gradient_flow_scale()
    ### sqrt_tau0 ###
    db.combine(leaf_prefix + "/E", f=lambda x: scale.compute_sqrt_tau0(tau, x), dst_tag=leaf_prefix + "/sqrt_tau0")
    ## a_inv_GeV from sqrt_tau0 ##
    mn, jks, _ = db.combine(leaf_prefix + "/sqrt_tau0", f=lambda x: scale.compute_a_inv_GeV_from_sqrt_tau0(x, scale.sqrt_t0_fm))
    mn_shifted = scale.compute_a_inv_GeV_from_sqrt_tau0(db.database[leaf_prefix + "/sqrt_tau0"].mean, scale.sqrt_t0_fm + scale.sqrt_t0_fm_std)
    misc = db.add_sys(mn, mn_shifted, "sqrt_t0")
    db.add_leaf(leaf_prefix + "/sqrt_tau0/a_inv_GeV", mn, jks, None, misc)
    if verbose:
        message(f"sqrt(tau0) = {db.database[leaf_prefix + '/sqrt_tau0'].mean:.4f} +- {db.jackknife_variance(leaf_prefix + '/sqrt_tau0', binsize)**.5:.4f}")
        message(f"a_inv_GeV from sqrt(tau0) (cutoff) = {db.database[leaf_prefix + '/sqrt_tau0/a_inv_GeV'].mean:.4f} +- {db.jackknife_variance(leaf_prefix + '/sqrt_tau0/a_inv_GeV', binsize)**.5:.4f} (STAT) +- {db.get_sys_var(leaf_prefix + '/sqrt_tau0/a_inv_GeV')**.5:.4f} (SYS) [{db.get_tot_var(leaf_prefix + '/sqrt_tau0/a_inv_GeV', binsize)**.5:.4f} (STAT+SYS)]")
    
    ### omega0 ###
    db.combine(leaf_prefix + "/E", f=lambda x: scale.compute_omega0(tau, x), dst_tag=leaf_prefix + "/omega0")
    ## a_inv_GeV from omega0 ##
    mn, jks, _ = db.combine(leaf_prefix + "/omega0", f=lambda x: scale.compute_a_inv_GeV_from_omega0(x, scale.w0_fm))
    mn_shifted = scale.compute_a_inv_GeV_from_omega0(db.database[leaf_prefix + "/omega0"].mean, scale.w0_fm + scale.w0_fm_std)
    misc = db.add_sys(mn, mn_shifted, "w0")
    db.add_leaf(leaf_prefix + "/omega0/a_inv_GeV", mn, jks, None, misc)
    if verbose:
        message(f"omega0 = {db.database[leaf_prefix + '/omega0'].mean:.4f} +- {db.jackknife_variance(leaf_prefix + '/omega0', binsize)**.5:.4f}")
        message(f"a_inv_GeV from omega0 (cutoff) = {db.database[leaf_prefix + '/omega0/a_inv_GeV'].mean:.4f} +- {db.jackknife_variance(leaf_prefix + '/omega0/a_inv_GeV', binsize)**.5:.4f} (STAT) +- {db.get_sys_var(leaf_prefix + '/omega0/a_inv_GeV')**.5:.4f} (SYS) [{(db.get_tot_var(leaf_prefix + '/omega0/a_inv_GeV', binsize))**.5:.4f} (STAT+SYS)]")

    ### difference between sqrt_t0 and omega0 ###
    diff_func = lambda x,y: abs(x-y)
    db.combine(leaf_prefix + "/sqrt_tau0", leaf_prefix + "/omega0", f=diff_func, dst_tag=leaf_prefix + "/abs(sqrt_tau0 - omega0)")

    ## difference between a_inv_GeVs from sqrt_t0 and omega0 ##
    db.combine(leaf_prefix + "/sqrt_tau0/a_inv_GeV", leaf_prefix + "/omega0/a_inv_GeV", f=diff_func, dst_tag=leaf_prefix + "/abs(sqrt_tau0/a_inv_GeV - omega0/a_inv_GeV)")
    if verbose:
        message(f"abs(sqrt_tau0 - omega0) = {db.database[leaf_prefix + '/abs(sqrt_tau0 - omega0)'].mean:.4f} +- {db.jackknife_variance(leaf_prefix + '/abs(sqrt_tau0 - omega0)', binsize)**.5:.4f}")
        message(f"abs(sqrt_tau0/a_inv_GeV - omega0/a_inv_GeV) = {db.database[leaf_prefix + '/abs(sqrt_tau0/a_inv_GeV - omega0/a_inv_GeV)'].mean:.4f} +- {db.jackknife_variance(leaf_prefix + '/abs(sqrt_tau0/a_inv_GeV - omega0/a_inv_GeV)', binsize)**.5:.4f} (STAT) +- {db.get_sys_var(leaf_prefix + '/abs(sqrt_tau0/a_inv_GeV - omega0/a_inv_GeV)')**.5:.4f} (SYS) [{(db.get_tot_var(leaf_prefix + '/abs(sqrt_tau0/a_inv_GeV - omega0/a_inv_GeV)', binsize))**.5:.4f} (STAT+SYS)]")

    ### ratio between sqrt_t0 and omega0 ###
    ratio_func = lambda x,y: x/y 
    db.combine(leaf_prefix + "/sqrt_tau0", leaf_prefix + "/omega0", f=ratio_func, dst_tag=leaf_prefix + "/sqrt_tau0_by_omega0")
    if verbose:
        message(f"sqrt_tau0_by_omega0 = {db.database[leaf_prefix + '/sqrt_tau0_by_omega0'].mean:.4f} +- {db.jackknife_variance(leaf_prefix + '/sqrt_tau0_by_omega0', binsize)**.5:.4f}")
