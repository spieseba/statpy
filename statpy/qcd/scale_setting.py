import numpy as np
from ..statistics.jackknife import jackknife


def derivative(farr, method="central", h=0.01):
    if method == "central":
        return (np.roll(farr, -1) - np.roll(farr, 1)) / (2*h)
    if method == "central4":
        return ( 0.25 * np.roll(farr, 2) - 2.0 * np.roll(farr, 1) + 2.0 * np.roll(farr, -1) - 0.25 * np.roll(farr, -2) ) / (3.0 * h)

class scale:
    """
    scale setting class using flow scales t0 and w0

        Parameters:
        -----------
                tau (numpy array): flow time array
                E (numpy array): 2D array containing energy density measurements on every configuration for every evaluated flow time tau. axis 0: data, axis 1: flow time
                
    """
    def __init__(self, tau, E, ct0=0.3, cw0=0.3):
        self.tau = tau
        self.E = E
        self.E_mean = np.mean(self.E, axis=0)
        self.ct0 = ct0        
        self.cw0 = cw0

    def comp_t2E(self, E_mean):
        return self.tau**2 * E_mean 

    def comp_tdt2E(self, E_mean):
        t2E = self.comp_t2E(E_mean)
        return self.tau * derivative(t2E, "central", self.tau[0])

    def lin_interp(self, x, f, c):
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
        if xa == 0.0:
            print(f"c = {c} not reached -- check plot")
            return None
        m = (c - fb) / (fa - fb)
        x0 = xa * m + xb * (1.0 - m)   
        return x0

    def comp_tau0(self, t2E):
        return self.lin_interp(self.tau, t2E, self.ct0) 

    def comp_wau0(self, tdt2E):
        return np.sqrt(self.lin_interp(self.tau[1:-1], tdt2E[1:-1], self.cw0))

    def get_cutoff_t0(self, tau0, tau0_std, verbose=False):
        sqrt_t0_fm = 0.1638; sqrt_t0_fm_err = 0.0010 # taken from arXiv:1401.3270
        afm = sqrt_t0_fm / np.sqrt(tau0) 
        afm_std = np.sqrt( sqrt_t0_fm_err**2 / tau0 + tau0_std**2 * sqrt_t0_fm**2 / (4 * tau0**3) )
        aGeV = afm / 0.1973; aGeV_std = afm_std / 0.1973
        aGeV_inv = 1.0/aGeV; aGeV_inv_std = abs(aGeV_std/aGeV**2)
        if verbose:
            print(f"scale: tau0 = {tau0:.6f} +- {tau0_std:.6f}")
            print(f"cutoff: {aGeV_inv:.6f} +- {aGeV_inv_std:.6f} GeV")
            print(f"lattice spacing: {aGeV:.6f} +- {aGeV_std:.6f} 1/GeV")
            print(f"lattice spacing: {afm:.6f} +- {afm_std:.6f} fm")  
        return aGeV_inv, aGeV_inv_std

    def get_cutoff_w0(self, wau0, wau0_std, verbose=False):
        w0_fm = 0.1670; w0_fm_err = 0.0010 # taken from arXiv:1401.3270
        afm = w0_fm / wau0
        afm_std = np.sqrt( w0_fm_err**2 / wau0**2 + wau0_std**2 * w0_fm**2 / wau0**4  )
        aGeV = afm / 0.1973; aGeV_std = afm_std / 0.1973
        aGeV_inv = 1.0/aGeV; aGeV_inv_std = abs(aGeV_std/aGeV**2)
        if verbose:
            print(f"scale: wau0 = {wau0:.6f} +- {wau0_std:.6f}")
            print(f"cutoff: {aGeV_inv:.6f} +- {aGeV_inv_std:.6f} GeV")
            print(f"lattice spacing: {aGeV:.6f} +- {aGeV_std:.6f} 1/GeV")
            print(f"lattice spacing: {afm:.6f} +- {afm_std:.6f} fm")  
        return aGeV_inv, aGeV_inv_std

    def comp_t0_phys(self):
        return np.sqrt(self.comp_tau0(self.comp_t2E(self.E_mean))) * 0.1973 / 0.1638 # taken from arXiv:1401.3270

    def comp_w0_phys(self):
        return self.comp_wau0(self.comp_tdt2E(self.E_mean)) * 0.1973 / 0.1670 # taken from arXiv:1401.3270
        
    def lattice_spacing(self, scale):
        jk = jackknife(self.E)
        if scale == 't0':
            self.t2E = self.comp_t2E(self.E_mean); self.t2E_std = np.sqrt( jk.variance(lambda E_mean: self.comp_t2E(E_mean)) )
            self.tau0 = self.comp_tau0(self.t2E); self.tau0_std = np.sqrt( jk.variance(lambda E_mean: self.comp_tau0(self.comp_t2E(E_mean))) )
            self.aGeV_inv_t0, self.aGeV_inv_std_t0 = self.get_cutoff_t0(self.tau0, self.tau0_std, verbose=True)
            return self.tau0, self.tau0_std, self.t2E, self.t2E_std, self.aGeV_inv_t0, self.aGeV_inv_std_t0

        if scale == 'w0':
            self.tdt2E = self.comp_tdt2E(self.E_mean); self.tdt2E_std = np.sqrt( jk.variance(lambda E_mean: self.comp_tdt2E(E_mean)) )
            self.wau0 = self.comp_wau0(self.tdt2E); self.wau0_std = np.sqrt( jk.variance(lambda E_mean: self.comp_wau0(self.comp_tdt2E(E_mean))) )
            self.aGeV_inv_w0, self.aGeV_inv_std_w0 = self.get_cutoff_w0(self.wau0, self.wau0_std, verbose=True)
            return self.wau0, self.wau0_std, self.tdt2E, self.tdt2E_std, self.aGeV_inv_w0, self.aGeV_inv_std_w0



