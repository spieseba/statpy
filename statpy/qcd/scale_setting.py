import numpy as np
from ..statistics.jackknife import variance, samples, variance_jks
import matplotlib.pyplot as plt

def derivative(farr, method="central", h=0.01):
    if method == "central":
        return (np.roll(farr, -1) - np.roll(farr, 1)) / (2*h)
    if method == "central4":
        return ( 0.25 * np.roll(farr, 2) - 2.0 * np.roll(farr, 1) + 2.0 * np.roll(farr, -1) - 0.25 * np.roll(farr, -2) ) / (3.0 * h)

def ratio_error_prop(a, std_a, b, std_b):
    return a**2./b**2. * (std_a**2./a**2. + std_b**2./b**2.)

class scale:
    """
    scale setting class using flow scales t0 and w0

        Parameters:
        -----------
                tau (numpy array): flow time array
                E (numpy array): 2D array containing energy density measurements on every configuration for every evaluated flow time tau. axis 0: data, axis 1: flow time
                
    """
    def __init__(self, tau, E, nskip=0, ct0=0.3, cw0=0.3):
        self.tau = tau
        self.E = E[nskip:]
        self.E_mean = np.mean(self.E, axis=0)
        self.ct0 = ct0        
        self.cw0 = cw0
        # taken from arXiv:1401.3270
        self.sqrt_t0_fm = 0.1638; self.sqrt_t0_fm_std = 0.0010
        self.w0_fm = 0.1670; self.w0_fm_std = 0.0010 # taken from arXiv:1401.3270

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
            self.make_plot()
            return None
        m = (c - fb) / (fa - fb)
        x0 = xa * m + xb * (1.0 - m)   
        return x0
    
    ########################################## t0 ###############################################
    
    def comp_t2E(self, E_mean):
        return self.tau**2 * E_mean 
    
    def comp_sqrt_tau0(self, t2E):
        return np.sqrt(self.lin_interp(self.tau, t2E, self.ct0))
    
    def comp_aGeV_inv_t0(self, sqrt_tau0):
        return sqrt_tau0 * 0.1973 / self.sqrt_t0_fm
    
    def comp_aGeV_inv_t0_std(self, aGeV_inv, sqrt_tau0, sqrt_tau0_std): 
        # afm = sqrt_t0 / sqrt_tau0
        afm_std = np.sqrt( ratio_error_prop(self.sqrt_t0_fm, self.sqrt_t0_fm_std, sqrt_tau0, sqrt_tau0_std)  )
        aGeV_std = afm_std / 0.1973; aGeV_inv_std = abs(aGeV_std * aGeV_inv**2)
        return aGeV_inv_std
    
    def get_cutoff_t0(self, sqrt_tau0, sqrt_tau0_std, verbose=False):
        aGeV_inv = self.comp_aGeV_inv_t0(sqrt_tau0)
        aGeV_inv_std = self.comp_aGeV_inv_t0_std(aGeV_inv, sqrt_tau0, sqrt_tau0_std)
        afm = self.sqrt_t0_fm / sqrt_tau0 
        afm_std = np.sqrt( ratio_error_prop(self.sqrt_t0_fm, self.sqrt_t0_fm_std, sqrt_tau0, sqrt_tau0_std)  )
        if verbose:
            print(f"scale: sqrt(tau0) = {sqrt_tau0:.6f} +- {sqrt_tau0_std:.6f}")
            print(f"cutoff: {aGeV_inv:.6f} +- {aGeV_inv_std:.6f} GeV")
            print(f"lattice spacing: {afm/0.1973:.6f} +- {afm_std/0.1973:.6f} 1/GeV")
            print(f"lattice spacing: {afm:.6f} +- {afm_std:.6f} fm")  
        return aGeV_inv, aGeV_inv_std

    def comp_t0_phys(self, E_mean):
        return self.comp_sqrt_tau0(self.comp_t2E(E_mean)) * 0.1973 / self.sqrt_t0_fm

    def sample_gaussian_sqrt_t0_fm(self, N, seed=0):
        np.random.seed(seed)
        return np.random.normal(loc=self.sqrt_t0_fm, scale=self.sqrt_t0_fm_std, size=N)

    ########################################## w0 ###############################################
    
    def comp_tdt2E(self, E_mean):
        t2E = self.comp_t2E(E_mean)
        return self.tau * derivative(t2E, "central", self.tau[0])

    def comp_wau0(self, tdt2E):
        return np.sqrt(self.lin_interp(self.tau[1:-1], tdt2E[1:-1], self.cw0))
    
    def comp_aGeV_inv_w0(self, wau0):
        return wau0 * 0.1973 / self.w0_fm

    def comp_aGeV_inv_w0_std(self, aGeV_inv, wau0, wau0_std):
        # afm = w0/wau0
        afm_std = np.sqrt( ratio_error_prop(self.w0_fm, self.w0_fm_std, wau0, self.wau0_std)  )
        aGeV_std = afm_std / 0.1973; aGeV_inv_std = abs(aGeV_std * aGeV_inv**2) 
        return aGeV_inv_std

    def get_cutoff_w0(self, wau0, wau0_std, verbose=False):
        aGeV_inv = self.comp_aGeV_inv_w0(wau0)
        aGeV_inv_std = self.comp_aGeV_inv_w0_std(aGeV_inv, wau0, wau0_std)
        afm = self.w0_fm / wau0
        afm_std = np.sqrt( ratio_error_prop(self.w0_fm, self.w0_fm_std, wau0, self.wau0_std)  )
        if verbose:
            print(f"scale: wau0 = {wau0:.6f} +- {wau0_std:.6f}")
            print(f"cutoff: {aGeV_inv:.6f} +- {aGeV_inv_std:.6f} GeV")
            print(f"lattice spacing: {afm/0.1973:.6f} +- {afm_std/0.1973:.6f} 1/GeV")
            print(f"lattice spacing: {afm:.6f} +- {afm_std:.6f} fm")  
        return aGeV_inv, aGeV_inv_std

    def comp_w0_phys(self, E_mean):
        return self.comp_wau0(self.comp_tdt2E(E_mean)) * 0.1973 / self.w0_fm

    ###################################################################################################################################

    def lattice_spacing(self, scale):
        self.scale = scale
        if self.scale == "t0":
            self.t2E = self.comp_t2E(self.E_mean)
            self.t2E_jks = samples(self.comp_t2E, self.E)
            self.t2E_std = np.sqrt(variance_jks(self.t2E, self.t2E_jks))        
        
            self.sqrt_tau0 = self.comp_sqrt_tau0(self.t2E)
            self.sqrt_tau0_jks = np.array([self.comp_sqrt_tau0(t2E) for t2E in self.t2E_jks])
            self.sqrt_tau0_std = np.sqrt(variance_jks(self.sqrt_tau0, self.sqrt_tau0_jks))
            
            self.aGeV_inv_t0, self.aGeV_inv_t0_std = self.get_cutoff_t0(self.sqrt_tau0, self.sqrt_tau0_std, verbose=True)
            self.aGeV_inv_t0_jks = np.array([self.comp_aGeV_inv_t0(sqrt_tau0) for sqrt_tau0 in self.sqrt_tau0_jks])

            #self.sqrt_t0_fake_jks = self.sample_gaussian_sqrt_t0_fm(len(self.sqrt_tau0_jks), seed=0)
            #def tmp(sqrt_tau0, sqrt_t0):
            #    return sqrt_tau0 * 0.1973 / sqrt_t0
            #self.aGeV_inv_t0_fake_jks = np.array([tmp(self.sqrt_tau0_jks[j], self.sqrt_t0_fake_jks[j]) for j in range(len(self.sqrt_tau0_jks))])
            #self.aGeV_inv_t0_fake = tmp(self.sqrt_tau0, np.mean(self.sqrt_t0_fake_jks))
            #self.aGeV_inv_t0_fake_std = np.sqrt(variance_jks(self.aGeV_inv_t0_fake, self.aGeV_inv_t0_fake_jks))
            #print(f"cutoff fake: {self.aGeV_inv_t0_fake:.6f} +- {self.aGeV_inv_t0_fake_std:.6f}")

            return self.sqrt_tau0, self.sqrt_tau0_std, self.t2E, self.t2E_std, self.aGeV_inv_t0, self.aGeV_inv_t0_std, self.aGeV_inv_t0_jks


        if self.scale == "w0":
            self.tdt2E = self.comp_tdt2E(self.E_mean)
            self.tdt2E_jks = samples(self.comp_tdt2E, self.E)
            self.tdt2E_std = np.sqrt(variance_jks(self.tdt2E, self.tdt2E_jks))

            self.wau0 = self.comp_wau0(self.tdt2E)
            self.wau0_jks = np.array([self.comp_wau0(tdt2E) for tdt2E in self.tdt2E_jks])
            self.wau0_std = np.sqrt(variance_jks(self.wau0, self.wau0_jks))
            
            self.aGeV_inv_w0, self.aGeV_inv_w0_std = self.get_cutoff_w0(self.wau0, self.wau0_std, verbose=True)
            self.aGeV_inv_w0_jks = np.array([self.comp_aGeV_inv_w0(wau0) for wau0 in self.wau0_jks])

            return self.wau0, self.wau0_std, self.tdt2E, self.tdt2E_std, self.aGeV_inv_w0, self.aGeV_inv_w0_std, self.aGeV_inv_w0_jks

