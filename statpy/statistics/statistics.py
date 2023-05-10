from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import statpy as sp

# standard error
def ste(data, tau_int=0.5):
    return np.sqrt(2.0*tau_int / len(data) ) * np.std(data, ddof=1, axis=0) 

# binning
def bin(data, b, *argv):
    if len(argv) != 0:
        w = argv[0]
    Nb = len(data) // b # cut off data of last incomplete bin
    bata = []
    for i in range(Nb):
        if len(argv) != 0:
            mean = np.average(data[i*b:(i+1)*b], weights=w[i*b:(i+1)*b], axis=0)
        else:
            mean = np.mean(data[i*b:(i+1)*b], axis=0)
        bata.append(mean)
    return np.array(bata) 

####################### INFINITE BINSIZE EXTRAPOLATION ########################

# one parameter model for variance ratio var[S]/var[1]
# assumes single dominant markov mode tau ~ tau_int
# can fit from S0=1a
# model = 2\tau \left[1 - \frac{\tau}{S}\left(1 - e^{-S/\tau} \right)\right]
# gradient = 2 - \frac{4\tau}{S}\left(1 - e^{-S/\tau}\right) + 2 e^{-S/\tau}
# infinite binsize extrapolation: 
#   \lim_{S \rightarrow \infty} 2\tau \left[1 - \frac{\tau}{S}\left(1 - e^{-S/\tau} \right)\right] = 2\tau
#   \lim_{S \rightarrow \infty} \frac{\partial}{\partial\tau} 2\tau \left[1 - \frac{\tau}{S}\left(1 - e^{-S/\tau} \right)\right] = 2
class singlemode_model:
    def __init__(self):
        pass
    def __call__(self, t, p):
        return 2. * p[0] * (1. - p[0]/t * (1. - np.exp(-t/p[0])))
    def parameter_gradient(self, t, p):
        return np.array([2. - 4.*p[0]/t * (1. - np.exp(-t/p[0])) + 2.*np.exp(-t/p[0])], dtype=object)
    
# two parameter model for variance ratio var[S]/var[1]
# leading order expectation
# start fit at S0 ~ 2tau
class twoparameter_model:
    def __init__(self):
        pass
    def __call__(self, t, p):
        return 2. * p[0] * (1 - np.sqrt(p[1]**2.)/t)
    def parameter_gradient(self, t, p):
        return np.array([2.*(1 - np.sqrt(p[1]**2.)/t), -2.*p[0]/t])

# three paramater model for variance ratio var[S]/var[1]
# takes exponential corrections to 2tau(1 - c/S) into account
# start fit at S0 ~ 2tau
# model = f(\tau_{A,int}, c_A, d_A) = 2\tau_{A,int} \left(1 - \frac{c_A}{S} + \frac{d_A}{S} e^{-S/\tau_{A,int}}\right)
# parameter gradient = [ 2\left(1 - \frac{c_A}{S}\right) + 2d_A\left(\frac{1}{S} + \frac{1}{\tau_{A,int}}\right) e^{-S/\tau_{A,int}},
# - \frac{2\tau_{A,int}}{S}, \frac{e^{-S/\tau_{A,int}}}{S}]
# infinite binsize extrapolation: 
#   \lim_{S \rightarrow \infty} f(\tau_{A,int}, c_A, d_A) = 2\tau
#   \lim_{S \rightarrow \infty} parameter_gradient  = [2, 0, 0]
class threeparam_model:
    def __init__(self):
        pass
    def __call__(self, t, p):
        return 2. * p[0] * (1. - np.sqrt(p[1]**2.)/t  + np.sqrt(p[2]**2.)/t * np.exp(-t/p[0]))
    def parameter_gradient(self, t, p):
        return np.array([2.*(1. - np.sqrt(p[1]**2.)/t) + 2.*np.sqrt(p[2]**2.)*(1./t + 1./p[0])*np.exp(-t/p[0]), -2.*p[0]/t, np.exp(-t/p[0])/t], dtype=object)

# theoretical std estimate of var
def var_of_var(var, J):
    # J = N//b
    return (np.sqrt(2.*(J-1)/J**2.) * var)**2.

def ratio_var(a, a_var, b, b_var):
    # f = A/B
    return (a/b)**2. * (a_var/a**2. + b_var/b**2.)

def infinite_binsize_extrapolation(var_dict, N, binsizes_to_be_fitted, fit_model, p0, fit_method="Migrad", fit_params={}, make_plot=True):
    # theoretical estimate of var on var
    var_var = {b:var_of_var(var_dict[b], N//b) for b in var_dict}
    # compute variance ratio and propagate error
    var_ratio = {b:var_dict[b]/var_dict[1] for b in var_dict}
    var_ratio_var = {b:ratio_var(var_dict[b], var_var[b], var_dict[1], var_var[1]) for b in var_dict}
    # fit model to var_ratio
    fit_models = {"singlemode": singlemode_model, "twoparam": twoparameter_model, "threeparam": threeparam_model} 
    model = fit_models[fit_model]()
    fitter = sp.fitting.Fitter(t=binsizes_to_be_fitted,
                               C=np.diag(np.array([var_ratio_var[b] for b in binsizes_to_be_fitted])), 
                               model=model, estimator=lambda x : x, method=fit_method, minimizer_params=fit_params)
    try:
        fitter.fit(np.array([var_ratio[b] for b in binsizes_to_be_fitted]), p0)
        if fit_model == "singlemode": 
            ratio_inf_b = 2. * fitter.best_parameter[0] #; ratio_inf_b_var = np.array([2.0]) @ fitter.best_parameter_cov @ np.array([2.0])
            model_label = r"$2\tau \left[1 - \frac{\tau}{S}\left(1 - e^{-S/\tau} \right)\right]$"
        if fit_model == "twoparam":
            ratio_inf_b = 2. * fitter.best_parameter[0] 
            model_label = r"$2\tau_{A,int} \left(1 - \frac{c_A}{S}\right)$"
        if fit_model == "threeparam":
            ratio_inf_b = 2. * fitter.best_parameter[0] #; ratio_inf_b_var = np.array([2.0, 0, 0]) @ fitter.best_parameter_cov @ np.array([2.0, 0, 0])
            model_label = r"$2\tau_{A,int} \left(1 - \frac{c_A}{S} + \frac{d_A}{S} e^{-S/\tau_{A,int}}\right)$"
        if make_plot:
            # figure
            fig, ax = plt.subplots(figsize=(16,5))
            ax.set_ylabel(r"$\sigma^2[S]\,/\,\sigma^2[1]$")
            ax.set_xlabel(r"binsize S")
            # data
            ax.errorbar(var_ratio.keys(), var_ratio.values(), np.array(list(var_ratio_var.values()))**.5, 
                        linestyle="", marker="+", capsize=5, color="C0", label="data")
            # fit
            brange = np.arange(binsizes_to_be_fitted[0], binsizes_to_be_fitted[-1], 0.01)
            fb = np.array([model(b, fitter.best_parameter) for b in brange])
            #fb_std = np.array([fitter.fit_var(b)**.5 for b in brange])
            ax.plot(brange, fb, color="C1") 
            #ax.fill_between(brange, fb-fb_std, fb+fb_std, alpha=0.5, color="C1")
            # infinite binsize extrapolation
            ax.plot(brange, [ratio_inf_b for b in brange], color="C2", 
                    label=r"$2\tau = $" + f"{ratio_inf_b:.2f}, fit model: " + model_label) # +- {ratio_inf_b_var**.5:.2f}, fit model: " + model_label)
            #ax.fill_between(brange, ratio_inf_b-ratio_inf_b_var**.5, ratio_inf_b+ratio_inf_b_var**.5, alpha=0.5, color="C2")
            # optics
            ax.grid()
            ax.legend(loc="upper left")
            plt.tight_layout()
            plt.show()
    except AssertionError:
        print("Fitter did not converge")
        ratio_inf_b = []; ratio_inf_b_var = []
        if make_plot:
            fig, ax = plt.subplots(figsize=(16,5))
            color = "C0"
            ax.set_ylabel(r"$\sigma^2[S]\,/\,\sigma^2[1]$")
            ax.set_xlabel(r"binsize S")
            ax.errorbar(var_ratio.keys(), var_ratio.values(), np.array(list(var_ratio_var.values()))**.5, 
                        linestyle="", marker="+", capsize=5, color="C0", label="data")
            ax.grid()
            ax.legend(loc="upper left")
            plt.tight_layout()
            plt.show()
    return ratio_inf_b #, ratio_inf_b_var