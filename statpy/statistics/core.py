#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from ..fitting.core import Fitter, model_prediction_var
from iminuit import Minuit

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
# for more details see: https://arxiv.org/pdf/2211.03744.pdf

# one parameter model for variance ratio var[S]/var[1]
# assumes single dominant markov mode tau ~ tau_int
# can fit from S0=1
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

def infinite_binsize_extrapolation(var_dict, N, fit_model, p0, make_plot=True):
    # theoretical estimate of var on var
    var_var = {b:var_of_var(var_dict[b], N//b) for b in var_dict}
    # compute variance ratio and propagate error
    var_ratio = {b:var_dict[b]/var_dict[1] for b in var_dict}
    var_ratio_var = {b:ratio_var(var_dict[b], var_var[b], var_dict[1], var_var[1]) for b in var_dict}
    # fit model to var_ratio
    fit_models = {"singlemode": singlemode_model, "twoparam": twoparameter_model, "threeparam": threeparam_model} 
    model = fit_models[fit_model]()
    # scan through all binsizes as lower boundary for fit range
    bs = sorted(list(var_dict.keys()))
    diff = 1.0
    best_parameter = np.zeros(len(p0)); best_parameter_cov = np.zeros((len(p0),len(p0))); binsizes_plot = np.arange(len(var_ratio))
    for b in bs:
        binsizes_to_be_fitted = np.arange(b, len(var_ratio))
        dof = len(binsizes_to_be_fitted) - len(best_parameter)
        if len(binsizes_to_be_fitted) < 2 or dof < 1:
            continue
        fitter = Fitter(C=np.diag(np.array([var_ratio_var[b] for b in binsizes_to_be_fitted])), model=model,
                        method="Migrad", minimizer_params=None)
        try:
            m = Minuit(lambda p: fitter.chi_squared(binsizes_to_be_fitted, p, np.array([var_ratio[b] for b in binsizes_to_be_fitted])), p0)
            m.migrad()
            assert m.valid == True
            new_diff = abs(2.0 * m.values[0] - b)
            print("fitted binsizes: ", binsizes_to_be_fitted)
            print(f"abs(2*tau_int - b_start) = abs({2.0*m.values[0]} - {b}) = {new_diff}")
            print(f"chi2 / dof = {m.fval} / {dof} = {m.fval/dof}")
            print("-------------------------------------------")
            if  new_diff < diff:
                best_parameter = np.array(m.values)
                best_parameter_cov = np.array(m.covariance)
                diff = new_diff
                binsizes_plot = binsizes_to_be_fitted
        except AssertionError:
            print("fitted binsizes: ", binsizes_to_be_fitted)
            print(f"Fitter did not converge.")
            print("-------------------------------------------")
            continue
    if fit_model == "singlemode": 
        ratio_inf_b = 2. * best_parameter[0]; ratio_inf_b_var = np.array([2.0]) @ best_parameter_cov @ np.array([2.0])
        model_label = r"$2\tau \left[1 - \frac{\tau}{S}\left(1 - e^{-S/\tau} \right)\right]$"
    if fit_model == "twoparam":
        ratio_inf_b = 2. * best_parameter[0] 
        model_label = r"$2\tau_{A,int} \left(1 - \frac{c_A}{S}\right)$"; ratio_inf_b_var = np.array([2.0, 0]) @ best_parameter_cov @ np.array([2.0, 0])
    if fit_model == "threeparam":
        ratio_inf_b = 2. * best_parameter[0]; ratio_inf_b_var = np.array([2.0, 0, 0]) @ best_parameter_cov @ np.array([2.0, 0, 0])
        model_label = r"$2\tau_{A,int} \left(1 - \frac{c_A}{S} + \frac{d_A}{S} e^{-S/\tau_{A,int}}\right)$"
    if make_plot:
        # figure
        fig, ax = plt.subplots(figsize=(12,5))
        ax.set_ylabel(r"$\sigma^2[S]\,/\,\sigma^2[1]$")
        ax.set_xlabel(r"binsize S")
        # data
        ax.errorbar(var_ratio.keys(), var_ratio.values(), np.array(list(var_ratio_var.values()))**.5, 
                    linestyle="", marker="+", capsize=5, color="C0", label="data")
        # fit
        brange = np.arange(binsizes_plot[0], binsizes_plot[-1], 0.01)
        fb = np.array([model(b, best_parameter) for b in brange])
        fb_std = np.array([model_prediction_var(b, best_parameter, best_parameter_cov, model.parameter_gradient) for b in brange])**.5
        ax.plot(brange, fb, color="C1") 
        ax.fill_between(brange, fb-fb_std, fb+fb_std, alpha=0.5, color="C1")
        # infinite binsize extrapolation
        ax.plot(brange, [ratio_inf_b for b in brange], color="C2", 
                label=r"$2\tau = $" + f"{ratio_inf_b:.2f} +- {ratio_inf_b_var**.5:.2f}, fit model: " + model_label)
        ax.fill_between(brange, ratio_inf_b-ratio_inf_b_var**.5, ratio_inf_b+ratio_inf_b_var**.5, alpha=0.5, color="C2")
        ax.legend(loc="upper left")
        plt.tight_layout()
        plt.show()