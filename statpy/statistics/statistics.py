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
    Nb = len(data) // b # cuts off data which do not constitute a complete bin
    bata = []
    for i in range(Nb):
        if len(argv) != 0:
            mean = np.average(data[i*b:(i+1)*b], weights=w[i*b:(i+1)*b], axis=0)
        else:
            mean = np.mean(data[i*b:(i+1)*b], axis=0)
        bata.append(mean)
    return np.array(bata) 

####################### INFINITE BINSIZE EXTRAPOLATION ########################
# theoretical std estimate of var
def std_of_var(var, J):
    # J = N//b
    return np.sqrt(2.*(J-1)/J**2.) * var

def ratio_err_prop(A, A_var, B, B_var):
    # f = A/B
    return (A/B)**2.0 * (A_var/A**2.0 + B_var/B**2.0)


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


def infinite_binsize_extrapolation(bs, fit_bs, var_dict, N, model, p0, fit_method="Migrad", fit_params={}, make_plot=True):
    def fit(t, y, cov, model, p0):
        models = {"singlemode": singlemode_model, "threeparam": threeparam_model}
        mod = models[model]()
        fitter = sp.fitting.fit(t, y, cov, mod, p0, lambda x: x, method=fit_method, minimizer_params=fit_params)
        fitter.fit()
        best_parameter = fitter.best_parameter; best_parameter_cov = fitter.best_parameter_cov
        fit_err = lambda t: (mod.parameter_gradient(t, best_parameter) @ best_parameter_cov @ mod.parameter_gradient(t, best_parameter))**0.5
        return best_parameter, best_parameter_cov, mod, fit_err
    var = np.array([var_dict[b] for b in bs])
    var_var = np.array([std_of_var(var_dict[b], N//b) for b in bs])**2.
    var_ratio = var/var[0]
    var_ratio_var = np.array([ratio_err_prop(var[b-bs[0]], var_var[b-bs[0]], var[0], var_var[0]) for b in bs])
    try:
        best_parameter, best_parameter_cov, model_func, fit_err = fit(bs[fit_bs], var_ratio[fit_bs], np.diag(var_ratio_var[fit_bs]), model, p0)
        assert model in ["singlemode", "threeparam"]
        if model == "singlemode":
            ratio_inf = 2. * best_parameter[0]; ratio_inf_var = np.array([2.0]) @ best_parameter_cov @ np.array([2.0])
            model_label = r"$2\tau \left[1 - \frac{\tau}{S}\left(1 - e^{-S/\tau} \right)\right]$"
        if model == "threeparam":
            ratio_inf = 2. * best_parameter[0]; ratio_inf_var = np.array([2.0, 0, 0]) @ best_parameter_cov @ np.array([2.0, 0, 0])
            model_label = r"$2\tau_{A,int} \left(1 - \frac{c_A}{S} + \frac{d_A}{S} e^{-S/\tau_{A,int}}\right)$"
        if make_plot:
            fig, ax = plt.subplots(figsize=(16,5))
            color = "C0"
            ax.set_ylabel(r"$\sigma^2[S]\,/\,\sigma^2[1]$")
            ax.set_xlabel(r"binsize S")
            ax.errorbar(bs, var_ratio, var_ratio_var**.5, linestyle="", marker="+", capsize=5, color=color, label="data")
            color = "C1"
            trange = np.arange(bs[fit_bs[0]], bs[fit_bs[-1]], 0.01)
            fy = np.array([model_func(t, best_parameter) for t in trange]); fyerr = np.array([fit_err(t) for t in trange])
            ax.plot(trange, fy, color=color)
            ax.fill_between(trange, fy-fyerr, fy+fyerr, alpha=0.5, color=color)
            color = "C2"
            ax.plot(trange, [ratio_inf for t in trange], color=color, label=r"$2\tau = $" + f"{ratio_inf:.2f} +- {ratio_inf_var**.5:.2f}, fit model: " + model_label)
            ax.fill_between(trange, ratio_inf-ratio_inf_var**.5, ratio_inf+ratio_inf_var**.5, alpha=0.5, color=color)
            ax.grid()
            ax.legend(loc="upper left")
            plt.tight_layout()
            plt.show()
    except AssertionError:
        ratio_inf = []; ratio_inf_var = []
        if make_plot:
            fig, ax = plt.subplots(figsize=(16,5))
            color = "C0"
            ax.set_ylabel(r"$\sigma^2[S]\,/\,\sigma^2[1]$")
            ax.set_xlabel(r"binsize S")
            ax.errorbar(bs, var_ratio, var_ratio_var**.5, linestyle="", marker="+", capsize=5, color=color, label="data")
            ax.grid()
            ax.legend(loc="upper left")
            plt.tight_layout()
            plt.show()
    return ratio_inf, ratio_inf_var
