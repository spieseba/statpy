"""Tests for variables on multiple ensembles: errors are computed per ensemble and added."""

import numpy as np

from statpy.simulation import ar1
from statpy.statistics.core import bin as bin_data
from statpy.variable import Layout, Variable


def _data():
    rng = np.random.default_rng(0)
    x = 5.0 + ar1.generate_chains(np.array(2.0), 400, rng)           # ensemble A
    y = 3.0 + ar1.generate_chains(np.array([1.0, 4.0]), 250, rng)    # ensemble B, two components
    return x, y


def _sum(x, y):
    variables = (Variable.from_data(x, Layout("A", np.arange(len(x)))),
                 Variable.from_data(y, Layout("B", np.arange(len(y)))))
    return Variable.derive(lambda x, y: x + 2 * y, *variables)


def test_jackknife_variance_adds_the_variances_of_the_means():
    x, y = _data()
    z = _sum(x, y)

    # variance of a mean: sample variance / N, per ensemble
    variance = np.var(x, ddof=1) / len(x) + 4 * np.var(y, axis=0, ddof=1) / len(y)
    covariance = np.var(x, ddof=1) / len(x) + 4 * np.cov(y.T, ddof=1) / len(y)
    np.testing.assert_allclose(z.jackknife_variance(), variance, rtol=1e-12)
    np.testing.assert_allclose(z.jackknife_covariance(), covariance, rtol=1e-12)


def test_delayed_binning_uses_the_bin_size_of_each_ensemble():
    x, y = _data()
    z = _sum(x, y)

    # bin sizes divide N: the same as binning the data of each ensemble up front
    x_bins = bin_data(x, 4, np.ones(len(x)))
    y_bins = bin_data(y, 5, np.ones(len(y)))
    variance = np.var(x_bins, ddof=1) / len(x_bins) + 4 * np.var(y_bins, axis=0, ddof=1) / len(y_bins)
    np.testing.assert_allclose(z.jackknife_variance({"A": 4, "B": 5}), variance, rtol=1e-12)


def test_gamma_method_adds_the_variances_of_the_ensembles():
    x, y = _data()
    z = _sum(x, y)
    result = z.gamma_method()
    x_result = Variable.from_data(x, Layout("A", np.arange(len(x)))).gamma_method()
    y_result = Variable.from_data(y, Layout("B", np.arange(len(y)))).gamma_method()

    # z = x + 2 y: errors add in quadrature, tau_int and window are those of each ensemble
    error = np.sqrt(x_result.error**2 + 4 * y_result.error**2)
    np.testing.assert_allclose(result.error, error, rtol=1e-12)
    np.testing.assert_allclose(result.tau_int["A"], np.full(2, x_result.tau_int["A"]), rtol=1e-12)
    np.testing.assert_allclose(result.tau_int["B"], y_result.tau_int["B"], rtol=1e-12)
    # error of the error from independent ensembles: sqrt(sum_e sigma_e^2 dsigma_e^2) / sigma
    error_of_error = np.sqrt(x_result.error**2 * x_result.error_of_error**2
                             + (2 * y_result.error)**2 * (2 * y_result.error_of_error)**2) / error
    np.testing.assert_allclose(result.error_of_error, error_of_error, rtol=1e-12)


def test_bias_correction_adds_the_corrections_of_the_ensembles():
    x, y = _data()
    x_layout = Layout("A", np.concatenate([np.arange(250), np.arange(150)]), replicas=np.repeat([0, 1], [250, 150]))
    y_layout = Layout("B", np.concatenate([np.arange(100), np.arange(100), np.arange(50)]),
                      replicas=np.repeat([0, 1, 2], [100, 100, 50]))
    z = Variable.derive(lambda x, y: x / y, Variable.from_data(x, x_layout), Variable.from_data(y, y_layout))

    # per ensemble: value - (Fbar_e - value) / (R_e - 1), Fbar_e = sum_r N_r F_r / N_e; F_r from the
    # mean of replica r of ensemble e and the full mean of the other ensemble
    x_mean, y_mean = x.mean(), y.mean(axis=0)
    value = x_mean / y_mean
    x_bar = (250 * x[:250].mean() / y_mean + 150 * x[250:].mean() / y_mean) / 400
    y_bar = (100 * x_mean / y[:100].mean(axis=0) + 100 * x_mean / y[100:200].mean(axis=0)
             + 50 * x_mean / y[200:].mean(axis=0)) / 250
    corrected = value - (x_bar - value) / (2 - 1) - (y_bar - value) / (3 - 1)
    np.testing.assert_allclose(z.bias_corrected_value(), corrected, rtol=1e-12)
