"""Tests for gamma."""

import numpy as np
import pytest

from statpy.simulation import ar1
from statpy.statistics import gamma


def test_mean_and_fluctuations_with_weights_by_hand():
    # mean = (1 + 2 + 2 * 3) / 4 = 9/4; delta = w (a - 9/4) / mean(w) with mean(w) = 4/3
    mean, fluctuations = gamma.mean_and_fluctuations(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 2.0]))
    np.testing.assert_allclose(mean, 9 / 4)
    np.testing.assert_allclose(fluctuations, [-15 / 16, -3 / 16, 18 / 16])


def test_lag_sum_fft_matches_direct_sum():
    rng = np.random.default_rng(0)
    for shape, max_lag in [((50,), 10), ((50, 3), 49), ((5, 3), 8)]:
        values = rng.normal(size=shape)
        np.testing.assert_allclose(gamma._lag_sum(values, max_lag),
                                   gamma._lag_sum(values, max_lag, fft=False), atol=1e-12)


def test_autocovariance_matches_direct_pair_sum():
    rng = np.random.default_rng(0)
    # Replica 0 has gaps at 3, 6, 7; replica 1 reuses Monte Carlo times, so pairs across replicas would show.
    replicas = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    monte_carlo_times = np.array([0, 1, 2, 4, 5, 8, 3, 4, 6, 7])
    fluctuations = rng.normal(size=(10, 2))
    # Default max_lag: 9 // 2 - 1 = 3; every lag up to it has pairs.
    expected = []
    for lag in range(4):
        pairs = [(i, j) for i in range(10) for j in range(10)
                 if replicas[i] == replicas[j] and monte_carlo_times[j] - monte_carlo_times[i] == lag]
        # Mean over pairs = product sum / number of pairs.
        expected.append(np.mean([fluctuations[i] * fluctuations[j] for i, j in pairs], axis=0))
    np.testing.assert_allclose(gamma.autocovariance(fluctuations, monte_carlo_times, replicas), expected)


def test_integrated_autocorrelation_times_by_hand():
    gamma_values = np.array([[1.0, 0.5, 0.25],
                             [2.0, 1.0, 0.5]]).T
    expected = np.array([[0.5, 1.0, 1.25],
                         [0.5, 1.0, 1.25]]).T
    np.testing.assert_allclose(gamma.integrated_autocorrelation_times(gamma_values), expected)


def test_window():
    tau_ints = np.array([[0.5, 0.4] + [1.0] * 8,    # tau_int <= 1/2 at W = 1
                         [0.5] + [1.0] * 9,         # g(7) > 0 > g(8)
                         [0.5] + [50.0] * 9]).T     # no sign change: largest W = 9
    with pytest.warns(UserWarning, match="No window found"):
        np.testing.assert_array_equal(gamma.window(tau_ints, 10_000, 1.5), [1, 8, 9])


def test_variance_and_errors_by_hand():
    gamma_values = np.array([[2.0, 4.0]])           # only gamma[0] enters
    tau_ints = np.array([[0.5, 1.0, 1.5, 2.0],
                         [0.5, 3.0, 4.5, 5.0]]).T
    W = np.array([1, 2])                            # tau_int(W) = 1.0 and 4.5
    # 2 tau_int gamma[0] (1 + (2W+1)/N) / N with N = 200
    np.testing.assert_allclose(gamma.variance(gamma_values, tau_ints, W, 200), [0.0203, 0.1845])
    # 2 tau_int sqrt(|W + 1/2 - tau_int| / N); component 1 needs the abs
    errors = gamma.error_integrated_autocorrelation_times(tau_ints, 200)
    np.testing.assert_allclose(errors[W, [0, 1]], [0.1, 0.9])
    # error sqrt((W + 1/2) / N) with N = 150
    np.testing.assert_allclose(gamma.error_of_error(np.array([2.0, 3.0]), np.array([1, 13]), 150), [0.2, 0.9])


def test_error_analysis_matches_exact_ar1():
    taus = np.array([0.5, 4.5])                     # one AR(1) chain per component
    num_configurations = 100_000
    values = ar1.generate_chains(taus, num_configurations, np.random.default_rng(0))
    _, fluctuations = gamma.mean_and_fluctuations(values, np.ones(num_configurations))
    result = gamma.error_analysis(fluctuations, np.arange(num_configurations), np.zeros(num_configurations, int), 1.5)
    # Wolff eq. 70: tau_int = tau; unit variance, so error = sqrt(2 tau / N)
    error = np.sqrt(2 * taus / num_configurations)
    # 3 sigma, with the gamma method's own error estimates
    assert np.all(np.abs(result.tau_int - taus) < 3 * result.error_tau_int)
    assert np.all(np.abs(result.error - error) < 3 * result.error_of_error)


@pytest.mark.filterwarnings("error")
def test_error_analysis_constant_component():
    rng = np.random.default_rng(0)
    fluctuations = np.stack([rng.normal(size=100), np.zeros(100)], axis=1)   # component 1 constant
    result = gamma.error_analysis(fluctuations, np.arange(100), np.zeros(100, int), 1.5)
    assert result.window[1] == 0
    np.testing.assert_array_equal([result.error[1], result.error_of_error[1], result.tau_int[1], result.error_tau_int[1]],
                                  [0, 0, 0.5, 0])


def test_autocovariance_independent_of_monte_carlo_time_step():
    fluctuations = np.random.default_rng(3).normal(size=(40, 2))
    replicas = np.repeat([0, 1], 20)
    times = np.concatenate([np.arange(20), np.arange(20)])
    times[25:] += 3  # gap in replica 1
    np.testing.assert_array_equal(gamma.autocovariance(fluctuations, 4 * times + 7, replicas),
                                  gamma.autocovariance(fluctuations, times, replicas))


def test_autocovariance_and_autocorrelation_errors_match_spread_over_chains():
    num_chains, num_configurations = 200, 1000
    chains = ar1.generate_chains(np.full(num_chains, 4.0), num_configurations, np.random.default_rng(0))
    _, fluctuations = gamma.mean_and_fluctuations(chains, np.ones(num_configurations))
    result = gamma.error_analysis(fluctuations, np.arange(num_configurations), np.zeros(num_configurations, int), 1.5)
    rho = gamma.autocorrelation(result.autocovariance)
    lags = [1, 4, 10, 20]
    error_gamma = gamma.error_autocovariance(result.autocovariance, result.window, num_configurations, max(lags))
    error_rho = gamma.error_autocorrelation(rho, result.window, num_configurations, max(lags))
    # each chain is one estimate; 3 sigma of the spread over chains, relative error 1 / sqrt(2 num_chains) (standard error of the standard deviation)
    rtol = 3 / np.sqrt(2 * num_chains)
    np.testing.assert_allclose(error_gamma[lags].mean(axis=1), result.autocovariance[lags].std(axis=1), rtol=rtol)
    np.testing.assert_allclose(error_rho[lags].mean(axis=1), rho[lags].std(axis=1), rtol=rtol)
