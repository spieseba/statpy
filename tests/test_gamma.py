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
        values_a, values_b = rng.normal(size=(2, *shape))
        np.testing.assert_allclose(gamma._lag_sum(values_a, values_b, max_lag),
                                   gamma._lag_sum(values_a, values_b, max_lag, fft=False), atol=1e-12)


def test_cross_covariance_matches_direct_pair_sum():
    rng = np.random.default_rng(0)
    # Replica 0 has gaps at 3, 6, 7; replica 1 reuses Monte Carlo times, so pairs across replicas would show.
    replicas = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    monte_carlo_times = np.array([0, 1, 2, 4, 5, 8, 3, 4, 6, 7])
    fluctuations_a, fluctuations_b = rng.normal(size=(2, 10, 2))
    # Default max_lag: 9 // 2 - 1 = 3; every lag up to it has pairs.
    expected = []
    for lag in range(4):
        # j is the later configuration: b is shifted by the lag.
        pairs = [(i, j) for i in range(10) for j in range(10)
                 if replicas[i] == replicas[j] and monte_carlo_times[j] - monte_carlo_times[i] == lag]
        # Mean over pairs = product sum / number of pairs.
        expected.append(np.mean([fluctuations_a[i] * fluctuations_b[j] for i, j in pairs], axis=0))
    np.testing.assert_allclose(gamma.cross_covariance(fluctuations_a, fluctuations_b, monte_carlo_times, replicas),
                               expected)


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
    # 2 |tau_int| sqrt((W + 1/2 - tau_int) / N) at W = 1; component 1 needs the abs
    tau_ints = np.array([[0.5, 1.0],
                         [0.5, -0.5]]).T
    errors = gamma.error_integrated_autocorrelation_times(tau_ints, 200)
    np.testing.assert_allclose(errors[1], [0.1, 0.1])
    # error sqrt((W + 1/2) / N) with N = 150
    np.testing.assert_allclose(gamma.error_of_error(np.array([2.0, 3.0]), np.array([1, 13]), 150), [0.2, 0.9])


def test_error_analysis_matches_exact_ar1():
    taus = np.array([0.5, 4.5])                     # one AR(1) chain per component
    num_configurations = 100_000
    values = ar1.generate_chains(taus, num_configurations, np.random.default_rng(0))
    _, fluctuations = gamma.mean_and_fluctuations(values, np.ones(num_configurations))
    result = gamma.error_analysis(fluctuations, np.full(num_configurations, "E"), np.arange(num_configurations),
                                  np.zeros(num_configurations, int), 1.5)
    # Wolff eq. 70: tau_int = tau; unit variance, so error = sqrt(2 tau / N)
    error = np.sqrt(2 * taus / num_configurations)
    # 3 sigma, with the gamma method's own error estimates
    assert np.all(np.abs(result.tau_int["E"] - taus) < 3 * result.error_tau_int["E"])
    assert np.all(np.abs(result.error - error) < 3 * result.error_of_error)


def test_rescaled_covariance_matches_exact_ar1():
    # a = x, b = 0.6 x + 0.8 y with independent unit-variance AR(1) chains x, y of tau_int 4: both components share one
    # normalized autocorrelation function, so the rescaled covariance is exact, C / N = 2 tau [[1, 0.6], [0.6, 1]] / N
    num_configurations = 100_000
    x, y = ar1.generate_chains(np.full(2, 4.0), num_configurations, np.random.default_rng(0)).T
    _, fluctuations = gamma.mean_and_fluctuations(np.stack([x, 0.6 * x + 0.8 * y], axis=1), np.ones(num_configurations))
    result = gamma.error_analysis(fluctuations, np.full(num_configurations, "E"), np.arange(num_configurations),
                                  np.zeros(num_configurations, int), 1.5)
    expected = 8 * np.array([[1.0, 0.6], [0.6, 1.0]]) / num_configurations
    # 3 sigma; covariance entries scale like error^2: relative error about 2 * error_of_error / error (Wolff eq. 40)
    rtol = 3 * 2 * np.max(result.error_of_error / result.error)
    np.testing.assert_allclose(gamma.rescaled_covariance(fluctuations, result.ensemble_errors["E"]), expected, rtol=rtol)


@pytest.mark.parametrize("taus, coefficients", [
    # a = x, b = 0.6 x + 0.8 y: one normalized autocorrelation function, C = [[8, 4.8], [4.8, 8]]
    ([4.0, 4.0], [[1.0, 0.0], [0.6, 0.8]]),
    # a = s + u, b = s + v, u and v white noise: C = [[9, 8], [8, 9]], correlation 8/9; Gamma(0) has correlation 1/2
    ([4.0, 0.5, 0.5], [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
])
def test_covariance_matches_exact_ar1(taus, coefficients):
    # each component is a linear combination of independent unit-variance AR(1) chains with tau_int = taus,
    # with one row of coefficients per component: C = 2 coefficients diag(taus) coefficients^T
    num_configurations = 100_000
    taus, coefficients = np.array(taus), np.array(coefficients)
    chains = ar1.generate_chains(taus, num_configurations, np.random.default_rng(0))
    _, fluctuations = gamma.mean_and_fluctuations(chains @ coefficients.T, np.ones(num_configurations))
    monte_carlo_times, replicas = np.arange(num_configurations), np.zeros(num_configurations, int)
    result = gamma.error_analysis(fluctuations, np.full(num_configurations, "E"), monte_carlo_times, replicas, 1.5)
    W = int(np.max(result.window["E"]))
    expected = 2 * coefficients @ np.diag(taus) @ coefficients.T / num_configurations
    # 3 sigma; covariance entries scale like error^2: relative error about 2 sqrt((W + 1/2) / N) (Wolff eq. 40)
    rtol = 3 * 2 * np.sqrt((W + 0.5) / num_configurations)
    np.testing.assert_allclose(gamma.covariance(fluctuations, monte_carlo_times, replicas, W), expected, rtol=rtol)


@pytest.mark.filterwarnings("error")
def test_replica_q_value_by_hand():
    # replica central values 1 and -1 (mean 0), N_r = 2, N = 4, error 1: chi^2 = (2 + 2) / 4 = 1 with 1 dof,
    # Q = erfc(1 / sqrt(2)); component 1 constant (error 0): NaN
    replica_central_values = np.array([[1.0, 0.0], [-1.0, 0.0]])
    q_value = gamma.replica_q_value(replica_central_values, np.array([2, 2]), np.array([1.0, 0.0]))
    np.testing.assert_allclose(q_value, [0.31731050786291404, np.nan])


@pytest.mark.filterwarnings("error")
def test_error_analysis_constant_component():
    rng = np.random.default_rng(0)
    fluctuations = np.stack([rng.normal(size=100), np.zeros(100)], axis=1)   # component 1 constant
    result = gamma.error_analysis(fluctuations, np.full(100, "E"), np.arange(100), np.zeros(100, int), 1.5)
    assert result.window["E"][1] == 0
    np.testing.assert_array_equal([result.error[1], result.error_of_error[1], result.tau_int["E"][1],
                                   result.error_tau_int["E"][1]], [0, 0, 0.5, 0])


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
    result = gamma.error_analysis(fluctuations, np.full(num_configurations, "E"), np.arange(num_configurations),
                                  np.zeros(num_configurations, int), 1.5)
    autocovariance, window = result.autocovariance["E"], result.window["E"]
    rho = gamma.autocorrelation(autocovariance)
    lags = [1, 4, 10, 20]
    error_gamma = gamma.error_autocovariance(autocovariance, window, num_configurations, max(lags))
    error_rho = gamma.error_autocorrelation(rho, window, num_configurations, max(lags))
    # each chain is one estimate; 3 sigma of the spread over chains, relative error 1 / sqrt(2 num_chains) (standard error of the standard deviation)
    rtol = 3 / np.sqrt(2 * num_chains)
    np.testing.assert_allclose(error_gamma[lags].mean(axis=1), autocovariance[lags].std(axis=1), rtol=rtol)
    np.testing.assert_allclose(error_rho[lags].mean(axis=1), rho[lags].std(axis=1), rtol=rtol)
