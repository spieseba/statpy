"""Tests for external inputs: variables without configurations, propagation of their covariance."""

import jax.numpy as jnp
import numpy as np

from statpy.simulation import ar1
from statpy.variable import Layout, Variable


def test_constant_enters_derive_at_its_central_value():
    rng = np.random.default_rng(0)
    x = 5.0 + ar1.generate_chains(np.array([1.0, 4.0]), 300, rng)
    data = Variable.from_data(x, Layout("A", np.arange(len(x))), rng.integers(0, len(x), (50, len(x))))
    constant = Variable.from_external("c", np.array([2.0, 3.0]))

    result = Variable.derive(lambda x, c: x * c**2, data, constant)
    expected = Variable.derive(lambda x: x * np.array([4.0, 9.0]), data)

    for field in ("central_value", "replica_central_values", "fluctuations", "jackknife_replicates",
                  "bootstrap_replicates"):
        np.testing.assert_allclose(getattr(result, field), getattr(expected, field), rtol=1e-14)


def _covariance(size, rng):
    a = rng.normal(size=(size, size))
    return a @ a.T


def test_linear_function_of_an_external_input_has_covariance_a_c_a_transpose():
    rng = np.random.default_rng(1)
    C = _covariance(3, rng)
    A = rng.normal(size=(2, 2, 3))
    c = Variable.from_external("c", rng.normal(size=3), C)

    # z is a 2 x 2 matrix; its covariance refers to z.reshape(-1)
    z = Variable.derive(lambda c: jnp.asarray(A) @ c, c)

    covariance = A.reshape(4, 3) @ C @ A.reshape(4, 3).T
    np.testing.assert_allclose(z.jackknife_covariance(), covariance, rtol=1e-12)
    np.testing.assert_allclose(z.jackknife_variance(), np.diag(covariance).reshape(2, 2), rtol=1e-12)
    np.testing.assert_allclose(z.gamma_method().error**2, np.diag(covariance).reshape(2, 2), rtol=1e-12)


def test_jacobians_of_the_same_external_input_are_added_before_the_variance():
    rng = np.random.default_rng(2)
    C = _covariance(2, rng)
    c = Variable.from_external("c", np.array([1.0, 2.0]), C)
    doubled = Variable.derive(lambda c: 2 * c, c)

    difference = Variable.derive(lambda a, b: a - 2 * b, doubled, c)
    total = Variable.derive(lambda a, b: a + b, doubled, c)

    np.testing.assert_allclose(difference.jackknife_variance(), 0, atol=1e-14)
    np.testing.assert_allclose(total.jackknife_covariance(), 9 * C, rtol=1e-12)


def test_external_variance_is_added_to_the_variance_from_the_configurations():
    rng = np.random.default_rng(3)
    x = 5.0 + ar1.generate_chains(np.array([1.0, 4.0]), 300, rng)
    data = Variable.from_data(x, Layout("A", np.arange(len(x))))
    mean, variance = 2.0, 0.09
    c = Variable.from_external("c", mean, [[variance]])

    z = Variable.derive(lambda x, c: x * c, data, c)
    constant = Variable.derive(lambda x, c: x * c, data, Variable.from_external("c", mean))

    external_variance = data.central_value**2 * variance
    np.testing.assert_allclose(z.jackknife_variance(external=False), constant.jackknife_variance(), rtol=1e-12)
    np.testing.assert_allclose(z.jackknife_variance(), constant.jackknife_variance() + external_variance, rtol=1e-12)
    np.testing.assert_allclose(z.gamma_method().error**2, constant.gamma_method().error**2 + external_variance,
                               rtol=1e-12)


def test_fit_of_a_constant_to_data_shifted_by_an_external_input():
    rng = np.random.default_rng(4)
    x = 5.0 + ar1.generate_chains(np.full(6, 1.0), 300, rng)
    data = Variable.from_data(x, Layout("A", np.arange(len(x))))
    C = _covariance(2, rng)
    c = Variable.from_external("c", np.array([3.0, 1.0]), C)
    # c[0] shifts the time slices 0 and 1, c[1] the time slices 2 to 5
    shifted = Variable.derive(lambda x, c: x + jnp.where(jnp.arange(6) < 2, c[0], c[1]), data, c)

    def model(t, parameters):
        return parameters[0] * jnp.ones(len(t))

    parameters = shifted.least_squares_fit(model, np.arange(2, 6), [6.0]).parameters

    # the fitted constant is a weighted mean of the time slices 2 to 5, so it shifts by c[1] alone
    np.testing.assert_allclose(parameters.jackknife_variance() - parameters.jackknife_variance(external=False),
                               C[1, 1], rtol=1e-10)


def test_drawn_bootstrap_replicates_have_the_given_mean_and_covariance():
    rng = np.random.default_rng(5)
    C = _covariance(2, rng)
    mean = np.array([1.0, -2.0])

    samples = Variable.from_external("c", mean, C, num_bootstraps=200_000).bootstrap_replicates
    assert samples is not None  # for the type checker

    # sampling errors: about sqrt(2 / 200000) = 0.3% of the variances, sqrt(variance / 200000) of the mean
    scale = np.max(np.diag(C))
    np.testing.assert_allclose(np.cov(samples.T), C, atol=0.03 * scale)
    np.testing.assert_allclose(np.mean(samples, axis=0), mean, atol=5 * np.sqrt(scale / 200_000))


def test_bootstrap_replicates_pair_data_and_external_input_by_sample():
    rng = np.random.default_rng(6)
    x = 5.0 + ar1.generate_chains(np.array([1.0, 4.0]), 300, rng)
    data = Variable.from_data(x, Layout("A", np.arange(len(x))), rng.integers(0, len(x), (50, len(x))))
    c = Variable.from_external("c", 2.0, [[0.09]], num_bootstraps=50)

    z = Variable.derive(lambda x, c: x * c, data, c)

    data_samples, c_samples, z_samples = data.bootstrap_replicates, c.bootstrap_replicates, z.bootstrap_replicates
    assert data_samples is not None and c_samples is not None and z_samples is not None  # for the type checker
    np.testing.assert_allclose(z_samples, data_samples * c_samples[:, None], rtol=1e-14)


def test_given_bootstrap_replicates_set_the_mean_and_the_covariance():
    rng = np.random.default_rng(7)
    samples = rng.normal(size=(500, 3)) @ rng.normal(size=(3, 3))
    A = rng.normal(size=(2, 3))

    c = Variable.from_external_replicates("c", samples)
    z = Variable.derive(lambda c: jnp.asarray(A) @ c, c)

    np.testing.assert_allclose(c.central_value, np.mean(samples, axis=0), rtol=1e-14)
    # for a linear function the linear propagation of the sample covariance equals the bootstrap covariance
    np.testing.assert_allclose(z.jackknife_covariance(), z.bootstrap_covariance(), rtol=1e-10)
