"""Tests for Variable.derive."""

import jax.numpy as jnp
import numpy as np
import pytest

from statpy.statistics import gamma
from statpy.variable import Layout, Variable


def test_weighted_ratio_matches_mean_and_fluctuations():
    num_configurations = 10
    rng = np.random.default_rng(0)
    a = rng.normal(size=num_configurations)
    w = rng.uniform(size=num_configurations)
    mean, fluctuations = gamma.mean_and_fluctuations(a, w)

    layout = Layout("E", np.arange(num_configurations))
    weighted = Variable.from_data(a * w, layout)
    weights = Variable.from_data(w, layout)

    # Linearising A / W around the means gives the weighted fluctuations w (a - mean) / mean(w).
    ratio = Variable.derive(lambda x, y: x / y, weighted, weights)

    np.testing.assert_allclose(ratio.central_value, mean)
    assert ratio.fluctuations is not None
    np.testing.assert_allclose(ratio.fluctuations, fluctuations, rtol=1e-12, atol=1e-14)


def _fewer_outputs(x, y):
    return jnp.array([jnp.sin(x[0]) * y[1, 2], jnp.cosh(y[1, 2])])


def _more_outputs(x, y):
    return jnp.array([
        [y[2, 1], 0, jnp.sin(x[1])],
        [0, 0, x[0] * jnp.cosh(y[1, 3])],
        [0, 0, 0],
        [jnp.sinh(x[1]) * jnp.cosh(y[1, 1]), 0, y[1, 2]],
        [0, x[1] * y[0, 0], 0],
    ])


# 2 and 15 outputs from 2 + 12 input components: fewer and more outputs than inputs.
@pytest.mark.parametrize("func", [_fewer_outputs, _more_outputs], ids=["fewer_outputs", "more_outputs"])
def test_derived_fluctuations_match_finite_difference_projection(func):
    num_configurations = 10
    rng = np.random.default_rng(0)
    x_observations = rng.normal(size=(num_configurations, 2))
    y_observations = rng.normal(size=(num_configurations, 3, 4))

    layout = Layout("E", np.arange(num_configurations))
    x = Variable.from_data(x_observations, layout)
    y = Variable.from_data(y_observations, layout)
    derived = Variable.derive(func, x, y)

    # Both inputs as one vector of 2 + 12 components; func_of_vector splits it back into x and y.
    observations = np.concatenate([x_observations, y_observations.reshape(num_configurations, 12)], axis=1)
    mean = observations.mean(axis=0)
    fluctuations = observations - mean

    def func_of_vector(vector):
        return np.asarray(func(vector[:2], vector[2:].reshape(3, 4)))

    # Sum over components of the fluctuation times the central-difference derivative of func.
    h = 1e-5
    expected = np.zeros((num_configurations,) + func_of_vector(mean).shape)
    for i, shift in enumerate(np.eye(len(mean)) * h):
        derivative = (func_of_vector(mean + shift) - func_of_vector(mean - shift)) / (2 * h)
        expected += np.multiply.outer(fluctuations[:, i], derivative)

    assert derived.fluctuations is not None
    np.testing.assert_allclose(derived.fluctuations, expected, rtol=1e-6)


def test_chained_derive_matches_single_derive():
    num_configurations = 10
    rng = np.random.default_rng(0)
    x_observations = rng.normal(size=(num_configurations, 2))
    y_observations = rng.normal(size=(num_configurations, 3))

    layout = Layout("E", np.arange(num_configurations))
    x = Variable.from_data(x_observations, layout)
    y = Variable.from_data(y_observations, layout)

    def f(a):
        return jnp.array([
            [a[1], a[0] * a[1]],
            [jnp.sin(a[1]), jnp.cosh(a[0]) * a[1]],
        ])

    def g(b, c):
        return jnp.array([
            jnp.cosh(b[1, 1] * b[0, 1]),
            b[0, 0] ** 2 * c[0],
            jnp.sin(b[1, 0]) * jnp.cos(c[1]),
            b[0, 0],
        ])

    # One step, and two steps reusing the intermediate's fluctuations; equal by the chain rule.
    one_step = Variable.derive(lambda a, c: g(f(a), c), x, y)
    assert one_step.fluctuations is not None

    intermediate = Variable.derive(f, x)
    two_step = Variable.derive(g, intermediate, y)
    assert two_step.fluctuations is not None

    np.testing.assert_allclose(two_step.central_value, one_step.central_value, rtol=1e-12)
    np.testing.assert_allclose(two_step.fluctuations, one_step.fluctuations, rtol=1e-12, atol=1e-14)
