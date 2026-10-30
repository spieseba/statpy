"""The jitted jnp models agree with the numba models."""
import itertools

import jax
import numpy as np
import pytest

from statpy.qcd.correlator import models, numba_models

NT, L = 64, 32
T = np.concatenate([np.arange(3, 15), L + np.arange(5, 18)])  # t >= L selects the second correlator of combined


@pytest.mark.parametrize(("name", "arguments", "parameters"), [
    ("exp", (), [1.2, 0.3]),
    ("cosh", (NT,), [1.2, 0.3]),
    ("sinh", (NT,), [1.2, 0.3]),
    ("double_exp", (), [1.2, 0.3, -0.7, 0.8]),
    ("double_cosh", (NT,), [1.2, 0.3, -0.7, 0.8]),
    ("double_sinh", (NT,), [1.2, 0.3, -0.7, 0.8]),
    ("const_plus_exp", (), [1.2, 0.3, 0.5]),
    *[("combined", (NT, L, signs), [1.2, -0.7, 0.3]) for signs in itertools.product([1.0, -1.0, 0.0], repeat=2)],
])
def test_jnp_model_matches_numba_model(name, arguments, parameters):
    parameters = np.array(parameters)
    jnp_values = jax.jit(getattr(models, name)(*arguments))(T, parameters)
    numba_values = getattr(numba_models, name)(*arguments)(T, parameters)
    np.testing.assert_allclose(jnp_values, numba_values, rtol=1e-14)
