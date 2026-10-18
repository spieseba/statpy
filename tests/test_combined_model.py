import numpy as np
import pytest

from statpy.qcd.correlator import models


@pytest.mark.parametrize("s0", [1.0, -1.0, 0.0])
@pytest.mark.parametrize("s1", [1.0, -1.0, 0.0])
def test_combined_model(s0, s1):
    Nt, L = 64, 32
    t0 = np.arange(3, 15)
    t1 = np.arange(5, 18)
    t = np.concatenate([t0, L + t1])
    parameters = np.array([1.2, -0.7, 0.3])
    single = {1.0: models.cosh(Nt), -1.0: models.sinh(Nt), 0.0: models.exp()}
    expected = np.concatenate([
        single[s0](t0, parameters[[0, 2]]),
        single[s1](t1, parameters[[1, 2]]),
    ])
    np.testing.assert_allclose(models.combined(Nt, L, (s0, s1))(t, parameters), expected)
