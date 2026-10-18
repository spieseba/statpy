"""Ground-state fit of synthetic data recovers the true parameters."""
import numpy as np
import pytest

from statpy.database import Database
from statpy.qcd.correlator import ground_state_fit
from statpy.qcd.correlator.fits import FitConfig
from statpy.variable import Layout, Variable

NT = 32
N_CFG = 50
TRUE_P = np.array([1.4, 0.25])
FIT_RANGE = np.arange(4, 10)
CONFIG = FitConfig()


def _model(name, t, amplitude, mass, Nt):
    sign = {"cosh": 1.0, "sinh": -1.0, "exp": 0.0}[name]
    return amplitude * (np.exp(-mass * t) + sign * np.exp(-mass * (Nt - t)))


def _synthetic_db():
    rng = np.random.default_rng(4)
    t = np.arange(NT)
    db = Database()
    sample = _model("cosh", t, TRUE_P[0], TRUE_P[1], NT) + rng.normal(scale=0.003, size=(N_CFG, NT))
    db["smsm"] = Variable.from_data(sample, Layout("E", np.arange(N_CFG)))
    return db


@pytest.mark.parametrize("with_bootstrap", [False, True])
def test_fit_recovers_true_parameters(with_bootstrap):
    db = _synthetic_db()
    bootstrap_indices = np.random.default_rng(8).integers(N_CFG, size=(20, N_CFG)) if with_bootstrap else None
    for bin_size in (1, 2):
        v = db["smsm"]
        if bin_size == 1:
            v = Variable.from_data(v.observations, v.layout, bootstrap_indices=bootstrap_indices)
        else:
            v = v.bin(bin_size)
        fit, bootstrap = ground_state_fit(
            v, FIT_RANGE, [1.3, 0.24], "cosh", CONFIG, Nt=NT,
        )
        np.testing.assert_allclose(fit.parameters.central_value, TRUE_P, rtol=0.08, atol=0.02)
        if with_bootstrap and bin_size == 1:
            np.testing.assert_allclose(bootstrap.parameters.central_value, TRUE_P, rtol=0.08, atol=0.02)
