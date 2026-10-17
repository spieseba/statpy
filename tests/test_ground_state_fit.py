"""Ground-state fit of synthetic data recovers the true parameters."""
import numpy as np
import pytest

from statpy.database import Database
from statpy.qcd.correlator import ground_state_fit
from statpy.qcd.correlator.fits import FitConfig
from statpy.variable import Layout, Variable

NT = 32
N_CFG = 50
TRUE_P = np.array([1.4, 0.8, 0.25])
FIT_RANGES = (np.arange(4, 10), np.arange(4, 10))
CONFIG = FitConfig()


def _model(name, t, amplitude, mass, Nt):
    sign = {"cosh": 1.0, "sinh": -1.0, "exp": 0.0}[name]
    return amplitude * (np.exp(-mass * t) + sign * np.exp(-mass * (Nt - t)))


def _synthetic_db(fit_models):
    rng = np.random.default_rng(4)
    cfgs = np.array([f"c{i}" for i in range(N_CFG)])
    weights = np.ones(N_CFG)
    t = np.arange(NT)
    db = Database()
    for tag, model, amplitude in zip(("smsm", "smloc"), fit_models, TRUE_P[:2]):
        sample = _model(model, t, amplitude, TRUE_P[2], NT) + rng.normal(scale=0.003, size=(N_CFG, NT))
        db[tag] = Variable.from_data(sample, Layout(cfgs, weights))
    return db


@pytest.mark.parametrize("with_bootstrap", [False, True])
def test_fit_recovers_true_parameters(with_bootstrap):
    db = _synthetic_db(("cosh", "cosh"))
    bootstraps = np.random.default_rng(8).integers(N_CFG, size=(20, N_CFG)) if with_bootstrap else None
    fits = []
    for bin_size in (1, 2):
        v = db["smsm"]
        if bin_size == 1:
            v = Variable.from_data(v.observations, v.layout, bootstraps=bootstraps)
        else:
            v = v.bin(bin_size, "smsm")
        fits.append(ground_state_fit(
            v, FIT_RANGES[0], [1.3, 0.24], "cosh", CONFIG, Nt=NT,
        ))
    truth = TRUE_P[[0, 2]]
    for bin_size, (fit, bootstrap) in enumerate(fits, start=1):
        np.testing.assert_allclose(fit.parameters.central_value, truth, rtol=0.08, atol=0.02)
        if with_bootstrap and bin_size == 1:
            np.testing.assert_allclose(bootstrap.parameters.central_value, truth, rtol=0.08, atol=0.02)
