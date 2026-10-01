"""Public-function tests for the two-correlator combined fit."""
import numpy as np
import pytest

from statpy.database.core import DB
from statpy.qcd.correlator import FitTags, ground_state_fit
from statpy.qcd.correlator.fits import FitConfig

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
    db = DB()
    for tag, model, amplitude in zip(("smsm", "smloc"), fit_models, TRUE_P[:2]):
        sample = _model(model, t, amplitude, TRUE_P[2], NT) + rng.normal(scale=0.003, size=(N_CFG, NT))
        db.add_entry(tag, sample=sample, weights=weights, cfgs=cfgs)
    return db


@pytest.mark.parametrize("with_bootstrap", [False, True])
def test_fit_references_resolve_resamples(with_bootstrap):
    db = _synthetic_db(("cosh", "cosh"))
    bootstraps = np.random.default_rng(8).integers(N_CFG, size=(20, N_CFG)) if with_bootstrap else None
    fits = [
        ground_state_fit(
            db, "smsm", binsize, FIT_RANGES[0], [1.3, 0.24], "cosh", CONFIG,
            Nt=NT, silent=True, bootstraps=bootstraps if binsize == 1 else None,
        )
        for binsize in (1, 2)
    ]
    truth = TRUE_P[[0, 2]]
    assert len(fits) == 2
    for binsize, fit in enumerate(fits, start=1):
        assert isinstance(fit, FitTags)
        entry = db.database[fit.jackknife]
        assert entry.jks.shape == (N_CFG // binsize, len(truth))
        np.testing.assert_allclose(entry.central_value, truth, rtol=0.08, atol=0.02)
        if with_bootstrap and binsize == 1:
            entry_bs = db.database[fit.bootstrap]
            assert entry_bs.bss.shape == (len(bootstraps), len(truth))
            np.testing.assert_allclose(entry_bs.central_value, truth, rtol=0.08, atol=0.02)
        else:
            assert fit.bootstrap is None


def test_ground_state_fit_rejects_bootstrap_above_binsize_one():
    db = _synthetic_db(("cosh", "cosh"))

    with pytest.raises(ValueError, match="binsize == 1"):
        ground_state_fit(
            db, "smsm", 5, FIT_RANGES[0], [1.3, 0.24], "cosh", CONFIG,
            Nt=NT, bootstraps=np.zeros((2, N_CFG), dtype=int), silent=True,
        )
