"""Public-function tests for the two-correlator combined fit."""
import numpy as np

from statpy.database.core import DB
from statpy.qcd.correlator.fits import FitConfig, correlator_combined_fit


NT = 32
N_CFG = 50
TRUE_P = np.array([1.4, 0.8, 0.25])
FIT_RANGES = (np.arange(4, 10), np.arange(4, 10))
CONFIG = FitConfig(bootstrap_available=False)


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


def test_combined_fit_cosh_cosh_tags_parameters_and_misc():
    db = _synthetic_db(("cosh", "cosh"))
    tags = ("smsm", "smloc")
    fit_models = ("cosh", "cosh")

    fit_tags = correlator_combined_fit(
        db, tags, "combined", FIT_RANGES, 2, [1.3, 0.75, 0.24], fit_models,
        CONFIG, silent=True,
    )

    expected = {
        "combined/combined-cosh-cosh_fit",
        "combined/binsize2/combined-cosh-cosh_fit",
    }
    assert set(fit_tags) == expected
    assert expected <= db.database.keys()
    for fit_tag in fit_tags:
        entry = db.database[fit_tag]
        np.testing.assert_allclose(entry.central_value, TRUE_P, rtol=0.08, atol=0.02)
        assert entry.misc["fit_model"] == "combined-cosh-cosh"
        assert entry.misc["fit_models"] == fit_models
        assert entry.misc["tags"] == tags
        for actual, expected_range in zip(entry.misc["t_blocks"], FIT_RANGES):
            np.testing.assert_array_equal(actual, expected_range)
    assert not any("afbare" in tag for tag in db.database)


def test_combined_fit_routes_cosh_sinh_signs():
    db = _synthetic_db(("cosh", "sinh"))

    fit_tags = correlator_combined_fit(
        db, ("smsm", "smloc"), "combined", FIT_RANGES, 1,
        [1.3, 0.75, 0.24], ("cosh", "sinh"), CONFIG, silent=True,
    )

    assert fit_tags == ["combined/combined-cosh-sinh_fit"]
    np.testing.assert_allclose(
        db.database[fit_tags[0]].central_value, TRUE_P, rtol=0.08, atol=0.02,
    )
