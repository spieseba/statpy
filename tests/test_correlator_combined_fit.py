"""Public-function tests for the two-correlator combined fit."""
import numpy as np
import pytest

from statpy.qcd.correlator import FitTags, ground_state_fit

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
    assert {fit.jackknife for fit in fit_tags} == expected
    assert all(fit.bootstrap is None for fit in fit_tags)
    assert expected <= db.database.keys()
    for fit in fit_tags:
        entry = db.database[fit.jackknife]
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

    assert [fit.jackknife for fit in fit_tags] == ["combined/combined-cosh-sinh_fit"]
    assert fit_tags[0].bootstrap is None
    np.testing.assert_allclose(
        db.database[fit_tags[0].jackknife].central_value, TRUE_P, rtol=0.08, atol=0.02,
    )


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("with_bootstrap", [False, True])
def test_fit_references_resolve_resamples(combined, with_bootstrap):
    db = _synthetic_db(("cosh", "cosh"))
    config = FitConfig(bootstrap_available=with_bootstrap)
    bootstraps = np.random.default_rng(8).integers(N_CFG, size=(20, N_CFG))
    if combined:
        fits = correlator_combined_fit(
            db, ("smsm", "smloc"), "joint", FIT_RANGES, 2,
            [1.3, 0.75, 0.24], ("cosh", "cosh"), config,
            silent=True, bootstraps=bootstraps,
        )
        truth = TRUE_P
    else:
        fits = ground_state_fit(
            db, "smsm", 2, FIT_RANGES[0], [1.3, 0.24], "cosh", config,
            silent=True, bootstraps=bootstraps,
        )
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
