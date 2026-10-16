"""Tests for deriving variables and explicitly storing their results."""
import numpy as np
import pytest

from statpy.database import Database
from statpy.variable import Layout, Variable


def _db():
    """A Database with two positive data variables 'a' and 'b' sharing cfgs (shape (N, 3))."""
    rng = np.random.default_rng(0)
    cfgs = np.array([f"c{i}" for i in range(50)])
    weights = np.ones(len(cfgs))
    db = Database()
    db["a"] = Variable.from_data(5.0 + rng.standard_normal((50, 3)), Layout(cfgs, weights))
    db["b"] = Variable.from_data(5.0 + rng.standard_normal((50, 3)), Layout(cfgs, weights))
    return db


def test_transform_returns_variable_without_storing():
    db = _db()
    num_before = len(db)
    out = Variable.derive(lambda x: 2 * x, db["a"])
    assert isinstance(out, Variable)
    mean, _, bss = out.central_value, out.jackknife_replicates, out.bootstrap_replicates
    np.testing.assert_allclose(mean, 2 * db["a"].central_value, rtol=1e-12)
    assert bss is None                                         # data variable has no bss
    assert len(db) == num_before                        # nothing stored


def test_transform_can_be_stored():
    db = _db()
    derived = Variable.derive(lambda x: 2 * x, db["a"])
    mean, jks, _ = derived.central_value, derived.jackknife_replicates, derived.bootstrap_replicates        # reference (compute mode)
    db["a2"] = Variable.derive(lambda x: 2 * x, db["a"])
    assert "a2" in db
    np.testing.assert_allclose(db["a2"].central_value, mean, rtol=1e-12)
    np.testing.assert_allclose(db["a2"].jackknife_replicates, jks, rtol=1e-12)


def test_combine_returns_variable_without_storing():
    db = _db()
    num_before = len(db)
    out = Variable.derive(lambda x, y: x / y, db["a"], db["b"])
    assert isinstance(out, Variable)
    assert len(db) == num_before                        # nothing stored


def test_combine_can_be_stored():
    db = _db()
    derived = Variable.derive(lambda x, y: x / y, db["a"], db["b"])
    mean, jks, _ = derived.central_value, derived.jackknife_replicates, derived.bootstrap_replicates  # reference (compute mode)
    db["r"] = Variable.derive(lambda x, y: x / y, db["a"], db["b"])
    assert "r" in db
    np.testing.assert_allclose(db["r"].central_value, mean, rtol=1e-12)
    np.testing.assert_allclose(db["r"].jackknife_replicates, jks, rtol=1e-12)


def test_derive_rejects_differing_configurations():
    db = _db()
    other = Variable.from_data(db["b"].observations, Layout(np.arange(50), np.ones(50)))
    with pytest.raises(ValueError, match="input 1 configurations differ"):
        Variable.derive(lambda x, y: x + y, db["a"], other)
