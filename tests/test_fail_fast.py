"""Tests for create-only Database semantics: duplicate tags raise instead of silently skipping."""

import os
import tempfile

import numpy as np

from statpy.database import Database, DuplicateTagError
from statpy.variable import Layout, Variable


def _data_variable(seed=0):
    rng = np.random.default_rng(seed)
    return Variable.from_data(rng.normal(size=(10, 4)),
                              Layout(np.array([f"c{i}" for i in range(10)]), np.ones(10)))


def test_duplicate_add_raises_and_preserves_original():
    db = Database()
    db["a"] = _data_variable(0)
    original_mean = db["a"].central_value.copy()
    try:
        db["a"] = _data_variable(1)
    except DuplicateTagError:
        pass
    else:
        raise AssertionError("expected DuplicateTagError on duplicate assignment")
    np.testing.assert_array_equal(db["a"].central_value, original_mean)


def test_rename_onto_existing_raises():
    db = Database()
    db["a"] = _data_variable(0)
    db["b"] = _data_variable(1)
    try:
        db.rename("a", "b")
    except DuplicateTagError:
        pass
    else:
        raise AssertionError("expected DuplicateTagError on rename onto existing tag")
    assert "a" in db and "b" in db


def test_merge_overlap_raises_and_names_tag():
    db1 = Database()
    db1["shared"] = _data_variable(0)
    db2 = Database()
    db2["shared"] = _data_variable(1)
    try:
        Database(db1, db2)
    except DuplicateTagError as exc:
        assert "shared" in str(exc)
    else:
        raise AssertionError("expected DuplicateTagError on overlapping merge")
    # disjoint merge still works
    db3 = Database()
    db3["other"] = _data_variable(2)
    merged = Database(db1, db3)
    assert set(merged) == {"shared", "other"}


def test_load_overlap_raises():
    db = Database()
    db["shared"] = _data_variable(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir, "db.pkl")
        db.save(fn)
        fresh = Database()
        fresh.load(fn)  # into empty: fine
        assert "shared" in fresh
        try:
            db.load(fn)  # into itself: overlap
        except DuplicateTagError as exc:
            assert "shared" in str(exc)
        else:
            raise AssertionError("expected DuplicateTagError on overlapping load")


def test_guarded_caching_idiom_still_works():
    # the documented pattern: check membership, add only if missing
    db = Database()
    db["a"] = _data_variable(0)
    for _ in range(2):
        if "a" not in db:
            db["a"] = _data_variable(1)
    assert len(db) == 1


def test_jackknife_replicates_require_configurations():
    db = Database()
    try:
        db["a"] = Variable.from_results(central_value=np.zeros(4), jackknife_replicates=np.zeros((10, 4)))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for jackknife_replicates without configurations")
    assert "a" not in db


def test_invalid_assignment_and_missing_tags():
    import pytest

    db = Database()
    with pytest.raises(TypeError):
        db["a"] = object()
    with pytest.raises(KeyError):
        del db["missing"]
