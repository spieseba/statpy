"""Tests for create-only DB semantics: duplicate tags raise instead of silently skipping."""

import os
import tempfile

import numpy as np

from statpy.database.core import DB, DuplicateTagError


def _data_entry_kwargs(seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        sample=rng.normal(size=(10, 4)),
        weights=np.ones(10),
        cfgs=np.array([f"c{i}" for i in range(10)]),
    )


def test_duplicate_add_raises_and_preserves_original():
    db = DB()
    db.add_entry("a", **_data_entry_kwargs(0))
    original_mean = db.database["a"].mean.copy()
    try:
        db.add_entry("a", **_data_entry_kwargs(1))
    except DuplicateTagError:
        pass
    else:
        raise AssertionError("expected DuplicateTagError on duplicate add_entry")
    np.testing.assert_array_equal(db.database["a"].mean, original_mean)


def test_rename_onto_existing_raises():
    db = DB()
    db.add_entry("a", **_data_entry_kwargs(0))
    db.add_entry("b", **_data_entry_kwargs(1))
    try:
        db.rename_entry("a", "b")
    except DuplicateTagError:
        pass
    else:
        raise AssertionError("expected DuplicateTagError on rename onto existing tag")
    assert "a" in db.database and "b" in db.database


def test_merge_overlap_raises_and_names_tag():
    db1 = DB()
    db1.add_entry("shared", **_data_entry_kwargs(0))
    db2 = DB()
    db2.add_entry("shared", **_data_entry_kwargs(1))
    try:
        DB(db1, db2)
    except DuplicateTagError as exc:
        assert "shared" in str(exc)
    else:
        raise AssertionError("expected DuplicateTagError on overlapping merge")
    # disjoint merge still works
    db3 = DB()
    db3.add_entry("other", **_data_entry_kwargs(2))
    merged = DB(db1, db3)
    assert set(merged.database) == {"shared", "other"}


def test_load_overlap_raises():
    db = DB()
    db.add_entry("shared", **_data_entry_kwargs(0))
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = os.path.join(tmpdir, "db.pkl")
        db.save(fn)
        fresh = DB()
        fresh.load(fn)  # into empty: fine
        assert "shared" in fresh.database
        try:
            db.load(fn)  # into itself: overlap
        except DuplicateTagError as exc:
            assert "shared" in str(exc)
        else:
            raise AssertionError("expected DuplicateTagError on overlapping load")


def test_guarded_caching_idiom_still_works():
    # the documented pattern: check membership, add only if missing
    db = DB()
    db.add_entry("a", **_data_entry_kwargs(0))
    for _ in range(2):
        if "a" not in db.database:
            db.add_entry("a", **_data_entry_kwargs(1))
    assert len(db.database) == 1
