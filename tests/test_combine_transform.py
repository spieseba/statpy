"""Tests for the DB.combine / DB.transform store-vs-return contract.

Both compute (mean, jks, bss) and then either:
  - return the (mean, jks, bss) tuple when ``store_as`` is omitted, or
  - store the result under ``store_as`` and return ``None``.
The stored entry must match what the return-mode produces. Runs under pytest or
as a plain script:

    python tests/test_combine_transform.py
"""
import numpy as np

from statpy.database.core import DB


def _db():
    """A DB with two positive data entries 'a' and 'b' sharing cfgs (shape (N, 3))."""
    rng = np.random.default_rng(0)
    cfgs = np.array([f"c{i}" for i in range(50)])
    weights = np.ones(len(cfgs))
    db = DB()
    db.add_entry("a", sample=5.0 + rng.standard_normal((50, 3)), weights=weights, cfgs=cfgs)
    db.add_entry("b", sample=5.0 + rng.standard_normal((50, 3)), weights=weights, cfgs=cfgs)
    return db


def test_transform_returns_tuple_without_store_as():
    db = _db()
    n_before = len(db.database)
    out = db.transform("a", f=lambda x: 2 * x)
    assert isinstance(out, tuple) and len(out) == 3            # (mean, jks, bss)
    mean, jks, bss = out
    np.testing.assert_allclose(mean, 2 * db.database["a"].mean, rtol=1e-12)
    assert bss is None                                         # data entry has no bss
    assert len(db.database) == n_before                        # nothing stored


def test_transform_stores_and_returns_none_with_store_as():
    db = _db()
    mean, jks, _ = db.transform("a", f=lambda x: 2 * x)        # reference (compute mode)
    ret = db.transform("a", f=lambda x: 2 * x, store_as="a2")
    assert ret is None                                         # store mode returns nothing
    assert "a2" in db.database
    np.testing.assert_allclose(db.database["a2"].mean, mean, rtol=1e-12)
    np.testing.assert_allclose(db.database["a2"].jks, jks, rtol=1e-12)


def test_combine_returns_tuple_without_store_as():
    db = _db()
    n_before = len(db.database)
    out = db.combine("a", "b", f=lambda x, y: x / y)
    assert isinstance(out, tuple) and len(out) == 3
    assert len(db.database) == n_before                        # nothing stored


def test_combine_stores_and_returns_none_with_store_as():
    db = _db()
    mean, jks, _ = db.combine("a", "b", f=lambda x, y: x / y)  # reference (compute mode)
    ret = db.combine("a", "b", f=lambda x, y: x / y, store_as="r")
    assert ret is None
    assert "r" in db.database
    np.testing.assert_allclose(db.database["r"].mean, mean, rtol=1e-12)
    np.testing.assert_allclose(db.database["r"].jks, jks, rtol=1e-12)


if __name__ == "__main__":
    test_transform_returns_tuple_without_store_as()
    test_transform_stores_and_returns_none_with_store_as()
    test_combine_returns_tuple_without_store_as()
    test_combine_stores_and_returns_none_with_store_as()
    print("OK: combine/transform return None + store with store_as; return the tuple without it")
