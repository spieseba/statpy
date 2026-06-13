"""Round-trip test for the v1-JSON -> v2 migrator (statpy.database.io.load_v1_json).

Synthesises a tiny in-memory v1 custom-JSON database (the retired format), runs
the migrator, and checks that the v2 entries reproduce the v1 mean/jackknife and
survive a save/load cycle. Runs under pytest or as a plain script:

    python tests/test_v1_migration.py
"""
import base64
import json
import os
import tempfile

import numpy as np

from statpy.database.core import DB
from statpy.database.io import load_v1_json


def _encode(arr):
    """Encode an ndarray the v1 custom-JSON way."""
    arr = np.ascontiguousarray(arr)
    return {
        "__ndarray__": base64.b64encode(arr.tobytes()).decode("ascii"),
        "dtype": arr.dtype.str,
        "shape": list(arr.shape),
    }


def _v1_leaf(sample_by_cfg, misc=None):
    """Build a v1 leaf dict (mean/jks/sample/misc/checksum) from cfg->array."""
    cfgs = list(sample_by_cfg)
    stacked = np.array([sample_by_cfg[c] for c in cfgs])
    mean = np.mean(stacked, axis=0)
    N = len(cfgs)
    jks = {c: (N * mean - sample_by_cfg[c]) / (N - 1) for c in cfgs}  # leave-one-out
    leaf = {
        "mean": _encode(mean),
        "jks": {c: _encode(jks[c]) for c in cfgs},
        "sample": {c: _encode(sample_by_cfg[c]) for c in cfgs},
        "misc": misc,
        "checksum": 0,
    }
    return {"__leaf__": leaf}, mean, stacked, jks


def _make_v1_db(rng):
    """A small v1 database covering value-shapes (nt,), (nsrc,nt) and misc passthrough."""
    cfgs = ["A-100", "A-110", "A-120", "A-130", "A-140"]
    db_json, expected = {}, {}

    # vector-valued leaf, shape (8,)
    sbc = {c: rng.standard_normal(8) for c in cfgs}
    leaf, mean, stacked, jks = _v1_leaf(sbc)
    db_json["A/vec"] = leaf
    expected["A/vec"] = (cfgs, mean, stacked, jks, None)

    # matrix-valued leaf with misc, shape (3, 4)
    misc = {"ptsrcs": {c: [f"src{i}" for i in range(3)] for c in cfgs}}
    sbc = {c: rng.standard_normal((3, 4)) for c in cfgs}
    leaf, mean, stacked, jks = _v1_leaf(sbc, misc=misc)
    db_json["A/mat"] = leaf
    expected["A/mat"] = (cfgs, mean, stacked, jks, misc)

    return db_json, expected


def _run(tmpdir):
    rng = np.random.default_rng(0)
    db_json, expected = _make_v1_db(rng)

    src = os.path.join(tmpdir, "fermionic.sample")
    with open(src, "w") as f:
        json.dump(db_json, f)

    db = load_v1_json(src, silent=True)

    assert set(db.database) == set(expected), "tag set must be preserved"
    for tag, (cfgs, mean, stacked, jks, misc) in expected.items():
        e = db.database[tag]
        assert list(e.cfgs) == cfgs, f"{tag}: cfg labels/order preserved"
        assert np.array_equal(e.sample, stacked), f"{tag}: sample decoded/stacked exactly"
        np.testing.assert_allclose(e.mean, mean, rtol=1e-12, atol=0,
                                   err_msg=f"{tag}: derived mean matches v1")
        v1_jks = np.array([jks[c] for c in cfgs])
        np.testing.assert_allclose(e.jks, v1_jks, rtol=1e-12, atol=0,
                                   err_msg=f"{tag}: derived jks matches v1 leave-one-out")
        assert np.array_equal(e.weights, np.ones(len(cfgs))), f"{tag}: uniform weights"
        assert e.misc == misc, f"{tag}: misc carried through unchanged"

    # save/load (v2 pickle+CRC32) round-trips
    dst = os.path.join(tmpdir, "fermionic.db")
    db.save(dst)
    reloaded = DB(dst)
    assert set(reloaded.database) == set(expected)
    for tag in expected:
        np.testing.assert_array_equal(reloaded.database[tag].sample, db.database[tag].sample)


def test_v1_migration_round_trip():
    with tempfile.TemporaryDirectory() as tmpdir:
        _run(tmpdir)


def test_ragged_leaf_rejected():
    rng = np.random.default_rng(1)
    bad = {
        "A/ragged": {"__leaf__": {
            "sample": {"A-1": _encode(rng.standard_normal(8)),
                       "A-2": _encode(rng.standard_normal(9))},  # mismatched shape
            "mean": _encode(rng.standard_normal(8)), "jks": {}, "misc": None, "checksum": 0,
        }}
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "bad.sample")
        with open(src, "w") as f:
            json.dump(bad, f)
        try:
            load_v1_json(src, silent=True)
        except ValueError as exc:
            assert "ragged" in str(exc)
        else:
            raise AssertionError("expected ValueError for ragged per-cfg samples")


if __name__ == "__main__":
    test_v1_migration_round_trip()
    test_ragged_leaf_rejected()
    print("OK: v1 migration round-trip + ragged rejection")
