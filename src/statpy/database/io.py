import h5py, os, sys
import numpy as np
from statpy.log import message
from statpy.database.core import DB

def load_CLS(fn, rwf_fn, tags, stream_tag, run_tag=None, cfgs_to_be_removed=None, verbosity=0, accept_cfg_mismatch=False):
    assert os.path.isfile(fn), f"{fn} not found!"
    assert isinstance(cfgs_to_be_removed, list) or isinstance(cfgs_to_be_removed, np.ndarray) or cfgs_to_be_removed is None
    message(f"---------------------------------")
    message(f"Load CLS data from {fn}")
    message(f"Load rw factors from: {rwf_fn}")
    message(f" -- tags: {tags}")
    message(f" -- ensemble tag = {stream_tag}")
    message(f" -- run tag: {run_tag}")
    message(f" -- cfgs to be removed: {cfgs_to_be_removed}")
    # data
    f = h5py.File(fn, "r")["messpec"]
    f_cfgs = np.array([int(cfg.decode("utf-8").split("n")[1]) for cfg in f.get("configlist")])
    f_cfgs_filtered = f_cfgs[~np.isin(f_cfgs, cfgs_to_be_removed)] if cfgs_to_be_removed is not None else f_cfgs
    message(f"Number of cfgs in hdf5 file: {len(f_cfgs)} | Number of filtered configs in hdf5 file: {len(f_cfgs_filtered)}")
    db = DB(verbosity=verbosity)
    # rwfs
    if rwf_fn is None:
        message(f"rwf file not available. Use rwf=1.0 for all configs.")
        common_cfgs = f_cfgs_filtered
        rwf = {f"{stream_tag}-{cfg}":1.0 for cfg in common_cfgs}
    else:
        assert os.path.isfile(rwf_fn) 
        rwf_cfgs = np.array(np.loadtxt(rwf_fn)[:,0], dtype=int)
        rwf_cfgs_filtered = rwf_cfgs[~np.isin(rwf_cfgs, cfgs_to_be_removed)] if cfgs_to_be_removed is not None else rwf_cfgs
        message(f"Number of cfgs in rwf file: {rwf_cfgs.shape[0]} | Number of filtered configs in rwf file : {rwf_cfgs_filtered.shape[0]}")
        if not np.array_equal(f_cfgs_filtered, rwf_cfgs_filtered):
            message(f"WARNING: filtered rwf file has different configs than filtered hdf5 file!")
            if not accept_cfg_mismatch: 
                sys.exit(1)
            message(f"---> Add only common configs to database.")
        common_cfgs = np.array([cfg for cfg in rwf_cfgs_filtered if cfg in f_cfgs_filtered])
        message(f"Number of filtered configs in hdf5 file and rwf file: {common_cfgs.shape[0]}")
        rwf = np.loadtxt(rwf_fn)[:,1] 
        rwf = {f"{stream_tag}-{cfg}":val for cfg,val in zip(rwf_cfgs, rwf) if cfg in common_cfgs} 
    db.add_leaf(tag=f"{stream_tag}/rwf", mean=None, jks=None, sample=rwf, misc=None)
    db.add_nrwf(rwf_tag=f"{stream_tag}/rwf")
    # data
    for t in tags:
        for key in f["data"].keys(): 
            if t in key:
                f_vals = f["data"].get(key)[:]
                sample = {f"{stream_tag}-{cfg}":val for cfg,val in zip(f_cfgs, f_vals) if cfg in common_cfgs}
                f_tag = f"{stream_tag}/{key}" if run_tag is None else f"{stream_tag}/{run_tag}/{key}"
                db.add_leaf(tag=f_tag, mean=None, jks=None, sample=sample, misc=None, verbosity=verbosity)
    message(f"---------------------------------")
    return db

def load_CLS_deprecated(fn, rwf_fn, tags, stream_tag, run_tag=None, verbosity=0, accept_cfg_mismatch=False):
    assert os.path.isfile(fn), f"{fn} not found!"
    message(f"---------------------------------")
    message(f"Load CLS data from {fn}")
    message(f"reweighting factors: {rwf_fn}")
    message(f"tags: {tags}")
    message(f"ensemble tag = {stream_tag}")
    message(f"run tag: {run_tag}")
    # data
    f = h5py.File(fn, "r")
    f_cfgs = np.array([int(cfg.decode("utf-8").split("n")[1]) for cfg in f.get("configlist")])
    message(f"# of cfgs in hdf5 file: {len(f_cfgs)}")
    db = DB(verbosity=verbosity)
    # rwfs
    if rwf_fn is None:
        message(f"rwf file not available. Use rwf=1.0 for all configs.")
        common_cfgs = f_cfgs
        rwf = {f"{stream_tag}-{cfg}":1.0 for cfg in common_cfgs}
    else:
        assert os.path.isfile(rwf_fn) 
        rwf_cfgs = np.array(np.loadtxt(rwf_fn)[:,0], dtype=int)
        common_cfgs = np.array([cfg for cfg in rwf_cfgs if cfg in f_cfgs])
        message(f"# of cfgs in rwf file: {rwf_cfgs.shape[0]} | # of common configs with hdf5 file : {common_cfgs.shape[0]}")
        if rwf_cfgs.shape[0] != common_cfgs.shape[0] or rwf_cfgs.shape[0] != f_cfgs.shape[0]:
            message(f"WARNING: rwf file has different configs than hdf5 file!")
            if not accept_cfg_mismatch: 
                sys.exit(1)
            message(f"---> Add only common configs to database.")
        rwf = np.loadtxt(rwf_fn)[:,1] 
        rwf = {f"{stream_tag}-{cfg}":val for cfg,val in zip(rwf_cfgs, rwf) if cfg in common_cfgs} 
    db.add_leaf(tag=f"{stream_tag}/rwf", mean=None, jks=None, sample=rwf, misc=None)
    db.add_nrwf(rwf_tag=f"{stream_tag}/rwf")
    # data
    for t in tags:
        for key in f.keys(): 
            if t in key:
                if "SRCPOS" in key:
                    continue
                f_vals = f.get(key)[:]
                sample = {f"{stream_tag}-{cfg}":val for cfg,val in zip(f_cfgs, f_vals) if cfg in common_cfgs}
                f_tag = f"{stream_tag}/{key}" if run_tag is None else f"{stream_tag}/{run_tag}/{key}"
                db.add_leaf(tag=f_tag, mean=None, jks=None, sample=sample, misc=None, verbosity=verbosity)
    message(f"---------------------------------")
    return db
