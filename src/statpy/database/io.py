import h5py, os, sys
import numpy as np
from statpy.log import message
from statpy.database.core import DB

def load_CLS(fn, rwf_fn, tags, branch_tag, exclude_SRCPOS=True, verbosity=0, accept_cfg_mismatch=False):
    assert os.path.isfile(fn), f"{fn} not found!"
    message(f"---------------------------------")
    message(f"Load CLS data from {fn}")
    message(f"reweighting factors: {rwf_fn}")
    message(f"tags: {tags}")
    message(f"store as branch tag: {branch_tag}")
    # data
    f = h5py.File(fn, "r")
    f_cfgs = np.array([int(cfg.decode("utf-8").split("n")[1]) for cfg in f.get("configlist")])
    message(f"# of cfgs in hdf5 file: {len(f_cfgs)}")
    db = DB(verbosity=verbosity)
    # rwfs
    rwf_branch_tag = branch_tag.split("/")[0]
    if rwf_fn is None:
        message(f"rwf file not available. Use rwf=1.0 for all configs.")
        common_cfgs = f_cfgs
        rwf = {f"{rwf_branch_tag}-{cfg}":1.0 for cfg in common_cfgs}
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
        rwf = {f"{rwf_branch_tag}-{cfg}":val for cfg,val in zip(rwf_cfgs, rwf) if cfg in common_cfgs} 
    db.add_leaf(tag=f"{rwf_branch_tag}/rwf", mean=None, jks=None, sample=rwf, misc=None)
    db.add_nrwf(rwf_tag=f"{rwf_branch_tag}/rwf")
    # data
    for t in tags:
        for key in f.keys(): # in future: additional layer /messpec
            if t in key:
                if "SRCPOS" in key and exclude_SRCPOS:
                    continue
                f_vals = f.get(key)[:]
                sample = {f"{branch_tag}-{cfg}":val for cfg,val in zip(f_cfgs, f_vals) if cfg in common_cfgs}
                db.add_leaf(tag=f"{branch_tag}/{key}", mean=None, jks=None, sample=sample, misc=None, verbosity=verbosity)
    message(f"---------------------------------")
    return db
