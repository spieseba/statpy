import h5py, os
import numpy as np
from statpy.log import message
from statpy.database.core import DB

def load_CLS(fn, rwf_fn, tags, branch_tag, exclude_SRCPOS=True, verbosity=0):
    assert os.path.isfile(fn), f"{fn} not found!"
    message(f"---------------------------------")
    message(f"Load CLS data from {fn}")
    message(f"reweighting factors: {rwf_fn}")
    message(f"tags: {tags}")
    message(f"store as branch tag: {branch_tag}")
    message(f"---------------------------------")
    # data
    f = h5py.File(fn, "r")
    f_cfgs = np.array([int(cfg.decode("utf-8").split("n")[1]) for cfg in f.get("configlist")])
    db = DB(verbosity=verbosity)
    # rwfs
    if rwf_fn is None:
        cfgs = f_cfgs 
        rwf = {f"{branch_tag}-{cfg}":1.0 for cfg in f_cfgs}
    else:
        assert os.path.isfile(rwf_fn) 
        rwf_cfgs = np.array(np.loadtxt(rwf_fn)[:,0], dtype=int)
        cfgs = np.array([cfg for cfg in rwf_cfgs if cfg in f_cfgs])
        rwf = np.loadtxt(rwf_fn)[:,1] 
        rwf = {f"{branch_tag}-{cfg}":val for cfg, val in zip(cfgs, rwf)} 
    db.add_leaf(tag=f"{branch_tag}/rwf", mean=None, jks=None, sample=rwf, misc=None)
    db.add_nrwf(rwf_tag=f"{branch_tag}/rwf")
    # data
    for t in tags:
        for key in f.keys():
            if t in key:
                if "SRCPOS" in key and exclude_SRCPOS:
                    continue
                vals = f.get(key)
                assert len(vals) == len(cfgs), "Missmatch between determined number of configs and stored values"
                sample = {f"{branch_tag}-{cfg}":val for cfg, val in zip(cfgs, vals)}
                db.add_leaf(tag=f"{branch_tag}/{key}", mean=None, jks=None, sample=sample, misc=None)
    return db
