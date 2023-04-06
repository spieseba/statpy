#!/usr/bin/env python3

import os, sys
import numpy as np
from statpy.dbpy import custom_json as json
from statpy.dbpy.leafs import Leaf, SampleLeaf

def create_sample_db(src_dir, meas_tag, src_tags, dst, dst_tags, ensemble_label):
    if os.path.isfile(dst):
        if query_yes_no(f"file {dst} already exists. Overwrite?"):
            os.remove(dst)
        else:        
            exit()
    # get filenames and cfgs
    filenames = sorted([os.path.join(src_dir, f) for f in os.listdir(src_dir) if (os.path.isfile(os.path.join(src_dir, f)) and meas_tag in f.split("."))], key=lambda x: int(x.split("ckpoint_lat.")[-1].split(".")[0]))
    cfgs = [ensemble_label + "-" + x.split("ckpoint_lat.")[-1].split(".")[0] for x in filenames]
    # create db
    database = {}
    for src_tag, dst_tag in zip(src_tags, dst_tags):
        print(f"reading src_tag={src_tag}")
        sample = {}
        for cfg, f in zip(cfgs, filenames):
            print(f"\tcfg={cfg}")
            sample[cfg] = load(f, src_tag)
        database[ensemble_label + dst_tag] = SampleLeaf(sample)
    with open(dst, "w") as f:
        json.dump(database, f)

def load(src_file, tag):
    with open(src_file) as f:
        src_data = json.load(f)
    return src_data[tag]

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.
    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).
    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")

#def convert(src, dst, src_tags, dst_tags, cfg_prefix="", verbose=False):
#    if os.path.isfile(dst):
#        if query_yes_no(f"file {dst} already exists. Overwrite?"):
#            pass
#        else:        
#            exit()
#    database = {}
#    for dst_tag in dst_tags:
#        database[dst_tag] = {}
#    ensemble = [os.path.join(src, f) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
#    ensemble.sort(key=lambda f: int(re.sub('\D', '', f)))
#    for cfg in ensemble:
#        if verbose:
#            print(cfg)
#        with open(cfg) as f:
#            data = json.load(f)
#        for src_tag, dst_tag in zip(src_tags, dst_tags):
#            database[dst_tag][cfg_prefix + cfg.split(".")[-1]] = data[src_tag]
#    with open(dst, "w") as f:
#        json.dump(database, f)

#def onvert_old_to_new(src, src_tags, dst_tags):
#    ensemble = [os.path.join(src, f) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
#    ensemble.sort(key=lambda f: int(re.sub('\D', '', f)))
#    for cfg in ensemble:
#        print(cfg)
#        database = {}
#        with open(cfg) as f:
#            data = json.load(f)
#        for src_tag, dst_tag in zip(src_tags, dst_tags):
#            database[dst_tag] = np.array(ast.literal_eval(data[src_tag]))
#        with open(cfg, "w") as f:
#            json.dump(database, f)
#
#def convert_from_old_format(src, dst, src_tags, dst_tags, cfg_prefix="", verbose=False):
#    if os.path.isfile(dst):
#        if query_yes_no(f"file {dst} already exists. Overwrite?"):
#            pass
#        else:        
#            exit()
#    else:
#        database = {}
#    for dst_tag in dst_tags:
#        database[dst_tag] = {}
#    ensemble = [os.path.join(src, f) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
#    # remove ,json
#    ensemble = [ x for x in ensemble if ".json" not in x ]
#    ensemble.sort(key=lambda f: int(re.sub('\D', '', f)))
#    for cfg in ensemble:
#        if verbose:
#            print(cfg)
#        with open(cfg) as f:
#            data = json.load(f)
#        for src_tag, dst_tag in zip(src_tags, dst_tags):
#            database[dst_tag][cfg_prefix + cfg.split(".")[-1]] = np.array(ast.literal_eval(data[src_tag]))
#    with open(dst, "w") as f:
#        json.dump(database, f)
