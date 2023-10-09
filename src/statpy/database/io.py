#!/usr/bin/env python3

import binascii, h5py, os, re, struct, sys
import numpy as np
from statpy.database import custom_json as json
from .leafs import Leaf

class Database_IO:
    def __init__(self):
        self.database = {}

    def store(self, key, value):
        self.database[key] = value

    def save(self, dst):
        with open(dst, "w") as f:
            json.dump(self.database, f)

    def create_SAMPLE_DB(self, src_dir, src_tags, branch_tag, leaf_prefix, dst, filter_str=None, dst_tags=None):
        print("THIS METHOD IS DEPRECATED AND WILL BE REMOVED SOMETIME IN THE FUTURE")
        filenames = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if (os.path.isfile(os.path.join(src_dir, f)))]
        if filter_str is not None:
            filenames = [x for x in filenames if filter_str in x]
        filenames = sorted(filenames, key=lambda x: int(x.split("ckpoint_lat.")[-1].split(".")[0]))
        cfgs = [branch_tag + "-" + x.split("ckpoint_lat.")[-1].split(".")[0] for x in filenames]
        if dst_tags is None:
            dst_tags = src_tags
        for stag, dtag in zip(src_tags, dst_tags):
            print(f"store {stag} as {dtag} in database")
            sample = {}
            for cfg, fn in zip(cfgs, filenames):
                with open(fn) as f:
                    data = json.load(f)
                sample[cfg] = data[stag] 
            self.database[f"{leaf_prefix}/{dtag}"] = Leaf(mean=None, jks=None, sample=sample)
        self.save(dst)

class GPT_IO:
    def __init__(self):
        self.R_NONE    = 0x00
        self.R_EMPTY   = 0x01
        self.R_REAL    = 0x02
        self.R_IMAG    = 0x04
        self.R_SYMM    = 0x08
        self.R_ASYMM   = 0x10

    def _flag_str(self, f):
        r=""
        if f & self.R_EMPTY:
            r += "empty "
        if f & self.R_REAL:
            r += "real "
        if f & self.R_IMAG:
            r += "imag "
        if f & self.R_SYMM:
            r += "symm "
        if f & self.R_ASYMM:
            r += "asymm "
        return r.strip()

    def _reconstruct_full(self, flags, i):
        if flags & self.R_SYMM:
            N=len(i)/2
            for j in range(N/2+1,N):
                jm=N - j
                i[2*j + 0] = i[2*jm + 0]
                i[2*j + 1] = -i[2*jm + 1]
        if flags & self.R_ASYMM:
            N=len(i)/2
            for j in range(N/2+1,N):
                jm=N - j
                i[2*j + 0] = -i[2*jm + 0]
                i[2*j + 1] = i[2*jm + 1]

    def _reconstruct_min(self, flags, i, NT):
        if flags & self.R_EMPTY:
            return [ 0.0 for l in 2*range(NT) ]
        if flags == 0:
            return i
        o=[ 0.0 for l in 2*range(NT) ]
        # first fill in data at right places
        i0=0
        istep=1
        if flags & self.R_REAL:
            istep=2
        if flags & self.R_IMAG:
            istep=2
            i0=1
        for j in range(len(i)):
            o[istep*j + i0] = i[j]
        return o

    def load(self, fn, pattern):
        f=open(fn,"rb")
        try:
            while True:
                rd=f.read(4)
                if len(rd) == 0:
                    break
                ntag=struct.unpack('i', rd)[0]
                tag=f.read(ntag)
                (crc32,ln,flags)=struct.unpack('IHH', f.read(4*2))
                nf = 1
                lnr = ln
                if flags & (self.R_REAL|self.R_IMAG):
                    nf = 2
                if flags & (self.R_SYMM|self.R_ASYMM):
                    lnr = ln//2+1
                if flags & self.R_EMPTY:
                    lnr = 0
                match = re.search(pattern, tag[0:-1].decode("ascii"))
                if match:
                    rd = self._reconstruct_min(flags, struct.unpack('d'*(2*lnr // nf), f.read(16*lnr // nf) ), ln)
                    crc32comp = (binascii.crc32(struct.pack('d'*2*ln,*rd)) & 0xffffffff)
                    self._reconstruct_full(flags, rd )
                    if crc32comp != crc32:
                        print("Data corrupted!")
                        f.close()
                        sys.exit(1)
                    print(f"Tag[{tag[0:-1].decode('ascii'):s}]] Size[{ln:d}] Flags[{self._flag_str(flags):s}] CRC32[{crc32:X}]")
                    corr = []
                    if flags != self.R_EMPTY:
                        for j in range(ln):
                            corr.append(rd[j*2+0]+1j*rd[j*2+1])
                    return match.string, np.array(corr)
                else:
                    f.seek(lnr*16 // nf,1)
            f.close()
        except:
           raise

def load_CLS(fn, rwf_fn, tags, branch_tag):
    assert os.path.isfile(fn)
    assert os.path.isfile(rwf_fn)
    measurements = {}
    # rwfs
    rwf_cfgs = np.array(np.loadtxt(rwf_fn)[:,0], dtype=int)
    rwf = np.loadtxt(rwf_fn)[:,1]; nrwf = rwf / np.mean(rwf)
    nrwf = {f"{branch_tag}-{cfg}":val for cfg, val in zip(rwf_cfgs, nrwf)} 
    measurements[f"{branch_tag}/nrwf"] = Leaf(mean=None, jks=None, sample=nrwf)
    # data
    f = h5py.File(fn, "r")
    f_cfgs = np.array([int(cfg.decode("utf-8").split("n")[1]) for cfg in f.get("configlist")])
    cfgs = np.array([cfg for cfg in rwf_cfgs if cfg in f_cfgs])
    for t in tags:
        for key in f.keys():
            if t in key:
                sample = {f"{branch_tag}-{cfg}":val for cfg, val in zip(cfgs, f.get(key)[cfgs-1])}
                measurements[f"{branch_tag}/{key}"] = Leaf(mean=None, jks=None, sample=sample)
    return measurements