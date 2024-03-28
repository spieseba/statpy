import binascii, os, re, struct, sys
import numpy as np
from statpy.log import message

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

    def load(self, fn, pattern, verbosity=-1):
        try:
            f=open(fn,"rb")
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
                    message(f"Tag[{tag[0:-1].decode('ascii'):s}]] Size[{ln:d}] Flags[{self._flag_str(flags):s}] CRC32[{crc32:X}]", verbosity=verbosity)
                    corr = []
                    if flags != self.R_EMPTY:
                        for j in range(ln):
                            corr.append(rd[j*2+0]+1j*rd[j*2+1])
                    f.close()
                    return match.string, np.array(corr)
                else:
                    f.seek(lnr*16 // nf,1)
            f.close()
        except:
           raise

    def get_tags(self, fn, pattern, verbosity=-1):
        try:
            f=open(fn,"rb")
            tags = []
            while True:
                rd = f.read(4)
                if len(rd) == 0:
                    break
                ntag=struct.unpack('i', rd)[0]
                tag=f.read(ntag)
                (crc32,ln,flags)=struct.unpack('IHH', f.read(4*2))
                nf = 1
                if flags & (self.R_REAL|self.R_IMAG):
                    nf = 2
                if flags & (self.R_SYMM|self.R_ASYMM):
                    ln = ln/2+1
                if flags & self.R_EMPTY:
                    ln = 0
                message(f"Tag[{tag[0:-1]}] Size[{ln}] Flags[{self._flag_str(flags)}] CRC32[{crc32}]", verbosity=verbosity)
                match = re.search(pattern, tag[0:-1].decode("ascii"))
                if match:
                    tags.append(match.string)
                f.seek(ln*16 // nf,1)
            f.close()
            return np.array(tags)
        except:
            raise
