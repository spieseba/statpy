#!/usr/bin/env python3

import os
from statpy.dbpy import custom_json as json

try:
    import gpt as g
    def rank_func():
        return g.rank()
    def barrier():
        return g.barrier()
except ImportError:
    def rank_func():
        return 0
    def barrier():
        pass

class IO:
    def __init__(self, path, filename):
        self.src = path
        self.dst = path + filename
        if rank_func() == 0:
            if os.path.isfile(self.dst):
                with open(self.dst) as f:
                    self.data = json.load(f)
            else:
                self.data = {}
    def store(self, key, value):
        if rank_func() == 0:
            if self.data is not None:
                    self.data[key] = value
    def save(self):
        if rank_func() == 0:
            with open(self.dst, "w") as f:
                json.dump(self.data, f)
        barrier()
