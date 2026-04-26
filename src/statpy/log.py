from time import time

t0 = time()

def message(s="", silent=False):
    if not silent:
        print(f"STATPY:\t\t{time()-t0:.6f} s: {s}")