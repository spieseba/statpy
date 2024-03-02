from time import time

t0 = time()

def message(s="", verbosity=0):
    if verbosity >= 0: 
        print(f"STATPY:\t\t{time()-t0:.6f} s: {s}")