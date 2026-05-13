from time import time

_t0 = None


def message(s="", silent=False):
    """Print a STATPY-prefixed log line with seconds since the first call.

    ``t0`` is captured lazily on the first invocation so the timestamp
    reflects "seconds since first log message", not import time.
    """
    global _t0
    if _t0 is None:
        _t0 = time()
    if not silent:
        print(f"STATPY:\t\t{time()-_t0:.6f} s: {s}")
