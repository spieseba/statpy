from time import perf_counter

_t0 = None


def message(s="", silent=False, *, continuation=False):
    """Print elapsed seconds with a statpy prefix and aligned continuations.

    The timer starts at the first visible message; silent calls do not start it.
    Set continuation=True to align under the preceding message without a prefix.
    """
    global _t0
    if silent:
        return
    now = perf_counter()
    if _t0 is None:
        _t0 = now
    prefix = f"statpy {now - _t0:9.3f} s: "
    indent = " " * len(prefix)
    print((indent if continuation else prefix) + str(s).replace("\n", "\n" + indent))
