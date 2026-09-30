from time import perf_counter

import numpy as np

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


def format_paren(val, err, sig=2):
    """Format ``val(err)`` with ``sig`` significant digits on ``err``, e.g. ``0.1234(56)``.

    Values with magnitude below 1e-3 or from 1e4 on get an exponent, e.g. ``3.548(48)e-07``.
    """
    if not np.isfinite(err) or err == 0:
        return f"{val}"
    exponent = int(np.floor(np.log10(abs(val)))) if val != 0 else 0
    if exponent < -3 or exponent >= 4:
        return f"{_paren(val / 10**exponent, err / 10**exponent, sig)}e{exponent:+03d}"
    return _paren(val, err, sig)


def _paren(val, err, sig):
    """Format ``val(err)`` in fixed-point notation."""
    k = int(np.floor(np.log10(abs(err))))
    decimals = max(0, -(k - (sig - 1)))
    err_digits = round(err * 10 ** decimals)
    if err_digits == 10 ** sig and decimals > 0:
        # rounding crossed a decade (e.g. 0.996e-3 -> "(100)"): re-round there
        decimals -= 1
        err_digits = round(err * 10 ** decimals)
    return f"{val:.{decimals}f}({err_digits})"
