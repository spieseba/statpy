import numpy as np


class Entry:
    """Atomic data unit in a :class:`DB`.

    A entry either carries per-cfg data (``sample`` + ``weights`` + ``cfgs``,
    plus the derived ``jks``) or a cfg-less result (``central_value`` and
    optional ``bss``, with ``cfgs=None``).

    ``central_value`` defaults to the weighted mean of ``sample`` (see
    :meth:`DB.add_entry`) but may be any central estimate, e.g. a fit result.

    Arrays are pre-sorted at insertion time and never re-sorted on read.
    Do not mutate them in place.
    """

    def __init__(self, *, central_value=None, jks=None, sample=None, weights=None,
                 cfgs=None, bss=None, misc=None, binsize=1):
        self.central_value = central_value  # np.ndarray, shape value_shape
        self.jks = jks          # np.ndarray, shape (n_cfgs, *value_shape)
        self.sample = sample    # np.ndarray, shape (n_cfgs, *value_shape)
        self.weights = weights  # np.ndarray, shape (n_cfgs,)
        self.cfgs = cfgs        # np.ndarray[str], shape (n_cfgs,)
        self.bss = bss          # np.ndarray, shape (n_bs, *value_shape)
        self.misc = misc        # dict or None
        self.binsize = binsize  # 1 = unbinned/raw, N>1 = binned (jks/sample have n_bins entries)
