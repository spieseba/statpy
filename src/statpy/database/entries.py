import numpy as np


class Entry:
    """Atomic data unit in a :class:`DB`.

    A entry either carries per-cfg data (``sample`` + ``weights`` + ``cfgs``,
    plus the derived ``jks``) or a cfg-less result (``mean`` and optional
    ``bss``, with ``cfgs=None``).

    Arrays are pre-sorted at insertion time and never re-sorted on read.
    Do not mutate them in place.
    """

    def __init__(self, *, mean=None, jks=None, sample=None, weights=None,
                 cfgs=None, bss=None, misc=None, binsize=1):
        self.mean = mean
        self.jks = jks          # np.ndarray, shape (n_cfgs, *value_shape)
        self.sample = sample    # np.ndarray, shape (n_cfgs, *value_shape)
        self.weights = weights  # np.ndarray, shape (n_cfgs,)
        self.cfgs = cfgs        # np.ndarray[str], shape (n_cfgs,)
        self.bss = bss          # np.ndarray, shape (n_bs, *value_shape)
        self.misc = misc        # dict or None
        self.binsize = binsize  # 1 = unbinned/raw, N>1 = binned (jks/sample have n_bins entries)
