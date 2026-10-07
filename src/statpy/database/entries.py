

class Entry:
    """Atomic data unit in a :class:`DB`.

    A entry either carries per-cfg data (``samples`` + ``weights`` +
    ``configurations``, plus the derived ``jackknife_samples``) or a cfg-less
    result (``central_value`` and optional ``bootstrap_samples``, with
    ``configurations=None``).

    ``central_value`` defaults to the weighted mean of ``samples`` (see
    :meth:`DB.add_entry`) but may be any central estimate, e.g. a fit result.

    Arrays are pre-sorted at insertion time and never re-sorted on read.
    Do not mutate them in place.
    """

    def __init__(self, *, central_value=None, jackknife_samples=None, samples=None, weights=None,
                 configurations=None, bootstrap_samples=None, metadata=None, bin_size=1):
        self.central_value = central_value          # np.ndarray, shape value_shape
        self.jackknife_samples = jackknife_samples  # np.ndarray, shape (num_configurations, *value_shape)
        self.samples = samples                      # np.ndarray, shape (num_configurations, *value_shape)
        self.weights = weights                      # np.ndarray, shape (num_configurations,)
        self.configurations = configurations        # np.ndarray[str], shape (num_configurations,)
        self.bootstrap_samples = bootstrap_samples  # np.ndarray, shape (num_bs, *value_shape)
        self.metadata = metadata                    # dict or None
        self.bin_size = bin_size  # 1 = unbinned/raw, N>1 = binned (samples/jackknife_samples have num_bins entries)
