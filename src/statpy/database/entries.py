

class Entry:
    """One quantity in a :class:`DB` (a correlator, a fit result, ...) with its
    central value and resamples.

    A data entry holds per-configuration measurements (``samples``, ``weights``,
    ``configurations``); its estimate is the weighted mean, with jackknife
    samples derived from it. Any other entry holds a computed result, e.g. from a
    fit, a transformation or another estimator: any of ``central_value``,
    ``jackknife_samples`` (one per label in ``configurations``) and
    ``bootstrap_samples``.

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
