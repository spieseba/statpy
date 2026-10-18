import numpy as np

from statpy.statistics import bootstrap, gamma, jackknife
from statpy.statistics import core as statistics


class Layout:
    """Ensemble, Monte Carlo times, replicas and weights of the configurations.

    Required: ``ensemble`` (name) and ``monte_carlo_times`` (one integer per configuration,
    strictly increasing in array order within each replica). Optional: ``replicas`` (one integer
    per configuration; default ``zeros(N)``, one replica) and ``weights`` (reweighting factors;
    default ``ones(N)``). A configuration is identified by (ensemble, replica, Monte Carlo time).
    """

    def __init__(self, ensemble, monte_carlo_times, *, replicas=None, weights=None):
        num_configurations = len(monte_carlo_times)
        if replicas is None:
            replicas = np.zeros(num_configurations, dtype=int)
        if weights is None:
            weights = np.ones(num_configurations)

        if not isinstance(ensemble, str):
            raise TypeError(f"Layout: ensemble must be str, got {type(ensemble).__name__}")
        arrays = {"monte_carlo_times": monte_carlo_times, "replicas": replicas, "weights": weights}
        for name, values in arrays.items():
            if not isinstance(values, np.ndarray):
                raise TypeError(f"Layout: {name} must be np.ndarray, got {type(values).__name__}")
            if values.ndim != 1:
                raise ValueError(f"Layout: {name} must be 1-D, got {values.ndim}-D")
            if len(values) != num_configurations:
                raise ValueError(f"Layout: {name} has length {len(values)}, expected {num_configurations}")
        for name in ("monte_carlo_times", "replicas"):
            if not np.issubdtype(arrays[name].dtype, np.integer):
                raise ValueError(f"Layout: {name} must be integer, got {arrays[name].dtype}")
        for replica in np.unique(replicas):
            if np.any(np.diff(monte_carlo_times[replicas == replica]) <= 0):
                raise ValueError(f"Layout: monte_carlo_times not strictly increasing in replica {replica}")

        self.ensemble = ensemble
        self.monte_carlo_times = monte_carlo_times
        self.replicas = replicas
        self.weights = weights


class Variable:
    """A quantity with its central value and replicates for error estimation.

    Attributes (``None`` if absent):
    - ``central_value``: estimate of the quantity.
    - ``observations``: data per configuration (configurations on axis 0).
    - ``layout``: ``Layout`` of the configurations.
    - ``jackknife_replicates``: delete-one replicates, one per configuration.
    - ``bootstrap_replicates``: one replicate per bootstrap sample.
    - ``fluctuations``: per-configuration fluctuations for the gamma method.

    A data variable (made by ``from_data``) has observations and fluctuations; ``derive`` and
    ``from_results`` return variables without them.
    """

    def __init__(self, *, central_value=None, observations=None, layout=None, jackknife_replicates=None,
                 bootstrap_replicates=None, fluctuations=None):
        """Internal constructor; create variables with the classmethods."""
        self.central_value = central_value
        self.observations = observations
        self.layout = layout
        self.jackknife_replicates = jackknife_replicates
        self.bootstrap_replicates = bootstrap_replicates
        self.fluctuations = fluctuations

    @classmethod
    def from_data(cls, observations, layout, bootstrap_indices=None):
        """Return a data variable: weighted mean, jackknife replicates and gamma-method fluctuations.

        ``observations`` has configurations on axis 0, in ``layout`` order. ``bootstrap_indices`` is an
        optional index matrix (num_bootstraps, N); with it, bootstrap replicates are computed too.
        """
        if layout is None:
            raise ValueError("from_data: observations requires layout")
        if not isinstance(observations, np.ndarray):
            raise TypeError(f"from_data: observations must be np.ndarray, got {type(observations).__name__}")
        num_configurations = len(layout.monte_carlo_times)
        if len(observations) != num_configurations:
            raise ValueError(f"from_data: observations have length {len(observations)}, expected {num_configurations}")

        weights = layout.weights
        jackknife_replicates = jackknife.mean_replicates(observations, weights=weights)
        bootstrap_replicates = None
        if bootstrap_indices is not None:
            bootstrap_replicates = bootstrap.mean_replicates(observations, bootstrap_indices, weights=weights)
        mean, fluctuations = gamma.mean_and_fluctuations(observations, weights)
        return cls(central_value=mean, observations=observations, layout=layout,
                   jackknife_replicates=jackknife_replicates, bootstrap_replicates=bootstrap_replicates,
                   fluctuations=fluctuations)

    @classmethod
    def from_results(cls, central_value=None, layout=None, jackknife_replicates=None, bootstrap_replicates=None):
        """Return a variable from a computed central value and replicates, stored unchanged."""
        if jackknife_replicates is not None:
            if layout is None:
                raise ValueError("from_results: jackknife_replicates requires layout")
            num_configurations = len(layout.monte_carlo_times)
            if len(jackknife_replicates) != num_configurations:
                raise ValueError(f"from_results: jackknife_replicates have length {len(jackknife_replicates)}, "
                                 f"expected {num_configurations}")
        if central_value is not None:
            shape = np.shape(central_value)
            for name, replicates in (("jackknife_replicates", jackknife_replicates),
                                     ("bootstrap_replicates", bootstrap_replicates)):
                if replicates is not None and replicates.shape[1:] != shape:
                    raise ValueError(f"from_results: {name} have shape {replicates.shape[1:]}, expected {shape}")
        return cls(central_value=central_value, layout=layout, jackknife_replicates=jackknife_replicates,
                   bootstrap_replicates=bootstrap_replicates)

    @classmethod
    def derive(cls, f, *variables):
        """Apply f to the central values and to the replicates present in all inputs; layouts must be equal."""
        layout = cls._common_layout(variables, "derive")
        central_value = f(*[variable.central_value for variable in variables])
        jackknife_replicates = cls._derive_jackknife(f, variables)
        bootstrap_replicates = cls._derive_bootstrap(f, variables)
        return cls.from_results(central_value=central_value, layout=layout,
                                jackknife_replicates=jackknife_replicates, bootstrap_replicates=bootstrap_replicates)

    @classmethod
    def map_observations(cls, f, *variables):
        """Apply f to the observations of data variables; layouts must be equal. Return a data variable.

        f takes the observation arrays (configurations on axis 0) and must act on each configuration separately.
        """
        if any(variable.observations is None for variable in variables):
            raise ValueError("map_observations: input has no observations")
        layout = cls._common_layout(variables, "map_observations")
        observations = f(*[variable.observations for variable in variables])
        return cls.from_data(observations, layout)

    @staticmethod
    def _common_layout(variables, name):
        layouts = [variable.layout for variable in variables]
        reference = next((layout for layout in layouts if layout is not None), None)
        for index, layout in enumerate(layouts):
            if layout is None:
                continue
            for field in ("ensemble", "monte_carlo_times", "replicas", "weights"):
                if not np.array_equal(getattr(layout, field), getattr(reference, field)):
                    raise ValueError(f"{name}: input {index} {field} differ")
        if any(layout is None for layout in layouts):
            return None
        return reference

    @staticmethod
    def _derive_jackknife(f, variables):
        if any(variable.jackknife_replicates is None for variable in variables):
            return None
        num_configurations = len(variables[0].jackknife_replicates)
        return np.array([f(*[variable.jackknife_replicates[i] for variable in variables])
                         for i in range(num_configurations)])

    @staticmethod
    def _derive_bootstrap(f, variables):
        if any(variable.bootstrap_replicates is None for variable in variables):
            return None
        num_bootstraps = len(variables[0].bootstrap_replicates)
        for index, variable in enumerate(variables):
            if len(variable.bootstrap_replicates) != num_bootstraps:
                raise ValueError(f"derive: input {index} has a different number of bootstrap replicates")
        return np.array([f(*[variable.bootstrap_replicates[i] for variable in variables])
                         for i in range(num_bootstraps)])

    @classmethod
    def concatenate_components(cls, *variables, axis=0):
        """Join data variables along component ``axis``; layouts must be equal."""
        observations_axis = axis + 1 if axis >= 0 else axis
        return cls.map_observations(lambda *observations: np.concatenate(observations, axis=observations_axis),
                                    *variables)

    def bin(self, bin_size):
        """Return the data variable binned in array order, truncating the trailing incomplete bin.

        A bin holds ``bin_size`` measured configurations; with gaps it spans more Monte Carlo time.
        Requires one replica. The result keeps the ensemble and the replica and has Monte Carlo
        times ``0, ..., num_bins - 1``. Binned layouts identify ensemble, replica and number of bins only.
        ``bin_size == 1`` returns the variable itself.
        """
        if bin_size < 1:
            raise ValueError(f"bin: bin_size must be >= 1, got {bin_size}")
        if self.observations is None:
            raise ValueError("bin: variable has no observations")
        if bin_size == 1:
            return self
        if len(np.unique(self.layout.replicas)) > 1:
            raise ValueError("bin: more than one replica; binning across replicas is not supported")

        weights = self.layout.weights
        num_bins = len(self.observations) // bin_size
        observations = statistics.bin(self.observations, bin_size, weights=weights)
        bin_weights = statistics.bin(weights, bin_size, weights=np.ones(len(weights)))
        replicas = np.full(num_bins, self.layout.replicas[0])
        layout = Layout(self.layout.ensemble, np.arange(num_bins), replicas=replicas, weights=bin_weights)
        return type(self).from_data(observations, layout)

    def _jackknife_replicates(self, bin_size):
        if self.jackknife_replicates is None:
            raise ValueError("jackknife: variable has no jackknife replicates")
        if bin_size == 1:
            return self.jackknife_replicates
        if len(np.unique(self.layout.replicas)) > 1:
            raise ValueError("Delayed binning: more than one replica; binning across replicas is not supported")
        return jackknife.delayed_binning(self.jackknife_replicates, bin_size, self.layout.weights)

    def jackknife_variance(self, bin_size=1):
        """Return the jackknife variance; delayed binning (one replica only) for ``bin_size > 1``."""
        return jackknife.variance(self._jackknife_replicates(bin_size))

    def jackknife_covariance(self, bin_size=1):
        """Return the jackknife covariance; delayed binning (one replica only) for ``bin_size > 1``."""
        return jackknife.covariance(self._jackknife_replicates(bin_size))

    def bootstrap_variance(self):
        """Return the bootstrap variance."""
        if self.bootstrap_replicates is None:
            raise ValueError("bootstrap_variance: variable has no bootstrap replicates")
        return bootstrap.variance(self.bootstrap_replicates)

    def bootstrap_covariance(self):
        """Return the bootstrap covariance."""
        if self.bootstrap_replicates is None:
            raise ValueError("bootstrap_covariance: variable has no bootstrap replicates")
        return bootstrap.covariance(self.bootstrap_replicates)

    def gamma_method(self, S=1.5) -> gamma.GammaErrorAnalysis:
        """Return the gamma-method error analysis per component (Wolff, hep-lat/0306017).

        ``S`` is the factor in Wolff's hypothesis tau ~ S tau_int (sec. 3.3) that sets the
        automatic window. Fields: see ``GammaErrorAnalysis``. The variable is unchanged.
        """
        if self.fluctuations is None:
            raise ValueError("gamma_method: variable has no fluctuations")
        layout = self.layout
        return gamma.error_analysis(self.fluctuations, layout.monte_carlo_times, layout.replicas, S)

    def __repr__(self):
        parts = [f"shape={np.shape(self.central_value)}"]
        for name, replicates, count in (("observations", self.observations, "N"),
                                        ("jackknife", self.jackknife_replicates, "N"),
                                        ("bootstrap", self.bootstrap_replicates, "B")):
            if replicates is not None:
                parts.append(f"{name} {count}={len(replicates)}")
        return f"Variable({', '.join(parts)})"
