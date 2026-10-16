import numpy as np

from statpy.statistics import bootstrap, jackknife
from statpy.statistics import core as statistics


def _check_unique_cfgs(context, cfgs):
    """Raise ValueError if ``cfgs`` contains duplicate labels."""
    uniq, counts = np.unique(cfgs, return_counts=True)
    dup = uniq[counts > 1].tolist()
    if dup:
        raise ValueError(
            f"{context}: duplicate cfg label(s): {dup[:5]}"
            + ("..." if len(dup) > 5 else "")
        )


class Layout:
    """Configuration labels and optional weights."""

    def __init__(self, configurations, weights=None):
        if not isinstance(configurations, np.ndarray):
            raise TypeError(f"Layout: configurations must be np.ndarray, got {type(configurations).__name__}")
        if weights is not None:
            if not isinstance(weights, np.ndarray):
                raise TypeError(f"Layout: weights must be np.ndarray, got {type(weights).__name__}")
            if len(weights) != len(configurations):
                raise ValueError("Layout: weights/configurations length mismatch")
        _check_unique_cfgs("Layout", configurations)
        self.configurations = configurations
        self.weights = weights


class Variable:
    """A central value, replicates and optional per-configuration data."""

    def __init__(self, *, central_value=None, jackknife_replicates=None, bootstrap_replicates=None,
                 observations=None, layout=None, metadata=None):
        """Internal constructor; create variables with the classmethods."""
        self.central_value = central_value
        self.jackknife_replicates = jackknife_replicates
        self.observations = observations
        self.layout = layout
        self.bootstrap_replicates = bootstrap_replicates
        self.metadata = metadata

    @classmethod
    def from_data(cls, observations, layout, metadata=None, bootstraps=None):
        """Compute the weighted mean and jackknife replicates of data."""
        if layout is None:
            raise ValueError("from_data: observations requires layout")
        if layout.weights is None:
            raise ValueError("from_data: observations requires weights")
        if not isinstance(observations, np.ndarray):
            raise TypeError(f"from_data: observations must be np.ndarray, got {type(observations).__name__}")
        if not (len(observations) == len(layout.weights) == len(layout.configurations)):
            raise ValueError(
                f"from_data: length mismatch observations={len(observations)} "
                f"weights={len(layout.weights)} configurations={len(layout.configurations)}"
            )
        jackknife_replicates = jackknife.mean_replicates(observations, weights=layout.weights)
        central_value = np.average(observations, axis=0, weights=layout.weights)
        bootstrap_replicates = None if bootstraps is None else bootstrap.mean_replicates(observations, bootstraps, weights=layout.weights)
        return cls(central_value=central_value, jackknife_replicates=jackknife_replicates,
                   bootstrap_replicates=bootstrap_replicates, observations=observations, layout=layout, metadata=metadata)

    @classmethod
    def from_results(cls, central_value=None, jackknife_replicates=None, bootstrap_replicates=None,
                     layout=None, metadata=None):
        """Store computed values and replicates unchanged."""
        if jackknife_replicates is not None and layout is None:
            raise ValueError("from_results: jackknife_replicates requires layout")
        if jackknife_replicates is not None and len(jackknife_replicates) != len(layout.configurations):
            raise ValueError(
                "from_results: jackknife_replicates/configurations length mismatch "
                f"jackknife_replicates={len(jackknife_replicates)} configurations={len(layout.configurations)}"
            )
        for name, replicates in (("jackknife_replicates", jackknife_replicates),
                                 ("bootstrap_replicates", bootstrap_replicates)):
            if central_value is not None and replicates is not None and replicates.shape[1:] != np.shape(central_value):
                raise ValueError(f"from_results: {name}/central_value shape mismatch")
        return cls(central_value=central_value, jackknife_replicates=jackknife_replicates,
                   bootstrap_replicates=bootstrap_replicates, layout=layout, metadata=metadata)

    @classmethod
    def derive(cls, f, *variables):
        """Apply f to central values and matching replicates."""
        layout = cls._common_layout(variables, "derive")
        central_value = f(*[variable.central_value for variable in variables])
        jackknife_replicates = cls._derive_jackknife(f, variables)
        bootstrap_replicates = cls._derive_bootstrap(f, variables)
        return cls.from_results(central_value, jackknife_replicates, bootstrap_replicates, layout)

    @staticmethod
    def _common_layout(variables, name):
        layout = None
        for index, variable in enumerate(variables):
            if variable.layout is None:
                continue
            if layout is None:
                layout = variable.layout
            if not np.array_equal(variable.layout.configurations, layout.configurations):
                raise ValueError(f"{name}: input {index} configurations differ")
            weights = variable.layout.weights
            if (weights is None) != (layout.weights is None) or not np.array_equal(weights, layout.weights):
                raise ValueError(f"{name}: input {index} weights differ")
        return layout if all(variable.layout is not None for variable in variables) else None

    @staticmethod
    def _derive_jackknife(f, variables):
        if any(variable.jackknife_replicates is None for variable in variables):
            return None
        num = len(variables[0].jackknife_replicates)
        return np.array([f(*[variable.jackknife_replicates[i] for variable in variables])
                         for i in range(num)])

    @staticmethod
    def _derive_bootstrap(f, variables):
        if any(variable.bootstrap_replicates is None for variable in variables):
            return None
        num_bootstraps = variables[0].bootstrap_replicates.shape[0]
        for index, variable in enumerate(variables):
            if len(variable.bootstrap_replicates) != num_bootstraps:
                raise ValueError(f"derive: input {index} has a different number of bootstrap replicates")
        return np.array([f(*[variable.bootstrap_replicates[i] for variable in variables])
                         for i in range(num_bootstraps)])

    @classmethod
    def concatenate_components(cls, *variables, axis=0):
        """Join data variables along component ``axis``; layouts must be equal."""
        if any(variable.observations is None for variable in variables):
            raise ValueError("concatenate_components: input has no observations")
        layout = cls._common_layout(variables, "concatenate_components")
        observations_axis = axis + 1 if axis >= 0 else axis
        observations = np.concatenate([variable.observations for variable in variables], axis=observations_axis)
        return cls(central_value=np.concatenate([variable.central_value for variable in variables], axis=axis),
                   jackknife_replicates=jackknife.mean_replicates(observations, weights=layout.weights),
                   observations=observations, layout=layout, metadata={"num_inputs": len(variables)})

    def bin(self, bin_size, prefix):
        """Bin data in simulation order, truncating the trailing incomplete bin."""
        if bin_size <= 1:
            raise ValueError(f"bin: bin_size must be > 1, got {bin_size}")
        if self.observations is None or self.layout is None or self.layout.weights is None:
            raise ValueError("bin: variable has no observations or weights")
        num_bins = len(self.observations) // bin_size
        observations = statistics.bin(self.observations, bin_size, weights=self.layout.weights)
        weights = statistics.bin(self.layout.weights, bin_size, weights=np.ones(len(self.layout.weights)))
        configurations = np.array([f"{prefix}/bin_size{bin_size}-{i}" for i in range(num_bins)])
        return type(self).from_data(observations, Layout(configurations, weights), self.metadata)

    def _jackknife_replicates(self, bin_size):
        if self.jackknife_replicates is None:
            raise ValueError("Missing jackknife_replicates; use from_data or from_results(jackknife_replicates=...)")
        if bin_size == 1:
            return self.jackknife_replicates
        if self.layout is None or self.layout.weights is None:
            raise ValueError("Delayed binning requires weights")
        return jackknife.delayed_binning(self.jackknife_replicates, bin_size, self.layout.weights)

    def jackknife_variance(self, bin_size=1):
        """Return the jackknife variance; delayed binning for ``bin_size > 1``."""
        return jackknife.variance(self._jackknife_replicates(bin_size))

    def jackknife_covariance(self, bin_size=1):
        """Return the jackknife covariance; delayed binning for ``bin_size > 1``."""
        return jackknife.covariance(self._jackknife_replicates(bin_size))

    def bootstrap_variance(self):
        """Return the variance of stored bootstrap replicates."""
        if self.bootstrap_replicates is None:
            raise ValueError("Missing bootstrap_replicates; use from_results(bootstrap_replicates=...)")
        return bootstrap.variance(self.bootstrap_replicates)

    def bootstrap_covariance(self):
        """Return the covariance of stored bootstrap replicates."""
        if self.bootstrap_replicates is None:
            raise ValueError("Missing bootstrap_replicates; use from_results(bootstrap_replicates=...)")
        return bootstrap.covariance(self.bootstrap_replicates)

    def __repr__(self):
        parts = [f"shape={np.shape(self.central_value)}"]
        for name, replicates, count in (("observations", self.observations, "N"),
                                        ("jackknife", self.jackknife_replicates, "N"),
                                        ("bootstrap", self.bootstrap_replicates, "B")):
            if replicates is not None:
                parts.append(f"{name} {count}={len(replicates)}")
        return f"Variable({', '.join(parts)})"
