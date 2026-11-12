import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Self

import jax
import numpy as np
from numpy.typing import ArrayLike, NDArray

from statpy import log
from statpy.plot import legend_above, select_component, style_axes
from statpy.statistics import bootstrap, gamma, jackknife, least_squares
from statpy.statistics import core as statistics

if TYPE_CHECKING:
    from matplotlib.axes import Axes


class Layout:
    """Ensembles, Monte Carlo times, replicas and weights of the configurations.

    Required: ``ensembles`` (one name for all configurations, or one name per configuration in
    sorted blocks) and ``monte_carlo_times`` (one integer per configuration, strictly increasing
    in array order within each replica of an ensemble). Optional: ``replicas`` (one integer
    per configuration; default ``zeros(N)``, one replica) and ``weights`` (reweighting factors;
    default ``ones(N)``). A configuration is identified by (ensemble, replica, Monte Carlo time).
    """

    def __init__(self, ensembles: str | NDArray[np.str_], monte_carlo_times: NDArray[np.integer], *,
                 replicas: NDArray[np.integer] | None = None, weights: np.ndarray | None = None) -> None:
        num_configurations = len(monte_carlo_times)
        if isinstance(ensembles, str):
            ensembles = np.full(num_configurations, ensembles)
        if replicas is None:
            replicas = np.zeros(num_configurations, dtype=int)
        if weights is None:
            weights = np.ones(num_configurations)

        arrays = {"ensembles": ensembles, "monte_carlo_times": monte_carlo_times, "replicas": replicas,
                  "weights": weights}
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
        if not np.issubdtype(ensembles.dtype, np.str_):
            raise ValueError(f"Layout: ensembles must be strings, got {ensembles.dtype}")
        if np.any(ensembles[:-1] > ensembles[1:]):
            raise ValueError("Layout: ensembles must be in sorted blocks")
        for ensemble in np.unique(ensembles):
            for replica in np.unique(replicas[ensembles == ensemble]):
                selected = (ensembles == ensemble) & (replicas == replica)
                if np.any(np.diff(monte_carlo_times[selected]) <= 0):
                    raise ValueError("Layout: monte_carlo_times not strictly increasing in "
                                     f"ensemble {ensemble}, replica {replica}")

        self.ensembles = ensembles
        self.monte_carlo_times = monte_carlo_times
        self.replicas = replicas
        self.weights = weights

    @property
    def ensemble_names(self) -> tuple[str, ...]:
        """The names of the ensembles in sorted order."""
        return tuple(np.unique(self.ensembles).tolist())

    @staticmethod
    def merge(*layouts: Layout) -> Layout:
        """Layout with the configurations of all ensembles of ``layouts``, ensembles in sorted order; layouts with
        the same ensemble must have the same configurations of it."""
        fields = ("ensembles", "monte_carlo_times", "replicas", "weights")
        parts = []
        for name in sorted({name for layout in layouts for name in layout.ensemble_names}):
            reference = None
            for index, layout in enumerate(layouts):
                if name not in layout.ensemble_names:
                    continue
                part = [getattr(layout, field)[layout.ensembles == name] for field in fields]
                if reference is None:
                    reference = part
                for field, values, reference_values in zip(fields, part, reference):
                    if not np.array_equal(values, reference_values):
                        raise ValueError(f"Layout.merge: layout {index} {field} differ in ensemble {name}")
            parts.append(reference)
        ensembles, monte_carlo_times, replicas, weights = (np.concatenate(values) for values in zip(*parts))
        return Layout(ensembles, monte_carlo_times, replicas=replicas, weights=weights)


class Variable:
    """A quantity with its central value and replicates for error estimation.

    Attributes (``None`` if absent; ``layout`` and ``central_value`` are always set):
    - ``observations``: data per configuration (configurations on axis 0).
    - ``layout``: ``Layout`` of the configurations.
    - ``central_value``: estimate of the quantity.
    - ``replica_central_values``: estimate from each replica alone, replicas in sorted label order,
      for the replica bias correction and check.
    - ``fluctuations``: per-configuration fluctuations for the gamma method.
    - ``jackknife_replicates``: delete-one replicates, one per configuration.
    - ``bootstrap_replicates``: one replicate per bootstrap sample.

    A data variable (made by ``from_data``) has observations, replica central values and fluctuations.
    ``derive`` and ``from_results`` return variables without observations; ``derive`` propagates
    replica central values and fluctuations when all inputs have them.
    """

    # ---------------------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------------------

    def __init__(self, *, observations: np.ndarray | None = None, layout: Layout, central_value: np.ndarray,
                 replica_central_values: np.ndarray | None = None, fluctuations: np.ndarray | None = None,
                 jackknife_replicates: np.ndarray | None = None, bootstrap_replicates: np.ndarray | None = None) -> None:
        """Internal constructor; create variables with the classmethods."""
        self.observations = observations
        self.layout = layout
        self.central_value = central_value
        self.replica_central_values = replica_central_values
        self.fluctuations = fluctuations
        self.jackknife_replicates = jackknife_replicates
        self.bootstrap_replicates = bootstrap_replicates

    def __repr__(self) -> str:
        layout = self.layout
        labels = np.unique(layout.replicas)
        reweighted = ", reweighted" if np.any(layout.weights != 1) else ""
        lines = [(f"Variable of shape {np.shape(self.central_value)} on ensemble {', '.join(layout.ensemble_names)}, "
                  f"{len(layout.replicas)} configurations{reweighted}")]
        for label in labels:
            times = layout.monte_carlo_times[layout.replicas == label]
            if len(times) == 1:
                line = f"Monte Carlo time {times[0]}"
            else:
                step = np.gcd.reduce(np.diff(times))
                missing = (times[-1] - times[0]) // step + 1 - len(times)
                line = f"Monte Carlo times {times[0]} to {times[-1]} in steps of {step}"
                line += f", {missing} missing" if missing else ""
            lines.append(line if len(labels) == 1 else f"replica {label}: {len(times)} configurations, {line}")
        bootstrap = "bootstrap" if self.bootstrap_replicates is None else f"bootstrap (B = {len(self.bootstrap_replicates)})"
        stored = [name for name, value in (("observations", self.observations),
                                           ("replica central values", self.replica_central_values),
                                           ("fluctuations", self.fluctuations), ("jackknife", self.jackknife_replicates),
                                           (bootstrap, self.bootstrap_replicates))
                  if value is not None]
        lines.append(f"stored: {', '.join(stored) or 'central value only'}")
        return "\n  ".join(lines)

    @classmethod
    def from_data(cls, observations: np.ndarray, layout: Layout,
                  bootstrap_indices: NDArray[np.integer] | None = None) -> Self:
        """Return a data variable: weighted mean, the weighted mean of each replica, gamma-method fluctuations
        and jackknife replicates.

        ``observations`` has configurations on axis 0, in ``layout`` order. ``bootstrap_indices`` is an
        optional index matrix (num_bootstraps, N); with it, bootstrap replicates are computed too.
        Jackknife and bootstrap need one replica; with several, the variable has no jackknife replicates and
        ``bootstrap_indices`` raises.
        """
        if not isinstance(observations, np.ndarray):
            raise TypeError(f"from_data: observations must be np.ndarray, got {type(observations).__name__}")
        if len(layout.ensemble_names) > 1:
            raise ValueError(f"from_data: observations need one ensemble, got {len(layout.ensemble_names)}")
        num_configurations = len(layout.monte_carlo_times)
        if len(observations) != num_configurations:
            raise ValueError(f"from_data: observations have length {len(observations)}, expected {num_configurations}")
        replicas = np.unique(layout.replicas)
        if len(replicas) > 1 and bootstrap_indices is not None:
            raise ValueError(f"from_data: bootstrap needs one replica, got {len(replicas)}")

        weights = layout.weights
        mean, fluctuations = gamma.mean_and_fluctuations(observations, weights)
        replica_central_values = np.stack([
            np.average(observations[layout.replicas == replica], axis=0, weights=weights[layout.replicas == replica])
            for replica in replicas])
        jackknife_replicates = jackknife.mean_replicates(observations, weights=weights) if len(replicas) == 1 else None
        bootstrap_replicates = None
        if bootstrap_indices is not None:
            bootstrap_replicates = bootstrap.mean_replicates(observations, bootstrap_indices, weights=weights)
        return cls(observations=observations, layout=layout, central_value=mean,
                   replica_central_values=replica_central_values, fluctuations=fluctuations,
                   jackknife_replicates=jackknife_replicates, bootstrap_replicates=bootstrap_replicates)

    @classmethod
    def from_results(cls, *, layout: Layout, central_value: np.ndarray,
                     replica_central_values: np.ndarray | None = None, fluctuations: np.ndarray | None = None,
                     jackknife_replicates: np.ndarray | None = None, bootstrap_replicates: np.ndarray | None = None) -> Self:
        """Return a variable from a computed central value, replica central values, fluctuations and replicates,
        stored unchanged."""
        num_configurations = len(layout.monte_carlo_times)
        num_replicas = len(_row_ensembles(layout, "replica_central_values"))
        for name, values, expected in (("replica_central_values", replica_central_values, num_replicas),
                                       ("fluctuations", fluctuations, num_configurations),
                                       ("jackknife_replicates", jackknife_replicates, num_configurations)):
            if values is not None and len(values) != expected:
                raise ValueError(f"from_results: {name} have length {len(values)}, expected {expected}")
        shape = np.shape(central_value)
        for name, replicates in (("replica_central_values", replica_central_values),
                                 ("fluctuations", fluctuations),
                                 ("jackknife_replicates", jackknife_replicates),
                                 ("bootstrap_replicates", bootstrap_replicates)):
            if replicates is not None and replicates.shape[1:] != shape:
                raise ValueError(f"from_results: {name} have shape {replicates.shape[1:]}, expected {shape}")
        return cls(layout=layout, central_value=central_value,
                   replica_central_values=replica_central_values, fluctuations=fluctuations,
                   jackknife_replicates=jackknife_replicates, bootstrap_replicates=bootstrap_replicates)

    # ---------------------------------------------------------------------------
    # Transformations
    # ---------------------------------------------------------------------------

    @classmethod
    def derive(cls, func: Callable[..., np.ndarray | jax.Array], *variables: Variable,
               jacobians: Literal["jax"] | Sequence[np.ndarray] | None = "jax", use_vmap: bool = True) -> Self:
        """Variable for ``func`` of the inputs: central value, replica central values, fluctuations and replicates.

        The result has the configurations of all ensembles of the inputs; inputs with the same ensemble must have
        the same configurations of it. On an ensemble an input does not have, the input enters at its central value.
        ``jacobians`` sets how fluctuations are propagated: ``"jax"`` computes the Jacobian, a list supplies it (one
        array per input, shape output + input), ``None`` propagates none. ``use_vmap`` evaluates the replicates in
        one jax batch, else in a loop. If jax cannot trace ``func`` (it needs jax.numpy), the loop is used and
        ``"jax"`` gives no fluctuations. Parts that cannot be propagated are dropped with a warning.
        """
        if isinstance(jacobians, str) and jacobians != "jax":
            raise ValueError(f'derive: jacobians must be "jax", a list of arrays or None, not {jacobians!r}')
        layout = Layout.merge(*(variable.layout for variable in variables))
        central_values = [variable.central_value for variable in variables]
        compatible = (jacobians == "jax" or use_vmap) and cls._is_jax_compatible(func, central_values, use_vmap)
        input_fluctuations = cls._field_on_merged_layout(variables, "fluctuations", layout)
        input_jacobians = None if jacobians == "jax" else jacobians
        if jacobians == "jax" and compatible and input_fluctuations is not None:
            central_value, input_jacobians = gamma.value_and_jacobians(func, central_values)
        else:
            central_value = np.asarray(func(*central_values))
        if input_fluctuations is not None and input_jacobians is not None:
            fluctuations = gamma.project_fluctuations(input_jacobians, input_fluctuations)
        else:
            fluctuations = None
            if jacobians is not None and any(variable.fluctuations is not None for variable in variables):
                warnings.warn("derive: gamma-method fluctuations not propagated", stacklevel=2)
        replicates = cls._derive_replicates(func, variables, layout, use_vmap and compatible)
        return cls.from_results(layout=layout, central_value=central_value, fluctuations=fluctuations, **replicates)

    @staticmethod
    def _is_jax_compatible(func: Callable[..., np.ndarray | jax.Array], central_values: Sequence[np.ndarray],
                           vmap: bool) -> bool:
        try:
            jax.eval_shape(func, *central_values)
        except Exception as error:  # noqa: BLE001 any tracing failure (jax, numpy, numba) means not traceable
            loop_note = "; replicates evaluated in a Python loop" if vmap else ""
            warnings.warn(f"derive: func not traceable by jax ({type(error).__name__}){loop_note}", stacklevel=3)
            return False
        else:
            return True

    @classmethod
    def _derive_replicates(cls, func: Callable[..., np.ndarray | jax.Array], variables: tuple[Variable, ...],
                           layout: Layout, compatible: bool) -> dict[str, np.ndarray]:
        replicates = {field: cls._field_on_merged_layout(variables, field, layout)
                      for field in ("replica_central_values", "jackknife_replicates", "bootstrap_replicates")}
        dropped = [field for field, values in replicates.items()
                   if values is None and any(getattr(variable, field) is not None for variable in variables)]
        if dropped:
            warnings.warn(f"derive: {', '.join(dropped)} not propagated (missing in some inputs)", stacklevel=3)
        present = {field: values for field, values in replicates.items() if values is not None}
        if not present:
            return {}
        batched_inputs = [np.concatenate(arrays) for arrays in zip(*present.values())]
        if compatible:
            batched_output = np.asarray(jax.vmap(func)(*batched_inputs))
        else:
            batched_output = np.array([func(*row) for row in zip(*batched_inputs)])
        sizes = [len(values[0]) for values in present.values()]
        pieces = np.split(batched_output, np.cumsum(sizes)[:-1])
        return dict(zip(present.keys(), pieces))

    @staticmethod
    def _field_on_merged_layout(variables: tuple[Variable, ...], field: str,
                                merged_layout: Layout) -> list[np.ndarray] | None:
        """``field`` of each input with the rows of ``merged_layout``, or None if an input lacks it. For an ensemble
        an input does not have, its rows are the central value (fluctuations: zero)."""
        values = [getattr(variable, field) for variable in variables]
        if any(value is None for value in values):
            return None
        if field == "bootstrap_replicates":
            for index, input_values in enumerate(values):
                if len(input_values) != len(values[0]):
                    raise ValueError(f"derive: input {index} has a different number of {field}")
            return values
        merged_row_ensembles = _row_ensembles(merged_layout, field)
        rows = []
        for variable, input_values in zip(variables, values):
            # same ensembles: the rows are already those of the merged layout
            if variable.layout.ensemble_names == merged_layout.ensemble_names:
                rows.append(input_values)
                continue
            input_row_ensembles = _row_ensembles(variable.layout, field)
            fill = np.zeros_like(variable.central_value) if field == "fluctuations" else variable.central_value
            blocks = []
            for name in merged_layout.ensemble_names:
                if name in variable.layout.ensemble_names:
                    blocks.append(input_values[input_row_ensembles == name])
                else:
                    num_rows = np.sum(merged_row_ensembles == name)
                    blocks.append(np.broadcast_to(fill, (num_rows, *np.shape(fill))))
            rows.append(np.concatenate(blocks))
        return rows

    @classmethod
    def map_observations(cls, func: Callable[..., np.ndarray], *variables: Variable) -> Self:
        """Apply func to the observations of data variables; layouts must be equal. Return a data variable.

        func takes the observation arrays (configurations on axis 0) and must act on each configuration separately.
        """
        if any(variable.observations is None for variable in variables):
            raise ValueError("map_observations: input has no observations")
        layout = Layout.merge(*(variable.layout for variable in variables))
        if len(layout.ensemble_names) > 1:
            raise ValueError("map_observations: inputs are on different ensembles")
        observations = func(*[variable.observations for variable in variables])
        return cls.from_data(observations, layout)

    @classmethod
    def concatenate_components(cls, *variables: Variable, axis: int = 0) -> Self:
        """Join data variables along component ``axis``; layouts must be equal."""
        observations_axis = axis + 1 if axis >= 0 else axis
        return cls.map_observations(lambda *observations: np.concatenate(observations, axis=observations_axis),
                                    *variables)

    def bin(self, bin_size: int, bootstrap_indices: NDArray[np.integer] | None = None) -> Self:
        """Return the data variable binned in array order, truncating the trailing incomplete bin.

        A bin holds ``bin_size`` measured configurations; gaps in the Monte Carlo times are ignored.
        Requires one replica. The result keeps the ensemble and the replica and has Monte Carlo
        times ``0, ..., num_bins - 1``. Binned layouts identify ensemble, replica and number of bins only.
        ``bin_size == 1`` returns the variable itself. ``bootstrap_indices`` (num_bootstraps, num_bins) is optional;
        with it, bootstrap replicates of the binned data are computed too.
        """
        if bin_size < 1:
            raise ValueError(f"bin: bin_size must be >= 1, got {bin_size}")
        if self.observations is None:
            raise ValueError("bin: variable has no observations")
        if len(np.unique(self.layout.replicas)) > 1:
            raise ValueError("bin: more than one replica; binning across replicas is not supported")
        if bin_size == 1:
            return self if bootstrap_indices is None else type(self).from_data(self.observations, self.layout, bootstrap_indices)

        weights = self.layout.weights
        num_bins = len(self.observations) // bin_size
        observations = statistics.bin(self.observations, bin_size, weights=weights)
        bin_weights = statistics.bin(weights, bin_size, weights=np.ones(len(weights)))
        replicas = np.full(num_bins, self.layout.replicas[0])
        (ensemble,) = self.layout.ensemble_names
        layout = Layout(ensemble, np.arange(num_bins), replicas=replicas, weights=bin_weights)
        return type(self).from_data(observations, layout, bootstrap_indices)

    def least_squares_fit(self, model: least_squares.Model, fit_range: ArrayLike, initial_parameters: ArrayLike, *,
                          numba_model: least_squares.Model | None = None, weight_matrix: np.ndarray | None = None,
                          tolerance: float = 1e-4, max_calls: int | None = None) -> least_squares.LeastSquaresFit:
        """Least-squares fit of the jnp function ``model(fit_range, parameters)``, e.g. from
        ``statpy.qcd.correlator.models``, to the central value at the component indices ``fit_range``; refit each
        replica central value and jackknife and bootstrap replicate present, starting from the central parameters,
        with the same ``weight_matrix``.

        ``numba_model``, the same function in numba, e.g. from ``statpy.qcd.correlator.numba_models``, replaces
        ``model`` in the fits to speed them up. Fluctuations present are propagated with ``model``. ``weight_matrix``
        defaults to the inverse jackknife variance on ``fit_range``. Observations are not used. Returns a
        ``LeastSquaresFit``; its parameters have this variable's layout. ``tolerance`` is Migrad's; its default of
        0.1 makes jackknife refits too small at large sample sizes.
        """
        fit_range = np.asarray(fit_range)
        initial_parameters = np.asarray(initial_parameters, dtype=float)
        dof = len(fit_range) - len(initial_parameters)
        if dof <= 0:
            raise ValueError(f"least_squares_fit: {len(fit_range)} points for {len(initial_parameters)} parameters, dof = {dof}")
        if weight_matrix is None:
            weight_matrix = np.diag(1.0 / self.jackknife_variance()[fit_range])

        chi2_function = least_squares.chi2(model if numba_model is None else numba_model)

        def objective(y: np.ndarray, parameters: np.ndarray) -> float:
            return float(chi2_function(fit_range, y, parameters, weight_matrix))

        def exact_gradient(y: np.ndarray, parameters: np.ndarray) -> jax.Array:
            return least_squares.chi2_gradient(model, fit_range, y, parameters, weight_matrix)

        objective_gradient = exact_gradient if numba_model is None else None

        parameters = least_squares.minimize(objective, self.central_value[fit_range], initial_parameters,
                                            tolerance=tolerance, max_calls=max_calls, objective_gradient=objective_gradient)
        chi2_value = objective(self.central_value[fit_range], parameters)
        if not np.isfinite(chi2_value):
            raise least_squares.ConvergenceError(f"least_squares_fit: non-finite chi^2 = {chi2_value}")
        replica_central_values = fluctuations = jackknife_replicates = bootstrap_replicates = None
        if self.fluctuations is not None:
            jacobian = least_squares.parameter_jacobian(model, fit_range, self.central_value[fit_range], parameters, weight_matrix)
            fluctuations = gamma.project_fluctuations([jacobian], [self.fluctuations[:, fit_range]])
        if self.replica_central_values is not None:
            replica_central_values = least_squares.fit_replicates(
                objective, self.replica_central_values[:, fit_range], parameters,
                tolerance=tolerance, max_calls=max_calls, objective_gradient=objective_gradient)
        if self.jackknife_replicates is not None:
            jackknife_replicates = least_squares.fit_replicates(
                objective, self.jackknife_replicates[:, fit_range], parameters,
                tolerance=tolerance, max_calls=max_calls, objective_gradient=objective_gradient)
        if self.bootstrap_replicates is not None:
            bootstrap_replicates = least_squares.fit_replicates(
                objective, self.bootstrap_replicates[:, fit_range], parameters,
                tolerance=tolerance, max_calls=max_calls, objective_gradient=objective_gradient)
        fitted = Variable.from_results(layout=self.layout, central_value=parameters,
                                       replica_central_values=replica_central_values,
                                       fluctuations=fluctuations,
                                       jackknife_replicates=jackknife_replicates,
                                       bootstrap_replicates=bootstrap_replicates)
        return least_squares.LeastSquaresFit(fitted, chi2_value, dof, least_squares.p_value(chi2_value, dof),
                                             model, fit_range, self)

    # ---------------------------------------------------------------------------
    # Error analysis
    # ---------------------------------------------------------------------------

    def jackknife_variance(self, bin_size: int = 1) -> np.ndarray:
        """Return the jackknife variance; delayed binning (one replica only) for ``bin_size > 1``."""
        return jackknife.variance(self._jackknife_replicates(bin_size))

    def jackknife_covariance(self, bin_size: int = 1) -> np.ndarray:
        """Return the jackknife covariance; delayed binning (one replica only) for ``bin_size > 1``."""
        return jackknife.covariance(self._jackknife_replicates(bin_size))

    def _jackknife_replicates(self, bin_size: int) -> np.ndarray:
        if self.jackknife_replicates is None:
            raise ValueError("jackknife: variable has no jackknife replicates")
        if bin_size == 1:
            return self.jackknife_replicates
        if len(np.unique(self.layout.replicas)) > 1:
            raise ValueError("Delayed binning: more than one replica; binning across replicas is not supported")
        return jackknife.delayed_binning(self.jackknife_replicates, bin_size, self.layout.weights)

    def bootstrap_variance(self) -> np.ndarray:
        """Return the bootstrap variance."""
        if self.bootstrap_replicates is None:
            raise ValueError("bootstrap_variance: variable has no bootstrap replicates")
        return bootstrap.variance(self.bootstrap_replicates)

    def bootstrap_covariance(self) -> np.ndarray:
        """Return the bootstrap covariance."""
        if self.bootstrap_replicates is None:
            raise ValueError("bootstrap_covariance: variable has no bootstrap replicates")
        return bootstrap.covariance(self.bootstrap_replicates)

    def gamma_method(self, S: float = 1.5) -> gamma.GammaErrorAnalysis:
        """Return the gamma-method error analysis per component.

        ``S`` sets the automatic summation window; a larger ``S`` gives a larger window.
        Fields: see ``GammaErrorAnalysis``. The variable is unchanged.
        """
        if self.fluctuations is None:
            raise ValueError("gamma_method: variable has no fluctuations")
        layout = self.layout
        return gamma.error_analysis(self.fluctuations, layout.monte_carlo_times, layout.replicas, S)

    def bias_corrected_value(self) -> np.ndarray:
        """Return the central value with the replica bias correction, which cancels the leading
        O(1/N) bias of nonlinear derived quantities. Needs replica central values of at least two replicas;
        the variable is unchanged."""
        _, sizes = np.unique(self.layout.replicas, return_counts=True)
        if self.replica_central_values is None or len(sizes) < 2:
            raise ValueError("bias_corrected_value: needs replica central values of at least two replicas")
        return gamma.bias_corrected_value(self.central_value, self.replica_central_values, sizes)

    def summarize(self, S: float = 1.5, components: Mapping[str, int | tuple[int, ...]] | Iterable[int | tuple[int, ...]] | None = None) -> None:
        """Print ensemble, configurations and replicas, then per component the central value, its bias-corrected
        value and the correction in units of the gamma error, the errors (gamma method at ``S``, jackknife,
        bootstrap), tau_int, window W and replica Q-value; columns without values are left out. ``components`` lists
        the rows to show (ints or index tuples), default all; a dict names them, e.g. ``{"A": 0, "m": 1}``."""
        layout = self.layout
        _, sizes = np.unique(layout.replicas, return_counts=True)
        shape = np.shape(self.central_value)
        replica_sizes = f" = {' + '.join(map(str, sizes))}" if len(sizes) > 1 else ""
        S_text = "" if self.fluctuations is None else f", S = {S}"
        print(f"{', '.join(layout.ensemble_names)}: N = {len(layout.replicas)}{replica_sizes}, shape {shape}{S_text}")
        result = None if self.fluctuations is None else self.gamma_method(S)
        corrected = bias_over_error = q_value = None
        if self.replica_central_values is not None and len(sizes) > 1:
            corrected = self.bias_corrected_value()
            if result is not None:
                # constant component (error 0): NaN
                bias_over_error = (self.central_value - corrected) / np.where(result.error == 0, np.nan, result.error)
                q_value = gamma.replica_q_value(self.replica_central_values, sizes, result.error)
        errors = {"gamma": None if result is None else result.error,
                  "jackknife": None if self.jackknife_replicates is None else np.sqrt(self.jackknife_variance()),
                  "bootstrap": None if self.bootstrap_replicates is None else np.sqrt(self.bootstrap_variance())}
        available = [error for error in errors.values() if error is not None]

        def with_error(value: float, error: float | None) -> str:
            return f"{value:.6g}" if error is None else log.format_paren(value, error)

        rows = [["component", "value", "corrected", "bias/σ", *(f"{method}_error" for method in errors), "tau_int", "W", "replica Q"]]
        if isinstance(components, Mapping):
            selected = [(name, np.index_exp[component]) for name, component in components.items()]
        else:
            indices = np.ndindex(shape) if components is None else [np.index_exp[component] for component in components]
            selected = [(", ".join(map(str, index)), index) for index in indices]
        for label, index in selected:
            value_error = available[0][index].item() if available else None
            rows.append([str(label),
                         with_error(self.central_value[index].item(), value_error),
                         "-" if corrected is None else with_error(corrected[index].item(), value_error),
                         "-" if bias_over_error is None else f"{bias_over_error[index]:.2f}",
                         *("-" if error is None else f"{error[index]:.2g}" for error in errors.values()),
                         "-" if result is None else log.format_paren(result.tau_int[index].item(),
                                                                     result.error_tau_int[index].item()),
                         "-" if result is None else str(result.window[index]),
                         "-" if q_value is None else f"{q_value[index]:.2f}"])
        columns = [i for i in range(len(rows[0])) if any(row[i] not in ("-", "") for row in rows[1:])]
        rows = [[row[i] for i in columns] for row in rows]
        widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
        for row in rows:
            print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())

    # ---------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------

    def plot_observations(self, ax: Axes, component: int | tuple[int, ...] = (), weighted: bool = False) -> Axes:
        """Plot the observations of one ``component`` into ``ax`` against Monte Carlo time with the central value;
        ``weighted=True`` plots ``w a / mean(w)``."""
        if self.observations is None:
            raise ValueError("plot_observations: variable has no observations")
        observations = select_component(self.observations, component)
        central_value = self.central_value[np.index_exp[component]].item()
        if not weighted:
            return _monte_carlo_history(ax, observations, self.layout, central_value, "observation")
        weights = self.layout.weights
        return _monte_carlo_history(ax, weights * observations / np.mean(weights), self.layout, central_value,
                                    "weighted observation")

    def plot_fluctuations(self, ax: Axes, component: int | tuple[int, ...] = ()) -> Axes:
        """Plot central value + fluctuations of one ``component`` into ``ax`` against Monte Carlo time."""
        if self.fluctuations is None:
            raise ValueError("plot_fluctuations: variable has no fluctuations")
        central_value = self.central_value[np.index_exp[component]].item()
        return _monte_carlo_history(ax, central_value + select_component(self.fluctuations, component), self.layout,
                                    central_value, "central value + fluctuation")

    def plot_weights(self, ax: Axes) -> Axes:
        """Plot the weights into ``ax`` against Monte Carlo time with their mean."""
        weights = self.layout.weights
        mean = np.mean(weights)
        return _monte_carlo_history(ax, weights, self.layout, mean, "weight", reference_name="mean")


def _row_ensembles(layout: Layout, field: str) -> np.ndarray:
    """Ensemble name of each row of ``field``: one row per (ensemble, replica) for replica central values, else
    one per configuration."""
    if field != "replica_central_values":
        return layout.ensembles
    return np.concatenate([np.full(len(np.unique(layout.replicas[layout.ensembles == name])), name)
                           for name in layout.ensemble_names])


def _monte_carlo_history(ax: Axes, values: np.ndarray, layout: Layout, reference: float, ylabel: str,
                         reference_name: str = "central value") -> Axes:
    """Plot ``values`` into ``ax`` against Monte Carlo time, replicas one after another, with a line at ``reference``;
    legend of the replicas (if more than one) and the reference line."""
    offset = 0
    several_replicas = len(np.unique(layout.replicas)) > 1
    for replica in np.unique(layout.replicas):
        in_replica = layout.replicas == replica
        times = layout.monte_carlo_times[in_replica] - layout.monte_carlo_times[in_replica][0]
        if offset:
            ax.axvline(offset - 0.5, color="0.6", linewidth=0.8)
        # NaN at gaps, so the line breaks there
        gaps = np.flatnonzero(np.diff(times) > np.diff(times).min()) + 1
        ax.plot(np.insert((times + offset).astype(float), gaps, np.nan), np.insert(values[in_replica], gaps, np.nan),
                linewidth=0.7, label=f"replica {replica}" if several_replicas else None)
        offset += times[-1] + 1
    ax.axhline(reference, color="0.15", linewidth=1, label=f"{reference_name} = {reference:.4g}")
    legend_above(ax, *ax.get_legend_handles_labels())
    ax.set(xlabel="Monte Carlo time", ylabel=ylabel)
    style_axes(ax)
    return ax
