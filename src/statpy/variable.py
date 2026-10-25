import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Self

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
    """Ensemble, Monte Carlo times, replicas and weights of the configurations.

    Required: ``ensemble`` (name) and ``monte_carlo_times`` (one integer per configuration,
    strictly increasing in array order within each replica). Optional: ``replicas`` (one integer
    per configuration; default ``zeros(N)``, one replica) and ``weights`` (reweighting factors;
    default ``ones(N)``). A configuration is identified by (ensemble, replica, Monte Carlo time).
    """

    def __init__(self, ensemble: str, monte_carlo_times: NDArray[np.integer], *,
                 replicas: NDArray[np.integer] | None = None, weights: np.ndarray | None = None) -> None:
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

    Attributes (``None`` if absent; ``central_value`` and ``layout`` are always set):
    - ``central_value``: estimate of the quantity.
    - ``observations``: data per configuration (configurations on axis 0).
    - ``layout``: ``Layout`` of the configurations.
    - ``jackknife_replicates``: delete-one replicates, one per configuration.
    - ``bootstrap_replicates``: one replicate per bootstrap sample.
    - ``fluctuations``: per-configuration fluctuations for the gamma method.
    - ``replica_central_values``: estimate from each replica alone, replicas in sorted label order
      (Wolff eq. 15), for the replica bias correction and check.

    A data variable (made by ``from_data``) has observations, fluctuations and replica central values.
    ``derive`` and ``from_results`` return variables without observations; ``derive`` propagates
    fluctuations and replica central values when all inputs have them.
    """

    # ---------------------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------------------

    def __init__(self, *, central_value: np.ndarray, observations: np.ndarray | None = None, layout: Layout,
                 jackknife_replicates: np.ndarray | None = None, bootstrap_replicates: np.ndarray | None = None,
                 fluctuations: np.ndarray | None = None, replica_central_values: np.ndarray | None = None) -> None:
        """Internal constructor; create variables with the classmethods."""
        self.central_value = central_value
        self.observations = observations
        self.layout = layout
        self.jackknife_replicates = jackknife_replicates
        self.bootstrap_replicates = bootstrap_replicates
        self.fluctuations = fluctuations
        self.replica_central_values = replica_central_values

    def __repr__(self) -> str:
        parts = [f"shape={np.shape(self.central_value)}"]
        for name, replicates, count in (("observations", self.observations, "N"),
                                        ("jackknife", self.jackknife_replicates, "N"),
                                        ("bootstrap", self.bootstrap_replicates, "B")):
            if replicates is not None:
                parts.append(f"{name} {count}={len(replicates)}")
        return f"Variable({', '.join(parts)})"

    @classmethod
    def from_data(cls, observations: np.ndarray, layout: Layout,
                  bootstrap_indices: NDArray[np.integer] | None = None) -> Self:
        """Return a data variable: weighted mean, jackknife replicates, gamma-method fluctuations and the
        weighted mean of each replica.

        ``observations`` has configurations on axis 0, in ``layout`` order. ``bootstrap_indices`` is an
        optional index matrix (num_bootstraps, N); with it, bootstrap replicates are computed too.
        """
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
        replica_central_values = np.stack([
            np.average(observations[layout.replicas == replica], axis=0, weights=weights[layout.replicas == replica])
            for replica in np.unique(layout.replicas)])
        return cls(central_value=mean, observations=observations, layout=layout,
                   jackknife_replicates=jackknife_replicates, bootstrap_replicates=bootstrap_replicates,
                   fluctuations=fluctuations, replica_central_values=replica_central_values)

    @classmethod
    def from_results(cls, *, central_value: np.ndarray, layout: Layout,
                     jackknife_replicates: np.ndarray | None = None, bootstrap_replicates: np.ndarray | None = None,
                     fluctuations: np.ndarray | None = None, replica_central_values: np.ndarray | None = None) -> Self:
        """Return a variable from a computed central value, replicates, fluctuations and replica central values,
        stored unchanged."""
        num_configurations = len(layout.monte_carlo_times)
        num_replicas = len(np.unique(layout.replicas))
        for name, values, expected in (("jackknife_replicates", jackknife_replicates, num_configurations),
                                       ("fluctuations", fluctuations, num_configurations),
                                       ("replica_central_values", replica_central_values, num_replicas)):
            if values is not None and len(values) != expected:
                raise ValueError(f"from_results: {name} have length {len(values)}, expected {expected}")
        shape = np.shape(central_value)
        for name, replicates in (("jackknife_replicates", jackknife_replicates),
                                 ("bootstrap_replicates", bootstrap_replicates),
                                 ("fluctuations", fluctuations),
                                 ("replica_central_values", replica_central_values)):
            if replicates is not None and replicates.shape[1:] != shape:
                raise ValueError(f"from_results: {name} have shape {replicates.shape[1:]}, expected {shape}")
        return cls(central_value=central_value, layout=layout,
                   jackknife_replicates=jackknife_replicates, bootstrap_replicates=bootstrap_replicates,
                   fluctuations=fluctuations, replica_central_values=replica_central_values)

    # ---------------------------------------------------------------------------
    # Transformations
    # ---------------------------------------------------------------------------

    @classmethod
    def derive(cls, func: Callable[..., np.ndarray | jax.Array], *variables: "Variable",
               jacobian: Sequence[np.ndarray] | None = None, use_jax: bool = True) -> Self:
        """Variable for ``func`` of the inputs: central value, fluctuations and replicates; layouts must be equal.

        The central value is ``func`` of the central values, each replicate kind (jackknife, bootstrap, replica
        central values) is ``func`` of the inputs' replicates, and the fluctuations are the inputs' fluctuations
        projected with the Jacobian of ``func``: ``jacobian`` if given (one array per input, shape output + input),
        else from jax. A part missing in some input is dropped with a warning.

        ``use_jax=True`` uses jax for the Jacobian and to evaluate all replicates in one batch; if jax cannot trace
        ``func`` (it needs jax.numpy, no in-place writes or Python control flow on values), a warning is given, the
        replicates are evaluated in a loop and fluctuations need ``jacobian``. ``use_jax=False`` skips jax.
        """
        layout = cls._common_layout(variables, "derive")
        central_values = [variable.central_value for variable in variables]
        compatible = use_jax and cls._is_jax_compatible(func, central_values)
        input_fluctuations = cls._common_field(variables, "fluctuations")
        if input_fluctuations is not None and (jacobian is not None or compatible):
            if jacobian is None:
                central_value, jacobian = gamma.value_and_jacobians(func, central_values)
            else:
                central_value = np.asarray(func(*central_values))
            fluctuations = gamma.project_fluctuations(jacobian, input_fluctuations)
        else:
            central_value, fluctuations = np.asarray(func(*central_values)), None
            if any(variable.fluctuations is not None for variable in variables):
                warnings.warn("derive: gamma-method fluctuations not propagated", stacklevel=2)
        replicates = cls._derive_replicates(func, variables, compatible)  # jackknife, bootstrap and replica central values
        return cls.from_results(central_value=central_value, layout=layout, fluctuations=fluctuations, **replicates)

    @staticmethod
    def _is_jax_compatible(func: Callable[..., np.ndarray | jax.Array], central_values: Sequence[np.ndarray]) -> bool:
        try:
            jax.eval_shape(func, *central_values)
        except Exception as error:  # noqa: BLE001 any tracing failure (jax, numpy, numba) means not traceable
            warnings.warn(f"derive: func not traceable by jax ({type(error).__name__}); replicates evaluated in a Python loop", stacklevel=3)
            return False
        else:
            return True

    @classmethod
    def _derive_replicates(cls, func: Callable[..., np.ndarray | jax.Array], variables: tuple["Variable", ...],
                           compatible: bool) -> dict[str, np.ndarray]:
        replicates = {field: cls._common_field(variables, field)
                      for field in ("jackknife_replicates", "bootstrap_replicates", "replica_central_values")}
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
    def _common_field(variables: tuple["Variable", ...], field: str) -> list[np.ndarray] | None:
        values = [getattr(variable, field) for variable in variables]
        if any(value is None for value in values):
            return None
        num_rows = len(values[0])
        for index, input_values in enumerate(values):
            if len(input_values) != num_rows:
                raise ValueError(f"derive: input {index} has a different number of {field}")
        return values

    @classmethod
    def map_observations(cls, func: Callable[..., np.ndarray], *variables: "Variable") -> Self:
        """Apply func to the observations of data variables; layouts must be equal. Return a data variable.

        func takes the observation arrays (configurations on axis 0) and must act on each configuration separately.
        """
        if any(variable.observations is None for variable in variables):
            raise ValueError("map_observations: input has no observations")
        layout = cls._common_layout(variables, "map_observations")
        observations = func(*[variable.observations for variable in variables])
        return cls.from_data(observations, layout)

    @staticmethod
    def _common_layout(variables: tuple["Variable", ...], name: str) -> Layout:
        reference = variables[0].layout
        for index, variable in enumerate(variables):
            for field in ("ensemble", "monte_carlo_times", "replicas", "weights"):
                if not np.array_equal(getattr(variable.layout, field), getattr(reference, field)):
                    raise ValueError(f"{name}: input {index} {field} differ")
        return reference

    @classmethod
    def concatenate_components(cls, *variables: "Variable", axis: int = 0) -> Self:
        """Join data variables along component ``axis``; layouts must be equal."""
        observations_axis = axis + 1 if axis >= 0 else axis
        return cls.map_observations(lambda *observations: np.concatenate(observations, axis=observations_axis),
                                    *variables)

    def bin(self, bin_size: int) -> Self:
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

    def least_squares_fit(self, model: least_squares.Model, fit_range: ArrayLike, initial_parameters: ArrayLike, *,
                          weight_matrix: np.ndarray | None = None, tolerance: float | None = None,
                          max_calls: int | None = None) -> least_squares.LeastSquaresFit:
        """Least-squares fit of the function ``model(fit_range, parameters)``, e.g. a numba model from
        ``statpy.qcd.correlator.models``, to the central value at the component indices ``fit_range``; refit each
        jackknife and bootstrap replicate and replica central value present, starting from the central parameters,
        with the same ``weight_matrix``.

        ``weight_matrix`` defaults to the inverse jackknife variance on ``fit_range``. Observations and fluctuations
        are not used. Returns a ``LeastSquaresFit``; its parameters have this variable's layout.
        """
        fit_range = np.asarray(fit_range)
        initial_parameters = np.asarray(initial_parameters, dtype=float)
        dof = len(fit_range) - len(initial_parameters)
        if dof <= 0:
            raise ValueError(f"least_squares_fit: {len(fit_range)} points for {len(initial_parameters)} parameters, dof = {dof}")
        if weight_matrix is None:
            weight_matrix = np.diag(1.0 / self.jackknife_variance()[fit_range])

        chi2_function = least_squares.chi2(model)

        def objective(y: np.ndarray, parameters: np.ndarray) -> float:
            return chi2_function(fit_range, y, parameters, weight_matrix)

        y = self.central_value[fit_range]
        parameters = least_squares.minimize(objective, y, initial_parameters, tolerance=tolerance,
                                            max_calls=max_calls)
        chi2_value = objective(y, parameters)
        if not np.isfinite(chi2_value):
            raise least_squares.ConvergenceError(f"least_squares_fit: non-finite chi^2 = {chi2_value}")
        jackknife_replicates = bootstrap_replicates = replica_central_values = None
        if self.jackknife_replicates is not None:
            jackknife_replicates = least_squares.fit_replicates(
                objective, self.jackknife_replicates[:, fit_range], parameters, tolerance=tolerance, max_calls=max_calls)
        if self.bootstrap_replicates is not None:
            bootstrap_replicates = least_squares.fit_replicates(
                objective, self.bootstrap_replicates[:, fit_range], parameters, tolerance=tolerance, max_calls=max_calls)
        if self.replica_central_values is not None:
            replica_central_values = least_squares.fit_replicates(
                objective, self.replica_central_values[:, fit_range], parameters, tolerance=tolerance,
                max_calls=max_calls)
        fitted = Variable.from_results(central_value=parameters, layout=self.layout,
                                       jackknife_replicates=jackknife_replicates,
                                       bootstrap_replicates=bootstrap_replicates,
                                       replica_central_values=replica_central_values)
        return least_squares.LeastSquaresFit(fitted, chi2_value, dof, least_squares.p_value(chi2_value, dof))

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
        """Return the gamma-method error analysis per component (Wolff, hep-lat/0306017).

        ``S`` is the factor in Wolff's hypothesis tau ~ S tau_int (sec. 3.3) that sets the
        automatic window. Fields: see ``GammaErrorAnalysis``. The variable is unchanged.
        """
        if self.fluctuations is None:
            raise ValueError("gamma_method: variable has no fluctuations")
        layout = self.layout
        return gamma.error_analysis(self.fluctuations, layout.monte_carlo_times, layout.replicas, S)

    def bias_corrected_value(self) -> np.ndarray:
        """Return the central value with Wolff's replica bias correction (eq. 20), which cancels the leading
        O(1/N) bias of nonlinear derived quantities. Needs replica central values of at least two replicas;
        the variable is unchanged."""
        _, sizes = np.unique(self.layout.replicas, return_counts=True)
        if self.replica_central_values is None or len(sizes) < 2:
            raise ValueError("bias_corrected_value: needs replica central values of at least two replicas")
        return gamma.bias_corrected_value(self.central_value, self.replica_central_values, sizes)

    def summarize(self, S: float = 1.5, components: Iterable[int | tuple[int, ...]] | None = None) -> None:
        """Print ensemble, configurations and replicas, then per component the central value, its bias-corrected
        value and the correction in units of the gamma error, the errors (gamma method at ``S``, jackknife,
        bootstrap), tau_int, window W and replica Q-value; ``-`` if absent. ``components`` lists the rows to show
        (ints or index tuples), default all."""
        layout = self.layout
        _, sizes = np.unique(layout.replicas, return_counts=True)
        shape = np.shape(self.central_value)
        replica_sizes = f" = {' + '.join(map(str, sizes))}" if len(sizes) > 1 else ""
        S_text = "" if self.fluctuations is None else f", S = {S}"
        print(f"{layout.ensemble}: N = {len(layout.replicas)}{replica_sizes}, shape {shape}{S_text}")
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

        rows = [["component", "value", "corrected", "bias/σ", *errors, "tau_int", "W", "replica Q"]]
        indices = np.ndindex(shape) if components is None else [np.index_exp[component] for component in components]
        for index in indices:
            value_error = available[0][index].item() if available else None
            rows.append([", ".join(map(str, index)),
                         with_error(self.central_value[index].item(), value_error),
                         "-" if corrected is None else with_error(corrected[index].item(), value_error),
                         "-" if bias_over_error is None else f"{bias_over_error[index]:.2f}",
                         *("-" if error is None else f"{error[index]:.2g}" for error in errors.values()),
                         "-" if result is None else log.format_paren(result.tau_int[index].item(),
                                                                     result.error_tau_int[index].item()),
                         "-" if result is None else str(result.window[index]),
                         "-" if q_value is None else f"{q_value[index]:.2f}"])
        widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
        for row in rows:
            print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())

    # ---------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------

    def plot_observations(self, ax: "Axes", component: int | tuple[int, ...] = (), weighted: bool = False) -> "Axes":
        """Plot the observations of one ``component`` into ``ax`` against Monte Carlo time with their mean;
        ``weighted=True`` plots ``w a / mean(w)`` with their weighted mean."""
        if self.observations is None:
            raise ValueError("plot_observations: variable has no observations")
        observations = select_component(self.observations, component)
        if not weighted:
            mean = np.mean(observations)
            return _monte_carlo_history(ax, observations, self.layout, mean, rf"$\bar a = {mean:.4g}$",
                                        r"observation $a_i$")
        weights = self.layout.weights
        central_value = self.central_value[np.index_exp[component]].item()
        return _monte_carlo_history(ax, weights * observations / np.mean(weights), self.layout, central_value,
                                    rf"$\bar F = {central_value:.4g}$", r"weighted observation $w_i a_i / \bar w$")

    def plot_fluctuations(self, ax: "Axes", component: int | tuple[int, ...] = ()) -> "Axes":
        """Plot the fluctuations of one ``component`` into ``ax`` against Monte Carlo time."""
        if self.fluctuations is None:
            raise ValueError("plot_fluctuations: variable has no fluctuations")
        central_value = self.central_value[np.index_exp[component]].item()
        return _monte_carlo_history(ax, select_component(self.fluctuations, component), self.layout, 0.0,
                                    rf"$0.0$ $(\bar F = {central_value:.4g})$", r"fluctuation $\delta_i$")

    def plot_weights(self, ax: "Axes") -> "Axes":
        """Plot the weights into ``ax`` against Monte Carlo time with their mean."""
        weights = self.layout.weights
        mean = np.mean(weights)
        return _monte_carlo_history(ax, weights, self.layout, mean, rf"$\bar w = {mean:.4g}$", r"weight $w_i$")


def _monte_carlo_history(ax: "Axes", values: np.ndarray, layout: Layout, reference: float, reference_label: str,
                         ylabel: str) -> "Axes":
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
    ax.axhline(reference, color="0.15", linewidth=1, label=reference_label)
    legend_above(ax, *ax.get_legend_handles_labels())
    ax.set(xlabel="Monte Carlo time", ylabel=ylabel)
    style_axes(ax)
    return ax
