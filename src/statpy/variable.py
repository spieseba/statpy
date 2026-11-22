import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
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
        if not parts:
            return Layout("", np.array([], dtype=int))
        ensembles, monte_carlo_times, replicas, weights = (np.concatenate(values) for values in zip(*parts))
        return Layout(ensembles, monte_carlo_times, replicas=replicas, weights=weights)


@dataclass(frozen=True)
class ExternalInput:
    """Dependence of a variable on an external input with n values: ``covariance`` of the input, shape (n, n), and
    ``jacobian`` of the variable with respect to the input, shape variable shape + (n,), or None if it was not
    propagated."""
    covariance: np.ndarray
    jacobian: np.ndarray | None


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
    - ``external_inputs``: dict by name of the ``ExternalInput`` the variable depends on, never ``None``; their
      variance J C J^T is added to the jackknife and gamma-method variances.

    A data variable (made by ``from_data``) has observations, replica central values and fluctuations.
    ``derive`` and ``from_results`` return variables without observations; ``derive`` propagates
    replica central values and fluctuations when all inputs have them.
    """

    # ---------------------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------------------

    def __init__(self, *, observations: np.ndarray | None = None, layout: Layout, central_value: np.ndarray,
                 replica_central_values: np.ndarray | None = None, fluctuations: np.ndarray | None = None,
                 jackknife_replicates: np.ndarray | None = None, bootstrap_replicates: np.ndarray | None = None,
                 external_inputs: Mapping[str, ExternalInput] | None = None) -> None:
        """Internal constructor; create variables with the classmethods."""
        self.observations = observations
        self.layout = layout
        self.central_value = central_value
        self.replica_central_values = replica_central_values
        self.fluctuations = fluctuations
        self.jackknife_replicates = jackknife_replicates
        self.bootstrap_replicates = bootstrap_replicates
        self.external_inputs = dict(external_inputs or {})

    def __repr__(self) -> str:
        layout = self.layout
        names = layout.ensemble_names
        reweighted = ", reweighted" if np.any(layout.weights != 1) else ""
        configurations = (f"on ensemble{'s' if len(names) > 1 else ''} {', '.join(names)}, "
                          f"{len(layout.replicas)} configurations{reweighted}") if names else "without configurations"
        lines = [f"Variable of shape {np.shape(self.central_value)} {configurations}"]
        for name, replica, rows in _replica_rows(layout):
            times = layout.monte_carlo_times[rows]
            if len(times) == 1:
                line = f"Monte Carlo time {times[0]}"
            else:
                step = np.gcd.reduce(np.diff(times))
                missing = (times[-1] - times[0]) // step + 1 - len(times)
                line = f"Monte Carlo times {times[0]} to {times[-1]} in steps of {step}"
                line += f", {missing} missing" if missing else ""
            label = ", ".join(([name] if name else []) + ([f"replica {replica}"] if replica is not None else []))
            lines.append(f"{label}: {len(times)} configurations, {line}" if label else line)
        bootstrap = "bootstrap" if self.bootstrap_replicates is None else f"bootstrap (B = {len(self.bootstrap_replicates)})"
        stored = [name for name, value in (("observations", self.observations),
                                           ("replica central values", self.replica_central_values),
                                           ("fluctuations", self.fluctuations), ("jackknife", self.jackknife_replicates),
                                           (bootstrap, self.bootstrap_replicates))
                  if value is not None]
        lines.append(f"stored: {', '.join(stored) or 'central value only'}")
        if self.external_inputs:
            lines.append(f"external inputs: {', '.join(self.external_inputs)}")
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
                     jackknife_replicates: np.ndarray | None = None, bootstrap_replicates: np.ndarray | None = None,
                     external_inputs: Mapping[str, ExternalInput] | None = None) -> Self:
        """Return a variable from a computed central value, replica central values, fluctuations, replicates and
        external inputs, stored unchanged."""
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
                   jackknife_replicates=jackknife_replicates, bootstrap_replicates=bootstrap_replicates,
                   external_inputs=external_inputs)

    @classmethod
    def from_external(cls, name: str, mean: ArrayLike, covariance: ArrayLike | None = None,
                      num_bootstraps: int | None = None, seed: int | None = None) -> Self:
        """Return a variable without configurations for the external input ``name`` with central value ``mean``.

        Without ``covariance`` it is a constant: its jackknife and gamma-method errors are 0. ``covariance`` of
        ``mean.reshape(-1)``, shape (n, n), is propagated by ``derive`` and fits to the jackknife and gamma-method
        errors. ``num_bootstraps`` also draws that many Gaussian samples with ``mean`` and ``covariance`` as
        bootstrap replicates; ``seed`` defaults to a seed from ``name``, so the same name gives the same samples.
        """
        central_value = np.asarray(mean, dtype=float)
        if covariance is None:
            if num_bootstraps is not None:
                raise ValueError("from_external: num_bootstraps needs a covariance")
            return cls._from_external(name, central_value, None, None)
        covariance = np.asarray(covariance, dtype=float)
        size = central_value.size
        if covariance.shape != (size, size):
            raise ValueError(f"from_external: covariance has shape {covariance.shape}, expected {(size, size)}")
        eigenvalues = np.linalg.eigvalsh(covariance)
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=0) or eigenvalues[0] < -1e-10 * eigenvalues[-1]:
            raise ValueError("from_external: covariance must be symmetric and positive semi-definite, "
                             f"eigenvalues from {eigenvalues[0]:.3g} to {eigenvalues[-1]:.3g}")
        bootstrap_replicates = None
        if num_bootstraps is not None:
            rng = np.random.default_rng(list(name.encode()) if seed is None else seed)
            samples = rng.multivariate_normal(central_value.reshape(-1), covariance, size=num_bootstraps,
                                              method="eigh", check_valid="ignore")
            bootstrap_replicates = samples.reshape(num_bootstraps, *central_value.shape)
        return cls._from_external(name, central_value, covariance, bootstrap_replicates)

    @classmethod
    def from_external_replicates(cls, name: str, bootstrap_replicates: np.ndarray,
                                 mean: ArrayLike | None = None) -> Self:
        """Return a variable without configurations for the external input ``name`` from its bootstrap replicates,
        shape (num_bootstraps,) + value shape, stored unchanged.

        ``mean`` is the central value, default the mean of the replicates. The covariance of the replicates is
        propagated by ``derive`` and fits to the jackknife and gamma-method errors.
        """
        central_value = np.mean(bootstrap_replicates, axis=0) if mean is None else np.asarray(mean, dtype=float)
        return cls._from_external(name, central_value, bootstrap.covariance(bootstrap_replicates), bootstrap_replicates)

    @classmethod
    def _from_external(cls, name: str, central_value: np.ndarray, covariance: np.ndarray | None,
                       bootstrap_replicates: np.ndarray | None) -> Self:
        """Variable without configurations; with ``covariance`` it depends on the external input ``name``."""
        no_rows = np.zeros((0, *central_value.shape))
        external_inputs = {}
        if covariance is not None:
            size = central_value.size
            external_inputs[name] = ExternalInput(covariance, np.eye(size).reshape(*central_value.shape, size))
        return cls.from_results(layout=Layout("", np.array([], dtype=int)), central_value=central_value,
                                replica_central_values=no_rows, fluctuations=no_rows, jackknife_replicates=no_rows,
                                bootstrap_replicates=bootstrap_replicates, external_inputs=external_inputs)

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
        ``"jax"`` gives no fluctuations. Parts that cannot be propagated are dropped with a warning. External inputs
        are propagated with the same Jacobian; without it they are kept with ``jacobian=None``.
        """
        if isinstance(jacobians, str) and jacobians != "jax":
            raise ValueError(f'derive: jacobians must be "jax", a list of arrays or None, not {jacobians!r}')
        layout = Layout.merge(*(variable.layout for variable in variables))
        central_values = [variable.central_value for variable in variables]
        compatible = (jacobians == "jax" or use_vmap) and cls._is_jax_compatible(func, central_values, use_vmap)
        input_fluctuations = cls._field_on_merged_layout(variables, "fluctuations", layout)
        input_jacobians = None if jacobians == "jax" else jacobians
        has_external_inputs = any(variable.external_inputs for variable in variables)
        if jacobians == "jax" and compatible and (input_fluctuations is not None or has_external_inputs):
            central_value, input_jacobians = gamma.value_and_jacobians(func, central_values)
        else:
            central_value = np.asarray(func(*central_values))
        if input_fluctuations is not None and input_jacobians is not None:
            fluctuations = gamma.project_fluctuations(input_jacobians, input_fluctuations)
        else:
            fluctuations = None
            if jacobians is not None and any(variable.fluctuations is not None for variable in variables):
                warnings.warn("derive: gamma-method fluctuations not propagated", stacklevel=2)
        if has_external_inputs and input_jacobians is None and jacobians is not None:
            warnings.warn("derive: external inputs not propagated to the jackknife and gamma-method errors",
                          stacklevel=2)
        external_inputs = cls._derive_external_inputs(variables, input_jacobians)
        replicates = cls._derive_replicates(func, variables, layout, use_vmap and compatible, central_value.shape)
        return cls.from_results(layout=layout, central_value=central_value, fluctuations=fluctuations,
                                external_inputs=external_inputs, **replicates)

    @staticmethod
    def _derive_external_inputs(variables: tuple[Variable, ...],
                                jacobians: Sequence[np.ndarray] | None) -> dict[str, ExternalInput]:
        """External inputs of the result: per name the unchanged covariance and the sum over the inputs of
        ``jacobians[k]`` applied to the Jacobian of input k; None if one of them is missing."""
        external_inputs: dict[str, ExternalInput] = {}
        for index, variable in enumerate(variables):
            for name, external in variable.external_inputs.items():
                jacobian = None
                if jacobians is not None and external.jacobian is not None:
                    # term k of the chain rule: (d result / d input k) (d input k / d external input)
                    jacobian = np.tensordot(jacobians[index], external.jacobian, axes=np.ndim(variable.central_value))
                if name in external_inputs:
                    previous = external_inputs[name]
                    if not np.array_equal(previous.covariance, external.covariance):
                        raise ValueError(f"derive: inputs have different covariances of external input {name}")
                    # sum of the terms over the inputs k
                    jacobian = None if previous.jacobian is None or jacobian is None else previous.jacobian + jacobian
                external_inputs[name] = ExternalInput(external.covariance, jacobian)
        return external_inputs

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
                           layout: Layout, compatible: bool, shape: tuple[int, ...]) -> dict[str, np.ndarray]:
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
            # reshape: without rows the list is empty and has no component axes
            batched_output = np.array([func(*row) for row in zip(*batched_inputs)]).reshape(-1, *shape)
        sizes = [len(values[0]) for values in present.values()]
        pieces = np.split(batched_output, np.cumsum(sizes)[:-1])
        return dict(zip(present.keys(), pieces))

    @staticmethod
    def _field_on_merged_layout(variables: tuple[Variable, ...], field: str,
                                merged_layout: Layout) -> list[np.ndarray] | None:
        """``field`` of each input with the rows of ``merged_layout``, or None if an input lacks it. For an ensemble
        an input does not have, its rows are the central value (fluctuations: zero); so are the bootstrap replicates
        of an input without configurations and external inputs."""
        values = [getattr(variable, field) for variable in variables]
        num_bootstraps = next((len(value) for value in values if value is not None), None)
        if field == "bootstrap_replicates" and num_bootstraps is not None:
            values = [np.broadcast_to(variable.central_value, (num_bootstraps, *np.shape(variable.central_value)))
                      if value is None and not variable.layout.ensemble_names and not variable.external_inputs
                      else value
                      for variable, value in zip(variables, values)]
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
        bin_weights = statistics.bin(weights, bin_size)
        observations = statistics.bin(self.observations, bin_size, weights)
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
        ``model`` in the fits to speed them up. Fluctuations and external inputs are propagated with ``model``.
        ``weight_matrix`` defaults to the inverse jackknife variance on ``fit_range``.
        Observations are not used. Returns a ``LeastSquaresFit``; its parameters have this variable's layout.
        ``tolerance`` is Migrad's; its default of 0.1 makes jackknife refits too small at large sample sizes.
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
        external_inputs = {}
        if self.fluctuations is not None or self.external_inputs:
            jacobian = least_squares.parameter_jacobian(model, fit_range, self.central_value[fit_range], parameters, weight_matrix)
            if self.fluctuations is not None:
                fluctuations = gamma.project_fluctuations([jacobian], [self.fluctuations[:, fit_range]])
            external_inputs = {
                name: ExternalInput(external.covariance,
                                    None if external.jacobian is None else jacobian @ external.jacobian[fit_range])
                for name, external in self.external_inputs.items()}
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
                                       bootstrap_replicates=bootstrap_replicates, external_inputs=external_inputs)
        return least_squares.LeastSquaresFit(fitted, chi2_value, dof, least_squares.p_value(chi2_value, dof),
                                             model, fit_range, self)

    # ---------------------------------------------------------------------------
    # Error analysis
    # ---------------------------------------------------------------------------

    def jackknife_variance(self, bin_size: int | Mapping[str, int] = 1) -> np.ndarray:
        """Return the jackknife variance, the sum of the variances from each ensemble, without external inputs.

        ``bin_size > 1`` applies delayed binning (ensembles with one replica only); a dict gives the bin size
        per ensemble name, 1 for names it does not contain.
        """
        return np.asarray(sum((jackknife.variance(replicates) for replicates in self._jackknife_replicates(bin_size)),
                              np.zeros(np.shape(self.central_value))))

    def jackknife_covariance(self, bin_size: int | Mapping[str, int] = 1) -> np.ndarray:
        """Return the jackknife covariance, the sum of the covariances from each ensemble, without external inputs;
        ``bin_size`` as in ``jackknife_variance``."""
        size = np.size(self.central_value)
        return sum((jackknife.covariance(replicates) for replicates in self._jackknife_replicates(bin_size)),
                   np.zeros((size, size)))

    def gamma_covariance(self, S: float | Mapping[str, float] = 1.5) -> np.ndarray:
        """Return the gamma-method covariance, the sum of the covariances from each ensemble, without external inputs:
        per ensemble, the correlations of Gamma(0) scaled with the gamma errors at ``S`` (exact if all components share
        one normalized autocorrelation function; positive semidefinite)."""
        if self.fluctuations is None:
            raise ValueError("gamma_covariance: variable has no fluctuations")
        errors = self.gamma_method(S).ensemble_errors
        layout = self.layout
        size = np.size(self.central_value)
        return sum((gamma.rescaled_covariance(self.fluctuations[layout.ensembles == name], errors[name])
                    for name in layout.ensemble_names), np.zeros((size, size)))

    def external_variances(self) -> dict[str, np.ndarray]:
        """Return the variance from each external input, the diagonal of J C J^T, by input name."""
        shape = np.shape(self.central_value)
        # diagonal of J C J^T, without building the matrix
        return {name: np.sum((jacobian @ covariance) * jacobian, axis=1).reshape(shape)
                for name, (jacobian, covariance) in self._external_jacobians().items()}

    def external_covariance(self) -> np.ndarray:
        """Return the covariance from the external inputs, J C J^T summed over them."""
        size = np.size(self.central_value)
        return sum((jacobian @ covariance @ jacobian.T for jacobian, covariance in self._external_jacobians().values()),
                   np.zeros((size, size)))

    def _external_jacobians(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Jacobian, shape (components, n), and covariance of each external input, by input name."""
        size = np.size(self.central_value)
        jacobians = {}
        for name, external in self.external_inputs.items():
            if external.jacobian is None:
                raise ValueError(f"variable has no Jacobian for the external input {name}")
            jacobians[name] = (external.jacobian.reshape(size, -1), external.covariance)
        return jacobians

    def total_variance(self, method: str = "gamma", S: float | Mapping[str, float] = 1.5,
                       bin_size: int | Mapping[str, int] = 1) -> np.ndarray:
        """Return the variance from the Monte Carlo data and the external inputs. ``method`` is "gamma" (at ``S``),
        "jackknife" (at ``bin_size``) or "bootstrap"."""
        if method == "bootstrap":
            # the replicates already contain the external inputs
            return self.bootstrap_variance()
        if method not in ("gamma", "jackknife"):
            raise ValueError(f"method must be gamma, jackknife or bootstrap, got {method!r}")
        return np.asarray(sum((np.square(error) for error in self.error_budget(method, S, bin_size).values()),
                              np.zeros(np.shape(self.central_value))))

    def total_covariance(self, method: str = "gamma", S: float | Mapping[str, float] = 1.5,
                         bin_size: int | Mapping[str, int] = 1) -> np.ndarray:
        """Return the covariance from the Monte Carlo data and the external inputs. ``method`` is "gamma" (at ``S``,
        see ``gamma_covariance``), "jackknife" (at ``bin_size``) or "bootstrap"."""
        if method == "bootstrap":
            # the replicates already contain the external inputs
            return self.bootstrap_covariance()
        if method == "gamma":
            statistical = self.gamma_covariance(S)
        elif method == "jackknife":
            statistical = self.jackknife_covariance(bin_size)
        else:
            raise ValueError(f"method must be gamma, jackknife or bootstrap, got {method!r}")
        return statistical + self.external_covariance()

    def error_budget(self, method: str = "gamma", S: float | Mapping[str, float] = 1.5,
                     bin_size: int | Mapping[str, int] = 1) -> dict[str, np.ndarray]:
        """Return the error from each source: the ensembles by name, then the external inputs as
        ``external <name>``. Their squares add up to ``total_variance``. ``method`` is "gamma" (at ``S``) or
        "jackknife" (at ``bin_size``)."""
        if method == "gamma":
            ensembles = self.gamma_method(S).ensemble_errors
        elif method == "jackknife":
            ensembles = self._jackknife_errors(bin_size)
        else:
            raise ValueError(f"method must be gamma or jackknife, got {method!r}")
        return _merge_sources(ensembles, self._external_errors())

    def _jackknife_errors(self, bin_size: int | Mapping[str, int]) -> dict[str, np.ndarray]:
        """Jackknife error from each ensemble, by ensemble name."""
        return {name: np.asarray(np.sqrt(jackknife.variance(replicates)))
                for name, replicates in zip(self.layout.ensemble_names, self._jackknife_replicates(bin_size))}

    def _external_errors(self) -> dict[str, np.ndarray]:
        """Error from each external input, by ``external <name>``."""
        return {f"external {name}": np.asarray(np.sqrt(variance))
                for name, variance in self.external_variances().items()}

    def _jackknife_replicates(self, bin_size: int | Mapping[str, int]) -> list[np.ndarray]:
        """Jackknife replicates of each ensemble, delayed-binned where its bin size is not 1."""
        if self.jackknife_replicates is None:
            raise ValueError("jackknife: variable has no jackknife replicates")
        layout = self.layout
        names = layout.ensemble_names
        ensemble_replicates = []
        for name in names:
            # one ensemble: all rows, without a copy
            rows = slice(None) if len(names) == 1 else layout.ensembles == name
            replicates = self.jackknife_replicates[rows]
            ensemble_bin_size = bin_size.get(name, 1) if isinstance(bin_size, Mapping) else bin_size
            if ensemble_bin_size != 1:
                if len(np.unique(layout.replicas[rows])) > 1:
                    raise ValueError("Delayed binning: more than one replica; binning across replicas is not supported")
                replicates = jackknife.delayed_binning(replicates, ensemble_bin_size, layout.weights[rows])
            ensemble_replicates.append(replicates)
        return ensemble_replicates

    def bootstrap_variance(self) -> np.ndarray:
        """Return the bootstrap variance."""
        if self.bootstrap_replicates is None:
            raise ValueError("bootstrap_variance: variable has no bootstrap replicates")
        return np.asarray(bootstrap.variance(self.bootstrap_replicates))

    def bootstrap_covariance(self) -> np.ndarray:
        """Return the bootstrap covariance."""
        if self.bootstrap_replicates is None:
            raise ValueError("bootstrap_covariance: variable has no bootstrap replicates")
        return bootstrap.covariance(self.bootstrap_replicates)

    def gamma_method(self, S: float | Mapping[str, float] = 1.5) -> gamma.GammaErrorAnalysis:
        """Return the gamma-method error analysis per component, without external inputs; the variances from each
        ensemble are added.

        ``S`` sets the automatic summation window; a larger ``S`` gives a larger window. A dict gives ``S`` per
        ensemble name, 1.5 for names it does not contain. Fields: see ``GammaErrorAnalysis``. The variable is
        unchanged.
        """
        if self.fluctuations is None:
            raise ValueError("gamma_method: variable has no fluctuations")
        layout = self.layout
        return gamma.error_analysis(self.fluctuations, layout.ensembles, layout.monte_carlo_times, layout.replicas, S)

    def bias_corrected_value(self) -> np.ndarray:
        """Return the central value with the replica bias correction, which cancels the leading
        O(1/N) bias of nonlinear derived quantities. The corrections of the ensembles with at least two replicas
        are added; needs replica central values and one such ensemble. The variable is unchanged."""
        sizes = self._replica_sizes()
        if self.replica_central_values is None or all(len(ensemble_sizes) < 2 for ensemble_sizes in sizes.values()):
            raise ValueError("bias_corrected_value: needs replica central values of at least two replicas")
        row_ensembles = _row_ensembles(self.layout, "replica_central_values")
        corrected = [gamma.bias_corrected_value(self.central_value, self.replica_central_values[row_ensembles == name],
                                                ensemble_sizes)
                     for name, ensemble_sizes in sizes.items() if len(ensemble_sizes) >= 2]
        # central value plus the correction of each ensemble
        return np.asarray(np.sum(corrected, axis=0) - (len(corrected) - 1) * self.central_value)

    def _replica_sizes(self) -> dict[str, np.ndarray]:
        """Number of configurations of each replica, by ensemble name."""
        layout = self.layout
        return {name: np.unique(layout.replicas[layout.ensembles == name], return_counts=True)[1]
                for name in layout.ensemble_names}

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------

    def summarize(self, gamma_S: float | Mapping[str, float] = 1.5,
                  jackknife_bin_size: int | Mapping[str, int] = 1,
                  components: Mapping[str, int | tuple[int, ...]] | Iterable[int | tuple[int, ...]] | None = None) -> None:
        """Print ensembles, configurations and replicas, then per component the central value, its bias-corrected
        value and the correction in units of the gamma error, the errors (gamma method at ``gamma_S``, jackknife at
        ``jackknife_bin_size``, bootstrap), tau_int, window W and replica Q-value; columns without values are left
        out. ``gamma_S`` and ``jackknife_bin_size`` are as ``S`` in ``gamma_method`` and ``bin_size`` in
        ``jackknife_variance``; each applies to its own column only. The gamma and
        jackknife errors include the external inputs. With more than one source (ensemble or external input), a row
        follows for each source the component depends on, with its gamma and jackknife errors and their shares of
        the total variance, for an ensemble also tau_int, W and replica Q-value. ``components`` lists the rows to show
        (ints or index tuples), default all; a dict names them, e.g. ``{"A": 0, "m": 1}``."""
        sizes = self._replica_sizes()
        shape = np.shape(self.central_value)
        sizes_text = "; ".join(f"{name}: N = {np.sum(ensemble_sizes)}"
                               + (f" = {' + '.join(map(str, ensemble_sizes))}" if len(ensemble_sizes) > 1 else "")
                               for name, ensemble_sizes in sizes.items())
        S_text = "" if self.fluctuations is None else f", gamma S = {gamma_S}"
        bin_size_text = ("" if self.jackknife_replicates is None or jackknife_bin_size == 1
                         else f", jackknife bin size = {jackknife_bin_size}")
        print(f"{sizes_text or 'no configurations'}, shape {shape}{S_text}{bin_size_text}")
        result = None if self.fluctuations is None else self.gamma_method(gamma_S)
        corrected = bias_over_error = None
        q_values = {}
        if self.replica_central_values is not None and any(len(ensemble_sizes) > 1 for ensemble_sizes in sizes.values()):
            corrected = self.bias_corrected_value()
            if result is not None:
                # constant component (error 0): NaN
                bias_over_error = (self.central_value - corrected) / np.where(result.error == 0, np.nan, result.error)
                row_ensembles = _row_ensembles(self.layout, "replica_central_values")
                q_values = {name: gamma.replica_q_value(self.replica_central_values[row_ensembles == name],
                                                        ensemble_sizes, result.ensemble_errors[name])
                            for name, ensemble_sizes in sizes.items() if len(ensemble_sizes) > 1}
        unpropagated = [name for name, external in self.external_inputs.items() if external.jacobian is None]
        if unpropagated:
            print(f"gamma and jackknife errors without the external inputs {', '.join(unpropagated)} (no Jacobian)")
        external_errors = {} if unpropagated else self._external_errors()
        # error from each source, for the methods that have it
        source_errors = {}
        if result is not None:
            source_errors["gamma"] = _merge_sources(result.ensemble_errors, external_errors)
        if self.jackknife_replicates is not None:
            source_errors["jackknife"] = _merge_sources(self._jackknife_errors(jackknife_bin_size), external_errors)
        # total error of each available method
        errors = {method: np.sqrt(sum(map(np.square, sources.values()), np.zeros(shape)))
                  for method, sources in source_errors.items()}
        if self.bootstrap_replicates is not None:
            errors["bootstrap"] = np.sqrt(self.bootstrap_variance())
        multiple = len(sizes) + len(external_errors) > 1

        def error_cells(index: tuple, source: str | None = None) -> list[str]:
            """Error of each method, total or from ``source``; with multiple sources and its share of the total
            variance in percent."""
            cells = []
            for method, total in errors.items():
                error = total if source is None else source_errors.get(method, {}).get(source)
                if error is None:
                    cells.append("")
                elif multiple and method in source_errors and total[index] != 0:
                    cells.append(f"{error[index]:.2g} ({100 * (error[index] / total[index])**2:.1f}%)")
                else:
                    cells.append(f"{error[index]:.2g}")
            return cells

        def diagnostics(name: str, index: tuple) -> list[str]:
            """tau_int, W and replica Q-value of one ensemble."""
            return ["" if result is None else log.format_paren(result.tau_int[name][index].item(),
                                                               result.error_tau_int[name][index].item()),
                    "" if result is None else str(result.window[name][index]),
                    f"{q_values[name][index]:.2f}" if name in q_values else ""]

        rows = [["component", "source", "value", "corrected", "bias/σ",
                 *(f"{method}_error" + (" (variance share)" if multiple and method in source_errors else "")
                   for method in errors),
                 "tau_int", "W", "replica Q"]]
        for label, index in _selected_components(shape, components):
            value_error = next(iter(errors.values()))[index].item() if errors else None
            rows.append([label, "total" if multiple else "",
                         _with_error(self.central_value[index].item(), value_error),
                         "" if corrected is None else _with_error(corrected[index].item(), value_error),
                         "" if bias_over_error is None else f"{bias_over_error[index]:.2f}",
                         *error_cells(index),
                         *(diagnostics(next(iter(sizes)), index) if len(sizes) == 1 and not multiple else [""] * 3)])
            for source in (*sizes, *external_errors) if multiple else ():
                # no row for a source with variance shares below 0.05%: the component does not depend on it
                if any(method_errors[source][index]**2 > 5e-4 * errors[method][index]**2
                       for method, method_errors in source_errors.items()):
                    rows.append(["", source, "", "", "", *error_cells(index, source),
                                 *(diagnostics(source, index) if source in sizes else [""] * 3)])
        _print_table(rows)

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
        if not self.layout.ensemble_names:
            raise ValueError("plot_fluctuations: variable has no configurations")
        central_value = self.central_value[np.index_exp[component]].item()
        return _monte_carlo_history(ax, central_value + select_component(self.fluctuations, component), self.layout,
                                    central_value, "central value + fluctuation")

    def plot_weights(self, ax: Axes) -> Axes:
        """Plot the weights into ``ax`` against Monte Carlo time with their mean per ensemble."""
        names = self.layout.ensemble_names
        if not names:
            raise ValueError("plot_weights: variable has no configurations")
        weights = self.layout.weights
        means = {name: np.mean(weights[self.layout.ensembles == name]).item() for name in names}
        return _monte_carlo_history(ax, weights, self.layout, means if len(names) > 1 else means[names[0]], "weight",
                                    reference_name="mean")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _merge_sources(ensembles: dict[str, np.ndarray], external: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Errors by source, the ensembles then the external inputs; raises if two sources have the same name."""
    clash = ensembles.keys() & external.keys()
    if clash:
        raise ValueError(f"an ensemble and an external input have the same source name: {', '.join(sorted(clash))}")
    return ensembles | external


def _selected_components(shape: tuple[int, ...], components: Mapping[str, int | tuple[int, ...]]
                         | Iterable[int | tuple[int, ...]] | None) -> list[tuple[str, tuple]]:
    """Label and index of each component to show: all of ``shape`` for None, the listed ones, or the named ones
    of a dict."""
    if isinstance(components, Mapping):
        return [(str(name), np.index_exp[component]) for name, component in components.items()]
    indices = np.ndindex(shape) if components is None else [np.index_exp[component] for component in components]
    return [(", ".join(map(str, index)), index) for index in indices]


def _with_error(value: float, error: float | None) -> str:
    """``value(error)``, or the value with six digits without an error."""
    return f"{value:.6g}" if error is None else log.format_paren(value, error)


def _print_table(rows: list[list[str]]) -> None:
    """Print ``rows`` (the first is the header) in aligned columns, without the columns that have no values."""
    columns = [i for i in range(len(rows[0])) if any(row[i] for row in rows[1:])]
    rows = [[row[i] for i in columns] for row in rows]
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())


def _row_ensembles(layout: Layout, field: str) -> np.ndarray:
    """Ensemble name of each row of ``field``: one row per (ensemble, replica) for replica central values, else
    one per configuration."""
    if field != "replica_central_values":
        return layout.ensembles
    # the empty first array keeps this defined without configurations
    return np.concatenate([layout.ensembles[:0]]
                          + [np.full(len(np.unique(layout.replicas[layout.ensembles == name])), name)
                             for name in layout.ensemble_names])


def _replica_rows(layout: Layout) -> list[tuple[str | None, int | None, np.ndarray]]:
    """Name, replica and rows of every replica of every ensemble, in layout order. The name is None for one
    ensemble, the replica is None if its ensemble has one replica."""
    names = layout.ensemble_names
    replica_rows = []
    for name in names:
        in_ensemble = layout.ensembles == name
        replicas = np.unique(layout.replicas[in_ensemble])
        for replica in replicas:
            replica_rows.append((name if len(names) > 1 else None, replica if len(replicas) > 1 else None,
                                 in_ensemble & (layout.replicas == replica)))
    return replica_rows


def _monte_carlo_history(ax: Axes, values: np.ndarray, layout: Layout, reference: float | Mapping[str, float],
                         ylabel: str, reference_name: str = "central value") -> Axes:
    """Plot ``values`` into ``ax`` against Monte Carlo time, the replicas of every ensemble one after another, with
    a line at ``reference`` (a dict by ensemble name: one line per ensemble, over its replicas); legend of the
    ensembles and replicas (if more than one) and the reference line."""
    offset = 0
    segments = []
    for name, replica, rows in _replica_rows(layout):
        times = layout.monte_carlo_times[rows] - layout.monte_carlo_times[rows][0]
        if offset:
            ax.axvline(offset - 0.5, color="0.6", linewidth=0.8)
        # NaN at gaps, so the line breaks there
        gaps = np.flatnonzero(np.diff(times) > np.diff(times).min()) + 1
        label = " ".join(([name] if name else []) + ([f"replica {replica}"] if replica is not None else []))
        ax.plot(np.insert((times + offset).astype(float), gaps, np.nan), np.insert(values[rows], gaps, np.nan),
                linewidth=0.7, label=label or None)
        segments.append((name, offset, offset + times[-1]))
        offset += times[-1] + 1
    if isinstance(reference, Mapping):
        names, starts, ends = zip(*segments)
        ax.hlines([reference[name] for name in names], starts, ends, color="0.15", linewidth=1, label=reference_name)
    else:
        ax.axhline(reference, color="0.15", linewidth=1, label=f"{reference_name} = {reference:.4g}")
    legend_above(ax, *ax.get_legend_handles_labels())
    ax.set(xlabel="Monte Carlo time", ylabel=ylabel)
    style_axes(ax)
    return ax
