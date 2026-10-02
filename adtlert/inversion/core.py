"""Log-space ERT inversion built on the differentiable forward operators.

Single-time and time-lapse inversions share one engine: models are stored as
``(n_cells, n_times)`` optimizer states, data as ``(n_times, n_measurements)``. Each
iteration stacks a linearized data term with spatial, temporal, and sensor terms and takes
either a Gauss-Newton/LM step or a matrix-free first-order step.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from adtlert.forward import ERTForward2p5D, ERTForwardModeling
from adtlert.inversion.config import (
    ArrayLike,
    ERTInversionResult,
    InversionConfig,
    TimeLapseERTInversionResult,
    _check_config,
    _emit,
)
from adtlert.inversion.forward_calls import (
    _cached_forward_and_jacobian,
    _forward_log_response,
    _forward_log_response_series,
)
from adtlert.inversion.inputs import (
    _as_log_model,
    _as_log_models,
    _as_observed_log,
    _config_for_time_index,
    _config_for_time_window,
    _measurement_count,
    _model_size,
    _weights,
)
from adtlert.inversion.matrix_free import (
    _normal_log_response_vjp_series,
    _supports_normal_matrix_free,
)
from adtlert.inversion.misfit import build_data_misfit
from adtlert.inversion.objective import (
    _coverage,
    _data_chi2,
    _data_cotangents,
    _data_phi,
    _data_system,
    _is_difference_misfit,
    _line_search_tau,
)
from adtlert.inversion.optimizers import (
    build_optimization_algorithm,
    first_order_step,
    linearized_gradient,
    linearized_step,
)
from adtlert.inversion.state import (
    _log_model_to_state,
    _petrophysics,
    _state_to_log_model,
)
from adtlert.inversion.terms import RegularizationTerms

# ---------------------------------------------------------------------------
# Parameterized forward (inversion parameters on a coarser mesh than the forward solve)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Configuration and input validation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Petrophysical state <-> model maps
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Forward adapters (log resistivity in, log apparent resistivity out)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Objective pieces
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Inversion engine
# ---------------------------------------------------------------------------


@dataclass
class _InversionRun:
    log_models: np.ndarray
    parameter_models: np.ndarray
    parameter_name: str
    predicted_log: np.ndarray
    coverage: list[np.ndarray]
    chi2: list[float]


def _invert(
    forward,
    observed_log: np.ndarray,
    initial_logs: np.ndarray,
    reference_logs: np.ndarray | None,
    config: InversionConfig,
    *,
    single: bool = False,
    jacobian_cache: OrderedDict | None = None,
    jacobian_cache_entries: int = 128,
) -> _InversionRun:
    """Shared engine: ``observed_log`` is ``(T, D)``, models are ``(C, T)``.

    ``single=True`` reproduces the single-time inversion: no temporal, joint-frame, sensor,
    or baseline-freeze terms, and ``single_*`` progress events.
    """

    n_times, n_cells = observed_log.shape[0], initial_logs.shape[0]
    prefix = "single" if single else "timelapse"
    weight = _weights(config.data_std, observed_log.shape)
    misfit = build_data_misfit(config.data_misfit)
    terms = RegularizationTerms(forward, config, n_cells, n_times, single=single)
    joint_frame = terms.joint_frame
    matrix_free = not build_optimization_algorithm(
        config.optimization_algorithm
    ).uses_linearized_solver
    matrix_free = matrix_free and _supports_normal_matrix_free(forward, config)
    time_configs = [_config_for_time_index(config, t) for t in range(n_times)]

    models = _log_model_to_state(initial_logs, config)
    frozen = (
        models[:, 0].copy() if config.freeze_first_timestep and not single else None
    )
    if frozen is not None:
        models[:, 0] = frozen
    if reference_logs is not None:
        reference = _log_model_to_state(reference_logs, config)
    elif config.spatial_regularization == "identity" and not joint_frame:
        reference = models.copy()
    else:
        reference = None

    def column_logs(states: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [_state_to_log_model(states[:, t], time_configs[t]) for t in range(n_times)]
        )

    def chain_factors(states: np.ndarray) -> list[np.ndarray]:
        return [
            _petrophysics(time_configs[t], states[:, t]).d_log_resistivity_d_state(
                states[:, t]
            )
            for t in range(n_times)
        ]

    def linearize(states: np.ndarray, iteration: int, stage: str):
        logs = column_logs(states)
        if matrix_free:
            return _forward_log_response_series(forward, logs.T), []
        rows, jacobians = [], []
        for t, factor in enumerate(chain_factors(states)):
            progress = dict(
                iteration=iteration,
                max_iterations=config.max_iterations,
                stage=stage,
                time_index=t,
                time_number=t + 1,
                n_times=n_times,
            )
            if not single:
                _emit(config, "timelapse_time_start", **progress)
            predicted, jacobian = _cached_forward_and_jacobian(
                jacobian_cache, jacobian_cache_entries, forward, logs[:, t],
                config.include_robin_boundary_derivative, config.normal_sensitivity,
            )  # fmt: skip
            rows.append(predicted)
            jacobians.append(jacobian * factor[None, :])
            if not single:
                _emit(config, "timelapse_time_done", **progress)
        return np.vstack(rows), jacobians

    _emit(
        config,
        f"{prefix}_start",
        n_cells=n_cells,
        **(
            {"n_data": observed_log.shape[1]}
            if single
            else {"n_measurements": observed_log.shape[1], "n_times": n_times}
        ),
        max_iterations=config.max_iterations,
    )
    timing = {} if single else {"n_times": n_times}
    optimizer_state: dict[str, Any] = {}
    predicted_log, jacobians, data_gradients, chi2_history = None, [], [], []
    stop_reason = "max_iterations"
    for iteration in range(1, config.max_iterations + 1):
        _emit(
            config,
            f"{prefix}_iteration_start",
            iteration=iteration,
            max_iterations=config.max_iterations,
            **timing,
        )
        if predicted_log is None:
            predicted_log, jacobians = linearize(models, iteration, "linearization")
        current = models.reshape(-1, order="F")
        blocks = terms.blocks(models, reference)
        if matrix_free:
            cotangents = _data_cotangents(misfit, predicted_log, observed_log, weight)
            gradient_rows = _normal_log_response_vjp_series(
                forward, column_logs(models).T, predicted_log, cotangents
            )
            data_gradients = [
                row * factor
                for row, factor in zip(
                    gradient_rows, chain_factors(models), strict=True
                )
            ]
            gradient = np.column_stack(data_gradients).reshape(-1, order="F")
            for matrix, rhs in blocks:
                gradient += linearized_gradient(matrix, rhs)
            delta = first_order_step(current, gradient, optimizer_state, config)
        else:
            system = [
                _data_system(misfit, predicted_log, observed_log, weight, jacobians),
                *blocks,
            ]
            matrix = sp.vstack([m for m, _ in system], format="csr")
            delta = linearized_step(
                matrix,
                np.concatenate([r for _, r in system]),
                current,
                optimizer_state,
                config,
            )

        def clipped(states: np.ndarray) -> np.ndarray:
            states = _petrophysics(config, states).clip_state(states)
            if frozen is not None:
                states[:, 0] = frozen
            return states

        candidate = clipped(
            models + config.step_length * delta.reshape((n_cells, n_times), order="F")
        )
        candidate_predicted, candidate_jacobians = linearize(
            candidate, iteration, "candidate"
        )
        step = candidate.reshape(-1, order="F") - current
        tau = 1.0
        if config.line_search:
            tau = _line_search_tau(
                current,
                step,
                predicted_log.ravel(),
                candidate_predicted.ravel(),
                sp.vstack([m for m, _ in blocks], format="csr")
                if blocks
                else sp.csr_matrix((0, current.size)),
                np.concatenate([m @ current + r for m, r in blocks])
                if blocks
                else np.zeros(0),
                lambda values: _data_phi(
                    misfit, values.reshape(observed_log.shape), observed_log, weight
                ),
            )
            step = tau * step
        if tau < 0.95:
            models = clipped((current + step).reshape((n_cells, n_times), order="F"))
            predicted_log, jacobians = linearize(models, iteration, "line_search")
        else:
            models, predicted_log, jacobians = (
                candidate,
                candidate_predicted,
                candidate_jacobians,
            )

        chi2 = _data_chi2(misfit, predicted_log, observed_log, weight)
        chi2_history.append(chi2)
        step_norm = float(np.linalg.norm(step) / max(float(np.sqrt(current.size)), 1.0))
        _emit(
            config,
            f"{prefix}_iteration_done",
            iteration=iteration,
            max_iterations=config.max_iterations,
            **timing,
            chi2=float(chi2),
            step_norm=step_norm,
            target_chi2=config.target_chi2,
        )
        if config.target_chi2 is not None and chi2 < config.target_chi2:
            stop_reason = "target_chi2"
            break
        if step_norm < config.step_tolerance:
            stop_reason = "step_tolerance"
            break

    coverage = [
        _coverage(forward, item)
        for item in (data_gradients if matrix_free else jacobians)
    ]
    transform = _petrophysics(config, models)
    run = _InversionRun(
        log_models=transform.log_resistivity_from_state(models),
        parameter_models=transform.parameter_from_state(models),
        parameter_name=transform.parameter_name,
        predicted_log=predicted_log,
        coverage=coverage,
        chi2=chi2_history,
    )
    _emit(
        config,
        f"{prefix}_done",
        iterations=len(chi2_history),
        max_iterations=config.max_iterations,
        final_chi2=chi2_history[-1] if chi2_history else None,
        stop_reason=stop_reason,
    )
    return run


def _timelapse_result(run: _InversionRun, **extra: Any) -> TimeLapseERTInversionResult:
    return TimeLapseERTInversionResult(
        final_models=np.exp(run.log_models),
        final_log_models=run.log_models,
        predicted_data=np.exp(run.predicted_log),
        predicted_log_data=run.predicted_log,
        coverage=np.nanmedian(np.column_stack(run.coverage), axis=1),
        all_coverage=run.coverage,
        all_chi2=np.asarray(run.chi2, dtype=float),
        iteration_chi2=run.chi2,
        final_parameter_models=run.parameter_models,
        final_parameter_name=run.parameter_name,
        **extra,
    )


def invert_single_log_resistivity(
    forward,
    observed_data: ArrayLike,
    initial_model: ArrayLike,
    *,
    reference_model: ArrayLike | None = None,
    config: InversionConfig | None = None,
    observed_log_data: bool = False,
    initial_log_model: bool = False,
    reference_log_model: bool = False,
) -> ERTInversionResult:
    """Invert one ERT dataset for cell log-resistivity.

    ``observed_data`` is apparent resistivity unless ``observed_log_data=True``;
    ``initial_model``/``reference_model`` are resistivity unless their ``*_log_model`` flag is set.
    """

    config = config or InversionConfig()
    _check_config(config)
    n_cells = _model_size(forward)
    observed = _as_observed_log(
        observed_data, observed_log_data, _measurement_count(forward), timelapse=False
    )
    if _is_difference_misfit(build_data_misfit(config.data_misfit)):
        raise ValueError(
            "log data-difference misfit is only defined for time-lapse inversions"
        )
    initial = _as_log_model(initial_model, n_cells, initial_log_model, "initial_model")
    reference = (
        None
        if reference_model is None
        else _as_log_model(
            reference_model, n_cells, reference_log_model, "reference_model"
        )
    )
    run = _invert(
        forward,
        observed[None, :],
        initial[:, None],
        None if reference is None else reference[:, None],
        config,
        single=True,
    )
    return ERTInversionResult(
        final_model=np.exp(run.log_models[:, 0]),
        final_log_model=run.log_models[:, 0],
        predicted_data=np.exp(run.predicted_log[0]),
        predicted_log_data=run.predicted_log[0],
        coverage=run.coverage[0],
        iteration_chi2=run.chi2,
        final_parameter_model=run.parameter_models[:, 0],
        final_parameter_name=run.parameter_name,
    )


def _timelapse_inputs(
    forward,
    observed_data,
    initial_model,
    reference_model,
    observed_log_data,
    initial_log_model,
    reference_log_model,
):
    n_cells = _model_size(forward)
    observed = _as_observed_log(
        observed_data, observed_log_data, _measurement_count(forward), timelapse=True
    )
    n_times = observed.shape[0]
    initial = _as_log_models(
        initial_model, n_cells, n_times, initial_log_model, "initial_model"
    )
    reference = (
        None
        if reference_model is None
        else _as_log_models(
            reference_model, n_cells, n_times, reference_log_model, "reference_model"
        )
    )
    return observed, initial, reference


def invert_timelapse_log_resistivity(
    forward,
    observed_data: ArrayLike,
    initial_model: ArrayLike,
    *,
    reference_model: ArrayLike | None = None,
    config: InversionConfig | None = None,
    observed_log_data: bool = False,
    initial_log_model: bool = False,
    reference_log_model: bool = False,
) -> TimeLapseERTInversionResult:
    """Jointly invert time-lapse ERT data with optional temporal regularization.

    Observations are ``(n_times, n_measurements)`` or ``(n_measurements, n_times)``;
    returned models are ``(n_cells, n_times)``.
    """

    config = config or InversionConfig(temporal_regularization=1.0)
    _check_config(config)
    observed, initial, reference = _timelapse_inputs(
        forward,
        observed_data,
        initial_model,
        reference_model,
        observed_log_data,
        initial_log_model,
        reference_log_model,
    )
    if observed.shape[0] < 2:
        raise ValueError("time-lapse inversion needs at least two timesteps")
    return _timelapse_result(_invert(forward, observed, initial, reference, config))


def _window_start_indices(
    n_times: int, window_size: int, window_step: int
) -> list[int]:
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    if window_size > n_times:
        raise ValueError(f"window_size={window_size} exceeds n_times={n_times}")
    starts = list(range(0, n_times - window_size + 1, max(1, int(window_step))))
    return sorted(set(starts + [n_times - window_size]))


def invert_windowed_timelapse_log_resistivity(
    forward,
    observed_data: ArrayLike,
    initial_model: ArrayLike,
    *,
    window_size: int = 3,
    window_step: int = 1,
    reference_model: ArrayLike | None = None,
    config: InversionConfig | None = None,
    observed_log_data: bool = False,
    initial_log_model: bool = False,
    reference_log_model: bool = False,
) -> TimeLapseERTInversionResult:
    """Sliding-window time-lapse inversion.

    Every overlapping window is inverted independently; each timestep's final log model is
    the mean over the windows containing it (the geometric mean in resistivity).
    """

    config = config or InversionConfig(temporal_regularization=1.0)
    _check_config(config)
    observed, initial, reference = _timelapse_inputs(
        forward,
        observed_data,
        initial_model,
        reference_model,
        observed_log_data,
        initial_log_model,
        reference_log_model,
    )
    n_cells, n_times = initial.shape
    window_size = int(window_size)
    starts = _window_start_indices(n_times, window_size, int(window_step))
    contributions: list[list[np.ndarray]] = [[] for _ in range(n_times)]
    coverage_bank, window_chi2, window_reports = [], [], []
    jacobian_cache: OrderedDict = OrderedDict()
    cache_entries = max(
        16, min(128, window_size * max(1, config.max_iterations + 2) * 4)
    )

    _emit(
        config,
        "windowed_start",
        n_cells=n_cells,
        n_measurements=observed.shape[1],
        n_times=n_times,
        n_windows=len(starts),
        window_size=window_size,
        window_step=int(window_step),
        max_iterations=config.max_iterations,
    )
    for window_index, start in enumerate(starts, start=1):
        end = start + window_size
        window = dict(
            window_index=window_index,
            n_windows=len(starts),
            start_idx=start,
            end_idx=end - 1,
        )
        _emit(
            config,
            "window_start",
            **window,
            window_size=window_size,
            max_iterations=config.max_iterations,
        )
        began = time.perf_counter()
        result = _timelapse_result(
            _invert(
                forward,
                observed[start:end],
                initial[:, start:end],
                None if reference is None else reference[:, start:end],
                _config_for_time_window(
                    config, observed_shape=observed.shape, start=start, end=end
                ),
                jacobian_cache=jacobian_cache,
                jacobian_cache_entries=cache_entries,
            )
        )
        for offset in range(window_size):
            contributions[start + offset].append(result.final_log_models[:, offset])
        coverage_bank.append(np.asarray(result.coverage, dtype=float).ravel())
        final_chi2 = float(result.iteration_chi2[-1]) if result.iteration_chi2 else None
        if final_chi2 is not None:
            window_chi2.append(final_chi2)
        window_reports.append(
            {
                "start_idx": start,
                "end_idx": end - 1,
                "final_chi2_data": final_chi2,
                "iterations": len(result.iteration_chi2),
                "elapsed_sec": float(time.perf_counter() - began),
            }
        )
        _emit(config, "window_done", **window, final_chi2=final_chi2)

    final_log_models = np.column_stack(
        [np.mean(np.column_stack(items), axis=1) for items in contributions]
    )
    transform = _petrophysics(config, final_log_models)
    final_states = transform.state_from_log_resistivity(final_log_models)
    _emit(config, "windowed_prediction_start", n_times=n_times)
    predicted_rows = []
    for t in range(n_times):
        _emit(
            config,
            "windowed_prediction_step",
            time_index=t,
            time_number=t + 1,
            n_times=n_times,
        )
        predicted_rows.append(_forward_log_response(forward, final_log_models[:, t]))
    predicted_log = np.vstack(predicted_rows)
    _emit(
        config,
        "windowed_done",
        n_windows=len(starts),
        final_chi2=window_chi2[-1] if window_chi2 else None,
    )
    return TimeLapseERTInversionResult(
        final_models=np.exp(final_log_models),
        final_log_models=final_log_models,
        predicted_data=np.exp(predicted_log),
        predicted_log_data=predicted_log,
        coverage=np.nanmedian(np.column_stack(coverage_bank), axis=1),
        all_coverage=coverage_bank,
        all_chi2=np.asarray(window_chi2, dtype=float),
        iteration_chi2=window_chi2,
        window_reports=window_reports,
        final_parameter_models=transform.parameter_from_state(final_states),
        final_parameter_name=transform.parameter_name,
    )


# ---------------------------------------------------------------------------
# Notebook-style ``setup()/run()`` wrappers
# ---------------------------------------------------------------------------


@dataclass
class ERTInversion:
    """Notebook-style wrapper for :func:`invert_single_log_resistivity`."""

    forward: ERTForward2p5D | ERTForwardModeling
    observed_data: ArrayLike
    config: InversionConfig = field(default_factory=InversionConfig)
    observed_log_data: bool = False

    def setup(self) -> ERTInversion:
        """Validate configuration and data dimensions; returns ``self``."""

        _check_config(self.config)
        _as_observed_log(
            self.observed_data,
            self.observed_log_data,
            _measurement_count(self.forward),
            timelapse=False,
        )
        return self

    def run(
        self,
        initial_model: ArrayLike,
        *,
        reference_model=None,
        initial_log_model=False,
        reference_log_model=False,
    ):
        return invert_single_log_resistivity(
            self.forward,
            self.observed_data,
            initial_model,
            reference_model=reference_model,
            config=self.config,
            observed_log_data=self.observed_log_data,
            initial_log_model=initial_log_model,
            reference_log_model=reference_log_model,
        )


@dataclass
class TimeLapseERTInversion:
    """Notebook-style wrapper for :func:`invert_timelapse_log_resistivity`."""

    forward: ERTForward2p5D | ERTForwardModeling
    observed_data: ArrayLike
    config: InversionConfig = field(
        default_factory=lambda: InversionConfig(temporal_regularization=1.0)
    )
    observed_log_data: bool = False

    def setup(self) -> TimeLapseERTInversion:
        """Validate configuration and data dimensions; returns ``self``."""

        _check_config(self.config)
        _as_observed_log(
            self.observed_data,
            self.observed_log_data,
            _measurement_count(self.forward),
            timelapse=True,
        )
        return self

    def run(
        self,
        initial_model: ArrayLike,
        *,
        reference_model=None,
        initial_log_model=False,
        reference_log_model=False,
    ):
        return invert_timelapse_log_resistivity(
            self.forward,
            self.observed_data,
            initial_model,
            reference_model=reference_model,
            config=self.config,
            observed_log_data=self.observed_log_data,
            initial_log_model=initial_log_model,
            reference_log_model=reference_log_model,
        )


@dataclass
class WindowedTimeLapseERTInversion:
    """Notebook-style wrapper for :func:`invert_windowed_timelapse_log_resistivity`."""

    forward: ERTForward2p5D | ERTForwardModeling
    observed_data: ArrayLike
    config: InversionConfig = field(
        default_factory=lambda: InversionConfig(temporal_regularization=1.0)
    )
    window_size: int = 3
    window_step: int = 1
    observed_log_data: bool = False

    def setup(self) -> WindowedTimeLapseERTInversion:
        """Validate configuration, data dimensions, and window controls; returns ``self``."""

        _check_config(self.config)
        observed = _as_observed_log(
            self.observed_data,
            self.observed_log_data,
            _measurement_count(self.forward),
            timelapse=True,
        )
        _window_start_indices(
            observed.shape[0], int(self.window_size), int(self.window_step)
        )
        return self

    def run(
        self,
        initial_model: ArrayLike,
        *,
        reference_model=None,
        initial_log_model=False,
        reference_log_model=False,
    ):
        return invert_windowed_timelapse_log_resistivity(
            self.forward,
            self.observed_data,
            initial_model,
            window_size=self.window_size,
            window_step=self.window_step,
            reference_model=reference_model,
            config=self.config,
            observed_log_data=self.observed_log_data,
            initial_log_model=initial_log_model,
            reference_log_model=reference_log_model,
        )
