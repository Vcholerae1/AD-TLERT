"""Fast GPU training loops for physics-coupled implicit ERT inversion."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np
import torch

from adtlert.inr.networks import MultiscaleINR
from adtlert.inr.physics import matrix_free_log_rhoa, matrix_free_log_rhoa_series, prepare_cuda_forward
from adtlert.inversion.regularization import first_order_constraint_matrix, regularization_mesh


ProgressCallback = Callable[[dict[str, Any]], None]
PenaltyCallback = Callable[[torch.Tensor, int], torch.Tensor]


@dataclass(frozen=True)
class INRConfig:
    """Controls for matrix-free INR inversion."""

    max_iterations: int = 300
    learning_rate: float = 3.0e-3
    optimizer: str = "adam"
    scheduler: str = "multistep"
    learning_rate_milestones: tuple[int, ...] = (100, 200)
    target_chi2: float | None = 1.0
    device: str = "cuda"
    require_cuda: bool = True
    gradient_clip_norm: float | None = 10.0
    weight_decay: float = 0.0
    plateau_patience: int = 20
    plateau_factor: float = 0.5
    minimum_learning_rate: float = 1.0e-5
    progressive_encoding: bool = True
    progressive_full_iteration: int = 150
    progressive_spatial_start_levels: int = 2
    progressive_temporal_start_levels: int = 1
    spatial_regularization: float = 0.0
    temporal_regularization: float = 0.0
    regularization_huber_delta: float = 0.1
    log_every: int = 10
    prepare_solver: bool = True
    time_window_size: int | None = None
    time_window_step: int = 1
    full_evaluation_interval: int = 25
    alternate_window_direction: bool = True
    snapshot_interval: int | None = None
    progress_callback: ProgressCallback | None = field(default=None, repr=False, compare=False)
    # Weighted scalar added to the objective; receives the full log-resistivity model and the iteration.
    extra_penalty: PenaltyCallback | None = field(default=None, repr=False, compare=False)

    def validate(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.optimizer not in {"adam", "adamw", "radam"}:
            raise ValueError("optimizer must be 'adam', 'adamw', or 'radam'")
        if self.scheduler not in {"multistep", "plateau", "none"}:
            raise ValueError("scheduler must be 'multistep', 'plateau', or 'none'")
        if any(step < 1 for step in self.learning_rate_milestones):
            raise ValueError("learning_rate_milestones must contain positive iterations")
        if tuple(sorted(set(self.learning_rate_milestones))) != self.learning_rate_milestones:
            raise ValueError("learning_rate_milestones must be strictly increasing")
        if self.target_chi2 is not None and self.target_chi2 <= 0.0:
            raise ValueError("target_chi2 must be positive")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive when set")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.plateau_patience < 1:
            raise ValueError("plateau_patience must be >= 1")
        if not (0.0 < self.plateau_factor < 1.0):
            raise ValueError("plateau_factor must be in (0, 1)")
        if not (0.0 < self.minimum_learning_rate <= self.learning_rate):
            raise ValueError("minimum_learning_rate must be positive and <= learning_rate")
        if self.log_every < 1:
            raise ValueError("log_every must be >= 1")
        if self.progressive_full_iteration < 1:
            raise ValueError("progressive_full_iteration must be >= 1")
        if self.progressive_spatial_start_levels < 0 or self.progressive_temporal_start_levels < 0:
            raise ValueError("progressive start levels must be non-negative")
        if self.spatial_regularization < 0.0 or self.temporal_regularization < 0.0:
            raise ValueError("regularization strengths must be non-negative")
        if self.regularization_huber_delta <= 0.0:
            raise ValueError("regularization_huber_delta must be positive")
        if self.require_cuda and not self.prepare_solver:
            raise ValueError("require_cuda=True requires prepare_solver=True so cuDSS can be verified")
        if self.time_window_size is not None and self.time_window_size < 1:
            raise ValueError("time_window_size must be positive when set")
        if self.time_window_step < 1:
            raise ValueError("time_window_step must be positive")
        if self.full_evaluation_interval < 1:
            raise ValueError("full_evaluation_interval must be positive")
        if self.snapshot_interval is not None and self.snapshot_interval < 1:
            raise ValueError("snapshot_interval must be positive when set")


@dataclass(frozen=True)
class INRResult:
    """Result and timing diagnostics for one INR optimization."""

    log_resistivity: np.ndarray
    resistivity: np.ndarray
    predicted_log_data: np.ndarray
    predicted_data: np.ndarray
    chi2_history: np.ndarray
    rms_history: np.ndarray
    iterations: int
    best_iteration: int
    best_chi2: float
    stop_reason: str
    elapsed_seconds: float
    forward_seconds: float
    backward_seconds: float
    optimizer_seconds: float
    device: str
    gpu_report: dict[str, Any]
    optimizer: str
    scheduler: str
    objective_history: np.ndarray
    spatial_penalty_history: np.ndarray
    temporal_penalty_history: np.ndarray
    extra_penalty_history: np.ndarray
    full_chi2_iterations: np.ndarray
    full_chi2_history: np.ndarray
    time_window_size: int | None
    time_window_step: int
    physics_forward_timesteps: int
    physics_vjp_timesteps: int


def _build_optimizer(config: INRConfig, network: torch.nn.Module) -> torch.optim.Optimizer:
    """Build a one-forward/one-VJP optimizer suitable for expensive ERT physics."""

    kwargs = {
        "lr": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
    }
    if config.optimizer == "radam":
        return torch.optim.RAdam(network.parameters(), **kwargs)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(network.parameters(), **kwargs)
    return torch.optim.Adam(network.parameters(), **kwargs)


def _build_scheduler(
    config: INRConfig,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if config.scheduler == "none":
        return None
    if config.scheduler == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(config.learning_rate_milestones),
            gamma=float(config.plateau_factor),
        )
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config.plateau_factor),
        patience=int(config.plateau_patience),
        min_lr=float(config.minimum_learning_rate),
    )


def _validated_data(
    observed_rhoa: np.ndarray,
    data_std: float | np.ndarray,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    observed = np.asarray(observed_rhoa, dtype=np.float32)
    if observed.ndim not in (1, 2) or not np.all(np.isfinite(observed)) or np.any(observed <= 0.0):
        raise ValueError("observed_rhoa must be a finite positive 1D or 2D array")
    std = np.asarray(data_std, dtype=np.float32)
    try:
        std = np.broadcast_to(std, observed.shape).copy()
    except ValueError as exc:
        raise ValueError(f"data_std cannot broadcast to observed shape {observed.shape}") from exc
    if not np.all(np.isfinite(std)) or np.any(std <= 0.0):
        raise ValueError("data_std must contain positive finite values")
    return (
        torch.as_tensor(np.log(observed), device=device, dtype=torch.float32),
        torch.as_tensor(std, device=device, dtype=torch.float32),
    )


def _emit(config: INRConfig, event: str, **payload: Any) -> None:
    if config.progress_callback is not None:
        config.progress_callback({"event": event, **payload})


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _spatial_operator(forward: Any, cell_count: int, device: torch.device) -> torch.Tensor:
    matrix = first_order_constraint_matrix(regularization_mesh(forward)).tocoo()
    if matrix.shape[1] != cell_count:
        raise ValueError("spatial regularization mesh does not match the INR parameter count")
    indices = torch.as_tensor(np.vstack((matrix.row, matrix.col)), dtype=torch.int64, device=device)
    values = torch.as_tensor(matrix.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, matrix.shape, device=device).coalesce()


def _huber_mean(values: torch.Tensor, delta: float) -> torch.Tensor:
    if values.numel() == 0:
        return values.sum()
    absolute = torch.abs(values)
    delta_t = torch.as_tensor(delta, dtype=values.dtype, device=values.device)
    return torch.mean(torch.where(absolute <= delta_t, 0.5 * values.square() / delta_t, absolute - 0.5 * delta_t))


def _detach_encoded(encoded: Any) -> Any:
    if isinstance(encoded, torch.Tensor):
        return encoded.detach()
    if isinstance(encoded, tuple):
        return tuple(_detach_encoded(value) for value in encoded)
    raise TypeError("network.encode must return a Tensor or tuple of Tensors")


def _window_schedule(n_times: int, size: int, step: int) -> tuple[list[np.ndarray], np.ndarray]:
    size = min(int(size), n_times)
    starts = list(range(0, max(n_times - size + 1, 1), int(step)))
    final_start = n_times - size
    if starts[-1] != final_start:
        starts.append(final_start)
    windows = [np.arange(start, start + size, dtype=np.int64) for start in starts]
    coverage = np.zeros(n_times, dtype=np.int64)
    for indices in windows:
        coverage[indices] += 1
    if np.any(coverage == 0):
        raise ValueError("time windows do not cover every timestep; reduce time_window_step")
    return windows, coverage


def _fit(
    *,
    forward: Any,
    network: torch.nn.Module,
    coordinates: np.ndarray,
    observed_rhoa: np.ndarray,
    data_std: float | np.ndarray,
    config: INRConfig,
    series: bool,
) -> INRResult:
    config.validate()
    device = torch.device(config.device)
    if config.require_cuda and device.type != "cuda":
        raise RuntimeError("require_cuda=True requires a CUDA network device")
    if config.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for INR inversion, but no CUDA device is available")

    network = network.to(device=device, dtype=torch.float32)
    coordinate_values = np.asarray(coordinates, dtype=np.float32)
    expected_ndim = 3 if series else 2
    if coordinate_values.ndim != expected_ndim or coordinate_values.shape[-1] != network.input_dimensions:
        raise ValueError(
            f"coordinates must have {expected_ndim} dimensions and end with {network.input_dimensions} features"
        )
    coordinates_t = torch.as_tensor(coordinate_values, device=device, dtype=torch.float32)
    with torch.no_grad():
        if config.progressive_encoding:
            network.set_progressive_levels(
                0.0,
                spatial_start_levels=config.progressive_spatial_start_levels,
                temporal_start_levels=config.progressive_temporal_start_levels,
            )
        encoded = _detach_encoded(network.encode(coordinates_t))
        initial_log = network.forward_encoded(encoded)

    observed_log, std = _validated_data(observed_rhoa, data_std, device=device)
    expected_model_shape = tuple(coordinate_values.shape[:-1])
    if tuple(initial_log.shape) != expected_model_shape:
        raise ValueError(f"network model shape {tuple(initial_log.shape)} does not match {expected_model_shape}")
    forward_cell_count = getattr(forward, "cell_count", None)
    if forward_cell_count is not None and expected_model_shape[-1] != int(forward_cell_count):
        raise ValueError(
            f"coordinates describe {expected_model_shape[-1]} cells, but forward expects {int(forward_cell_count)}"
        )
    expected_data_ndim = 2 if series else 1
    if observed_log.ndim != expected_data_ndim:
        raise ValueError(f"observed_rhoa must be {expected_data_ndim}D for this training mode")
    if series and observed_log.shape[0] != expected_model_shape[0]:
        raise ValueError("observed_rhoa timestep count must match spatiotemporal coordinates")
    if not series and config.time_window_size is not None:
        raise ValueError("time_window_size is only valid for time-lapse inversion")
    n_times = expected_model_shape[0] if series else 1
    windowed = bool(series and config.time_window_size is not None and config.time_window_size < n_times)
    if windowed:
        windows, window_coverage = _window_schedule(
            n_times,
            int(config.time_window_size),
            int(config.time_window_step),
        )
        inverse_window_coverage = torch.as_tensor(
            1.0 / window_coverage,
            device=device,
            dtype=torch.float32,
        )
    else:
        windows = [np.arange(n_times, dtype=np.int64)]
        inverse_window_coverage = torch.ones(n_times, device=device, dtype=torch.float32)
    # The VJP reuses forward fields from the operator LRU; a cache smaller than one
    # forward/VJP batch evicts them and silently re-solves every forward in backward.
    operator = getattr(forward, "forward_operator", forward)
    cache_entries = getattr(operator, "normal_field_cache_max_entries", None)
    batch_timesteps = max(int(window.size) for window in windows)
    if cache_entries is not None and int(cache_entries) < batch_timesteps:
        raise ValueError(
            f"normal_field_cache_max_entries={int(cache_entries)} is smaller than the {batch_timesteps} "
            "timesteps per forward/VJP batch; set it to at least that value when building the forward"
        )
    survey = getattr(forward, "survey", None)
    measurement_count = getattr(survey, "measurement_count", None)
    if measurement_count is not None and observed_log.shape[-1] != int(measurement_count):
        raise ValueError(
            f"observed_rhoa has {observed_log.shape[-1]} measurements, but survey expects {int(measurement_count)}"
        )

    gpu_report: dict[str, Any] = {
        "network_cuda_available": bool(torch.cuda.is_available()),
        "linear_solver_backend": None,
        "cudss_gpu_enabled": False,
        "cudss_zero_copy": False,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    if config.prepare_solver:
        preparation_model = initial_log.detach().to(device="cpu", dtype=torch.float64).numpy()
        if series:
            preparation_model = preparation_model[0]
        gpu_report = prepare_cuda_forward(
            forward,
            preparation_model,
            require_cuda=config.require_cuda,
            device=device,
        )

    optimizer = _build_optimizer(config, network)
    scheduler = _build_scheduler(config, optimizer)
    physics = matrix_free_log_rhoa_series if series else matrix_free_log_rhoa
    spatial_operator = None
    if config.spatial_regularization > 0.0:
        spatial_operator = _spatial_operator(forward, expected_model_shape[-1], device)
    chi2_history: list[float] = []
    rms_history: list[float] = []
    objective_history: list[float] = []
    spatial_penalty_history: list[float] = []
    temporal_penalty_history: list[float] = []
    extra_penalty_history: list[float] = []
    full_chi2_iterations: list[int] = []
    full_chi2_history: list[float] = []
    best: dict[str, Any] = {"chi2": float("inf"), "iteration": 0}

    def remember(iteration: int, chi2_value: float, model: torch.Tensor, prediction: torch.Tensor) -> None:
        """Record a full-data chi-square; keep the best model, prediction, and network state."""

        full_chi2_iterations.append(iteration)
        full_chi2_history.append(chi2_value)
        if chi2_value < best["chi2"]:
            best.update(
                chi2=chi2_value,
                iteration=iteration,
                model=model.detach().clone(),
                prediction=prediction.detach().clone(),
                state={name: value.detach().clone() for name, value in network.state_dict().items()},
            )
    stop_reason = "max_iterations"
    forward_seconds = 0.0
    backward_seconds = 0.0
    optimizer_seconds = 0.0
    physics_forward_timesteps = 0
    physics_vjp_timesteps = 0

    _synchronize(device)
    started = time.perf_counter()
    for iteration in range(config.max_iterations + 1):
        if config.progressive_encoding:
            network.set_progressive_levels(
                min(iteration / config.progressive_full_iteration, 1.0),
                spatial_start_levels=config.progressive_spatial_start_levels,
                temporal_start_levels=config.progressive_temporal_start_levels,
            )
        forward_started = time.perf_counter()
        log_model = network.forward_encoded(encoded)
        snapshot_due = (
            config.snapshot_interval is not None
            and iteration > 0
            and iteration % config.snapshot_interval == 0
        )
        evaluate_full = not windowed or iteration == 0 or iteration == config.max_iterations
        evaluate_full = evaluate_full or iteration % config.full_evaluation_interval == 0 or snapshot_due
        full_chi2: float | None = None
        full_rms: float | None = None
        if windowed and evaluate_full:
            with torch.no_grad():
                full_prediction = matrix_free_log_rhoa_series(log_model.detach(), forward)
                full_residual = (full_prediction - observed_log) / std
                full_chi2 = float(torch.mean(full_residual.square()).item())
                full_rms = float(
                    (100.0 * torch.sqrt(torch.mean(torch.expm1(full_prediction - observed_log).square()))).item()
                )
            physics_forward_timesteps += n_times
            remember(iteration, full_chi2, log_model, full_prediction)
        if windowed:
            cycle, position = divmod(iteration, len(windows))
            schedule_position = len(windows) - 1 - position if config.alternate_window_direction and cycle % 2 else position
            batch_indices_np = windows[schedule_position]
            batch_indices = torch.as_tensor(batch_indices_np, device=device, dtype=torch.int64)
            batch_model = log_model.index_select(0, batch_indices)
            batch_observed = observed_log.index_select(0, batch_indices)
            batch_std = std.index_select(0, batch_indices)
            predicted_log = matrix_free_log_rhoa_series(batch_model, forward)
            residual = (predicted_log - batch_observed) / batch_std
            time_weights = inverse_window_coverage.index_select(0, batch_indices)
            chi2_tensor = torch.sum(residual.square() * time_weights[:, None]) / (
                residual.shape[1] * torch.sum(time_weights)
            )
            rms_tensor = 100.0 * torch.sqrt(
                torch.sum(torch.expm1(predicted_log - batch_observed).square() * time_weights[:, None])
                / (residual.shape[1] * torch.sum(time_weights))
            )
            batch_timestep_count = int(batch_indices.numel())
        else:
            batch_indices_np = np.arange(n_times, dtype=np.int64)
            predicted_log = physics(log_model, forward)
            residual = (predicted_log - observed_log) / std
            chi2_tensor = torch.mean(torch.square(residual))
            rms_tensor = 100.0 * torch.sqrt(torch.mean(torch.square(torch.expm1(predicted_log - observed_log))))
            batch_timestep_count = n_times
        physics_forward_timesteps += batch_timestep_count
        spatial_penalty = torch.zeros((), device=device)
        if spatial_operator is not None:
            model_columns = log_model.transpose(0, 1) if series else log_model[:, None]
            spatial_differences = torch.sparse.mm(spatial_operator, model_columns)
            spatial_penalty = _huber_mean(spatial_differences, config.regularization_huber_delta)
        temporal_penalty = torch.zeros((), device=device)
        if series and config.temporal_regularization > 0.0:
            temporal_penalty = _huber_mean(
                log_model[1:] - log_model[:-1],
                config.regularization_huber_delta,
            )
        extra_penalty = torch.zeros((), device=device)
        if config.extra_penalty is not None:
            extra_penalty = config.extra_penalty(log_model, iteration)
        objective = (
            0.5 * chi2_tensor
            + config.spatial_regularization * spatial_penalty
            + config.temporal_regularization * temporal_penalty
            + extra_penalty
        )
        chi2 = float(chi2_tensor.detach().item())
        rms = float(rms_tensor.detach().item())
        objective_value = float(objective.detach().item())
        _synchronize(device)
        forward_seconds += time.perf_counter() - forward_started
        if not np.isfinite(chi2):
            raise FloatingPointError("INR inversion produced a non-finite chi-square")
        chi2_history.append(chi2)
        rms_history.append(rms)
        objective_history.append(objective_value)
        spatial_penalty_history.append(float(spatial_penalty.detach().item()))
        temporal_penalty_history.append(float(temporal_penalty.detach().item()))
        extra_penalty_history.append(float(extra_penalty.detach().item()))

        if not windowed:
            full_prediction = predicted_log
            full_chi2 = chi2
            full_rms = rms
            remember(iteration, full_chi2, log_model, full_prediction)

        target_reached = config.target_chi2 is not None and full_chi2 is not None and full_chi2 <= config.target_chi2
        if iteration % config.log_every == 0 or target_reached or iteration == config.max_iterations:
            _emit(
                config,
                "iteration",
                iteration=iteration,
                chi2=chi2,
                rms=rms,
                full_chi2=full_chi2,
                full_rms=full_rms,
                window_start=int(batch_indices_np[0]) if windowed else None,
                window_stop=int(batch_indices_np[-1] + 1) if windowed else None,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                objective=objective_value,
                spatial_penalty=spatial_penalty_history[-1],
                temporal_penalty=temporal_penalty_history[-1],
                extra_penalty=extra_penalty_history[-1],
            )
        if snapshot_due:
            _emit(
                config,
                "snapshot",
                iteration=iteration,
                chi2=full_chi2,
                rms=full_rms,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                log_resistivity=log_model.detach().to(device="cpu", dtype=torch.float64).numpy(),
            )
        if target_reached:
            stop_reason = "target_chi2"
            break
        if iteration == config.max_iterations:
            break

        optimizer.zero_grad(set_to_none=True)
        backward_started = time.perf_counter()
        objective.backward()
        physics_vjp_timesteps += batch_timestep_count
        if config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(network.parameters(), float(config.gradient_clip_norm))
        _synchronize(device)
        backward_seconds += time.perf_counter() - backward_started
        optimizer_started = time.perf_counter()
        optimizer.step()
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(chi2)
        elif scheduler is not None:
            scheduler.step()
        _synchronize(device)
        optimizer_seconds += time.perf_counter() - optimizer_started

    _synchronize(device)
    elapsed = time.perf_counter() - started
    if "model" not in best:
        raise RuntimeError("INR optimization did not evaluate a model")
    network.load_state_dict(best["state"])
    log_model_np = best["model"].to(device="cpu", dtype=torch.float64).numpy()
    predicted_np = best["prediction"].to(device="cpu", dtype=torch.float64).numpy()
    return INRResult(
        log_resistivity=log_model_np,
        resistivity=np.exp(log_model_np),
        predicted_log_data=predicted_np,
        predicted_data=np.exp(predicted_np),
        chi2_history=np.asarray(chi2_history, dtype=float),
        rms_history=np.asarray(rms_history, dtype=float),
        iterations=len(chi2_history) - 1,
        best_iteration=int(best["iteration"]),
        best_chi2=float(best["chi2"]),
        stop_reason=stop_reason,
        elapsed_seconds=float(elapsed),
        forward_seconds=float(forward_seconds),
        backward_seconds=float(backward_seconds),
        optimizer_seconds=float(optimizer_seconds),
        device=str(device),
        gpu_report=gpu_report,
        optimizer=config.optimizer,
        scheduler=config.scheduler,
        objective_history=np.asarray(objective_history, dtype=float),
        spatial_penalty_history=np.asarray(spatial_penalty_history, dtype=float),
        temporal_penalty_history=np.asarray(temporal_penalty_history, dtype=float),
        extra_penalty_history=np.asarray(extra_penalty_history, dtype=float),
        full_chi2_iterations=np.asarray(full_chi2_iterations, dtype=np.int32),
        full_chi2_history=np.asarray(full_chi2_history, dtype=float),
        time_window_size=None if not windowed else int(config.time_window_size),
        time_window_step=int(config.time_window_step),
        physics_forward_timesteps=int(physics_forward_timesteps),
        physics_vjp_timesteps=int(physics_vjp_timesteps),
    )


def fit_inr(
    forward: Any,
    network: MultiscaleINR,
    coordinates: np.ndarray,
    observed_rhoa: np.ndarray,
    data_std: float | np.ndarray,
    *,
    config: INRConfig | None = None,
) -> INRResult:
    """Invert one survey with a matrix-free coordinate network."""

    return _fit(
        forward=forward,
        network=network,
        coordinates=coordinates,
        observed_rhoa=observed_rhoa,
        data_std=data_std,
        config=INRConfig() if config is None else config,
        series=False,
    )


def fit_timelapse_inr(
    forward: Any,
    network: torch.nn.Module,
    coordinates: np.ndarray,
    observed_rhoa: np.ndarray,
    data_std: float | np.ndarray,
    *,
    config: INRConfig | None = None,
) -> INRResult:
    """Invert a complete time series, optionally using windowed physics loss.

    ``INRConfig.time_window_size`` changes only which timesteps enter each
    forward/VJP call. The network, its global time coordinates, and optimizer
    state persist for the complete run.
    """

    if network.input_dimensions != 3:
        raise ValueError("time-lapse INR requires a network with input_dimensions=3")
    return _fit(
        forward=forward,
        network=network,
        coordinates=coordinates,
        observed_rhoa=observed_rhoa,
        data_std=data_std,
        config=INRConfig() if config is None else config,
        series=True,
    )


__all__ = ["INRConfig", "INRResult", "fit_inr", "fit_timelapse_inr"]
