"""Training of coordinate networks against ERT data through the differentiable physics.

One loop serves single surveys and time series: a single survey is a series of length one.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from adtlert.inr.networks import MultiscaleINR
from adtlert.inr.physics import (
    forward_operator,
    matrix_free_log_rhoa_series,
    prepare_cuda_forward,
    select_cuda_device,
)
from adtlert.inversion.regularization import (
    first_order_constraint_matrix,
    regularization_mesh,
)

ProgressCallback = Callable[[dict[str, Any]], None]
PenaltyCallback = Callable[[torch.Tensor, int], torch.Tensor]

_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "radam": torch.optim.RAdam,
}
_SCHEDULERS = ("multistep", "plateau", "none")


def _check(*checks: tuple[bool, str]) -> None:
    for valid, message in checks:
        if not valid:
            raise ValueError(message)


def _optional_positive(value) -> bool:
    return value is None or value > 0


@dataclass(frozen=True)
class Optimization:
    """Optimizer, learning-rate schedule and gradient clipping."""

    optimizer: str = "adam"
    learning_rate: float = 3.0e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = 10.0
    scheduler: str = "multistep"
    learning_rate_milestones: tuple[int, ...] = (100, 200)
    plateau_patience: int = 20
    plateau_factor: float = 0.5
    minimum_learning_rate: float = 1.0e-5

    def validate(self) -> None:
        _check(
            (self.optimizer in _OPTIMIZERS, "optimizer must be 'adam', 'adamw', or 'radam'"),
            (self.scheduler in _SCHEDULERS, "scheduler must be 'multistep', 'plateau', or 'none'"),
            (self.learning_rate > 0.0, "learning_rate must be positive"),
            (self.weight_decay >= 0.0, "weight_decay must be non-negative"),
            (_optional_positive(self.gradient_clip_norm), "gradient_clip_norm must be positive when set"),
            (all(step >= 1 for step in self.learning_rate_milestones), "learning_rate_milestones must contain positive iterations"),
            (tuple(sorted(set(self.learning_rate_milestones))) == self.learning_rate_milestones, "learning_rate_milestones must be strictly increasing"),
            (self.plateau_patience >= 1, "plateau_patience must be >= 1"),
            (0.0 < self.plateau_factor < 1.0, "plateau_factor must be in (0, 1)"),
            (0.0 < self.minimum_learning_rate <= self.learning_rate, "minimum_learning_rate must be positive and <= learning_rate"),
        )  # fmt: skip


@dataclass(frozen=True)
class Progressive:
    """Coarse-to-fine unlocking of the Fourier levels over the first ``full_iteration`` steps."""

    enabled: bool = True
    full_iteration: int = 150
    spatial_start_levels: int = 2
    temporal_start_levels: int = 1

    def apply(self, network: torch.nn.Module, iteration: int) -> None:
        if self.enabled:
            network.set_progressive_levels(
                min(iteration / self.full_iteration, 1.0),
                spatial_start_levels=self.spatial_start_levels,
                temporal_start_levels=self.temporal_start_levels,
            )

    def validate(self) -> None:
        _check(
            (self.full_iteration >= 1, "progressive full_iteration must be >= 1"),
            (self.spatial_start_levels >= 0 and self.temporal_start_levels >= 0, "progressive start levels must be non-negative"),
        )  # fmt: skip


@dataclass(frozen=True)
class Regularization:
    """Huber smoothness penalties on the log-resistivity model."""

    spatial: float = 0.0
    temporal: float = 0.0
    huber_delta: float = 0.1

    def validate(self) -> None:
        _check(
            (self.spatial >= 0.0 and self.temporal >= 0.0, "regularization strengths must be non-negative"),
            (self.huber_delta > 0.0, "regularization huber_delta must be positive"),
        )  # fmt: skip


@dataclass(frozen=True)
class Windows:
    """Windowed physics loss for time series (``size=None`` uses every timestep per call)."""

    size: int | None = None
    step: int = 1
    full_evaluation_interval: int = 25
    alternate_direction: bool = True

    def validate(self) -> None:
        _check(
            (_optional_positive(self.size), "time window size must be positive when set"),
            (self.step >= 1, "time window step must be positive"),
            (self.full_evaluation_interval >= 1, "full_evaluation_interval must be positive"),
        )  # fmt: skip


@dataclass(frozen=True)
class INRConfig:
    """Controls for matrix-free INR inversion; the tunables live in the grouped sub-configs."""

    max_iterations: int = 300
    target_chi2: float | None = 1.0
    device: str = "cuda"
    log_every: int = 10
    prepare_solver: bool = True
    snapshot_interval: int | None = None
    optimization: Optimization = field(default_factory=Optimization)
    progressive: Progressive = field(default_factory=Progressive)
    regularization: Regularization = field(default_factory=Regularization)
    windows: Windows = field(default_factory=Windows)
    progress_callback: ProgressCallback | None = field(
        default=None, repr=False, compare=False
    )
    # Added to the objective; receives the network output (log resistivity) and the iteration.
    extra_penalty: PenaltyCallback | None = field(
        default=None, repr=False, compare=False
    )

    def validate(self) -> None:
        _check(
            (self.max_iterations >= 1, "max_iterations must be >= 1"),
            (_optional_positive(self.target_chi2), "target_chi2 must be positive"),
            (self.log_every >= 1, "log_every must be >= 1"),
            (_optional_positive(self.snapshot_interval), "snapshot_interval must be positive when set"),
        )  # fmt: skip
        for group in (
            self.optimization,
            self.progressive,
            self.regularization,
            self.windows,
        ):
            group.validate()


@dataclass(frozen=True)
class History:
    """Per-iteration series (``full_*`` hold the all-timestep evaluations of windowed runs)."""

    chi2: np.ndarray
    rms: np.ndarray
    objective: np.ndarray
    spatial_penalty: np.ndarray
    temporal_penalty: np.ndarray
    extra_penalty: np.ndarray
    full_iterations: np.ndarray
    full_chi2: np.ndarray


@dataclass(frozen=True)
class Timing:
    """Wall-clock seconds by phase and the number of timesteps the physics processed."""

    elapsed: float
    forward: float
    backward: float
    optimizer: float
    forward_timesteps: int
    vjp_timesteps: int


@dataclass(frozen=True)
class INRResult:
    """The best full-data model of one INR optimization, with its history and timing."""

    log_resistivity: np.ndarray
    resistivity: np.ndarray
    predicted_log_data: np.ndarray
    predicted_data: np.ndarray
    iterations: int
    best_iteration: int
    best_chi2: float
    stop_reason: str
    device: str
    gpu_report: dict[str, Any]
    history: History
    timing: Timing
    config: INRConfig


def _scheduler(config: INRConfig, optimizer: torch.optim.Optimizer):
    if config.optimization.scheduler == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(config.optimization.learning_rate_milestones),
            gamma=config.optimization.plateau_factor,
        )
    if config.optimization.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.optimization.plateau_factor,
            patience=config.optimization.plateau_patience,
            min_lr=config.optimization.minimum_learning_rate,
        )
    return None


def _log_data(
    observed_rhoa, data_std, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(log observed, std)`` as float32 device tensors of shape ``(T, D)`` (1D input -> ``T = 1``)."""

    observed = np.atleast_2d(np.asarray(observed_rhoa, dtype=np.float32))
    if (
        observed.ndim != 2
        or not np.all(np.isfinite(observed))
        or np.any(observed <= 0.0)
    ):
        raise ValueError("observed_rhoa must be a finite positive 1D or 2D array")
    try:
        std = np.broadcast_to(np.asarray(data_std, dtype=np.float32), observed.shape)
    except ValueError as exc:
        raise ValueError(
            f"data_std cannot broadcast to observed shape {observed.shape}"
        ) from exc
    if not np.all(np.isfinite(std)) or np.any(std <= 0.0):
        raise ValueError("data_std must contain positive finite values")
    return (
        torch.as_tensor(np.log(observed), device=device),
        torch.as_tensor(np.ascontiguousarray(std), device=device),
    )


def _spatial_operator(
    forward: Any, cell_count: int, device: torch.device
) -> torch.Tensor:
    matrix = first_order_constraint_matrix(regularization_mesh(forward)).tocoo()
    if matrix.shape[1] != cell_count:
        raise ValueError(
            "spatial regularization mesh does not match the INR parameter count"
        )
    indices = torch.as_tensor(
        np.vstack((matrix.row, matrix.col)), dtype=torch.int64, device=device
    )
    values = torch.as_tensor(matrix.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        indices, values, matrix.shape, device=device
    ).coalesce()


def _huber_mean(values: torch.Tensor, delta: float) -> torch.Tensor:
    if values.numel() == 0:
        return values.sum()
    magnitude = values.abs()
    return torch.where(
        magnitude <= delta, 0.5 * values.square() / delta, magnitude - 0.5 * delta
    ).mean()


def _misfit(prediction, observed, std, weights) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted ``(chi2, rms %)`` of log-data residuals; ``weights`` has one entry per timestep."""

    difference = prediction - observed
    norm = difference.shape[1] * weights.sum()
    chi2 = ((difference / std).square() * weights[:, None]).sum() / norm
    rms = 100.0 * torch.sqrt(
        (torch.expm1(difference).square() * weights[:, None]).sum() / norm
    )
    return chi2, rms


def _detach(encoded: Any) -> Any:
    if isinstance(encoded, torch.Tensor):
        return encoded.detach()
    if isinstance(encoded, tuple):
        return tuple(_detach(value) for value in encoded)
    raise TypeError("network.encode must return a Tensor or tuple of Tensors")


class _Schedule:
    """Which timesteps enter each physics call.

    Without windows every call uses all steps. With windows (``windows.size`` below the
    number of steps) calls cycle through overlapping windows, optionally alternating
    direction each cycle; each step's loss is weighted by ``1 / (number of windows covering it)``.
    """

    def __init__(self, n_times: int, config: INRConfig, device: torch.device):
        size = config.windows.size
        self.windowed = size is not None and size < n_times
        self.alternate = config.windows.alternate_direction
        self.device = device
        if self.windowed:
            size = min(int(size), n_times)
            starts = list(range(0, max(n_times - size + 1, 1), config.windows.step))
            if starts[-1] != n_times - size:
                starts.append(n_times - size)
            self.windows = [np.arange(start, start + size) for start in starts]
            coverage = np.zeros(n_times, dtype=np.int64)
            for window in self.windows:
                coverage[window] += 1
            if np.any(coverage == 0):
                raise ValueError(
                    "time windows do not cover every timestep; reduce windows.step"
                )
        else:
            self.windows, coverage = (
                [np.arange(n_times)],
                np.ones(n_times, dtype=np.int64),
            )
        self.weights = torch.as_tensor(
            1.0 / coverage, dtype=torch.float32, device=device
        )

    @property
    def batch_size(self) -> int:
        return max(window.size for window in self.windows)

    def window(self, iteration: int) -> np.ndarray:
        cycle, position = divmod(iteration, len(self.windows))
        return self.windows[
            len(self.windows) - 1 - position
            if self.alternate and cycle % 2
            else position
        ]

    def rows(self, window: np.ndarray) -> torch.Tensor | slice:
        return (
            torch.as_tensor(window, device=self.device)
            if self.windowed
            else slice(None)
        )


class _Recorder:
    """Per-iteration histories, phase timers, and the best full-data model."""

    def __init__(self, device: torch.device):
        self.device = device
        self.series: dict[str, list[float]] = {
            name: []
            for name in ("chi2", "rms", "objective", "spatial", "temporal", "extra")
        }
        self.full_iterations: list[int] = []
        self.full_chi2: list[float] = []
        self.best: dict[str, Any] = {"chi2": float("inf"), "iteration": 0}
        self.seconds = {"forward": 0.0, "backward": 0.0, "optimizer": 0.0}
        self.timesteps = {"forward": 0, "vjp": 0}

    @contextmanager
    def timed(self, phase: str):
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        yield
        torch.cuda.synchronize(self.device)
        self.seconds[phase] += time.perf_counter() - started

    def remember(
        self, iteration: int, chi2: float, model, prediction, network: torch.nn.Module
    ) -> None:
        self.full_iterations.append(iteration)
        self.full_chi2.append(chi2)
        if chi2 < self.best["chi2"]:
            self.best.update(
                chi2=chi2,
                iteration=iteration,
                model=model.detach().clone(),
                prediction=prediction.detach().clone(),
                state={
                    name: value.detach().clone()
                    for name, value in network.state_dict().items()
                },
            )


@dataclass
class _Problem:
    forward: Any
    network: torch.nn.Module
    encoded: Any
    observed: torch.Tensor
    std: torch.Tensor
    schedule: _Schedule
    series: bool
    spatial_operator: torch.Tensor | None

    @property
    def n_times(self) -> int:
        return self.observed.shape[0]


def _prepare(
    forward, network, coordinates, observed_rhoa, data_std, config, series: bool, device
) -> _Problem:
    """Validate the inputs and precompute everything that is fixed during training."""

    network = network.to(device=device, dtype=torch.float32)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    expected_ndim = 3 if series else 2
    if (
        coordinates.ndim != expected_ndim
        or coordinates.shape[-1] != network.input_dimensions
    ):
        raise ValueError(
            f"coordinates must have {expected_ndim} dimensions and end with {network.input_dimensions} features"
        )
    if not series and config.windows.size is not None:
        raise ValueError("windows.size is only valid for time-lapse inversion")
    if np.ndim(observed_rhoa) != (2 if series else 1):
        raise ValueError(
            f"observed_rhoa must be {2 if series else 1}D for this training mode"
        )

    with torch.no_grad():
        config.progressive.apply(network, 0)
        encoded = _detach(network.encode(torch.as_tensor(coordinates, device=device)))
        initial_model = network.forward_encoded(encoded)
    model_shape = tuple(coordinates.shape[:-1])
    if tuple(initial_model.shape) != model_shape:
        raise ValueError(
            f"network model shape {tuple(initial_model.shape)} does not match {model_shape}"
        )
    cell_count = getattr(forward, "cell_count", None)
    if cell_count is not None and model_shape[-1] != int(cell_count):
        raise ValueError(
            f"coordinates describe {model_shape[-1]} cells, but forward expects {int(cell_count)}"
        )

    observed, std = _log_data(observed_rhoa, data_std, device)
    n_times = model_shape[0] if series else 1
    if observed.shape[0] != n_times:
        raise ValueError(
            "observed_rhoa timestep count must match spatiotemporal coordinates"
        )
    measurements = getattr(getattr(forward, "survey", None), "measurement_count", None)
    if measurements is not None and observed.shape[1] != int(measurements):
        raise ValueError(
            f"observed_rhoa has {observed.shape[1]} measurements, but survey expects {int(measurements)}"
        )

    schedule = _Schedule(n_times, config, device)
    # The VJP reuses forward fields from the operator's LRU cache; one smaller than a
    # forward/VJP batch evicts them and silently re-solves every forward during backward.
    entries = getattr(forward_operator(forward), "normal_field_cache_max_entries", None)
    if entries is not None and int(entries) < schedule.batch_size:
        raise ValueError(
            f"normal_field_cache_max_entries={int(entries)} is smaller than the {schedule.batch_size} timesteps "
            "per forward/VJP batch; set it to at least that value when building the forward"
        )
    if config.prepare_solver:
        first = initial_model[0] if series else initial_model
        prepare_cuda_forward(
            forward, first.detach().double().cpu().numpy(), device=device
        )
    spatial = (
        _spatial_operator(forward, model_shape[-1], device)
        if config.regularization.spatial > 0.0
        else None
    )
    return _Problem(forward, network, encoded, observed, std, schedule, series, spatial)


def _penalties(
    problem: _Problem, config: INRConfig, log_model: torch.Tensor, iteration: int
) -> dict[str, torch.Tensor]:
    """Weighted regularization terms of the objective."""

    zero = torch.zeros((), device=log_model.device)
    model = log_model.reshape(problem.n_times, -1)
    terms = {"spatial": zero, "temporal": zero, "extra": zero}
    if problem.spatial_operator is not None:
        differences = torch.sparse.mm(problem.spatial_operator, model.T)
        terms["spatial"] = config.regularization.spatial * _huber_mean(
            differences, config.regularization.huber_delta
        )
    if problem.series and config.regularization.temporal > 0.0:
        terms["temporal"] = config.regularization.temporal * _huber_mean(
            model[1:] - model[:-1], config.regularization.huber_delta
        )
    if config.extra_penalty is not None:
        terms["extra"] = config.extra_penalty(log_model, iteration)
    return terms


def _fit(
    *,
    forward,
    network,
    coordinates,
    observed_rhoa,
    data_std,
    config: INRConfig,
    series: bool,
) -> INRResult:
    config.validate()
    device = torch.device(config.device)
    gpu_report = select_cuda_device(device)
    problem = _prepare(
        forward, network, coordinates, observed_rhoa, data_std, config, series, device
    )
    network, schedule = problem.network, problem.schedule
    tuning = config.optimization
    optimizer = _OPTIMIZERS[tuning.optimizer](
        network.parameters(), lr=tuning.learning_rate, weight_decay=tuning.weight_decay
    )
    scheduler = _scheduler(config, optimizer)
    record = _Recorder(device)
    emit = config.progress_callback or (lambda payload: None)
    stop_reason = "max_iterations"

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for iteration in range(config.max_iterations + 1):
        config.progressive.apply(network, iteration)
        with record.timed("forward"):
            log_model = network.forward_encoded(problem.encoded)
            model = log_model.reshape(problem.n_times, -1)
            snapshot_due = bool(
                config.snapshot_interval
                and iteration > 0
                and iteration % config.snapshot_interval == 0
            )
            full = (
                None  # (chi2, rms) over every timestep, when evaluated this iteration
            )
            if schedule.windowed and (
                iteration in (0, config.max_iterations)
                or iteration % config.windows.full_evaluation_interval == 0
                or snapshot_due
            ):
                with torch.no_grad():
                    full_prediction = matrix_free_log_rhoa_series(
                        model.detach(), forward
                    )
                    full_misfit = _misfit(
                        full_prediction,
                        problem.observed,
                        problem.std,
                        torch.ones_like(schedule.weights),
                    )
                full = tuple(torch.stack(full_misfit).tolist())
                record.timesteps["forward"] += problem.n_times
                record.remember(iteration, full[0], log_model, full_prediction, network)

            window = schedule.window(iteration)
            rows = schedule.rows(window)
            prediction = matrix_free_log_rhoa_series(model[rows], forward)
            chi2_t, rms_t = _misfit(
                prediction,
                problem.observed[rows],
                problem.std[rows],
                schedule.weights[rows],
            )
            record.timesteps["forward"] += window.size
            terms = _penalties(problem, config, log_model, iteration)
            objective = 0.5 * chi2_t + sum(terms.values())
            values = (
                torch.stack((chi2_t, rms_t, objective, *terms.values()))
                .detach()
                .tolist()
            )
        chi2, rms, objective_value, spatial, temporal, extra = values
        if not np.isfinite(chi2):
            raise FloatingPointError("INR inversion produced a non-finite chi-square")
        for name, value in zip(record.series, values, strict=True):
            record.series[name].append(value)
        if not schedule.windowed:
            full = (chi2, rms)
            record.remember(iteration, chi2, log_model, prediction, network)

        full_chi2, full_rms = full or (None, None)
        target_reached = (
            config.target_chi2 is not None
            and full_chi2 is not None
            and full_chi2 <= config.target_chi2
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        if (
            iteration % config.log_every == 0
            or target_reached
            or iteration == config.max_iterations
        ):
            emit(
                {
                    "event": "iteration",
                    "iteration": iteration,
                    "chi2": chi2,
                    "rms": rms,
                    "full_chi2": full_chi2,
                    "full_rms": full_rms,
                    "window_start": int(window[0]) if schedule.windowed else None,
                    "window_stop": int(window[-1] + 1) if schedule.windowed else None,
                    "learning_rate": learning_rate,
                    "objective": objective_value,
                    "spatial_penalty": spatial,
                    "temporal_penalty": temporal,
                    "extra_penalty": extra,
                }
            )
        if snapshot_due:
            emit(
                {
                    "event": "snapshot",
                    "iteration": iteration,
                    "chi2": full_chi2,
                    "rms": full_rms,
                    "learning_rate": learning_rate,
                    "log_resistivity": log_model.detach()
                    .to(device="cpu", dtype=torch.float64)
                    .numpy(),
                }
            )
        if target_reached:
            stop_reason = "target_chi2"
            break
        if iteration == config.max_iterations:
            break

        with record.timed("backward"):
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            record.timesteps["vjp"] += window.size
            if config.optimization.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(), config.optimization.gradient_clip_norm
                )
        with record.timed("optimizer"):
            optimizer.step()
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(chi2)
            elif scheduler is not None:
                scheduler.step()

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    best = record.best
    network.load_state_dict(best["state"])
    log_model = best["model"].to(device="cpu", dtype=torch.float64).numpy()
    predicted = best["prediction"].to(device="cpu", dtype=torch.float64).numpy()
    predicted = predicted if series else predicted[0]
    series = {
        name: np.asarray(values, dtype=float) for name, values in record.series.items()
    }
    history = History(
        chi2=series["chi2"],
        rms=series["rms"],
        objective=series["objective"],
        spatial_penalty=series["spatial"],
        temporal_penalty=series["temporal"],
        extra_penalty=series["extra"],
        full_iterations=np.asarray(record.full_iterations, dtype=np.int32),
        full_chi2=np.asarray(record.full_chi2, dtype=float),
    )
    timing = Timing(
        elapsed=float(elapsed),
        forward=record.seconds["forward"],
        backward=record.seconds["backward"],
        optimizer=record.seconds["optimizer"],
        forward_timesteps=record.timesteps["forward"],
        vjp_timesteps=record.timesteps["vjp"],
    )
    return INRResult(
        log_resistivity=log_model,
        resistivity=np.exp(log_model),
        predicted_log_data=predicted,
        predicted_data=np.exp(predicted),
        iterations=series["chi2"].size - 1,
        best_iteration=int(best["iteration"]),
        best_chi2=float(best["chi2"]),
        stop_reason=stop_reason,
        device=str(device),
        gpu_report=gpu_report,
        history=history,
        timing=timing,
        config=config,
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
        config=config or INRConfig(),
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
    """Invert a complete time series, optionally using a windowed physics loss.

    ``INRConfig.windows.size`` changes only which timesteps enter each forward/VJP call.
    The network, its global time coordinates, and the optimizer state persist for the run.
    """

    if network.input_dimensions != 3:
        raise ValueError("time-lapse INR requires a network with input_dimensions=3")
    return _fit(
        forward=forward,
        network=network,
        coordinates=coordinates,
        observed_rhoa=observed_rhoa,
        data_std=data_std,
        config=config or INRConfig(),
        series=True,
    )


__all__ = [
    "History",
    "INRConfig",
    "INRResult",
    "Optimization",
    "Progressive",
    "Regularization",
    "Timing",
    "Windows",
    "fit_inr",
    "fit_timelapse_inr",
]
