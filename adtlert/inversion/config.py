"""Configuration, results, progress events, and validation of an inversion.

Internal module: names with a leading underscore are shared inside the ``adtlert.inversion`` package.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from adtlert.inversion.misfit import build_data_misfit
from adtlert.inversion.optimizers import (
    build_linearized_optimizer,
    build_optimization_algorithm,
)
from adtlert.inversion.petrophysics import (
    available_petrophysical_transforms,
)
from adtlert.inversion.regularization import (
    build_spatial_regularization,
    build_temporal_regularization,
)

ArrayLike = Any


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class InversionConfig:
    """Controls for nonlinear log-resistivity inversion."""

    max_iterations: int = 8
    data_std: float | ArrayLike = 0.05
    data_misfit: str = "weighted_log_l2"
    regularization: float = 1.0e-2
    regularization_mode: str = "model"
    temporal_regularization: float = 0.0
    temporal_regularization_mode: str = "separate"
    temporal_regularization_type: str = "temporal_smoothness"
    spatial_regularization: str = "damping"
    regularization_domain: str = "state"
    physical_regularization_quantity: str = "parameter"
    z_weight: float = 1.0
    model_transform: str = "log"
    model_bounds: tuple[float, float] | None = None
    petrophysical_transform: str = "log_resistivity"
    petrophysical_parameters: dict[str, ArrayLike] | None = field(
        default=None, repr=False, compare=False
    )
    saturation_floor: float = 1.0e-4
    step_length: float = 1.0
    max_log_step: float | None = 1.0
    line_search: bool = False
    target_chi2: float | None = None
    step_tolerance: float = 1.0e-4
    active_time_threshold: float = 0.05
    active_time_minimum_weight: float = 0.05
    optimization_algorithm: str = "gauss_newton_cgls"
    linearized_solver: str = "lsqr"
    lm_damping: float = 1.0e-2
    optimizer_max_step: float = 1.0
    lbfgs_history: int = 10
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8
    lsqr_atol: float = 1.0e-6
    lsqr_btol: float = 1.0e-6
    lsqr_iter_limit: int | None = None
    cgls_max_iterations: int = 2000
    cgls_tolerance: float = 1.0e-8
    include_robin_boundary_derivative: bool = False
    normal_sensitivity: bool = True
    # Keep the first time-lapse state fixed at its initial value (baseline-anchored inversion).
    freeze_first_timestep: bool = False
    # Soft constraint from external observations: lambda_s * ||H parameter[:, t] - targets[:, t]||^2.
    sensor_constraint: float = 0.0
    sensor_constraint_operator: ArrayLike | None = field(
        default=None, repr=False, compare=False
    )
    sensor_constraint_targets: ArrayLike | None = field(
        default=None, repr=False, compare=False
    )
    sensor_constraint_weights: ArrayLike | None = field(
        default=None, repr=False, compare=False
    )
    progress_callback: ProgressCallback | None = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True)
class ERTInversionResult:
    """Single-time inversion result."""

    final_model: np.ndarray
    final_log_model: np.ndarray
    predicted_data: np.ndarray
    predicted_log_data: np.ndarray
    coverage: np.ndarray
    iteration_chi2: list[float]
    final_parameter_model: np.ndarray | None = None
    final_parameter_name: str = "resistivity"


@dataclass(frozen=True)
class TimeLapseERTInversionResult:
    """Time-lapse inversion result with models stored as ``(n_cells, n_times)``."""

    final_models: np.ndarray
    final_log_models: np.ndarray
    predicted_data: np.ndarray
    predicted_log_data: np.ndarray
    coverage: np.ndarray
    all_coverage: list[np.ndarray]
    all_chi2: np.ndarray
    iteration_chi2: list[float]
    window_reports: list[dict[str, float | int | None]] = field(default_factory=list)
    final_parameter_models: np.ndarray | None = None
    final_parameter_name: str = "resistivity"


def _emit(config: InversionConfig, event: str, **payload: Any) -> None:
    if config.progress_callback is not None:
        config.progress_callback({"event": event, **payload})


_PETROPHYSICAL_ALIASES = {
    "resistivity",
    "rho",
    "conductivity",
    "sigma",
    "water_saturation",
    "relative_archie",
    "water_content",
    "theta",
}


_WATER_CONTENT_TRANSFORMS = {
    "saturation",
    "water_saturation",
    "relative_archie_water_content",
    "relative_archie",
    "water_content",
    "theta",
}


def _key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def _regularization_domain(config: InversionConfig) -> str:
    return _key(config.regularization_domain)


def _physical_quantity(config: InversionConfig) -> str:
    key = _key(config.physical_regularization_quantity)
    if key in ("theta", "water_content", "moisture_content"):
        return "water_content"
    return "parameter" if key in ("parameter", "native", "physical_parameter") else key


def _check_config(config: InversionConfig) -> None:
    domain, quantity = _regularization_domain(config), _physical_quantity(config)
    petrophysical = _key(config.petrophysical_transform)
    build_data_misfit(config.data_misfit)
    build_temporal_regularization(config.temporal_regularization_type)
    build_spatial_regularization(config.spatial_regularization)
    build_optimization_algorithm(config.optimization_algorithm)
    build_linearized_optimizer(config.linearized_solver)
    if (
        petrophysical
        not in set(available_petrophysical_transforms()) | _PETROPHYSICAL_ALIASES
    ):
        raise ValueError(
            f"unknown petrophysical_transform={config.petrophysical_transform!r}; "
            f"available choices: {', '.join(available_petrophysical_transforms())}"
        )
    checks = (
        (config.max_iterations >= 1, "max_iterations must be >= 1"),
        (config.regularization >= 0.0, "regularization must be non-negative"),
        (config.regularization_mode in ("model", "update"), "regularization_mode must be 'model' or 'update'"),
        (domain in ("state", "physical"), "regularization_domain must be 'state' or 'physical'"),
        (quantity in ("parameter", "water_content"), "physical_regularization_quantity must be 'parameter' or 'theta'/'water_content'"),
        (config.temporal_regularization >= 0.0, "temporal_regularization must be non-negative"),
        (config.temporal_regularization_mode in ("separate", "joint_frame"), "temporal_regularization_mode must be 'separate' or 'joint_frame'"),
        (isinstance(config.freeze_first_timestep, (bool, np.bool_)), "freeze_first_timestep must be a boolean flag"),
        (config.sensor_constraint >= 0.0, "sensor_constraint must be non-negative"),
        (config.sensor_constraint <= 0.0 or config.sensor_constraint_operator is not None, "sensor_constraint_operator is required when sensor_constraint > 0"),
        (config.sensor_constraint <= 0.0 or config.sensor_constraint_targets is not None, "sensor_constraint_targets is required when sensor_constraint > 0"),
        (config.sensor_constraint_weights is None or config.sensor_constraint_targets is not None, "sensor_constraint_weights requires sensor_constraint_targets"),
        (config.sensor_constraint <= 0.0 or domain == "physical", "sensor_constraint currently requires regularization_domain='physical'"),
        (config.z_weight > 0.0, "z_weight must be positive"),
        (config.model_transform in ("log", "log_lu"), "model_transform must be 'log' or 'log_lu'"),
        (config.model_transform != "log_lu" or config.model_bounds is not None, "model_bounds are required for model_transform='log_lu'"),
        (
            domain != "physical" or quantity != "water_content" or petrophysical in _WATER_CONTENT_TRANSFORMS,
            "physical_regularization_quantity='theta'/'water_content' requires "
            "petrophysical_transform='saturation' or 'relative_archie_water_content'",
        ),
        (0.0 < config.saturation_floor < 1.0, "saturation_floor must be in (0, 1)"),
        (config.step_length > 0.0, "step_length must be positive"),
        (config.max_log_step is None or config.max_log_step > 0.0, "max_log_step must be positive when set"),
        (config.target_chi2 is None or config.target_chi2 > 0.0, "target_chi2 must be positive when set"),
        (config.active_time_threshold > 0.0, "active_time_threshold must be positive"),
        (0.0 <= config.active_time_minimum_weight <= 1.0, "active_time_minimum_weight must be in [0, 1]"),
        (config.lm_damping >= 0.0, "lm_damping must be non-negative"),
        (config.optimizer_max_step > 0.0, "optimizer_max_step must be positive"),
        (config.lbfgs_history >= 1, "lbfgs_history must be >= 1"),
        (0.0 <= config.adam_beta1 < 1.0, "adam_beta1 must be in [0, 1)"),
        (0.0 <= config.adam_beta2 < 1.0, "adam_beta2 must be in [0, 1)"),
        (config.adam_epsilon > 0.0, "adam_epsilon must be positive"),
        (config.cgls_max_iterations >= 1, "cgls_max_iterations must be >= 1"),
        (config.cgls_tolerance > 0.0, "cgls_tolerance must be positive"),
        (config.progress_callback is None or callable(config.progress_callback), "progress_callback must be callable when set"),
        (config.model_bounds is None or 0.0 < config.model_bounds[0] < config.model_bounds[1], "model_bounds must be positive and ordered as (min, max)"),
    )  # fmt: skip
    for valid, message in checks:
        if not valid:
            raise ValueError(message)
