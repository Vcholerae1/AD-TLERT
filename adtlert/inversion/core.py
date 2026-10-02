"""Log-space ERT inversion built on the differentiable forward operators.

Single-time and time-lapse inversions share one engine: models are stored as
``(n_cells, n_times)`` optimizer states, data as ``(n_times, n_measurements)``. Each
iteration stacks a linearized data term with spatial, temporal, and sensor terms and takes
either a Gauss-Newton/LM step or a matrix-free first-order step.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree

from adtlert.forward import ERTForward2p5D, ERTForwardModeling
from adtlert.forward.modeling import log_response_and_jacobian
from adtlert.inversion.misfit import DataMisfit, build_data_misfit
from adtlert.inversion.optimizers import (
    build_linearized_optimizer,
    build_optimization_algorithm,
    first_order_step,
    linearized_gradient,
    linearized_step,
)
from adtlert.inversion.petrophysics import (
    available_petrophysical_transforms,
    build_petrophysical_transform,
)
from adtlert.inversion.regularization import (
    build_spatial_regularization,
    build_temporal_regularization,
    cell_edges,
    regularization_mesh,
)
from adtlert.mesh import Mesh, Mesh3D
from adtlert.utils.dtypes import FLOAT_DTYPE

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


# ---------------------------------------------------------------------------
# Parameterized forward (inversion parameters on a coarser mesh than the forward solve)
# ---------------------------------------------------------------------------


class ParameterizedERTForward2p5D:
    """ERT forward wrapper with a full solve mesh and a smaller parameter mesh.

    ``background_mode`` sets forward cells outside the parameter domain from:
    ``"pygimli_prolongation"`` (pyGIMLi's neighbor-weighted resistivity prolongation),
    ``"nearest"`` (nearest parameter center), or ``"fixed_mean"`` (mean log-resistivity).
    """

    def __init__(
        self,
        forward: ERTForward2p5D,
        parameter_cell_ids: ArrayLike,
        *,
        regularization_mesh: Mesh | None = None,
        forward_cell_parameter_ids: ArrayLike | None = None,
        background_mode: str = "pygimli_prolongation",
    ) -> None:
        self.forward_operator = forward
        self.survey = forward.survey
        self.mesh = forward.mesh
        self.regularization_mesh = regularization_mesh
        cell_count = int(forward.mesh.cell_count)
        self.parameter_cell_ids = np.asarray(parameter_cell_ids, dtype=np.int32).ravel()
        if self.parameter_cell_ids.size == 0:
            raise ValueError("parameter_cell_ids must not be empty")
        if np.any(self.parameter_cell_ids < 0) or np.any(
            self.parameter_cell_ids >= cell_count
        ):
            raise ValueError(
                "parameter_cell_ids reference cells outside the forward mesh"
            )
        if np.unique(self.parameter_cell_ids).size != self.parameter_cell_ids.size:
            raise ValueError("parameter_cell_ids must be unique")
        if background_mode not in ("pygimli_prolongation", "nearest", "fixed_mean"):
            raise ValueError(
                "background_mode must be 'pygimli_prolongation', 'nearest', or 'fixed_mean'"
            )
        self.background_mode = background_mode

        if forward_cell_parameter_ids is None:
            ids = np.full(cell_count, -1, dtype=np.int32)
            ids[self.parameter_cell_ids] = np.arange(
                self.parameter_cell_ids.size, dtype=np.int32
            )
        else:
            ids = np.asarray(forward_cell_parameter_ids, dtype=np.int32).ravel()
            if ids.shape != (cell_count,):
                raise ValueError(
                    f"forward_cell_parameter_ids must have one entry per forward mesh cell ({ids.shape} != ({cell_count},))"
                )
            if np.any(ids < -1):
                raise ValueError(
                    "forward_cell_parameter_ids may only contain -1 or non-negative parameter ids"
                )
            active_ids = np.unique(ids[ids >= 0])
            if active_ids.size == 0:
                raise ValueError(
                    "forward_cell_parameter_ids must contain at least one active parameter cell"
                )
            if not np.array_equal(active_ids, np.arange(active_ids[-1] + 1)):
                raise ValueError(
                    "forward_cell_parameter_ids must use contiguous ids starting at 0"
                )
        self.forward_cell_parameter_ids = ids
        self._n_parameters = int(ids.max()) + 1
        if self.parameter_cell_ids.size != self._n_parameters:
            raise ValueError(
                "parameter_cell_ids must contain one representative forward cell per inversion parameter "
                f"({self.parameter_cell_ids.size} != {self._n_parameters})"
            )
        if not np.array_equal(
            ids[self.parameter_cell_ids], np.arange(self._n_parameters)
        ):
            raise ValueError(
                "parameter_cell_ids must be ordered representatives of forward_cell_parameter_ids"
            )
        if (
            regularization_mesh is not None
            and int(regularization_mesh.cell_count) != self._n_parameters
        ):
            raise ValueError(
                "regularization mesh cell count must match inversion parameter count "
                f"({regularization_mesh.cell_count} != {self._n_parameters})"
            )

        cells = np.arange(cell_count, dtype=np.int32)
        self._active_forward_mask = ids >= 0
        self._active_forward_cell_ids = cells[self._active_forward_mask]
        self._active_parameter_ids = ids[self._active_forward_cell_ids]
        self.background_cell_ids = cells[~self._active_forward_mask]
        self._background_parameter_ids = (
            self._nearest_parameters()
            if background_mode == "nearest"
            else np.empty(0, dtype=np.int32)
        )
        self._resistivity_prolongation_matrix = (
            self._prolongation() if background_mode == "pygimli_prolongation" else None
        )
        self._jacobian_projection = self._projection()
        self._torch_maps: dict[str, torch.Tensor] = {}

    @classmethod
    def from_mesh_survey(
        cls,
        mesh: Mesh,
        survey,
        parameter_cell_ids: ArrayLike,
        *,
        regularization_mesh: Mesh | None = None,
        background_mode: str = "pygimli_prolongation",
        **forward_kwargs: Any,
    ) -> ParameterizedERTForward2p5D:
        forward_cell_parameter_ids = forward_kwargs.pop(
            "forward_cell_parameter_ids", None
        )
        return cls(
            ERTForward2p5D.from_mesh_survey(mesh, survey, **forward_kwargs),
            parameter_cell_ids,
            regularization_mesh=regularization_mesh,
            forward_cell_parameter_ids=forward_cell_parameter_ids,
            background_mode=background_mode,
        )

    @property
    def cell_count(self) -> int:
        return self._n_parameters

    def close(self) -> None:
        self.forward_operator.close()

    def _nearest_parameters(self) -> np.ndarray:
        if self.background_cell_ids.size == 0:
            return np.empty(0, dtype=np.int32)
        centers = np.mean(
            np.asarray(self.mesh.nodes, dtype=float)[np.asarray(self.mesh.cells)],
            axis=1,
        )
        sums = np.zeros((self.cell_count, centers.shape[1]))
        counts = np.zeros(self.cell_count)
        np.add.at(
            sums, self._active_parameter_ids, centers[self._active_forward_cell_ids]
        )
        np.add.at(counts, self._active_parameter_ids, 1.0)
        if np.any(counts <= 0.0):
            raise ValueError(
                "each inversion parameter must own at least one forward mesh cell"
            )
        _, nearest = cKDTree(sums / counts[:, None]).query(
            centers[self.background_cell_ids]
        )
        return np.asarray(nearest, dtype=np.int32)

    def _projection(self) -> sp.csr_matrix:
        """``d log(rho_forward) / d log(rho_parameter)`` used to project Jacobian columns."""

        rows, cols = [self._active_forward_cell_ids], [self._active_parameter_ids]
        data = [np.ones(self._active_forward_cell_ids.size)]
        background = self.background_cell_ids
        if background.size and self.background_mode == "nearest":
            rows.append(background)
            cols.append(self._background_parameter_ids)
            data.append(np.ones(background.size))
        elif background.size and self.background_mode == "fixed_mean":
            rows.append(np.repeat(background, self.cell_count))
            cols.append(
                np.tile(np.arange(self.cell_count, dtype=np.int32), background.size)
            )
            data.append(
                np.full(background.size * self.cell_count, 1.0 / self.cell_count)
            )
        shape = (int(self.mesh.cell_count), self.cell_count)
        return sp.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=shape,
        ).tocsr()

    def _prolongation(self) -> sp.csr_matrix:
        """Replicate pyGIMLi's marker prolongation: background rows are neighbor-weighted averages."""

        cell_count = int(self.mesh.cell_count)
        matrix = np.zeros((cell_count, self.cell_count))
        matrix[self._active_forward_cell_ids, self._active_parameter_ids] = 1.0
        if self.background_cell_ids.size == 0:
            return sp.csr_matrix(matrix)

        nodes = np.asarray(self.mesh.nodes, dtype=float)
        edge_cells: dict[tuple[int, ...], list[int]] = {}
        for cell_id, cell in enumerate(np.asarray(self.mesh.cells, dtype=np.int32)):
            for edge in cell_edges(cell):
                edge_cells.setdefault(tuple(sorted(edge)), []).append(cell_id)
        neighbors: list[list[tuple[int, float]]] = [[] for _ in range(cell_count)]
        for edge, owners in edge_cells.items():
            tangent = nodes[edge[1]] - nodes[edge[0]]
            length = float(np.linalg.norm(tangent))
            if len(owners) != 2 or length <= 0.0:
                continue
            weight = abs(float(tangent[1])) / length + 1.0e-6
            neighbors[owners[0]].append((owners[1], weight))
            neighbors[owners[1]].append((owners[0], weight))

        known = self._active_forward_mask.copy()
        unknown = set(self.background_cell_ids.tolist())
        while unknown:
            assignments = []
            for cell_id in sorted(unknown):
                row, total = np.zeros(self.cell_count), 0.0
                for neighbor, weight in neighbors[cell_id]:
                    if known[neighbor]:
                        row += matrix[neighbor] * weight
                        total += weight
                if total > 1.0e-8:
                    assignments.append((cell_id, row / total))
            if not assignments:
                raise ValueError(
                    "could not prolongate background forward cells from active parameter cells"
                )
            for cell_id, row in assignments:
                matrix[cell_id], known[cell_id] = row, True
                unknown.remove(cell_id)
        return sp.csr_matrix(matrix)

    def _full_log_model(self, log_resistivity: ArrayLike) -> np.ndarray:
        return self._full_log_model_and_projection(log_resistivity)[0]

    def _full_log_model_and_projection(
        self, log_resistivity: ArrayLike
    ) -> tuple[np.ndarray, sp.csr_matrix]:
        parameter_log = np.asarray(log_resistivity, dtype=float).ravel()
        if parameter_log.shape != (self.cell_count,):
            raise ValueError(f"log_resistivity must have shape ({self.cell_count},)")
        if self.background_mode == "pygimli_prolongation":
            full = np.asarray(
                self._resistivity_prolongation_matrix @ np.exp(parameter_log),
                dtype=float,
            ).ravel()
            if np.any(full <= 0.0) or not np.all(np.isfinite(full)):
                raise ValueError(
                    "prolongated forward resistivity contains non-positive or non-finite values"
                )
            return np.log(full), self._jacobian_projection
        full_log = np.empty(int(self.mesh.cell_count))
        full_log[self._active_forward_cell_ids] = parameter_log[
            self._active_parameter_ids
        ]
        if self.background_cell_ids.size:
            full_log[self.background_cell_ids] = (
                parameter_log[self._background_parameter_ids]
                if self.background_mode == "nearest"
                else float(np.mean(parameter_log))
            )
        return full_log, self._jacobian_projection

    def log_model_to_full(self, log_parameters: torch.Tensor) -> torch.Tensor:
        """Differentiable ``(..., n_parameters) -> (..., n_forward_cells)`` log-resistivity map.

        The torch counterpart of :meth:`_full_log_model` (float64, on the input's device),
        used to backpropagate through the background extension.
        """

        exponential = self.background_mode == "pygimli_prolongation"
        device = log_parameters.device
        if str(device) not in self._torch_maps:
            matrix = (
                self._resistivity_prolongation_matrix
                if exponential
                else self._jacobian_projection
            ).tocoo()
            self._torch_maps[str(device)] = torch.sparse_coo_tensor(
                np.vstack((matrix.row, matrix.col)),
                matrix.data,
                matrix.shape,
                dtype=torch.float64,
                device=device,
            ).coalesce()
        flat = log_parameters.to(torch.float64).reshape(-1, log_parameters.shape[-1]).T
        full = torch.sparse.mm(
            self._torch_maps[str(device)], torch.exp(flat) if exponential else flat
        ).T
        if exponential:
            full = torch.log(full)
        return full.reshape(*log_parameters.shape[:-1], full.shape[-1])

    def forward_and_jacobian(
        self,
        resistivity_model: ArrayLike,
        log_transform: bool = True,
        *,
        include_robin_boundary_derivative: bool | None = None,
        normal_sensitivity: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        log_model = _as_log_model(
            resistivity_model, self.cell_count, log_transform, "resistivity_model"
        )
        full_log, projection = self._full_log_model_and_projection(log_model)
        robin = bool(include_robin_boundary_derivative)
        normal = True if normal_sensitivity is None else bool(normal_sensitivity)
        if self.background_mode == "pygimli_prolongation" and normal and not robin:
            # Fused aggregation of forward-cell sensitivities into parameter columns.
            return log_response_and_jacobian(
                self.forward_operator,
                torch.as_tensor(np.exp(-full_log), dtype=FLOAT_DTYPE),
                chain=np.exp(-log_model),
                jacobian_cell_parameter_ids=self.forward_cell_parameter_ids,
                jacobian_parameter_count=self.cell_count,
            )
        predicted, full_jacobian = _forward_and_jacobian_log(
            self.forward_operator, full_log, robin=robin, normal=normal
        )
        return predicted, np.asarray((projection.T @ full_jacobian.T).T, dtype=float)

    def response(self, resistivity_model: ArrayLike) -> np.ndarray:
        return self.forward(resistivity_model, log_transform=False)

    def forward(
        self, resistivity_model: ArrayLike, log_transform: bool = True
    ) -> np.ndarray:
        log_model = _as_log_model(
            resistivity_model, self.cell_count, log_transform, "resistivity_model"
        )
        response = _forward_log_response(
            self.forward_operator, self._full_log_model(log_model)
        )
        return response if log_transform else np.exp(response)


# ---------------------------------------------------------------------------
# Configuration and input validation
# ---------------------------------------------------------------------------

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


def _model_size(forward) -> int:
    count = getattr(forward, "cell_count", None)
    if count is None:
        count = getattr(getattr(forward, "mesh", None), "cell_count", None)
    if count is None:
        raise TypeError("forward must expose a adtlert-compatible cell count")
    return int(count)


def _measurement_count(forward) -> int:
    survey = getattr(forward, "survey", None) or getattr(
        getattr(forward, "forward_operator", None), "survey", None
    )
    if survey is None:
        raise TypeError("forward must expose a adtlert-compatible measurement count")
    return int(survey.measurement_count)


def _logged(
    values: np.ndarray, *, already_log: bool, name: str, quantity: str
) -> np.ndarray:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    if already_log:
        return values.copy()
    if np.any(values <= 0.0):
        raise ValueError(f"{name} must contain positive {quantity} values")
    return np.log(values)


def _as_log_model(
    model: ArrayLike, expected_size: int, log_model: bool, name: str
) -> np.ndarray:
    values = np.asarray(model, dtype=float).ravel()
    if values.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},)")
    return _logged(values, already_log=log_model, name=name, quantity="resistivity")


def _as_log_models(
    model: ArrayLike, expected_size: int, n_times: int, log_model: bool, name: str
) -> np.ndarray:
    """Return ``(n_cells, n_times)`` log models from a shared or per-time model array."""

    values = np.asarray(model, dtype=float)
    if values.ndim == 1:
        return np.column_stack(
            [_as_log_model(values, expected_size, log_model, name)] * n_times
        )
    if values.shape == (expected_size, n_times):
        matrix = values
    elif values.shape == (n_times, expected_size):
        matrix = values.T
    else:
        raise ValueError(
            f"{name} must have shape ({expected_size},), ({expected_size}, {n_times}), or ({n_times}, {expected_size})"
        )
    return _logged(matrix, already_log=log_model, name=name, quantity="resistivity")


def _as_observed_log(
    data: ArrayLike, log_data: bool, measurement_count: int, *, timelapse: bool
) -> np.ndarray:
    """Return ``(n_times, n_measurements)`` log apparent resistivities (``(n,)`` for single-time)."""

    values = np.asarray(data, dtype=float)
    if not timelapse:
        values = values.ravel()
        if values.shape != (measurement_count,):
            raise ValueError(f"observed_data must have shape ({measurement_count},)")
    elif values.ndim != 2:
        raise ValueError("observed_data must be a 2D array for time-lapse inversion")
    elif values.shape[1] != measurement_count:
        if values.shape[0] != measurement_count:
            raise ValueError(
                "observed_data must have shape (n_times, n_measurements) or (n_measurements, n_times)"
            )
        values = values.T
    return _logged(
        values,
        already_log=log_data,
        name="observed_data",
        quantity="apparent resistivity",
    )


def _weights(data_std: float | ArrayLike, shape: tuple[int, ...]) -> np.ndarray:
    std = np.broadcast_to(np.asarray(data_std, dtype=float), shape)
    if not np.all(np.isfinite(std)):
        raise ValueError("data_std contains non-finite values")
    if np.any(std <= 0.0):
        raise ValueError("data_std must be positive")
    return 1.0 / std


# ---------------------------------------------------------------------------
# Petrophysical state <-> model maps
# ---------------------------------------------------------------------------


def _petrophysics(config: InversionConfig, values: np.ndarray):
    """Petrophysical transform sized for a ``(n_cells,)`` or ``(n_cells, n_times)`` array."""

    values = np.asarray(values)
    return build_petrophysical_transform(
        config.petrophysical_transform,
        n_cells=int(values.shape[0] if values.ndim == 2 else values.size),
        model_transform=config.model_transform,
        model_bounds=config.model_bounds,
        saturation_floor=float(config.saturation_floor),
        parameters=config.petrophysical_parameters,
    )


def _log_model_to_state(log_model: np.ndarray, config: InversionConfig) -> np.ndarray:
    return _petrophysics(config, log_model).state_from_log_resistivity(
        np.asarray(log_model, dtype=float)
    )


def _state_to_log_model(state: np.ndarray, config: InversionConfig) -> np.ndarray:
    return _petrophysics(config, state).log_resistivity_from_state(
        np.asarray(state, dtype=float)
    )


def _cell_parameter_array(values: ArrayLike, *, n_cells: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    result = np.full(n_cells, float(array)) if array.ndim == 0 else array.reshape(-1)
    if result.shape != (n_cells,):
        raise ValueError(f"{name} must be scalar or have shape ({n_cells},)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _regularization_value_and_derivative(
    state: np.ndarray, config: InversionConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Quantity regularized in the physical domain (parameter or water content) and its state derivative."""

    transform = _petrophysics(config, state)
    parameter, derivative = (
        transform.parameter_from_state(state),
        transform.d_parameter_d_state(state),
    )
    if (
        _physical_quantity(config) == "parameter"
        or transform.parameter_name == "water_content"
    ):
        return np.asarray(parameter, dtype=float), np.asarray(derivative, dtype=float)
    if transform.parameter_name != "saturation":
        raise ValueError(
            "physical_regularization_quantity='theta'/'water_content' requires "
            "petrophysical_transform='saturation' or a transform whose parameter is water_content"
        )
    phi = (config.petrophysical_parameters or {}).get("phi")
    if phi is None:
        raise ValueError(
            "physical_regularization_quantity='theta'/'water_content' requires petrophysical_parameters['phi']"
        )
    phi = _cell_parameter_array(phi, n_cells=int(state.shape[0]), name="phi")
    phi = phi[:, None] if state.ndim == 2 else phi
    return np.asarray(phi * parameter, dtype=float), np.asarray(
        phi * derivative, dtype=float
    )


def _diagonal(derivative: np.ndarray) -> sp.csr_matrix:
    diagonal = np.asarray(derivative, dtype=float).reshape(-1, order="F")
    if not np.all(np.isfinite(diagonal)):
        raise ValueError(
            "regularization projection derivative contains non-finite values"
        )
    return sp.diags(diagonal, format="csr")


def _config_for_time_index(config: InversionConfig, time_index: int) -> InversionConfig:
    """Slice time-varying ``(n_cells, n_times)`` petrophysical parameters to one timestep."""

    if not config.petrophysical_parameters:
        return config
    sliced, changed = {}, False
    for key, value in config.petrophysical_parameters.items():
        array = None if value is None else np.asarray(value)
        if array is not None and array.ndim >= 2:
            if not 0 <= time_index < array.shape[1]:
                raise IndexError(
                    f"time_index={time_index} outside petrophysical parameter {key!r} shape {array.shape}"
                )
            value, changed = array[:, time_index].copy(), True
        sliced[key] = value
    return replace(config, petrophysical_parameters=sliced) if changed else config


def _config_for_time_window(
    config: InversionConfig, *, observed_shape: tuple[int, int], start: int, end: int
) -> InversionConfig:
    """Slice every time-indexed config array to the window ``[start, end)``."""

    n_times = int(observed_shape[0])
    updates: dict[str, Any] = {}
    if np.ndim(config.data_std):
        updates["data_std"] = np.broadcast_to(
            np.asarray(config.data_std, dtype=float), observed_shape
        )[start:end].copy()
    if config.petrophysical_parameters:
        sliced = {
            key: np.asarray(value)[:, start:end].copy()
            if value is not None
            and np.ndim(value) >= 2
            and np.shape(value)[1] == n_times
            else value
            for key, value in config.petrophysical_parameters.items()
        }
        if any(
            sliced[key] is not value
            for key, value in config.petrophysical_parameters.items()
        ):
            updates["petrophysical_parameters"] = sliced
    for name in ("sensor_constraint_targets", "sensor_constraint_weights"):
        value = getattr(config, name)
        if value is None:
            continue
        array = np.asarray(value, dtype=float)
        if array.ndim >= 2 and array.shape[1] == n_times:
            updates[name] = array[:, start:end].copy()
        elif array.ndim == 1 and array.shape[0] == n_times:
            updates[name] = array[start:end].copy()
    if (
        config.freeze_first_timestep
    ):  # only the window containing the global first timestep is anchored
        updates["freeze_first_timestep"] = start == 0
    return replace(config, **updates) if updates else config


def _sensor_system(
    config: InversionConfig, n_cells: int, n_times: int
) -> tuple[sp.csr_matrix, np.ndarray] | None:
    """Row-scaled ``(kron(I_T, H), targets)`` restricted to valid observations."""

    if config.sensor_constraint <= 0.0:
        return None
    operator = config.sensor_constraint_operator
    if sp.issparse(operator):
        operator = operator.tocsr()
    else:
        operator = np.asarray(operator, dtype=float)
        if operator.ndim != 2:
            raise ValueError("sensor_constraint_operator must be a 2D matrix")
        operator = sp.csr_matrix(operator)
    if operator.nnz and not np.all(np.isfinite(operator.data)):
        raise ValueError("sensor_constraint_operator contains non-finite entries")
    if operator.shape[1] != n_cells:
        raise ValueError(
            f"sensor_constraint_operator second dimension must match n_cells ({operator.shape[1]} != {n_cells})"
        )
    targets = np.asarray(config.sensor_constraint_targets, dtype=float)
    targets = targets.reshape(-1, 1) if targets.ndim == 1 else targets
    if targets.ndim != 2:
        raise ValueError(
            "sensor_constraint_targets must be a 2D matrix [n_constraints, n_times]"
        )
    if targets.shape[0] != operator.shape[0]:
        raise ValueError(
            f"sensor_constraint_targets first dimension must match operator rows ({targets.shape[0]} != {operator.shape[0]})"
        )
    if targets.shape[1] == 1 and n_times > 1:
        targets = np.repeat(targets, n_times, axis=1)
    if targets.shape[1] != n_times:
        raise ValueError(
            f"sensor_constraint_targets second dimension must match n_times ({targets.shape[1]} != {n_times})"
        )

    shape = (operator.shape[0], n_times)
    weights = (
        np.ones(shape)
        if config.sensor_constraint_weights is None
        else np.asarray(config.sensor_constraint_weights, dtype=float)
    )
    if weights.ndim == 1:
        if weights.shape[0] == shape[0]:
            weights = weights[:, None]
        elif weights.shape[0] == shape[1]:
            weights = weights[None, :]
        elif weights.shape[0] == shape[0] * shape[1]:
            weights = weights.reshape(shape, order="F")
        else:
            raise ValueError(
                "sensor_constraint_weights 1D shape must match n_constraints, n_times, or n_constraints*n_times"
            )
    elif weights.ndim > 2:
        raise ValueError("sensor_constraint_weights must be scalar, 1D, or 2D")
    weights = np.broadcast_to(weights, shape)
    if not np.all(np.isfinite(weights)):
        raise ValueError("sensor_constraint_weights contains non-finite values")
    if np.any(weights < 0.0):
        raise ValueError("sensor_constraint_weights must be non-negative")
    valid = (np.isfinite(targets) & (weights > 0.0)).reshape(-1, order="F")
    if not np.any(valid):
        raise ValueError(
            "sensor_constraint has no valid (finite, positive-weight) entries"
        )

    rows = np.flatnonzero(valid)
    matrix = sp.kron(sp.eye(n_times, format="csr"), operator, format="csr")[
        rows
    ].tocsr()
    scale = np.sqrt(weights.reshape(-1, order="F")[rows])
    if scale.size and not np.allclose(scale, 1.0):
        matrix = sp.diags(scale, format="csr") @ matrix
    return matrix.tocsr(), scale * targets.reshape(-1, order="F")[rows]


# ---------------------------------------------------------------------------
# Forward adapters (log resistivity in, log apparent resistivity out)
# ---------------------------------------------------------------------------


def _forward_log_response(forward, log_resistivity: np.ndarray) -> np.ndarray:
    if isinstance(forward, ERTForward2p5D):
        conductivity = torch.as_tensor(1.0 / np.exp(log_resistivity), dtype=FLOAT_DTYPE)
        return np.log(
            np.asarray(forward.apparent_resistivity_values(conductivity), dtype=float)
        )
    if not hasattr(forward, "forward"):
        raise TypeError(
            "forward must be ERTForward2p5D, ERTForwardModeling, or expose forward"
        )
    return np.asarray(forward.forward(log_resistivity, log_transform=True), dtype=float)


def _forward_log_response_series(forward, log_resistivity: np.ndarray) -> np.ndarray:
    """``(n_steps, n_data)`` log responses of a ``(n_steps, n_cells)`` log-model series."""

    log_models = np.asarray(log_resistivity, dtype=float)
    if log_models.ndim != 2:
        raise ValueError("log_resistivity series must be a 2D array")
    return np.vstack([_forward_log_response(forward, row) for row in log_models])


def _forward_and_jacobian_log(
    forward, log_resistivity: np.ndarray, *, robin: bool = False, normal: bool = True
):
    options = {"include_robin_boundary_derivative": robin, "normal_sensitivity": normal}
    if isinstance(forward, ERTForward2p5D):
        return log_response_and_jacobian(
            forward,
            torch.as_tensor(1.0 / np.exp(log_resistivity), dtype=FLOAT_DTYPE),
            **options,
        )
    if not hasattr(forward, "forward_and_jacobian"):
        raise TypeError(
            "forward must be ERTForward2p5D, ERTForwardModeling, or expose forward_and_jacobian"
        )
    return forward.forward_and_jacobian(log_resistivity, log_transform=True, **options)


def _cached_forward_and_jacobian(
    cache: OrderedDict | None, max_entries: int, forward, log_model, robin, normal
):
    """LRU memoization of linearizations shared by overlapping sliding windows."""

    if cache is None or max_entries < 1:
        return _forward_and_jacobian_log(forward, log_model, robin=robin, normal=normal)
    key = (np.ascontiguousarray(log_model, dtype=np.float64).tobytes(), robin, normal)
    if key not in cache:
        cache[key] = _forward_and_jacobian_log(
            forward, log_model, robin=robin, normal=normal
        )
        while len(cache) > max_entries:
            cache.popitem(last=False)
    cache.move_to_end(key)
    return cache[key]


def _operator(forward) -> ERTForward2p5D | None:
    """2.5D operator whose cells map onto the inversion model (``None`` for other forwards)."""

    if isinstance(forward, ERTForward2p5D):
        return forward
    if isinstance(forward, (ERTForwardModeling, ParameterizedERTForward2p5D)):
        return (
            forward.forward_operator
            if isinstance(forward.forward_operator, ERTForward2p5D)
            else None
        )
    return None


def _supports_normal_matrix_free(forward, config: InversionConfig) -> bool:
    """Whether the exact normal-sensitivity VJP is available (2.5D, no Robin derivative)."""

    return (
        config.normal_sensitivity
        and not config.include_robin_boundary_derivative
        and _operator(forward) is not None
    )


def _normal_log_response_vjp_series(
    forward, log_resistivity, predicted_log, cotangents
) -> np.ndarray:
    """Rows of ``(d log(rhoa) / d log(rho))^T cotangent`` without materializing Jacobians.

    The PyGIMLi-prolongation parameterization follows the fused active-cell aggregation of
    :meth:`ParameterizedERTForward2p5D.forward_and_jacobian`.
    """

    operator = _operator(forward)
    if operator is None:
        raise TypeError(
            "matrix-free normal VJP is currently available for 2.5D ERT only"
        )
    parameter_logs = np.asarray(log_resistivity, dtype=float)
    predicted = np.asarray(predicted_log, dtype=float)
    cotangents = np.asarray(cotangents, dtype=float)
    if (
        parameter_logs.ndim != 2
        or predicted.ndim != 2
        or cotangents.shape != predicted.shape
    ):
        raise ValueError("matrix-free VJP series inputs must be compatible 2D arrays")
    factors = np.abs(np.asarray(operator._geometric_factors(), dtype=float)).reshape(
        1, -1
    )
    resistance_cotangents = cotangents * factors / np.exp(predicted)
    parameterized = isinstance(forward, ParameterizedERTForward2p5D)
    fused = parameterized and forward.background_mode == "pygimli_prolongation"
    rows = []
    for log_model, cotangent in zip(parameter_logs, resistance_cotangents, strict=True):
        full_log, projection = (
            forward._full_log_model_and_projection(log_model)
            if parameterized
            else (log_model, None)
        )
        options = (
            {
                "cell_parameter_ids": forward.forward_cell_parameter_ids,
                "parameter_count": forward.cell_count,
            }
            if fused
            else {}
        )
        gradient_sigma = np.asarray(
            operator.normal_vjp(
                torch.as_tensor(np.exp(-full_log), dtype=FLOAT_DTYPE),
                torch.as_tensor(cotangent, dtype=FLOAT_DTYPE),
                **options,
            ),
            dtype=float,
        ).reshape(-1)
        if fused:
            rows.append(-np.exp(-log_model) * gradient_sigma)
        else:
            gradient = -np.exp(-full_log) * gradient_sigma
            rows.append(
                gradient
                if projection is None
                else np.asarray(projection.T @ gradient, dtype=float).reshape(-1)
            )
    return np.vstack(rows)


def _normal_log_response_vjp(
    forward, log_resistivity, predicted_log, cotangent
) -> np.ndarray:
    return _normal_log_response_vjp_series(
        forward,
        np.reshape(log_resistivity, (1, -1)),
        np.reshape(predicted_log, (1, -1)),
        np.reshape(cotangent, (1, -1)),
    )[0]


# ---------------------------------------------------------------------------
# Objective pieces
# ---------------------------------------------------------------------------


def _is_difference_misfit(data_misfit: DataMisfit) -> bool:
    return getattr(data_misfit, "name", "") == "log_data_difference_l2"


def _difference_weights(weights: np.ndarray) -> np.ndarray:
    sigma = 1.0 / np.asarray(weights, dtype=float)
    return 1.0 / np.sqrt(sigma[1:] ** 2 + sigma[0][None, :] ** 2)


def _difference_residual(
    predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Baseline residual followed by weighted residuals of changes relative to the first survey."""

    if predicted.shape != observed.shape:
        raise ValueError("predicted_log and observed_log must have the same shape")
    if predicted.ndim != 2 or predicted.shape[0] < 2:
        raise ValueError("log data-difference misfit requires at least two time steps")
    base = ((predicted[0] - observed[0]) * weights[0]).reshape(1, -1)
    change = (predicted[1:] - predicted[0][None, :]) - (
        observed[1:] - observed[0][None, :]
    )
    return np.vstack((base, change * _difference_weights(weights)))


def _data_phi(misfit: DataMisfit, predicted, observed, weights) -> float:
    if _is_difference_misfit(misfit):
        residual = _difference_residual(predicted, observed, weights).reshape(-1)
        return float(np.dot(residual, residual))
    return misfit.phi(predicted, observed, weights)


def _data_chi2(misfit: DataMisfit, predicted, observed, weights) -> float:
    if _is_difference_misfit(misfit):
        return float(np.mean(_difference_residual(predicted, observed, weights) ** 2))
    return misfit.chi2(predicted, observed, weights)


def _data_system(
    misfit: DataMisfit, predicted, observed, weights, jacobians
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Linearized data term over all timesteps (block-coupled for the data-difference misfit)."""

    if not _is_difference_misfit(misfit):
        blocks = [
            misfit.linearized_system(predicted[t], observed[t], weights[t], jac)
            for t, jac in enumerate(jacobians)
        ]
        return sp.block_diag(
            [sp.csr_matrix(matrix) for matrix, _ in blocks], format="csr"
        ), np.concatenate([rhs for _, rhs in blocks])

    n_times = predicted.shape[0]
    if n_times < 2:
        raise ValueError("log data-difference misfit requires at least two time steps")
    zero = sp.csr_matrix((predicted.shape[1], jacobians[0].shape[1]))
    rows = [
        sp.hstack(
            [sp.csr_matrix(jacobians[0] * weights[0][:, None])]
            + [zero] * (n_times - 1),
            format="csr",
        )
    ]
    rhs = [-((predicted[0] - observed[0]) * weights[0])]
    for t, w_t in enumerate(_difference_weights(weights), start=1):
        residual = (predicted[t] - predicted[0]) - (observed[t] - observed[0])
        blocks = [zero] * n_times
        blocks[0] = sp.csr_matrix(-jacobians[0] * w_t[:, None])
        blocks[t] = sp.csr_matrix(jacobians[t] * w_t[:, None])
        rows.append(sp.hstack(blocks, format="csr"))
        rhs.append(-(residual * w_t))
    return sp.vstack(rows, format="csr"), np.concatenate(rhs)


def _data_cotangents(misfit: DataMisfit, predicted, observed, weights) -> np.ndarray:
    """Per-timestep data cotangents whose VJPs give the data-term gradient."""

    if not _is_difference_misfit(misfit):
        return np.vstack(
            [
                misfit.linearized_cotangent(predicted[t], observed[t], weights[t])
                for t in range(predicted.shape[0])
            ]
        )
    cotangents = np.zeros_like(predicted, dtype=float)
    cotangents[0] += weights[0] ** 2 * (predicted[0] - observed[0])
    for t, w_t in enumerate(_difference_weights(weights), start=1):
        contribution = w_t**2 * (
            (predicted[t] - predicted[0]) - (observed[t] - observed[0])
        )
        cotangents[t] += contribution
        cotangents[0] -= contribution
    return cotangents


def _reference_roughness(
    mode: str, current: np.ndarray, matrix, reference, *, identity: bool = False
) -> np.ndarray:
    if mode == "update" or (reference is None and identity):
        return current
    return np.zeros_like(current) if reference is None else matrix @ reference


def _model_difference_operator(base: sp.spmatrix, n_times: int) -> sp.csr_matrix:
    """Spatial roughness of the baseline model and of every change relative to it."""

    base = base.tocsr()
    zero = sp.csr_matrix(base.shape)
    rows = [sp.hstack([base] + [zero] * (n_times - 1), format="csr")]
    for t in range(1, n_times):
        blocks = [zero] * n_times
        blocks[0], blocks[t] = -base, base
        rows.append(sp.hstack(blocks, format="csr"))
    return sp.vstack(rows, format="csr")


def _mesh_cell_areas(mesh: Mesh | Mesh3D) -> np.ndarray:
    if isinstance(mesh, Mesh3D):
        areas = np.asarray(mesh.cell_volumes, dtype=float)
    else:
        cell_nodes = np.asarray(mesh.nodes, dtype=float)[
            np.asarray(mesh.cells, dtype=np.int32)
        ]
        x, y = cell_nodes[:, :, 0], cell_nodes[:, :, 1]
        areas = 0.5 * np.abs(
            np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
        )
    if np.any(areas <= 0.0) or not np.all(np.isfinite(areas)):
        raise ValueError(
            "regularization mesh contains non-positive or non-finite cell areas"
        )
    return areas


def _coverage(forward, sensitivity: np.ndarray) -> np.ndarray:
    """PyGIMLi-style log10 coverage: summed |sensitivity| over data, normalized by cell size.

    ``sensitivity`` is a Jacobian (summed over rows) or, for matrix-free runs, a gradient.
    """

    sensitivity = np.abs(np.asarray(sensitivity, dtype=float))
    total = np.sum(sensitivity, axis=0) if sensitivity.ndim == 2 else sensitivity
    areas = _mesh_cell_areas(regularization_mesh(forward))
    if total.shape != areas.shape:
        raise ValueError(
            f"coverage Jacobian column count does not match regularization mesh cell count ({total.shape[0]} != {areas.shape[0]})"
        )
    return np.log10(np.maximum(total / areas, np.finfo(float).tiny))


def _line_search_tau(
    state,
    step,
    predicted_log,
    candidate_predicted_log,
    objective_matrix,
    objective_reference,
    data_phi,
) -> float:
    """Grid search on ``tau`` in ``[0, 1]`` using a linear model of the data along the step."""

    def phi(tau: float, predicted) -> float:
        roughness = objective_matrix @ (state + tau * step) - objective_reference
        return data_phi(predicted) + float(np.dot(roughness, roughness))

    best_phi = phi(0.0, predicted_log)
    if phi(1.0, candidate_predicted_log) < best_phi:
        return 1.0
    best_tau, direction = 0.0, candidate_predicted_log - predicted_log
    for index in range(1, 101):
        tau = 0.01 * index
        value = phi(tau, predicted_log + tau * direction)
        if value < best_phi:
            best_phi, best_tau = value, tau
    return 0.03 if 0.0 < best_tau < 0.03 else best_tau


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
    joint_frame = not single and config.temporal_regularization_mode == "joint_frame"
    temporal_weight = 0.0 if single else config.temporal_regularization
    spatial = build_spatial_regularization(config.spatial_regularization)
    temporal = build_temporal_regularization(config.temporal_regularization_type)
    if (
        joint_frame
        and temporal_weight > 0.0
        and temporal.name not in ("first_order_l2", "second_order_l2")
    ):
        raise ValueError(
            "robust temporal regularization is currently supported for temporal_regularization_mode='separate' only"
        )
    if (
        joint_frame
        and config.regularization > 0.0
        and spatial.name not in ("identity", "first_order")
    ):
        raise ValueError(
            "robust spatial regularization is currently supported for temporal_regularization_mode='separate' only"
        )

    roughness = spatial.matrix(forward, n_cells, z_weight=config.z_weight)
    roughness_all = sp.block_diag([roughness] * n_times, format="csr")
    sensor = None if single else _sensor_system(config, n_cells, n_times)
    physical = _regularization_domain(config) == "physical" and (
        config.regularization > 0.0
        or (not joint_frame and temporal_weight > 0.0)
        or sensor is not None
    )
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

    def regularization_blocks(
        states: np.ndarray,
    ) -> list[tuple[sp.spmatrix, np.ndarray]]:
        """Scaled ``(A, b)`` blocks of every model-space term, linearized at ``states``."""

        if physical:
            domain, derivative = _regularization_value_and_derivative(states, config)
            projection = _diagonal(derivative)
            reference_domain = (
                None
                if reference is None
                else _regularization_value_and_derivative(reference, config)[0]
            )
        else:
            domain, derivative, projection, reference_domain = (
                states,
                np.ones_like(states),
                None,
                reference,
            )
        domain_vec = np.asarray(domain, dtype=float).reshape(-1, order="F")
        reference_vec = (
            None
            if reference_domain is None
            else np.asarray(reference_domain, dtype=float).reshape(-1, order="F")
        )

        def project(matrix, columns=None):
            if projection is None:
                return matrix
            chain = projection if columns is None else _diagonal(derivative[:, columns])
            return (matrix @ chain).tocsr()

        def linear_block(operator, scale, *, identity=False):
            current = operator @ domain_vec
            target = _reference_roughness(
                config.regularization_mode,
                current,
                operator,
                reference_vec,
                identity=identity,
            )
            return project(scale * operator), scale * (target - current)

        blocks = []
        scale = float(np.sqrt(config.regularization))
        if config.regularization > 0.0 and joint_frame:
            frame = [roughness_all]
            if temporal_weight > 0.0:
                frame.append(
                    temporal.matrix(n_cells, n_times, scale=float(temporal_weight))
                )
            blocks.append(linear_block(sp.vstack(frame, format="csr"), scale))
        elif (
            config.regularization > 0.0
            and spatial.name == "model_difference_smoothness"
            and not single
        ):
            blocks.append(
                linear_block(_model_difference_operator(roughness, n_times), scale)
            )
        elif config.regularization > 0.0 and spatial.name in (
            "identity",
            "first_order",
        ):
            blocks.append(
                linear_block(roughness_all, scale, identity=spatial.name == "identity")
            )
        elif (
            config.regularization > 0.0
        ):  # IRLS-reweighted robust spatial terms, one block per timestep
            per_time = []
            for t in range(n_times):
                domain_t = np.asarray(domain[:, t], dtype=float)
                current = roughness @ domain_t
                target = _reference_roughness(
                    config.regularization_mode, current, roughness,
                    None if reference_domain is None else np.asarray(reference_domain[:, t], dtype=float),
                )  # fmt: skip
                matrix, rhs = spatial.linearized_system(
                    forward,
                    domain_t,
                    n_cells,
                    reference_roughness=target,
                    scale=scale,
                    z_weight=config.z_weight,
                )
                per_time.append((project(matrix, t), rhs))
            blocks.append(
                (
                    sp.block_diag([m for m, _ in per_time], format="csr"),
                    np.concatenate([r for _, r in per_time]),
                )
            )

        if not joint_frame and temporal_weight > 0.0:
            options = {
                "threshold": config.active_time_threshold,
                "minimum_weight": config.active_time_minimum_weight,
            }
            matrix, rhs = temporal.linearized_system(
                domain_vec, n_cells, n_times, scale=float(np.sqrt(temporal_weight)),
                **(options if temporal.name == "active_time_constraint" else {}),
            )  # fmt: skip
            blocks.append((project(matrix), rhs))

        if sensor is not None:
            matrix, target = sensor
            sensor_scale = float(np.sqrt(config.sensor_constraint))
            blocks.append(
                (
                    sensor_scale * project(matrix),
                    sensor_scale * (target - matrix @ domain_vec),
                )
            )
        return blocks

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
        blocks = regularization_blocks(models)
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
