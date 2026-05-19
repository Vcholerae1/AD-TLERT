"""Log-space ERT inversion routines built on the native differentiable forward path."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
import time
from typing import Any

from deepert.utils.torch_compat import jnp
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.sparse.linalg import cg, lsqr

from deepert.forward import ERTForward2p5D, ERTForwardModeling
from deepert.mesh import Mesh
from deepert.utils.dtypes import FLOAT_DTYPE


ArrayLike = Any
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class InversionConfig:
    """Controls for damped Gauss-Newton inversion in log-resistivity space."""

    max_iterations: int = 8
    data_std: float | ArrayLike = 0.05
    regularization: float = 1.0e-2
    regularization_mode: str = "model"
    temporal_regularization: float = 0.0
    temporal_regularization_mode: str = "separate"
    spatial_regularization: str = "identity"
    z_weight: float = 1.0
    model_transform: str = "log"
    model_bounds: tuple[float, float] | None = None
    step_length: float = 1.0
    max_log_step: float | None = 1.0
    line_search: bool = False
    target_chi2: float | None = None
    step_tolerance: float = 1.0e-4
    linearized_solver: str = "lsqr"
    lsqr_atol: float = 1.0e-6
    lsqr_btol: float = 1.0e-6
    lsqr_iter_limit: int | None = None
    cgls_max_iterations: int = 2000
    cgls_tolerance: float = 1.0e-8
    include_robin_boundary_derivative: bool = False
    normal_sensitivity: bool = True
    progress_callback: ProgressCallback | None = field(default=None, repr=False, compare=False)


def _emit_progress(config: InversionConfig, event: str, **payload: Any) -> None:
    callback = config.progress_callback
    if callback is not None:
        callback({"event": event, **payload})


@dataclass(frozen=True)
class ERTInversionResult:
    """Single-time inversion result."""

    final_model: np.ndarray
    final_log_model: np.ndarray
    predicted_data: np.ndarray
    predicted_log_data: np.ndarray
    coverage: np.ndarray
    iteration_chi2: list[float]


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


class ParameterizedERTForward2p5D:
    """ERT forward wrapper with a full solve mesh and a smaller parameter mesh."""

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
        forward_cell_count = int(forward.mesh.cell_count)
        self.parameter_cell_ids = np.asarray(parameter_cell_ids, dtype=np.int32).ravel()
        if self.parameter_cell_ids.ndim != 1:
            raise ValueError("parameter_cell_ids must be a 1D array")
        if self.parameter_cell_ids.size == 0:
            raise ValueError("parameter_cell_ids must not be empty")
        if np.any(self.parameter_cell_ids < 0) or np.any(self.parameter_cell_ids >= forward_cell_count):
            raise ValueError("parameter_cell_ids reference cells outside the forward mesh")
        if np.unique(self.parameter_cell_ids).size != self.parameter_cell_ids.size:
            raise ValueError("parameter_cell_ids must be unique")
        if background_mode not in ("pygimli_prolongation", "nearest", "fixed_mean"):
            raise ValueError("background_mode must be 'pygimli_prolongation', 'nearest', or 'fixed_mean'")
        self.background_mode = background_mode
        self.forward_cell_parameter_ids = self._resolve_forward_cell_parameter_ids(
            forward_cell_parameter_ids,
            forward_cell_count=forward_cell_count,
        )
        self._n_parameters = int(np.max(self.forward_cell_parameter_ids)) + 1
        if self.parameter_cell_ids.size != self._n_parameters:
            raise ValueError(
                "parameter_cell_ids must contain one representative forward cell per inversion parameter "
                f"({self.parameter_cell_ids.size} != {self._n_parameters})"
            )
        representative_ids = self.forward_cell_parameter_ids[self.parameter_cell_ids]
        expected_ids = np.arange(self._n_parameters, dtype=np.int32)
        if not np.array_equal(representative_ids, expected_ids):
            raise ValueError("parameter_cell_ids must be ordered representatives of forward_cell_parameter_ids")
        if regularization_mesh is not None and int(regularization_mesh.cell_count) != self._n_parameters:
            raise ValueError(
                "regularization mesh cell count must match inversion parameter count "
                f"({regularization_mesh.cell_count} != {self._n_parameters})"
            )
        all_cell_ids = np.arange(forward_cell_count, dtype=np.int32)
        self._active_forward_mask = self.forward_cell_parameter_ids >= 0
        self._active_forward_cell_ids = all_cell_ids[self._active_forward_mask]
        self._active_parameter_ids = self.forward_cell_parameter_ids[self._active_forward_cell_ids]
        self.background_cell_ids = all_cell_ids[~self._active_forward_mask]
        self._background_parameter_ids = (
            self._build_background_parameter_ids()
            if self.background_mode == "nearest"
            else np.empty((0,), dtype=np.int32)
        )
        self._resistivity_prolongation_matrix = (
            self._build_resistivity_prolongation_matrix()
            if self.background_mode == "pygimli_prolongation"
            else None
        )
        self._jacobian_projection = self._build_jacobian_projection(forward_cell_count)

    def _resolve_forward_cell_parameter_ids(
        self,
        forward_cell_parameter_ids: ArrayLike | None,
        *,
        forward_cell_count: int,
    ) -> np.ndarray:
        if forward_cell_parameter_ids is None:
            cell_parameter_ids = np.full((forward_cell_count,), -1, dtype=np.int32)
            cell_parameter_ids[self.parameter_cell_ids] = np.arange(self.parameter_cell_ids.size, dtype=np.int32)
            return cell_parameter_ids

        cell_parameter_ids = np.asarray(forward_cell_parameter_ids, dtype=np.int32).ravel()
        if cell_parameter_ids.shape != (forward_cell_count,):
            raise ValueError(
                "forward_cell_parameter_ids must have one entry per forward mesh cell "
                f"({cell_parameter_ids.shape} != ({forward_cell_count},))"
            )
        if np.any(cell_parameter_ids < -1):
            raise ValueError("forward_cell_parameter_ids may only contain -1 or non-negative parameter ids")
        active_ids = cell_parameter_ids[cell_parameter_ids >= 0]
        if active_ids.size == 0:
            raise ValueError("forward_cell_parameter_ids must contain at least one active parameter cell")
        unique_ids = np.unique(active_ids)
        expected_ids = np.arange(int(unique_ids[-1]) + 1, dtype=np.int32)
        if not np.array_equal(unique_ids, expected_ids):
            raise ValueError("forward_cell_parameter_ids must use contiguous ids starting at 0")
        return cell_parameter_ids

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
    ) -> "ParameterizedERTForward2p5D":
        forward_cell_parameter_ids = forward_kwargs.pop("forward_cell_parameter_ids", None)
        forward = ERTForward2p5D.from_mesh_survey(mesh, survey, **forward_kwargs)
        return cls(
            forward,
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

    def _cell_centers(self) -> np.ndarray:
        nodes = np.asarray(self.forward_operator.mesh.nodes, dtype=float)
        cells = np.asarray(self.forward_operator.mesh.cells, dtype=np.int32)
        return np.mean(nodes[cells], axis=1)

    def _parameter_centers(self, centers: np.ndarray) -> np.ndarray:
        sums = np.zeros((self.cell_count, centers.shape[1]), dtype=float)
        counts = np.zeros((self.cell_count,), dtype=float)
        np.add.at(sums, self._active_parameter_ids, centers[self._active_forward_cell_ids])
        np.add.at(counts, self._active_parameter_ids, 1.0)
        if np.any(counts <= 0.0):
            raise ValueError("each inversion parameter must own at least one forward mesh cell")
        return sums / counts[:, None]

    def _build_background_parameter_ids(self) -> np.ndarray:
        if self.background_cell_ids.size == 0:
            return np.empty((0,), dtype=np.int32)
        centers = self._cell_centers()
        parameter_centers = self._parameter_centers(centers)
        _, nearest = cKDTree(parameter_centers).query(centers[self.background_cell_ids])
        return np.asarray(nearest, dtype=np.int32)

    def _build_jacobian_projection(self, forward_cell_count: int) -> sp.csr_matrix:
        rows = [self._active_forward_cell_ids]
        cols = [self._active_parameter_ids]
        data = [np.ones(self._active_forward_cell_ids.size, dtype=float)]
        if self.background_mode == "pygimli_prolongation":
            return sp.coo_matrix(
                (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                shape=(forward_cell_count, self.cell_count),
            ).tocsr()
        if self.background_cell_ids.size and self.background_mode == "nearest":
            rows.append(self.background_cell_ids)
            cols.append(self._background_parameter_ids)
            data.append(np.ones(self.background_cell_ids.size, dtype=float))
        elif self.background_cell_ids.size and self.background_mode == "fixed_mean":
            rows.append(np.repeat(self.background_cell_ids, self.cell_count))
            cols.append(np.tile(np.arange(self.cell_count, dtype=np.int32), self.background_cell_ids.size))
            data.append(np.full(self.background_cell_ids.size * self.cell_count, 1.0 / self.cell_count, dtype=float))
        return sp.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(forward_cell_count, self.cell_count),
        ).tocsr()

    def _build_resistivity_prolongation_matrix(self) -> sp.csr_matrix:
        """Replicate PyGIMLi's marker model prolongation for background cells."""

        forward_cell_count = int(self.forward_operator.mesh.cell_count)
        matrix = np.zeros((forward_cell_count, self.cell_count), dtype=float)
        matrix[self._active_forward_cell_ids, self._active_parameter_ids] = 1.0
        if self.background_cell_ids.size == 0:
            return sp.csr_matrix(matrix)

        nodes = np.asarray(self.forward_operator.mesh.nodes, dtype=float)
        cells = np.asarray(self.forward_operator.mesh.cells, dtype=np.int32)
        edge_cells: dict[tuple[int, int], list[int]] = {}
        for cell_id, cell in enumerate(cells):
            for edge in _cell_edges(cell):
                edge_cells.setdefault(tuple(sorted(edge)), []).append(cell_id)

        neighbors: list[list[tuple[int, float]]] = [[] for _ in range(forward_cell_count)]
        for edge, owners in edge_cells.items():
            if len(owners) != 2:
                continue
            p0, p1 = nodes[list(edge)]
            tangent = p1 - p0
            length = float(np.linalg.norm(tangent))
            if length <= 0.0:
                continue
            weight = abs(float(tangent[1])) / length + 1.0e-6
            left, right = int(owners[0]), int(owners[1])
            neighbors[left].append((right, weight))
            neighbors[right].append((left, weight))

        known = self._active_forward_mask.copy()
        unknown = set(int(cell_id) for cell_id in self.background_cell_ids)
        while unknown:
            assignments: list[tuple[int, np.ndarray]] = []
            for cell_id in sorted(unknown):
                weighted = np.zeros((self.cell_count,), dtype=float)
                total_weight = 0.0
                for neighbor_id, weight in neighbors[cell_id]:
                    if known[neighbor_id]:
                        weighted += matrix[neighbor_id] * weight
                        total_weight += weight
                if total_weight > 1.0e-8:
                    assignments.append((cell_id, weighted / total_weight))
            if not assignments:
                raise ValueError("could not prolongate background forward cells from active parameter cells")
            for cell_id, row in assignments:
                matrix[cell_id] = row
                known[cell_id] = True
                unknown.remove(cell_id)

        return sp.csr_matrix(matrix)

    def _full_log_model(self, log_resistivity: ArrayLike) -> np.ndarray:
        full_log, _ = self._full_log_model_and_projection(log_resistivity)
        return full_log

    def _full_log_model_and_projection(self, log_resistivity: ArrayLike) -> tuple[np.ndarray, sp.csr_matrix]:
        parameter_log = np.asarray(log_resistivity, dtype=float).ravel()
        if parameter_log.shape != (self.cell_count,):
            raise ValueError(f"log_resistivity must have shape ({self.cell_count},)")
        if self.background_mode == "pygimli_prolongation":
            if self._resistivity_prolongation_matrix is None:
                raise ValueError("resistivity prolongation matrix has not been initialized")
            parameter_resistivity = np.exp(parameter_log)
            full_resistivity = np.asarray(self._resistivity_prolongation_matrix @ parameter_resistivity, dtype=float).ravel()
            if np.any(full_resistivity <= 0.0) or not np.all(np.isfinite(full_resistivity)):
                raise ValueError("prolongated forward resistivity contains non-positive or non-finite values")
            if self._jacobian_projection is None:
                raise ValueError("Jacobian projection has not been initialized")
            return np.log(full_resistivity), self._jacobian_projection

        full_log = np.empty((self.forward_operator.mesh.cell_count,), dtype=float)
        full_log[self._active_forward_cell_ids] = parameter_log[self._active_parameter_ids]
        if self.background_cell_ids.size:
            if self.background_mode == "nearest":
                full_log[self.background_cell_ids] = parameter_log[self._background_parameter_ids]
            else:
                full_log[self.background_cell_ids] = float(np.mean(parameter_log))
        if self._jacobian_projection is None:
            raise ValueError("Jacobian projection has not been initialized")
        return full_log, self._jacobian_projection

    def forward_and_jacobian(
        self,
        resistivity_model: ArrayLike,
        log_transform: bool = True,
        *,
        include_robin_boundary_derivative: bool | None = None,
        normal_sensitivity: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        log_model = _as_log_model(
            resistivity_model,
            expected_size=self.cell_count,
            log_model=log_transform,
            name="resistivity_model",
        )
        full_log_model, projection = self._full_log_model_and_projection(log_model)
        if (
            self.background_mode == "pygimli_prolongation"
            and (normal_sensitivity is None or bool(normal_sensitivity))
            and not (bool(include_robin_boundary_derivative) if include_robin_boundary_derivative is not None else False)
        ):
            conductivity = jnp.asarray(np.exp(-full_log_model), dtype=FLOAT_DTYPE)
            response, resistance_jacobian = self.forward_operator.solve_with_jacobian(
                conductivity,
                include_robin_boundary_derivative=False,
                normal_sensitivity=True,
                jacobian_cell_parameter_ids=self.forward_cell_parameter_ids,
                jacobian_parameter_count=self.cell_count,
            )
            apparent_jacobian_sigma = jnp.abs(self.forward_operator._geometric_factors())[:, None] * resistance_jacobian
            parameter_conductivity = jnp.asarray(np.exp(-log_model), dtype=apparent_jacobian_sigma.dtype)
            jacobian = apparent_jacobian_sigma * (-parameter_conductivity[None, :])
            jacobian = jacobian / response.apparent_resistivity[:, None]
            return (
                np.log(np.asarray(response.apparent_resistivity, dtype=float)),
                np.asarray(jacobian, dtype=float),
            )
        predicted, full_jacobian = _forward_and_jacobian_log(
            self.forward_operator,
            full_log_model,
            include_robin_boundary_derivative=bool(include_robin_boundary_derivative)
            if include_robin_boundary_derivative is not None
            else False,
            normal_sensitivity=bool(normal_sensitivity) if normal_sensitivity is not None else True,
        )
        jacobian = np.asarray((projection.T @ np.asarray(full_jacobian, dtype=float).T).T, dtype=float)
        return predicted, jacobian

    def response(self, resistivity_model: ArrayLike) -> np.ndarray:
        return self.forward(resistivity_model, log_transform=False)

    def forward(self, resistivity_model: ArrayLike, log_transform: bool = True) -> np.ndarray:
        log_model = _as_log_model(
            resistivity_model,
            expected_size=self.cell_count,
            log_model=log_transform,
            name="resistivity_model",
        )
        response = _forward_log_response(self.forward_operator, self._full_log_model(log_model))
        if log_transform:
            return response
        return np.exp(response)


def _check_config(config: InversionConfig) -> None:
    if config.max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if config.regularization < 0.0:
        raise ValueError("regularization must be non-negative")
    if config.regularization_mode not in ("model", "update"):
        raise ValueError("regularization_mode must be 'model' or 'update'")
    if config.temporal_regularization < 0.0:
        raise ValueError("temporal_regularization must be non-negative")
    if config.temporal_regularization_mode not in ("separate", "joint_frame"):
        raise ValueError("temporal_regularization_mode must be 'separate' or 'joint_frame'")
    if config.spatial_regularization not in ("identity", "first_order"):
        raise ValueError("spatial_regularization must be 'identity' or 'first_order'")
    if config.z_weight <= 0.0:
        raise ValueError("z_weight must be positive")
    if config.model_transform not in ("log", "log_lu"):
        raise ValueError("model_transform must be 'log' or 'log_lu'")
    if config.model_transform == "log_lu" and config.model_bounds is None:
        raise ValueError("model_bounds are required for model_transform='log_lu'")
    if config.step_length <= 0.0:
        raise ValueError("step_length must be positive")
    if config.max_log_step is not None and config.max_log_step <= 0.0:
        raise ValueError("max_log_step must be positive when set")
    if config.target_chi2 is not None and config.target_chi2 <= 0.0:
        raise ValueError("target_chi2 must be positive when set")
    if config.linearized_solver not in ("lsqr", "pyhydro_cgls", "normal_cg", "gpu_cgls", "gpu_timelapse_cgls"):
        raise ValueError(
            "linearized_solver must be 'lsqr', 'pyhydro_cgls', 'normal_cg', 'gpu_cgls', or 'gpu_timelapse_cgls'"
        )
    if config.cgls_max_iterations < 1:
        raise ValueError("cgls_max_iterations must be >= 1")
    if config.cgls_tolerance <= 0.0:
        raise ValueError("cgls_tolerance must be positive")
    if config.progress_callback is not None and not callable(config.progress_callback):
        raise ValueError("progress_callback must be callable when set")
    if config.model_bounds is not None:
        lo, hi = config.model_bounds
        if not (0.0 < lo < hi):
            raise ValueError("model_bounds must be positive and ordered as (min, max)")


def _model_size(forward: ERTForward2p5D | ERTForwardModeling) -> int:
    if isinstance(forward, ERTForward2p5D):
        return forward.mesh.cell_count
    if isinstance(forward, ERTForwardModeling):
        return forward.cell_count
    if hasattr(forward, "cell_count"):
        return int(forward.cell_count)
    if hasattr(forward, "mesh") and hasattr(forward.mesh, "cell_count"):
        return int(forward.mesh.cell_count)
    raise TypeError("forward must expose a deepert-compatible cell count")


def _measurement_count(forward: ERTForward2p5D | ERTForwardModeling) -> int:
    if isinstance(forward, ERTForward2p5D):
        return forward.survey.measurement_count
    if isinstance(forward, ERTForwardModeling):
        return forward.forward_operator.survey.measurement_count
    if hasattr(forward, "survey") and hasattr(forward.survey, "measurement_count"):
        return int(forward.survey.measurement_count)
    if hasattr(forward, "forward_operator") and hasattr(forward.forward_operator, "survey"):
        return int(forward.forward_operator.survey.measurement_count)
    raise TypeError("forward must expose a deepert-compatible measurement count")


def _as_log_model(
    model: ArrayLike,
    *,
    expected_size: int,
    log_model: bool,
    name: str,
) -> np.ndarray:
    values = np.asarray(model, dtype=float).ravel()
    if values.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},)")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    if log_model:
        return values.copy()
    if np.any(values <= 0.0):
        raise ValueError(f"{name} must contain positive resistivity values")
    return np.log(values)


def _as_log_model_matrix(
    model: ArrayLike,
    *,
    expected_size: int,
    n_times: int,
    log_model: bool,
    name: str,
) -> np.ndarray:
    values = np.asarray(model, dtype=float)
    if values.ndim == 1:
        return np.column_stack(
            [
                _as_log_model(values, expected_size=expected_size, log_model=log_model, name=name)
                for _ in range(n_times)
            ]
        )
    if values.shape == (expected_size, n_times):
        matrix = values.copy()
    elif values.shape == (n_times, expected_size):
        matrix = values.T.copy()
    else:
        raise ValueError(f"{name} must have shape ({expected_size},), ({expected_size}, {n_times}), or ({n_times}, {expected_size})")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    if log_model:
        return matrix
    if np.any(matrix <= 0.0):
        raise ValueError(f"{name} must contain positive resistivity values")
    return np.log(matrix)


def _as_observed_log_vector(data: ArrayLike, *, log_data: bool, expected_size: int | None = None) -> np.ndarray:
    values = np.asarray(data, dtype=float).ravel()
    if expected_size is not None and values.shape != (expected_size,):
        raise ValueError(f"observed_data must have shape ({expected_size},)")
    if not np.all(np.isfinite(values)):
        raise ValueError("observed_data contains non-finite values")
    if log_data:
        return values.copy()
    if np.any(values <= 0.0):
        raise ValueError("observed_data must contain positive apparent resistivity values")
    return np.log(values)


def _as_observed_log_matrix(
    data: ArrayLike,
    *,
    log_data: bool,
    measurement_count: int,
) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    if values.ndim != 2:
        raise ValueError("observed_data must be a 2D array for time-lapse inversion")
    if values.shape[1] == measurement_count:
        matrix = values.copy()
    elif values.shape[0] == measurement_count:
        matrix = values.T.copy()
    else:
        raise ValueError(
            "observed_data must have shape (n_times, n_measurements) "
            "or (n_measurements, n_times)"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("observed_data contains non-finite values")
    if log_data:
        return matrix
    if np.any(matrix <= 0.0):
        raise ValueError("observed_data must contain positive apparent resistivity values")
    return np.log(matrix)


def _weights(data_std: float | ArrayLike, shape: tuple[int, ...]) -> np.ndarray:
    std = np.asarray(data_std, dtype=float)
    if std.ndim == 0:
        std = np.full(shape, float(std), dtype=float)
    else:
        std = np.broadcast_to(std, shape).astype(float, copy=True)
    if not np.all(np.isfinite(std)):
        raise ValueError("data_std contains non-finite values")
    if np.any(std <= 0.0):
        raise ValueError("data_std must be positive")
    return 1.0 / std


def _log_bounds(bounds: tuple[float, float] | None) -> tuple[float, float] | None:
    if bounds is None:
        return None
    return float(np.log(bounds[0])), float(np.log(bounds[1]))


def _clip_log_model(log_model: np.ndarray, bounds: tuple[float, float] | None) -> np.ndarray:
    log_bounds = _log_bounds(bounds)
    if log_bounds is None:
        return log_model
    lo, hi = log_bounds
    return np.clip(log_model, lo, hi)


def _log_model_to_state(log_model: np.ndarray, config: InversionConfig) -> np.ndarray:
    if config.model_transform == "log":
        return _clip_log_model(log_model, config.model_bounds)

    if config.model_bounds is None:
        raise ValueError("model_bounds are required for the LogLU model transform")
    lo, hi = config.model_bounds
    span = hi - lo
    rho = np.exp(log_model)
    rho = np.clip(rho, lo + span * 1.0e-12, hi - span * 1.0e-12)
    return np.log(rho - lo) - np.log(hi - rho)


def _state_to_log_model(state: np.ndarray, config: InversionConfig) -> np.ndarray:
    if config.model_transform == "log":
        return _clip_log_model(state, config.model_bounds)

    if config.model_bounds is None:
        raise ValueError("model_bounds are required for the LogLU model transform")
    lo, hi = config.model_bounds
    exp_state = np.exp(np.clip(state, -50.0, 50.0))
    rho = (exp_state * hi + lo) / (exp_state + 1.0)
    return np.log(rho)


def _clip_model_state(state: np.ndarray, config: InversionConfig) -> np.ndarray:
    if config.model_transform == "log":
        return _clip_log_model(state, config.model_bounds)
    return state


def _d_log_model_d_state(state: np.ndarray, config: InversionConfig) -> np.ndarray:
    if config.model_transform == "log":
        return np.ones_like(state, dtype=float)

    if config.model_bounds is None:
        raise ValueError("model_bounds are required for the LogLU model transform")
    lo, hi = config.model_bounds
    rho = np.exp(_state_to_log_model(state, config))
    return ((rho - lo) * (hi - rho)) / ((hi - lo) * rho)


def _limit_delta(delta: np.ndarray, max_log_step: float | None) -> np.ndarray:
    if max_log_step is None:
        return delta
    max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
    if max_abs <= max_log_step:
        return delta
    return delta * (max_log_step / max_abs)


def _weighted_chi2(predicted_log_data: np.ndarray, observed_log_data: np.ndarray, weights: np.ndarray) -> float:
    residual = (predicted_log_data - observed_log_data) * weights
    return float(np.mean(residual**2))


def _mesh_cell_areas_np(mesh: Mesh) -> np.ndarray:
    nodes = np.asarray(mesh.nodes, dtype=float)
    cells = np.asarray(mesh.cells, dtype=np.int32)
    cell_nodes = nodes[cells]
    x_values = cell_nodes[:, :, 0]
    y_values = cell_nodes[:, :, 1]
    cross_sum = np.sum(
        x_values * np.roll(y_values, -1, axis=1) - np.roll(x_values, -1, axis=1) * y_values,
        axis=1,
    )
    areas = 0.5 * np.abs(cross_sum)
    if np.any(areas <= 0.0) or not np.all(np.isfinite(areas)):
        raise ValueError("regularization mesh contains non-positive or non-finite cell areas")
    return areas


def _pygimli_style_coverage_from_jacobian(
    forward: ERTForward2p5D | ERTForwardModeling,
    jacobian: np.ndarray,
) -> np.ndarray:
    """Return PyGIMLi-style log10 coverage used for default plot masking.

    PyGIMLi's ERT coverage path applies the data/model log transform to the
    sensitivity matrix, sums absolute transformed sensitivities over data, and
    normalizes by parameter-cell size before taking log10.
    """

    matrix = np.asarray(jacobian, dtype=float)
    mesh = _regularization_mesh(forward)
    areas = _mesh_cell_areas_np(mesh)
    if matrix.shape[1] != areas.shape[0]:
        raise ValueError(
            "coverage Jacobian column count does not match regularization mesh cell count "
            f"({matrix.shape[1]} != {areas.shape[0]})"
        )
    sensitivity_sum = np.sum(np.abs(matrix), axis=0)
    normalized = np.maximum(sensitivity_sum / areas, np.finfo(float).tiny)
    return np.log10(normalized)


def _data_phi(predicted_log_data: np.ndarray, observed_log_data: np.ndarray, weights: np.ndarray) -> float:
    residual = (predicted_log_data - observed_log_data) * weights
    return float(np.sum(residual**2))


def _model_phi(
    state: np.ndarray,
    regularization_matrix: sp.spmatrix,
    reference_roughness: np.ndarray,
) -> float:
    roughness = regularization_matrix @ state - reference_roughness
    return float(np.dot(roughness, roughness))


def _line_search_tau(
    *,
    state: np.ndarray,
    step: np.ndarray,
    predicted_log: np.ndarray,
    candidate_predicted_log: np.ndarray,
    observed_log: np.ndarray,
    weights: np.ndarray,
    regularization_matrix: sp.spmatrix,
    reference_roughness: np.ndarray,
    regularization: float,
) -> float:
    data_direction = candidate_predicted_log - predicted_log
    best_tau = 0.0
    best_phi = _data_phi(predicted_log, observed_log, weights) + regularization * _model_phi(
        state,
        regularization_matrix,
        reference_roughness,
    )
    candidate_phi = _data_phi(candidate_predicted_log, observed_log, weights) + regularization * _model_phi(
        state + step,
        regularization_matrix,
        reference_roughness,
    )
    if candidate_phi < best_phi:
        return 1.0
    for index in range(1, 101):
        tau = 0.01 * index
        state_tau = state + tau * step
        predicted_tau = predicted_log + tau * data_direction
        phi = _data_phi(predicted_tau, observed_log, weights) + regularization * _model_phi(
            state_tau,
            regularization_matrix,
            reference_roughness,
        )
        if phi < best_phi:
            best_phi = phi
            best_tau = tau
    if 0.0 < best_tau < 0.03:
        return 0.03
    return best_tau


def _regularization_mesh(forward: ERTForward2p5D | ERTForwardModeling) -> Mesh:
    mesh = getattr(forward, "regularization_mesh", None)
    if mesh is not None:
        if isinstance(mesh, Mesh):
            return mesh
        raise TypeError("forward.regularization_mesh must be a deepert Mesh")
    if isinstance(forward, ERTForward2p5D):
        return forward.mesh
    if isinstance(forward, ERTForwardModeling):
        return forward._resolved_mesh()
    if hasattr(forward, "_resolved_mesh"):
        resolved = forward._resolved_mesh()
        if isinstance(resolved, Mesh):
            return resolved
    mesh = getattr(forward, "mesh", None)
    if isinstance(mesh, Mesh):
        return mesh
    raise TypeError("forward must expose a Mesh for first-order regularization")


def _cell_edges(cell: np.ndarray) -> list[tuple[int, int]]:
    return [
        (int(cell[index]), int(cell[(index + 1) % cell.size]))
        for index in range(cell.size)
    ]


def _first_order_constraint_matrix(mesh: Mesh, *, z_weight: float = 1.0) -> sp.csr_matrix:
    """Build first-order neighbor constraints for cell models."""

    nodes = np.asarray(mesh.nodes, dtype=float)
    cells = np.asarray(mesh.cells, dtype=np.int32)
    edge_cells: dict[tuple[int, int], list[int]] = {}
    for cell_index, cell in enumerate(cells):
        for edge in _cell_edges(cell):
            key = tuple(sorted(edge))
            edge_cells.setdefault(key, []).append(cell_index)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    row = 0
    for edge, owners in edge_cells.items():
        if len(owners) != 2:
            continue
        p0, p1 = nodes[list(edge)]
        tangent = p1 - p0
        length = float(np.linalg.norm(tangent))
        if length <= 0.0:
            continue
        normal_z = abs(float(tangent[0])) / length
        weight = 1.0 + normal_z * (float(z_weight) - 1.0)
        left, right = owners
        rows.extend((row, row))
        cols.extend((left, right))
        data.extend((weight, -weight))
        row += 1

    return sp.coo_matrix((data, (rows, cols)), shape=(row, int(mesh.cell_count))).tocsr()


def _spatial_regularization_matrix(
    forward: ERTForward2p5D | ERTForwardModeling,
    config: InversionConfig,
    n_cells: int,
) -> sp.csr_matrix:
    if config.spatial_regularization == "identity":
        return sp.eye(n_cells, format="csr")
    matrix = _first_order_constraint_matrix(
        _regularization_mesh(forward),
        z_weight=config.z_weight,
    )
    if matrix.shape[1] != n_cells:
        raise ValueError(
            "regularization mesh cell count does not match inversion model size "
            f"({matrix.shape[1]} != {n_cells})"
        )
    return matrix


def _forward_and_jacobian_log(
    forward: ERTForward2p5D | ERTForwardModeling,
    log_resistivity: np.ndarray,
    *,
    include_robin_boundary_derivative: bool = False,
    normal_sensitivity: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(forward, ERTForwardModeling):
        return forward.forward_and_jacobian(
            log_resistivity,
            log_transform=True,
            include_robin_boundary_derivative=include_robin_boundary_derivative,
            normal_sensitivity=normal_sensitivity,
        )

    if not isinstance(forward, ERTForward2p5D):
        if hasattr(forward, "forward_and_jacobian"):
            return forward.forward_and_jacobian(
                log_resistivity,
                log_transform=True,
                include_robin_boundary_derivative=include_robin_boundary_derivative,
                normal_sensitivity=normal_sensitivity,
            )
        raise TypeError("forward must be ERTForward2p5D, ERTForwardModeling, or expose forward_and_jacobian")

    resistivity = np.exp(log_resistivity)
    conductivity = jnp.asarray(1.0 / resistivity, dtype=FLOAT_DTYPE)
    response, resistance_jacobian = forward.solve_with_jacobian(
        conductivity,
        include_robin_boundary_derivative=include_robin_boundary_derivative,
        normal_sensitivity=normal_sensitivity,
    )
    apparent_jacobian_sigma = jnp.abs(forward._geometric_factors())[:, None] * resistance_jacobian
    jacobian = apparent_jacobian_sigma * (-conductivity[None, :])
    jacobian = jacobian / response.apparent_resistivity[:, None]
    return (
        np.log(np.asarray(response.apparent_resistivity, dtype=float)),
        np.asarray(jacobian, dtype=float),
    )


ForwardJacobianCache = OrderedDict[tuple[tuple[int, ...], str, bytes, bool, bool], tuple[np.ndarray, np.ndarray]]


def _forward_and_jacobian_log_cached(
    forward: ERTForward2p5D | ERTForwardModeling,
    log_resistivity: np.ndarray,
    *,
    include_robin_boundary_derivative: bool = False,
    normal_sensitivity: bool = True,
    cache: ForwardJacobianCache | None = None,
    max_entries: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    if cache is None or max_entries < 1:
        return _forward_and_jacobian_log(
            forward,
            log_resistivity,
            include_robin_boundary_derivative=include_robin_boundary_derivative,
            normal_sensitivity=normal_sensitivity,
        )

    key_array = np.ascontiguousarray(log_resistivity, dtype=np.float64)
    key = (
        tuple(int(size) for size in key_array.shape),
        key_array.dtype.str,
        key_array.tobytes(),
        bool(include_robin_boundary_derivative),
        bool(normal_sensitivity),
    )
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return cached

    result = _forward_and_jacobian_log(
        forward,
        log_resistivity,
        include_robin_boundary_derivative=include_robin_boundary_derivative,
        normal_sensitivity=normal_sensitivity,
    )
    cache[key] = result
    cache.move_to_end(key)
    while len(cache) > max_entries:
        cache.popitem(last=False)
    return result


def _forward_log_response(
    forward: ERTForward2p5D | ERTForwardModeling,
    log_resistivity: np.ndarray,
) -> np.ndarray:
    if isinstance(forward, ERTForwardModeling):
        return np.asarray(forward.forward(log_resistivity, log_transform=True), dtype=float)

    if not isinstance(forward, ERTForward2p5D):
        if hasattr(forward, "forward"):
            return np.asarray(forward.forward(log_resistivity, log_transform=True), dtype=float)
        raise TypeError("forward must be ERTForward2p5D, ERTForwardModeling, or expose forward")

    resistivity = np.exp(log_resistivity)
    conductivity = jnp.asarray(1.0 / resistivity, dtype=FLOAT_DTYPE)
    response = forward.apparent_resistivity_values(conductivity)
    return np.log(np.asarray(response, dtype=float))


def _solve_increment(
    matrix: sp.spmatrix,
    rhs: np.ndarray,
    config: InversionConfig,
) -> np.ndarray:
    if config.linearized_solver == "gpu_cgls":
        return _cupy_cgls(
            matrix,
            rhs,
            max_iterations=config.cgls_max_iterations,
            tolerance=config.cgls_tolerance,
        )

    if config.linearized_solver == "pyhydro_cgls":
        normal_matrix = (matrix.T @ matrix).tocsr()
        normal_rhs = np.asarray(matrix.T @ rhs, dtype=float).reshape(-1, 1)
        solution = _pyhydro_cgls(
            normal_matrix,
            normal_rhs,
            max_iterations=config.cgls_max_iterations,
            tolerance=config.cgls_tolerance,
        ).ravel()
        if not np.all(np.isfinite(solution)):
            raise ValueError("linearized inversion update contains non-finite values")
        return solution

    if config.linearized_solver == "normal_cg":
        normal_matrix = (matrix.T @ matrix).tocsr()
        normal_rhs = np.asarray(matrix.T @ rhs, dtype=float).ravel()
        solution, info = cg(
            normal_matrix,
            normal_rhs,
            rtol=config.cgls_tolerance,
            atol=0.0,
            maxiter=config.cgls_max_iterations,
        )
        if info < 0:
            raise ValueError(f"normal_cg failed with illegal input/info={info}")
        if not np.all(np.isfinite(solution)):
            raise ValueError("linearized inversion update contains non-finite values")
        return np.asarray(solution, dtype=float)

    solution = lsqr(
        matrix,
        rhs,
        atol=config.lsqr_atol,
        btol=config.lsqr_btol,
        iter_lim=config.lsqr_iter_limit,
    )[0]
    if not np.all(np.isfinite(solution)):
        raise ValueError("linearized inversion update contains non-finite values")
    return solution


def _cupy_cgls(
    matrix: sp.spmatrix,
    rhs: np.ndarray,
    *,
    max_iterations: int,
    tolerance: float,
) -> np.ndarray:
    """Solve ``min ||A x - b||`` with CGLS using CuPy sparse matvecs."""

    try:
        import cupy as cp
        import cupyx.scipy.sparse as cupy_sparse
    except ImportError as exc:
        raise ImportError("linearized_solver='gpu_cgls' requires CuPy") from exc

    system_cpu = matrix.tocsr()
    dtype = np.float64 if system_cpu.dtype == np.float64 or np.asarray(rhs).dtype == np.float64 else np.float32
    system = cupy_sparse.csr_matrix(
        (
            cp.asarray(system_cpu.data, dtype=dtype),
            cp.asarray(system_cpu.indices, dtype=cp.int32),
            cp.asarray(system_cpu.indptr, dtype=cp.int32),
        ),
        shape=system_cpu.shape,
    )
    b = cp.asarray(np.asarray(rhs, dtype=dtype).ravel())
    x = cp.zeros((system.shape[1],), dtype=dtype)
    r = b.copy()
    s = system.T @ r
    p = s.copy()
    gamma = cp.dot(s, s)
    rr0 = cp.dot(r, r)
    gamma_value = float(gamma)
    gamma0_value = gamma_value
    rr0_value = float(rr0)
    if gamma_value <= 0.0 or rr0_value <= 0.0:
        return cp.asnumpy(x)

    for _ in range(int(max_iterations)):
        q = system @ p
        denominator = cp.dot(q, q)
        denominator_value = float(denominator)
        if denominator_value <= 0.0:
            break
        alpha = gamma / denominator
        x = x + alpha * p
        r = r - alpha * q
        s = system.T @ r
        gamma_new = cp.dot(s, s)
        gamma_new_value = float(gamma_new)
        if gamma_new_value <= 0.0:
            break
        if gamma_new_value / gamma0_value < float(tolerance):
            break
        p = s + (gamma_new / gamma) * p
        gamma = gamma_new

    solution = cp.asnumpy(x)
    if not np.all(np.isfinite(solution)):
        raise ValueError("linearized inversion update contains non-finite values")
    return np.asarray(solution, dtype=float)


def _cupy_timelapse_cgls(
    *,
    jacobians: list[np.ndarray],
    weight: np.ndarray,
    data_rhs: np.ndarray,
    spatial_regularization: sp.spmatrix | None,
    spatial_scale: float,
    spatial_rhs: np.ndarray | None,
    temporal_scale: float,
    temporal_rhs: np.ndarray | None,
    max_iterations: int,
    tolerance: float,
) -> np.ndarray:
    """Matrix-free CGLS for the default sliding-window time-lapse system."""

    try:
        import cupy as cp
        import cupyx.scipy.sparse as cupy_sparse
    except ImportError as exc:
        raise ImportError("linearized_solver='gpu_cgls' requires CuPy") from exc

    jacobian_cpu = np.stack(
        [
            np.asarray(jac_t, dtype=np.float64) * np.asarray(w_t, dtype=np.float64)[:, None]
            for jac_t, w_t in zip(jacobians, weight)
        ]
    )
    data_rhs_cpu = np.asarray(data_rhs, dtype=np.float64)
    n_times, n_measurements, n_cells = jacobian_cpu.shape
    total_size = n_times * n_cells

    jacobian_gpu = cp.asarray(jacobian_cpu)
    data_rhs_gpu = cp.asarray(data_rhs_cpu)
    rhs_parts = [data_rhs_gpu.reshape(-1)]

    spatial_gpu = None
    if spatial_regularization is not None and spatial_scale > 0.0 and spatial_rhs is not None:
        spatial_cpu = spatial_regularization.tocsr()
        spatial_gpu = cupy_sparse.csr_matrix(
            (
                cp.asarray(spatial_cpu.data, dtype=cp.float64),
                cp.asarray(spatial_cpu.indices, dtype=cp.int32),
                cp.asarray(spatial_cpu.indptr, dtype=cp.int32),
            ),
            shape=spatial_cpu.shape,
        )
        rhs_parts.append(cp.asarray(np.asarray(spatial_rhs, dtype=np.float64).reshape(-1)))

    has_temporal = temporal_scale > 0.0 and temporal_rhs is not None and n_times > 1
    if has_temporal:
        rhs_parts.append(cp.asarray(np.asarray(temporal_rhs, dtype=np.float64).reshape(-1)))

    b = cp.concatenate(rhs_parts)

    def matvec(vector):
        model = vector.reshape((n_times, n_cells))
        parts = [cp.einsum("tmc,tc->tm", jacobian_gpu, model).reshape(-1)]
        if spatial_gpu is not None:
            spatial_rows = [spatial_scale * (spatial_gpu @ model[time_index]) for time_index in range(n_times)]
            parts.append(cp.stack(spatial_rows, axis=0).reshape(-1))
        if has_temporal:
            parts.append((temporal_scale * (model[1:] - model[:-1])).reshape(-1))
        return cp.concatenate(parts)

    def rmatvec(vector):
        offset = 0
        data_size = n_times * n_measurements
        data_part = vector[offset : offset + data_size].reshape((n_times, n_measurements))
        offset += data_size
        gradient = cp.einsum("tmc,tm->tc", jacobian_gpu, data_part)
        if spatial_gpu is not None:
            spatial_rows = int(spatial_gpu.shape[0])
            spatial_part = vector[offset : offset + n_times * spatial_rows].reshape((n_times, spatial_rows))
            offset += n_times * spatial_rows
            for time_index in range(n_times):
                gradient[time_index] += spatial_scale * (spatial_gpu.T @ spatial_part[time_index])
        if has_temporal:
            temporal_part = vector[offset:].reshape((n_times - 1, n_cells))
            gradient[:-1] -= temporal_scale * temporal_part
            gradient[1:] += temporal_scale * temporal_part
        return gradient.reshape(-1)

    x = cp.zeros((total_size,), dtype=cp.float64)
    r = b.copy()
    s = rmatvec(r)
    p = s.copy()
    gamma = cp.dot(s, s)
    gamma0_value = float(gamma)
    if gamma0_value <= 0.0:
        return cp.asnumpy(x)

    for _ in range(int(max_iterations)):
        q = matvec(p)
        denominator = cp.dot(q, q)
        denominator_value = float(denominator)
        if denominator_value <= 0.0:
            break
        alpha = gamma / denominator
        x = x + alpha * p
        r = r - alpha * q
        s = rmatvec(r)
        gamma_new = cp.dot(s, s)
        gamma_new_value = float(gamma_new)
        if gamma_new_value <= 0.0:
            break
        if gamma_new_value / gamma0_value < float(tolerance):
            break
        p = s + (gamma_new / gamma) * p
        gamma = gamma_new

    solution = cp.asnumpy(x)
    if not np.all(np.isfinite(solution)):
        raise ValueError("linearized inversion update contains non-finite values")
    return np.asarray(solution, dtype=float)


def _pyhydro_cgls(
    matrix: sp.spmatrix,
    rhs: np.ndarray,
    *,
    max_iterations: int,
    tolerance: float,
) -> np.ndarray:
    """Replicate PyHydroGeophysX's CGLS routine for the linearized update."""

    system = matrix.tocsr() if sp.issparse(matrix) else np.asarray(matrix, dtype=float)
    b = np.asarray(rhs, dtype=float)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    x = np.zeros((system.shape[1], 1), dtype=float)
    r = b.copy()
    s = system.T.dot(r)
    if np.ndim(s) == 1:
        s = np.asarray(s, dtype=float).reshape(-1, 1)
    else:
        s = np.asarray(s, dtype=float)
    p = s.copy()
    gamma = float((s.T @ s).item())
    rr = float((r.T @ r).item())
    rr0 = rr
    if rr0 <= 0.0 or gamma <= 0.0:
        return x

    for _ in range(int(max_iterations)):
        q = system.dot(p)
        if np.ndim(q) == 1:
            q = np.asarray(q, dtype=float).reshape(-1, 1)
        else:
            q = np.asarray(q, dtype=float)
        denominator = float((q.T @ q).item())
        if denominator <= 0.0:
            break
        alpha = gamma / denominator
        x += alpha * p
        r -= alpha * q
        s = system.T.dot(r)
        if np.ndim(s) == 1:
            s = np.asarray(s, dtype=float).reshape(-1, 1)
        else:
            s = np.asarray(s, dtype=float)
        gamma_new = float((s.T @ s).item())
        if gamma <= 0.0:
            break
        p = s + float(gamma_new / gamma) * p
        gamma = gamma_new
        rr = float((r.T @ r).item())
        if rr / rr0 < float(tolerance):
            break
    return x


def invert_single_log_resistivity(
    forward: ERTForward2p5D | ERTForwardModeling,
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

    ``observed_data`` is interpreted as apparent resistivity unless
    ``observed_log_data=True``. ``initial_model`` and ``reference_model`` are
    interpreted as resistivity unless their corresponding ``*_log_model`` flag
    is set.
    """

    config = config or InversionConfig()
    _check_config(config)

    n_cells = _model_size(forward)
    n_data = _measurement_count(forward)
    observed_log = _as_observed_log_vector(observed_data, log_data=observed_log_data, expected_size=n_data)
    weight = _weights(config.data_std, observed_log.shape)
    initial_log = _as_log_model(
        initial_model,
        expected_size=n_cells,
        log_model=initial_log_model,
        name="initial_model",
    )
    model = _log_model_to_state(initial_log, config)
    if reference_model is None:
        reference = model.copy() if config.spatial_regularization == "identity" else None
    else:
        reference_log = _as_log_model(
            reference_model,
            expected_size=n_cells,
            log_model=reference_log_model,
            name="reference_model",
        )
        reference = _log_model_to_state(reference_log, config)

    iteration_chi2: list[float] = []
    predicted_log = np.empty_like(observed_log)
    jacobian = np.empty((n_data, n_cells), dtype=float)
    linearization_valid = False
    regularization_matrix = _spatial_regularization_matrix(forward, config, n_cells)

    _emit_progress(
        config,
        "single_start",
        n_cells=int(n_cells),
        n_data=int(n_data),
        max_iterations=int(config.max_iterations),
    )
    stop_reason = "max_iterations"
    for iteration_index in range(config.max_iterations):
        iteration = iteration_index + 1
        _emit_progress(
            config,
            "single_iteration_start",
            iteration=int(iteration),
            max_iterations=int(config.max_iterations),
        )
        if not linearization_valid:
            log_model = _state_to_log_model(model, config)
            predicted_log, jacobian_log = _forward_and_jacobian_log(
                forward,
                log_model,
                include_robin_boundary_derivative=config.include_robin_boundary_derivative,
                normal_sensitivity=config.normal_sensitivity,
            )
            jacobian = jacobian_log * _d_log_model_d_state(model, config)[None, :]
        data_matrix = sp.csr_matrix(jacobian * weight[:, None])
        rhs_blocks = [(observed_log - predicted_log) * weight]
        matrix_blocks: list[sp.spmatrix] = [data_matrix]
        reference_roughness = np.zeros(regularization_matrix.shape[0], dtype=float)

        if config.regularization > 0.0:
            scale = float(np.sqrt(config.regularization))
            matrix_blocks.append(scale * regularization_matrix)
            current_roughness = regularization_matrix @ model
            if config.regularization_mode == "update":
                reference_roughness = current_roughness
            elif reference is None:
                if config.spatial_regularization == "identity":
                    reference_roughness = current_roughness
                else:
                    reference_roughness = np.zeros_like(current_roughness)
            else:
                reference_roughness = regularization_matrix @ reference
            rhs_blocks.append(scale * (reference_roughness - current_roughness))

        matrix = sp.vstack(matrix_blocks, format="csr")
        rhs = np.concatenate(rhs_blocks)
        delta = _solve_increment(matrix, rhs, config)
        delta = _limit_delta(delta, config.max_log_step)
        step = config.step_length * delta
        candidate_model = _clip_model_state(model + step, config)
        candidate_step = candidate_model - model

        log_model = _state_to_log_model(candidate_model, config)
        candidate_predicted_log, candidate_jacobian_log = _forward_and_jacobian_log(
            forward,
            log_model,
            include_robin_boundary_derivative=config.include_robin_boundary_derivative,
            normal_sensitivity=config.normal_sensitivity,
        )
        candidate_jacobian = candidate_jacobian_log * _d_log_model_d_state(candidate_model, config)[None, :]

        actual_step = candidate_step
        if config.line_search:
            tau = _line_search_tau(
                state=model,
                step=candidate_step,
                predicted_log=predicted_log,
                candidate_predicted_log=candidate_predicted_log,
                observed_log=observed_log,
                weights=weight,
                regularization_matrix=regularization_matrix,
                reference_roughness=reference_roughness,
                regularization=config.regularization,
            )
            actual_step = tau * candidate_step
            if tau < 0.95:
                model = _clip_model_state(model + actual_step, config)
                log_model = _state_to_log_model(model, config)
                predicted_log, jacobian_log = _forward_and_jacobian_log(
                    forward,
                    log_model,
                    include_robin_boundary_derivative=config.include_robin_boundary_derivative,
                    normal_sensitivity=config.normal_sensitivity,
                )
                jacobian = jacobian_log * _d_log_model_d_state(model, config)[None, :]
            else:
                model = candidate_model
                predicted_log = candidate_predicted_log
                jacobian = candidate_jacobian
        else:
            model = candidate_model
            predicted_log = candidate_predicted_log
            jacobian = candidate_jacobian

        linearization_valid = True
        chi2 = _weighted_chi2(predicted_log, observed_log, weight)
        iteration_chi2.append(chi2)
        step_metric = float(np.linalg.norm(actual_step) / max(float(np.sqrt(n_cells)), 1.0))
        _emit_progress(
            config,
            "single_iteration_done",
            iteration=int(iteration),
            max_iterations=int(config.max_iterations),
            chi2=float(chi2),
            step_norm=step_metric,
            target_chi2=None if config.target_chi2 is None else float(config.target_chi2),
        )
        if config.target_chi2 is not None and chi2 < config.target_chi2:
            stop_reason = "target_chi2"
            break
        if step_metric < config.step_tolerance:
            stop_reason = "step_tolerance"
            break

    coverage = _pygimli_style_coverage_from_jacobian(forward, jacobian)
    predicted_data = np.exp(predicted_log)
    final_log_model = _state_to_log_model(model, config)
    _emit_progress(
        config,
        "single_done",
        iterations=int(len(iteration_chi2)),
        max_iterations=int(config.max_iterations),
        final_chi2=float(iteration_chi2[-1]) if iteration_chi2 else None,
        stop_reason=stop_reason,
    )
    return ERTInversionResult(
        final_model=np.exp(final_log_model),
        final_log_model=final_log_model,
        predicted_data=predicted_data,
        predicted_log_data=predicted_log,
        coverage=coverage,
        iteration_chi2=iteration_chi2,
    )


def _temporal_difference_matrix(n_cells: int, n_times: int, scale: float) -> sp.csr_matrix:
    row_count = n_cells * (n_times - 1)
    col_count = n_cells * n_times
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for time_index in range(1, n_times):
        row_offset = (time_index - 1) * n_cells
        previous_offset = (time_index - 1) * n_cells
        current_offset = time_index * n_cells
        for cell_index in range(n_cells):
            row = row_offset + cell_index
            rows.extend((row, row))
            cols.extend((current_offset + cell_index, previous_offset + cell_index))
            data.extend((scale, -scale))
    return sp.coo_matrix((data, (rows, cols)), shape=(row_count, col_count)).tocsr()


def invert_timelapse_log_resistivity(
    forward: ERTForward2p5D | ERTForwardModeling,
    observed_data: ArrayLike,
    initial_model: ArrayLike,
    *,
    reference_model: ArrayLike | None = None,
    config: InversionConfig | None = None,
    observed_log_data: bool = False,
    initial_log_model: bool = False,
    reference_log_model: bool = False,
    _forward_jacobian_cache: ForwardJacobianCache | None = None,
    _forward_jacobian_cache_max_entries: int = 128,
) -> TimeLapseERTInversionResult:
    """Jointly invert time-lapse ERT data with optional temporal smoothing.

    Observations are accepted as ``(n_times, n_measurements)`` or
    ``(n_measurements, n_times)``. Returned models are shaped
    ``(n_cells, n_times)`` to match the notebook artifact convention.
    """

    config = config or InversionConfig(temporal_regularization=1.0)
    _check_config(config)

    n_cells = _model_size(forward)
    n_measurements = _measurement_count(forward)
    observed_log = _as_observed_log_matrix(
        observed_data,
        log_data=observed_log_data,
        measurement_count=n_measurements,
    )
    n_times = int(observed_log.shape[0])
    if n_times < 2:
        raise ValueError("time-lapse inversion needs at least two timesteps")

    weight = _weights(config.data_std, observed_log.shape)
    initial_logs = _as_log_model_matrix(
        initial_model,
        expected_size=n_cells,
        n_times=n_times,
        log_model=initial_log_model,
        name="initial_model",
    )
    models = _log_model_to_state(initial_logs, config)
    if reference_model is None:
        reference = (
            models.copy()
            if config.spatial_regularization == "identity"
            and config.temporal_regularization_mode == "separate"
            else None
        )
    else:
        reference_logs = _as_log_model_matrix(
            reference_model,
            expected_size=n_cells,
            n_times=n_times,
            log_model=reference_log_model,
            name="reference_model",
        )
        reference = _log_model_to_state(reference_logs, config)

    total_size = n_cells * n_times
    iteration_chi2: list[float] = []
    predicted_log = np.empty_like(observed_log)
    jacobians: list[np.ndarray] = []
    linearization_valid = False
    spatial_regularization = _spatial_regularization_matrix(forward, config, n_cells)
    spatial_regularization_all = sp.block_diag(
        [spatial_regularization] * n_times,
        format="csr",
    )
    temporal_difference = _temporal_difference_matrix(n_cells, n_times, 1.0)

    _emit_progress(
        config,
        "timelapse_start",
        n_cells=int(n_cells),
        n_measurements=int(n_measurements),
        n_times=int(n_times),
        max_iterations=int(config.max_iterations),
    )
    stop_reason = "max_iterations"
    for iteration_index in range(config.max_iterations):
        iteration = iteration_index + 1
        _emit_progress(
            config,
            "timelapse_iteration_start",
            iteration=int(iteration),
            max_iterations=int(config.max_iterations),
            n_times=int(n_times),
        )
        if not linearization_valid:
            predicted_rows: list[np.ndarray] = []
            jacobians = []
            for time_index in range(n_times):
                _emit_progress(
                    config,
                    "timelapse_time_start",
                    iteration=int(iteration),
                    max_iterations=int(config.max_iterations),
                    stage="linearization",
                    time_index=int(time_index),
                    time_number=int(time_index + 1),
                    n_times=int(n_times),
                )
                state_t = models[:, time_index]
                pred_t, jac_log_t = _forward_and_jacobian_log_cached(
                    forward,
                    _state_to_log_model(state_t, config),
                    include_robin_boundary_derivative=config.include_robin_boundary_derivative,
                    normal_sensitivity=config.normal_sensitivity,
                    cache=_forward_jacobian_cache,
                    max_entries=_forward_jacobian_cache_max_entries,
                )
                predicted_rows.append(pred_t)
                jacobians.append(jac_log_t * _d_log_model_d_state(state_t, config)[None, :])
                _emit_progress(
                    config,
                    "timelapse_time_done",
                    iteration=int(iteration),
                    max_iterations=int(config.max_iterations),
                    stage="linearization",
                    time_index=int(time_index),
                    time_number=int(time_index + 1),
                    n_times=int(n_times),
                )
            predicted_log = np.vstack(predicted_rows)

        data_blocks: list[sp.csr_matrix] = []
        rhs_blocks: list[np.ndarray] = []
        for time_index, jac_t in enumerate(jacobians):
            w_t = weight[time_index]
            data_blocks.append(sp.csr_matrix(jac_t * w_t[:, None]))
            rhs_blocks.append((observed_log[time_index] - predicted_log[time_index]) * w_t)
        data_rhs_matrix = np.vstack(rhs_blocks)

        matrix_blocks: list[sp.spmatrix] = [sp.block_diag(data_blocks, format="csr")]
        rhs_all: list[np.ndarray] = [np.concatenate(rhs_blocks)]
        objective_blocks: list[sp.spmatrix] = []
        objective_references: list[np.ndarray] = []
        gpu_spatial_rhs: np.ndarray | None = None
        gpu_spatial_scale = 0.0
        gpu_temporal_rhs: np.ndarray | None = None
        gpu_temporal_scale = 0.0

        current_vec = models.reshape(total_size, order="F")
        reference_vec = None if reference is None else reference.reshape(total_size, order="F")
        if config.temporal_regularization_mode == "joint_frame":
            if config.regularization > 0.0:
                frame_blocks: list[sp.spmatrix] = [spatial_regularization_all]
                if config.temporal_regularization > 0.0:
                    frame_blocks.append(
                        _temporal_difference_matrix(
                            n_cells,
                            n_times,
                            float(config.temporal_regularization),
                        )
                    )
                frame_constraint = sp.vstack(frame_blocks, format="csr")
                current_roughness = frame_constraint @ current_vec
                if config.regularization_mode == "update":
                    reference_roughness = current_roughness
                elif reference_vec is None:
                    reference_roughness = np.zeros_like(current_roughness)
                else:
                    reference_roughness = frame_constraint @ reference_vec
                scale = float(np.sqrt(config.regularization))
                matrix_blocks.append(scale * frame_constraint)
                rhs_all.append(scale * (reference_roughness - current_roughness))
                objective_blocks.append(scale * frame_constraint)
                objective_references.append(scale * reference_roughness)
        elif config.regularization > 0.0:
            scale = float(np.sqrt(config.regularization))
            matrix_blocks.append(scale * spatial_regularization_all)
            current_roughness = spatial_regularization_all @ current_vec
            if config.regularization_mode == "update":
                reference_roughness = current_roughness
            elif reference is None:
                if config.spatial_regularization == "identity":
                    reference_roughness = current_roughness
                else:
                    reference_roughness = np.zeros_like(current_roughness)
            else:
                reference_roughness = spatial_regularization_all @ reference_vec
            rhs_all.append(scale * (reference_roughness - current_roughness))
            objective_blocks.append(scale * spatial_regularization_all)
            objective_references.append(scale * reference_roughness)
            gpu_spatial_scale = scale
            gpu_spatial_rhs = (scale * (reference_roughness - current_roughness)).reshape(
                (n_times, spatial_regularization.shape[0])
            )

        if config.temporal_regularization_mode == "separate" and config.temporal_regularization > 0.0:
            scale = float(np.sqrt(config.temporal_regularization))
            matrix_blocks.append(scale * temporal_difference)
            temporal_roughness = temporal_difference @ current_vec
            temporal_reference = np.zeros_like(temporal_roughness)
            rhs_all.append(scale * (temporal_reference - temporal_roughness))
            objective_blocks.append(scale * temporal_difference)
            objective_references.append(scale * temporal_reference)
            gpu_temporal_scale = scale
            gpu_temporal_rhs = (scale * (temporal_reference - temporal_roughness)).reshape((n_times - 1, n_cells))

        if config.linearized_solver == "gpu_timelapse_cgls" and config.temporal_regularization_mode == "separate":
            delta_vec = _cupy_timelapse_cgls(
                jacobians=jacobians,
                weight=weight,
                data_rhs=data_rhs_matrix,
                spatial_regularization=spatial_regularization if config.regularization > 0.0 else None,
                spatial_scale=gpu_spatial_scale,
                spatial_rhs=gpu_spatial_rhs,
                temporal_scale=gpu_temporal_scale,
                temporal_rhs=gpu_temporal_rhs,
                max_iterations=config.cgls_max_iterations,
                tolerance=config.cgls_tolerance,
            )
        else:
            matrix = sp.vstack(matrix_blocks, format="csr")
            rhs = np.concatenate(rhs_all)
            delta_vec = _solve_increment(matrix, rhs, config)
        delta_vec = _limit_delta(delta_vec, config.max_log_step)
        delta = delta_vec.reshape((n_cells, n_times), order="F")
        step = config.step_length * delta
        candidate_models = _clip_model_state(models + step, config)
        candidate_step_vec = candidate_models.reshape(total_size, order="F") - current_vec

        candidate_rows = []
        candidate_jacobians = []
        for time_index in range(n_times):
            _emit_progress(
                config,
                "timelapse_time_start",
                iteration=int(iteration),
                max_iterations=int(config.max_iterations),
                stage="candidate",
                time_index=int(time_index),
                time_number=int(time_index + 1),
                n_times=int(n_times),
            )
            state_t = candidate_models[:, time_index]
            pred_t, jac_log_t = _forward_and_jacobian_log_cached(
                forward,
                _state_to_log_model(state_t, config),
                include_robin_boundary_derivative=config.include_robin_boundary_derivative,
                normal_sensitivity=config.normal_sensitivity,
                cache=_forward_jacobian_cache,
                max_entries=_forward_jacobian_cache_max_entries,
            )
            candidate_rows.append(pred_t)
            candidate_jacobians.append(jac_log_t * _d_log_model_d_state(state_t, config)[None, :])
            _emit_progress(
                config,
                "timelapse_time_done",
                iteration=int(iteration),
                max_iterations=int(config.max_iterations),
                stage="candidate",
                time_index=int(time_index),
                time_number=int(time_index + 1),
                n_times=int(n_times),
            )
        candidate_predicted_log = np.vstack(candidate_rows)

        actual_step_vec = candidate_step_vec
        if config.line_search:
            if objective_blocks:
                objective_matrix = sp.vstack(objective_blocks, format="csr")
                objective_reference = np.concatenate(objective_references)
            else:
                objective_matrix = sp.csr_matrix((0, total_size))
                objective_reference = np.zeros((0,), dtype=float)
            tau = _line_search_tau(
                state=current_vec,
                step=candidate_step_vec,
                predicted_log=predicted_log.ravel(),
                candidate_predicted_log=candidate_predicted_log.ravel(),
                observed_log=observed_log.ravel(),
                weights=weight.ravel(),
                regularization_matrix=objective_matrix,
                reference_roughness=objective_reference,
                regularization=1.0,
            )
            actual_step_vec = tau * candidate_step_vec
            if tau < 0.95:
                models = _clip_model_state((current_vec + actual_step_vec).reshape((n_cells, n_times), order="F"), config)
                predicted_rows = []
                jacobians = []
                for time_index in range(n_times):
                    _emit_progress(
                        config,
                        "timelapse_time_start",
                        iteration=int(iteration),
                        max_iterations=int(config.max_iterations),
                        stage="line_search",
                        time_index=int(time_index),
                        time_number=int(time_index + 1),
                        n_times=int(n_times),
                    )
                    state_t = models[:, time_index]
                    pred_t, jac_log_t = _forward_and_jacobian_log_cached(
                        forward,
                        _state_to_log_model(state_t, config),
                        include_robin_boundary_derivative=config.include_robin_boundary_derivative,
                        normal_sensitivity=config.normal_sensitivity,
                        cache=_forward_jacobian_cache,
                        max_entries=_forward_jacobian_cache_max_entries,
                    )
                    predicted_rows.append(pred_t)
                    jacobians.append(jac_log_t * _d_log_model_d_state(state_t, config)[None, :])
                    _emit_progress(
                        config,
                        "timelapse_time_done",
                        iteration=int(iteration),
                        max_iterations=int(config.max_iterations),
                        stage="line_search",
                        time_index=int(time_index),
                        time_number=int(time_index + 1),
                        n_times=int(n_times),
                    )
                predicted_log = np.vstack(predicted_rows)
            else:
                models = candidate_models
                predicted_log = candidate_predicted_log
                jacobians = candidate_jacobians
        else:
            models = candidate_models
            predicted_log = candidate_predicted_log
            jacobians = candidate_jacobians

        linearization_valid = True
        chi2 = _weighted_chi2(predicted_log, observed_log, weight)
        iteration_chi2.append(chi2)
        step_metric = float(np.linalg.norm(actual_step_vec) / max(float(np.sqrt(total_size)), 1.0))
        _emit_progress(
            config,
            "timelapse_iteration_done",
            iteration=int(iteration),
            max_iterations=int(config.max_iterations),
            n_times=int(n_times),
            chi2=float(chi2),
            step_norm=step_metric,
            target_chi2=None if config.target_chi2 is None else float(config.target_chi2),
        )
        if config.target_chi2 is not None and chi2 < config.target_chi2:
            stop_reason = "target_chi2"
            break
        if step_metric < config.step_tolerance:
            stop_reason = "step_tolerance"
            break

    all_coverage = [_pygimli_style_coverage_from_jacobian(forward, jac_t) for jac_t in jacobians]
    coverage = np.nanmedian(np.column_stack(all_coverage), axis=1)
    final_log_models = _state_to_log_model(models, config)
    _emit_progress(
        config,
        "timelapse_done",
        iterations=int(len(iteration_chi2)),
        max_iterations=int(config.max_iterations),
        final_chi2=float(iteration_chi2[-1]) if iteration_chi2 else None,
        stop_reason=stop_reason,
    )
    return TimeLapseERTInversionResult(
        final_models=np.exp(final_log_models),
        final_log_models=final_log_models,
        predicted_data=np.exp(predicted_log),
        predicted_log_data=predicted_log,
        coverage=coverage,
        all_coverage=all_coverage,
        all_chi2=np.asarray(iteration_chi2, dtype=float),
        iteration_chi2=iteration_chi2,
    )


def _window_start_indices(n_times: int, window_size: int, window_step: int) -> list[int]:
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    if window_size > n_times:
        raise ValueError(f"window_size={window_size} exceeds n_times={n_times}")
    step = max(1, int(window_step))
    starts = list(range(0, n_times - window_size + 1, step))
    tail_start = n_times - window_size
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return sorted(set(starts))


def _config_for_time_window(
    config: InversionConfig,
    *,
    observed_shape: tuple[int, int],
    start: int,
    end: int,
) -> InversionConfig:
    data_std = np.asarray(config.data_std, dtype=float)
    if data_std.ndim == 0:
        return config
    window_std = np.broadcast_to(data_std, observed_shape)[start:end].copy()
    return replace(config, data_std=window_std)


def invert_windowed_timelapse_log_resistivity(
    forward: ERTForward2p5D | ERTForwardModeling,
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
    """Run notebook-style sliding-window time-lapse inversion.

    Every overlapping window is inverted independently with
    :func:`invert_timelapse_log_resistivity`. The global model for each
    timestep is the geometric mean of all window contributions, matching the
    project notebook aggregation.
    """

    config = config or InversionConfig(temporal_regularization=1.0)
    _check_config(config)

    n_cells = _model_size(forward)
    n_measurements = _measurement_count(forward)
    observed_log = _as_observed_log_matrix(
        observed_data,
        log_data=observed_log_data,
        measurement_count=n_measurements,
    )
    n_times = int(observed_log.shape[0])
    starts = _window_start_indices(n_times, int(window_size), int(window_step))
    initial_logs = _as_log_model_matrix(
        initial_model,
        expected_size=n_cells,
        n_times=n_times,
        log_model=initial_log_model,
        name="initial_model",
    )
    reference_logs = None
    if reference_model is not None:
        reference_logs = _as_log_model_matrix(
            reference_model,
            expected_size=n_cells,
            n_times=n_times,
            log_model=reference_log_model,
            name="reference_model",
        )

    contributions: list[list[np.ndarray]] = [[] for _ in range(n_times)]
    coverage_bank: list[np.ndarray] = []
    window_final_chi2: list[float] = []
    window_reports: list[dict[str, float | int | None]] = []
    forward_jacobian_cache: ForwardJacobianCache = OrderedDict()
    forward_jacobian_cache_entries = max(
        16,
        min(128, int(window_size) * max(1, int(config.max_iterations) + 2) * 4),
    )

    _emit_progress(
        config,
        "windowed_start",
        n_cells=int(n_cells),
        n_measurements=int(n_measurements),
        n_times=int(n_times),
        n_windows=int(len(starts)),
        window_size=int(window_size),
        window_step=int(window_step),
        max_iterations=int(config.max_iterations),
    )
    for window_index, start in enumerate(starts, start=1):
        end = start + int(window_size)
        _emit_progress(
            config,
            "window_start",
            window_index=int(window_index),
            n_windows=int(len(starts)),
            start_idx=int(start),
            end_idx=int(end - 1),
            window_size=int(window_size),
            max_iterations=int(config.max_iterations),
        )
        window_config = _config_for_time_window(
            config,
            observed_shape=observed_log.shape,
            start=start,
            end=end,
        )
        window_start_time = time.perf_counter()
        window_result = invert_timelapse_log_resistivity(
            forward,
            observed_log[start:end],
            initial_logs[:, start:end],
            reference_model=None if reference_logs is None else reference_logs[:, start:end],
            config=window_config,
            observed_log_data=True,
            initial_log_model=True,
            reference_log_model=True,
            _forward_jacobian_cache=forward_jacobian_cache,
            _forward_jacobian_cache_max_entries=forward_jacobian_cache_entries,
        )
        window_elapsed_sec = time.perf_counter() - window_start_time
        for local_index in range(window_result.final_log_models.shape[1]):
            global_index = start + local_index
            contributions[global_index].append(window_result.final_log_models[:, local_index])
        if window_result.coverage is not None:
            coverage_bank.append(np.asarray(window_result.coverage, dtype=float).ravel())
        final_chi2 = float(window_result.iteration_chi2[-1]) if window_result.iteration_chi2 else None
        if final_chi2 is not None:
            window_final_chi2.append(final_chi2)
        window_reports.append(
            {
                "start_idx": int(start),
                "end_idx": int(end - 1),
                "final_chi2_data": final_chi2,
                "iterations": int(len(window_result.iteration_chi2)),
                "elapsed_sec": float(window_elapsed_sec),
            }
        )
        _emit_progress(
            config,
            "window_done",
            window_index=int(window_index),
            n_windows=int(len(starts)),
            start_idx=int(start),
            end_idx=int(end - 1),
            final_chi2=final_chi2,
        )

    final_log_columns: list[np.ndarray] = []
    for time_index, timestep_contributions in enumerate(contributions):
        if not timestep_contributions:
            raise ValueError(f"no window contribution for timestep index={time_index}")
        stack = np.column_stack(timestep_contributions)
        final_log_columns.append(np.mean(stack, axis=1))
    final_log_models = np.column_stack(final_log_columns)

    _emit_progress(
        config,
        "windowed_prediction_start",
        n_times=int(n_times),
    )
    predicted_rows: list[np.ndarray] = []
    for time_index in range(n_times):
        _emit_progress(
            config,
            "windowed_prediction_step",
            time_index=int(time_index),
            time_number=int(time_index + 1),
            n_times=int(n_times),
        )
        predicted_rows.append(_forward_log_response(forward, final_log_models[:, time_index]))
    predicted_log = np.vstack(predicted_rows)
    if coverage_bank:
        coverage = np.nanmedian(np.column_stack(coverage_bank), axis=1)
        all_coverage = coverage_bank
    else:
        coverage = np.zeros((n_cells,), dtype=float)
        all_coverage = []

    _emit_progress(
        config,
        "windowed_done",
        n_windows=int(len(starts)),
        final_chi2=float(window_final_chi2[-1]) if window_final_chi2 else None,
    )
    return TimeLapseERTInversionResult(
        final_models=np.exp(final_log_models),
        final_log_models=final_log_models,
        predicted_data=np.exp(predicted_log),
        predicted_log_data=predicted_log,
        coverage=coverage,
        all_coverage=all_coverage,
        all_chi2=np.asarray(window_final_chi2, dtype=float),
        iteration_chi2=window_final_chi2,
        window_reports=window_reports,
    )


@dataclass
class ERTInversion:
    """Small class wrapper matching the notebook-style ``setup/run`` flow."""

    forward: ERTForward2p5D | ERTForwardModeling
    observed_data: ArrayLike
    config: InversionConfig = field(default_factory=InversionConfig)
    observed_log_data: bool = False

    def setup(self) -> "ERTInversion":
        """Validate basic dimensions and return ``self`` for notebook ergonomics."""

        _check_config(self.config)
        _as_observed_log_vector(
            self.observed_data,
            log_data=self.observed_log_data,
            expected_size=_measurement_count(self.forward),
        )
        return self

    def run(
        self,
        initial_model: ArrayLike,
        *,
        reference_model: ArrayLike | None = None,
        initial_log_model: bool = False,
        reference_log_model: bool = False,
    ) -> ERTInversionResult:
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
    """Notebook-style wrapper for joint time-lapse log-resistivity inversion."""

    forward: ERTForward2p5D | ERTForwardModeling
    observed_data: ArrayLike
    config: InversionConfig = field(default_factory=lambda: InversionConfig(temporal_regularization=1.0))
    observed_log_data: bool = False

    def setup(self) -> "TimeLapseERTInversion":
        """Validate basic dimensions and return ``self`` for notebook ergonomics."""

        _check_config(self.config)
        _as_observed_log_matrix(
            self.observed_data,
            log_data=self.observed_log_data,
            measurement_count=_measurement_count(self.forward),
        )
        return self

    def run(
        self,
        initial_model: ArrayLike,
        *,
        reference_model: ArrayLike | None = None,
        initial_log_model: bool = False,
        reference_log_model: bool = False,
    ) -> TimeLapseERTInversionResult:
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
    """Notebook-style sliding-window wrapper for large time-lapse inversions."""

    forward: ERTForward2p5D | ERTForwardModeling
    observed_data: ArrayLike
    config: InversionConfig = field(default_factory=lambda: InversionConfig(temporal_regularization=1.0))
    window_size: int = 3
    window_step: int = 1
    observed_log_data: bool = False

    def setup(self) -> "WindowedTimeLapseERTInversion":
        """Validate dimensions and window controls."""

        _check_config(self.config)
        observed_log = _as_observed_log_matrix(
            self.observed_data,
            log_data=self.observed_log_data,
            measurement_count=_measurement_count(self.forward),
        )
        _window_start_indices(
            int(observed_log.shape[0]),
            int(self.window_size),
            int(self.window_step),
        )
        return self

    def run(
        self,
        initial_model: ArrayLike,
        *,
        reference_model: ArrayLike | None = None,
        initial_log_model: bool = False,
        reference_log_model: bool = False,
    ) -> TimeLapseERTInversionResult:
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
