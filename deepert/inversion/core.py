"""Log-space ERT inversion routines built on the native differentiable forward path."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from deepert.forward import ERTForward2p5D, ERTForwardModeling
from deepert.mesh import Mesh
from deepert.utils.dtypes import FLOAT_DTYPE


ArrayLike = Any


@dataclass(frozen=True)
class InversionConfig:
    """Controls for damped Gauss-Newton inversion in log-resistivity space."""

    max_iterations: int = 8
    data_std: float | ArrayLike = 0.05
    regularization: float = 1.0e-2
    temporal_regularization: float = 0.0
    temporal_regularization_mode: str = "separate"
    spatial_regularization: str = "identity"
    z_weight: float = 1.0
    model_transform: str = "log"
    model_bounds: tuple[float, float] | None = None
    step_length: float = 1.0
    max_log_step: float | None = 1.0
    line_search: bool = False
    step_tolerance: float = 1.0e-4
    lsqr_atol: float = 1.0e-6
    lsqr_btol: float = 1.0e-6
    lsqr_iter_limit: int | None = None
    include_robin_boundary_derivative: bool = False
    normal_sensitivity: bool = True


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


def _check_config(config: InversionConfig) -> None:
    if config.max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if config.regularization < 0.0:
        raise ValueError("regularization must be non-negative")
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


def _coverage_from_jacobian(jacobian: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weighted_jacobian = jacobian * weights[:, None]
    return np.sqrt(np.sum(weighted_jacobian**2, axis=0))


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

    for _ in range(config.max_iterations):
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
            if reference is None:
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
        iteration_chi2.append(_weighted_chi2(predicted_log, observed_log, weight))
        if np.linalg.norm(actual_step) / max(float(np.sqrt(n_cells)), 1.0) < config.step_tolerance:
            break

    coverage = _coverage_from_jacobian(jacobian, weight)
    predicted_data = np.exp(predicted_log)
    final_log_model = _state_to_log_model(model, config)
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

    for _ in range(config.max_iterations):
        if not linearization_valid:
            predicted_rows: list[np.ndarray] = []
            jacobians = []
            for time_index in range(n_times):
                state_t = models[:, time_index]
                pred_t, jac_log_t = _forward_and_jacobian_log(
                    forward,
                    _state_to_log_model(state_t, config),
                    include_robin_boundary_derivative=config.include_robin_boundary_derivative,
                    normal_sensitivity=config.normal_sensitivity,
                )
                predicted_rows.append(pred_t)
                jacobians.append(jac_log_t * _d_log_model_d_state(state_t, config)[None, :])
            predicted_log = np.vstack(predicted_rows)

        data_blocks: list[sp.csr_matrix] = []
        rhs_blocks: list[np.ndarray] = []
        for time_index, jac_t in enumerate(jacobians):
            w_t = weight[time_index]
            data_blocks.append(sp.csr_matrix(jac_t * w_t[:, None]))
            rhs_blocks.append((observed_log[time_index] - predicted_log[time_index]) * w_t)

        matrix_blocks: list[sp.spmatrix] = [sp.block_diag(data_blocks, format="csr")]
        rhs_all: list[np.ndarray] = [np.concatenate(rhs_blocks)]
        objective_blocks: list[sp.spmatrix] = []
        objective_references: list[np.ndarray] = []

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
                if reference_vec is None:
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
            if reference is None:
                if config.spatial_regularization == "identity":
                    reference_roughness = current_roughness
                else:
                    reference_roughness = np.zeros_like(current_roughness)
            else:
                reference_roughness = spatial_regularization_all @ reference_vec
            rhs_all.append(scale * (reference_roughness - current_roughness))
            objective_blocks.append(scale * spatial_regularization_all)
            objective_references.append(scale * reference_roughness)

        if config.temporal_regularization_mode == "separate" and config.temporal_regularization > 0.0:
            scale = float(np.sqrt(config.temporal_regularization))
            matrix_blocks.append(scale * temporal_difference)
            temporal_roughness = temporal_difference @ current_vec
            temporal_reference = np.zeros_like(temporal_roughness)
            rhs_all.append(scale * (temporal_reference - temporal_roughness))
            objective_blocks.append(scale * temporal_difference)
            objective_references.append(scale * temporal_reference)

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
            state_t = candidate_models[:, time_index]
            pred_t, jac_log_t = _forward_and_jacobian_log(
                forward,
                _state_to_log_model(state_t, config),
                include_robin_boundary_derivative=config.include_robin_boundary_derivative,
                normal_sensitivity=config.normal_sensitivity,
            )
            candidate_rows.append(pred_t)
            candidate_jacobians.append(jac_log_t * _d_log_model_d_state(state_t, config)[None, :])
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
                    state_t = models[:, time_index]
                    pred_t, jac_log_t = _forward_and_jacobian_log(
                        forward,
                        _state_to_log_model(state_t, config),
                        include_robin_boundary_derivative=config.include_robin_boundary_derivative,
                        normal_sensitivity=config.normal_sensitivity,
                    )
                    predicted_rows.append(pred_t)
                    jacobians.append(jac_log_t * _d_log_model_d_state(state_t, config)[None, :])
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
        iteration_chi2.append(_weighted_chi2(predicted_log, observed_log, weight))
        if np.linalg.norm(actual_step_vec) / max(float(np.sqrt(total_size)), 1.0) < config.step_tolerance:
            break

    all_coverage = [
        _coverage_from_jacobian(jac_t, weight[time_index])
        for time_index, jac_t in enumerate(jacobians)
    ]
    coverage = np.nanmedian(np.column_stack(all_coverage), axis=1)
    final_log_models = _state_to_log_model(models, config)
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

    for start in starts:
        end = start + int(window_size)
        window_config = _config_for_time_window(
            config,
            observed_shape=observed_log.shape,
            start=start,
            end=end,
        )
        window_result = invert_timelapse_log_resistivity(
            forward,
            observed_log[start:end],
            initial_logs[:, start:end],
            reference_model=None if reference_logs is None else reference_logs[:, start:end],
            config=window_config,
            observed_log_data=True,
            initial_log_model=True,
            reference_log_model=True,
        )
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
            }
        )

    final_log_columns: list[np.ndarray] = []
    for time_index, timestep_contributions in enumerate(contributions):
        if not timestep_contributions:
            raise ValueError(f"no window contribution for timestep index={time_index}")
        stack = np.column_stack(timestep_contributions)
        final_log_columns.append(np.mean(stack, axis=1))
    final_log_models = np.column_stack(final_log_columns)

    predicted_log = np.vstack(
        [
            _forward_log_response(forward, final_log_models[:, time_index])
            for time_index in range(n_times)
        ]
    )
    if coverage_bank:
        coverage = np.nanmedian(np.column_stack(coverage_bank), axis=1)
        all_coverage = coverage_bank
    else:
        coverage = np.zeros((n_cells,), dtype=float)
        all_coverage = []

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
