"""ERT forward modelling facade without external modelling dependencies."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from adtlert.forward.ert2p5d import ERTForward2p5D
from adtlert.forward.ert3d import ERTForward3D
from adtlert.mesh import Mesh, Mesh3D
from adtlert.survey import Survey
from adtlert.utils.dtypes import FLOAT_DTYPE


def _call(value: Any) -> Any:
    return value() if callable(value) else value


def _point_coordinates(point: Any, *, dimension: int | None = None) -> list[float]:
    if hasattr(point, "pos"):
        point = point.pos()
    try:
        coordinate_count = len(point)
    except TypeError:
        coordinate_count = 3 if hasattr(point, "z") else 2
    width = dimension or (3 if coordinate_count >= 3 else 2)
    return [float(point[index]) for index in range(width)]


def _cell_node_ids(cell: Any) -> list[int]:
    if hasattr(cell, "nodeCount") and hasattr(cell, "node"):
        node_ids = [int(cell.node(local_id).id()) for local_id in range(int(cell.nodeCount()))]
    else:
        node_ids = [int(node_id) for node_id in cell]
    if len(node_ids) < 3:
        raise ValueError("mesh cells must have at least three nodes")
    if len(node_ids) > 4:
        raise ValueError("only triangular and quadrilateral cells are supported")
    return node_ids


def _mesh_from_arrays(nodes, cells) -> Mesh | Mesh3D:
    nodes = np.asarray(nodes)
    return (Mesh3D if nodes.ndim == 2 and nodes.shape[1] == 3 else Mesh).from_arrays(nodes, cells)


def mesh_to_adtlert(mesh: Any) -> Mesh | Mesh3D:
    """Convert common mesh-like objects into :class:`adtlert.mesh.Mesh` or :class:`Mesh3D`.

    Supported inputs are ``Mesh``/``Mesh3D`` instances, ``(nodes, cells)`` tuples, meshio
    meshes, and objects exposing ``nodes``/``cells`` arrays or methods (duck typing only).
    """

    if isinstance(mesh, (Mesh, Mesh3D)):
        return mesh
    if isinstance(mesh, tuple) and len(mesh) == 2:
        return _mesh_from_arrays(*mesh)
    if hasattr(mesh, "cells_dict"):
        return (Mesh3D if "tetra" in mesh.cells_dict else Mesh).from_meshio(mesh)
    if getattr(mesh, "nodes", None) is None or getattr(mesh, "cells", None) is None:
        raise TypeError("mesh must be a Mesh, (nodes, cells), meshio mesh, or mesh-like object")
    dimension = _call(getattr(mesh, "dimension", getattr(mesh, "dim", None)))
    dimension = None if dimension is None else int(dimension)
    nodes = np.asarray([_point_coordinates(node, dimension=dimension) for node in _call(mesh.nodes)], dtype=float)
    cells = np.asarray([_cell_node_ids(cell) for cell in _call(mesh.cells)], dtype=np.int32)
    return _mesh_from_arrays(nodes, cells)


def survey_to_adtlert(data: Any, *, dimension: int | None = None) -> Survey:
    """Convert common survey/data-like objects into :class:`adtlert.survey.Survey`.

    Supported inputs are ``Survey`` instances, ``(electrodes, abmn)`` tuples, objects with
    ``electrode_positions``/``measurements``, and DataContainer-like objects exposing
    ``sensors()`` or ``sensorPositions()`` plus ``a/b/m/n`` fields.
    """

    if isinstance(data, Survey):
        return data
    if isinstance(data, tuple) and len(data) == 2:
        positions, measurements = data
    elif hasattr(data, "electrode_positions") and hasattr(data, "measurements"):
        positions, measurements = data.electrode_positions, data.measurements
    else:
        sensors = getattr(data, "sensors", None)
        sensors = getattr(data, "sensorPositions", None) if sensors is None else sensors
        if sensors is None:
            raise TypeError("data must be a Survey, (electrodes, abmn), or data-like object")
        electrodes = [_point_coordinates(sensor, dimension=dimension) for sensor in _call(sensors)]
        measurements = np.column_stack([np.asarray(data[key], dtype=np.int32) for key in "abmn"])
        return Survey.from_arrays(np.asarray(electrodes, dtype=float), measurements)
    positions = np.asarray(positions, dtype=float)
    return Survey.from_arrays(positions if dimension is None else positions[:, :dimension], measurements)


def _resistivity(model: Any, *, log_transform: bool, expected_size: int) -> np.ndarray:
    values = np.asarray(model, dtype=float).ravel()
    if log_transform:
        if not np.all(np.isfinite(values)):
            raise ValueError("resistivity_model contains non-finite log-resistivity values")
        values = np.exp(values)
    if values.shape != (expected_size,):
        raise ValueError(f"resistivity_model must have shape ({expected_size},)")
    if not np.all(np.isfinite(values)):
        raise ValueError("resistivity_model contains non-finite resistivity values")
    if np.any(values <= 0.0):
        raise ValueError(f"resistivity_model must contain positive resistivity values (min={float(np.min(values)):.6e})")
    return values


def log_response_and_jacobian(operator, conductivity, *, chain=None, **solve_kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Return ``log(rhoa)`` and ``d log(rhoa) / d log(rho)`` from an explicit resistance Jacobian.

    ``chain`` is the conductivity of each Jacobian column (defaults to ``conductivity``),
    which differs when the operator aggregates columns into inversion parameters.
    """

    response, resistance_jacobian = operator.solve_with_jacobian(conductivity, **solve_kwargs)
    apparent_jacobian = torch.abs(torch.as_tensor(operator._geometric_factors()))[:, None] * resistance_jacobian
    chain = conductivity if chain is None else torch.as_tensor(chain, dtype=apparent_jacobian.dtype)
    jacobian = apparent_jacobian * (-chain[None, :]) / response.apparent_resistivity[:, None]
    return np.log(np.asarray(response.apparent_resistivity, dtype=float)), np.asarray(jacobian, dtype=float)


@dataclass
class ERTForwardModeling:
    """Small forward wrapper with the common ``setData/setMesh/forward`` shape."""

    mesh: Any | None = None
    data: Any | None = None
    quadrature_order: int = 2
    numerical_h2_refined: bool = True
    numerical_p2_refined: bool = True
    topographic_geometric_factor_mode: str = "analytic"
    terrain_cache_dir: str | Path | None = None
    include_robin_boundary_derivative: bool = False
    normal_sensitivity: bool = True
    boundary_mode_3d: str = "mixed"
    singularity_removal_3d: bool = True
    element_order_3d: int = 1
    geometric_factor_mode_3d: str = "auto"

    def __post_init__(self) -> None:
        self._forward: ERTForward2p5D | ERTForward3D | None = None
        self._mesh: Mesh | Mesh3D | None = None
        self._survey: Survey | None = None

    def set_data(self, data: Any) -> None:
        """Set the ERT survey/data object."""

        self.data, self._survey, self._forward = data, None, None

    def set_mesh(self, mesh: Any) -> None:
        """Set the forward mesh."""

        self.mesh, self._mesh, self._forward = mesh, None, None

    setData = set_data  # noqa: N815 - compatibility with common ERT APIs
    setMesh = set_mesh  # noqa: N815

    def _resolved_mesh(self) -> Mesh | Mesh3D:
        if self.mesh is None:
            raise ValueError("mesh has not been set")
        if self._mesh is None:
            self._mesh = mesh_to_adtlert(self.mesh)
        return self._mesh

    def _resolved_survey(self) -> Survey:
        if self.data is None:
            raise ValueError("data has not been set")
        if self._survey is None:
            self._survey = survey_to_adtlert(self.data, dimension=3 if isinstance(self._resolved_mesh(), Mesh3D) else 2)
        return self._survey

    @property
    def cell_count(self) -> int:
        return self._resolved_mesh().cell_count

    @property
    def forward_operator(self) -> ERTForward2p5D | ERTForward3D:
        if self._forward is None:
            mesh, survey = self._resolved_mesh(), self._resolved_survey()
            if isinstance(mesh, Mesh3D):
                self._forward = ERTForward3D.from_mesh_survey(
                    mesh,
                    survey,
                    boundary_mode=self.boundary_mode_3d,
                    singularity_removal=self.singularity_removal_3d,
                    element_order=self.element_order_3d,
                    geometric_factor_mode=self.geometric_factor_mode_3d,
                )
            else:
                self._forward = ERTForward2p5D.from_mesh_survey(
                    mesh,
                    survey,
                    quadrature_order=self.quadrature_order,
                    numerical_h2_refined=self.numerical_h2_refined,
                    numerical_p2_refined=self.numerical_p2_refined,
                    topographic_geometric_factor_mode=self.topographic_geometric_factor_mode,
                    terrain_cache_dir=self.terrain_cache_dir,
                )
        return self._forward

    def _conductivity(self, model: Any, log_transform: bool) -> torch.Tensor:
        return torch.as_tensor(1.0 / _resistivity(model, log_transform=log_transform, expected_size=self.cell_count), dtype=FLOAT_DTYPE)

    def prepare(self, resistivity_model: Any | None = None, log_transform: bool = True, *, include_solver_state: bool = True) -> None:
        """Warm geometry, cache, and optional solver state."""

        conductivity = None if resistivity_model is None else self._conductivity(resistivity_model, log_transform)
        self.forward_operator.prepare(conductivity, include_solver_state=include_solver_state)

    def forward(self, resistivity_model: Any, log_transform: bool = True) -> np.ndarray:
        """Compute apparent resistivity (its log when ``log_transform``) for the current mesh and survey."""

        conductivity = self._conductivity(resistivity_model, log_transform)
        values = np.asarray(self.forward_operator.apparent_resistivity_values(conductivity), dtype=float)
        return np.log(values) if log_transform else values

    def response(self, resistivity_model: Any) -> np.ndarray:
        """Return non-log apparent resistivity values."""

        return self.forward(resistivity_model, log_transform=False)

    def forward_and_jacobian(
        self,
        resistivity_model: Any,
        log_transform: bool = True,
        *,
        include_robin_boundary_derivative: bool | None = None,
        normal_sensitivity: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute apparent resistivity and its cellwise Jacobian.

        With ``log_transform=True`` the model and response are logarithmic and the Jacobian
        is ``d log(rhoa) / d log(rho)``; otherwise it is ``d rhoa / d rho``. The default
        normal-quadrupole sensitivity omits the mixed Robin boundary derivative.
        """

        conductivity = self._conductivity(resistivity_model, log_transform)
        forward = self.forward_operator
        options = {
            "include_robin_boundary_derivative": self.include_robin_boundary_derivative
            if include_robin_boundary_derivative is None
            else include_robin_boundary_derivative,
            "normal_sensitivity": self.normal_sensitivity if normal_sensitivity is None else normal_sensitivity,
        }
        if isinstance(forward, ERTForward3D):
            options["include_fields"] = False
        if log_transform:
            return log_response_and_jacobian(forward, conductivity, **options)
        response, resistance_jacobian = forward.solve_with_jacobian(conductivity, **options)
        jacobian = torch.abs(torch.as_tensor(forward._geometric_factors()))[:, None] * resistance_jacobian
        jacobian = jacobian * (-(conductivity**2)[None, :])
        return np.asarray(response.apparent_resistivity, dtype=float), np.asarray(jacobian, dtype=float)


_MODELING_OPTIONS = tuple(item.name for item in fields(ERTForwardModeling) if item.name not in ("mesh", "data"))


@dataclass
class MappedERTForwardModeling:
    """Forward facade whose model vector covers only ``active_cell_ids`` of the forward mesh.

    Cells outside the active set are kept at ``inactive_resistivity``.
    """

    mesh: Any
    data: Any
    active_cell_ids: Any
    inactive_resistivity: Any
    regularization_mesh: Any | None = None
    quadrature_order: int = 2
    numerical_h2_refined: bool = True
    numerical_p2_refined: bool = True
    topographic_geometric_factor_mode: str = "analytic"
    terrain_cache_dir: str | Path | None = None
    include_robin_boundary_derivative: bool = False
    normal_sensitivity: bool = True
    boundary_mode_3d: str = "mixed"
    singularity_removal_3d: bool = True
    element_order_3d: int = 1
    geometric_factor_mode_3d: str = "auto"

    def __post_init__(self) -> None:
        self._forward_modeling = ERTForwardModeling(
            mesh=self.mesh, data=self.data, **{name: getattr(self, name) for name in _MODELING_OPTIONS}
        )
        full_cell_count = self._forward_modeling.cell_count
        active = np.asarray(self.active_cell_ids, dtype=np.int32).ravel()
        if active.size == 0:
            raise ValueError("active_cell_ids must be non-empty")
        if np.any(active < 0):
            raise ValueError("active_cell_ids contains negative indices")
        if np.unique(active).size != active.size:
            raise ValueError("active_cell_ids must be unique")
        if np.any(active >= full_cell_count):
            raise ValueError("active_cell_ids references cells outside the forward mesh")
        self._active_cell_ids = active

        inactive = np.asarray(self.inactive_resistivity, dtype=float)
        inactive = np.full(full_cell_count, float(inactive)) if inactive.ndim == 0 else inactive.ravel().copy()
        if inactive.shape != (full_cell_count,):
            raise ValueError(f"inactive_resistivity must be scalar or shape ({full_cell_count},)")
        if not np.all(np.isfinite(inactive)) or np.any(inactive <= 0.0):
            raise ValueError("inactive_resistivity must contain positive finite values")
        self._inactive_resistivity = inactive
        if self.regularization_mesh is not None:
            self.regularization_mesh = mesh_to_adtlert(self.regularization_mesh)

    @property
    def cell_count(self) -> int:
        return int(self._active_cell_ids.size)

    @property
    def forward_operator(self) -> ERTForward2p5D | ERTForward3D:
        return self._forward_modeling.forward_operator

    @property
    def active_cell_ids_array(self) -> np.ndarray:
        return self._active_cell_ids.copy()

    def _expand(self, resistivity_model: Any, log_transform: bool) -> np.ndarray:
        full = self._inactive_resistivity.copy()
        full[self._active_cell_ids] = _resistivity(resistivity_model, log_transform=log_transform, expected_size=self.cell_count)
        return full

    def prepare(self, resistivity_model: Any | None = None, log_transform: bool = True, *, include_solver_state: bool = True) -> None:
        """Warm geometry, cache, and optional solver state."""

        full = None if resistivity_model is None else self._expand(resistivity_model, log_transform)
        self._forward_modeling.prepare(full, log_transform=False, include_solver_state=include_solver_state)

    def forward(self, resistivity_model: Any, log_transform: bool = True) -> np.ndarray:
        values = self._forward_modeling.forward(self._expand(resistivity_model, log_transform), log_transform=False)
        return np.log(values) if log_transform else values

    def response(self, resistivity_model: Any) -> np.ndarray:
        return self.forward(resistivity_model, log_transform=False)

    def forward_and_jacobian(
        self,
        resistivity_model: Any,
        log_transform: bool = True,
        *,
        include_robin_boundary_derivative: bool | None = None,
        normal_sensitivity: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        full = self._expand(resistivity_model, log_transform)
        response, jacobian = self._forward_modeling.forward_and_jacobian(
            np.log(full) if log_transform else full,
            log_transform=log_transform,
            include_robin_boundary_derivative=include_robin_boundary_derivative,
            normal_sensitivity=normal_sensitivity,
        )
        return response, np.asarray(jacobian[:, self._active_cell_ids], dtype=float)
