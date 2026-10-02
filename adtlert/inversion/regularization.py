"""Spatial and temporal regularization operators for ADTLERT inversions.

Every term exposes ``matrix`` (the linear operator) and ``linearized_system`` (the
scaled ``(A, b)`` block of the current Gauss-Newton step). Robust variants reweight the
rows of the first-order operator by IRLS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import scipy.sparse as sp

from adtlert.inversion.misfit import huber_irls_weight, l1_irls_weight
from adtlert.mesh import Mesh, Mesh3D


class SpatialRegularization(Protocol):
    """Interface for spatial regularization terms that have a linear operator."""

    name: str

    def matrix(self, forward, n_cells: int, *, z_weight: float = 1.0) -> sp.csr_matrix:
        """Build the spatial regularization matrix for the current parameter mesh."""

    def linearized_system(
        self,
        forward,
        current_model,
        n_cells: int,
        *,
        reference_roughness,
        scale: float = 1.0,
        z_weight: float = 1.0,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        """Build ``(A, b)`` for the current spatial regularization update."""


class TemporalRegularization(Protocol):
    """Interface for temporal regularization terms that have a linear operator."""

    name: str

    def matrix(
        self, n_cells: int, n_times: int, *, scale: float = 1.0
    ) -> sp.csr_matrix:
        """Build the temporal regularization matrix."""

    def linearized_system(
        self, current_model, n_cells: int, n_times: int, *, scale: float = 1.0
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        """Build ``(A, b)`` for the current temporal regularization update."""


def regularization_mesh(forward) -> Mesh | Mesh3D:
    """Return the mesh used for spatial regularization."""

    mesh = getattr(forward, "regularization_mesh", None)
    if mesh is not None:
        if isinstance(mesh, (Mesh, Mesh3D)):
            return mesh
        raise TypeError("forward.regularization_mesh must be a adtlert Mesh or Mesh3D")
    if hasattr(forward, "_resolved_mesh") and isinstance(
        resolved := forward._resolved_mesh(), (Mesh, Mesh3D)
    ):
        return resolved
    mesh = getattr(forward, "mesh", None)
    if isinstance(mesh, (Mesh, Mesh3D)):
        return mesh
    raise TypeError(
        "forward must expose a Mesh or Mesh3D for first-order regularization"
    )


def cell_edges(cell: np.ndarray) -> list[tuple[int, int]]:
    """Return the edges of a polygon cell as node-id pairs."""

    return [
        (int(cell[index]), int(cell[(index + 1) % cell.size]))
        for index in range(cell.size)
    ]


def tetrahedron_faces(cell: np.ndarray) -> list[tuple[int, int, int]]:
    """Return the four triangular faces of a tetrahedron."""

    a, b, c, d = (int(node) for node in cell)
    return [(b, c, d), (a, d, c), (a, b, d), (a, c, b)]


def _neighbor_pairs(mesh: Mesh | Mesh3D) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Cells sharing an interior edge/face and the |z|-component of that interface's normal."""

    nodes = np.asarray(mesh.nodes, dtype=float)
    three_d = isinstance(mesh, Mesh3D)
    owners: dict[tuple[int, ...], list[int]] = {}
    for cell_id, cell in enumerate(np.asarray(mesh.cells, dtype=np.int32)):
        for entity in tetrahedron_faces(cell) if three_d else cell_edges(cell):
            owners.setdefault(tuple(sorted(entity)), []).append(cell_id)
    pairs, normal_z = [], []
    for entity, cells in owners.items():
        if len(cells) != 2:
            continue
        points = nodes[list(entity)]
        if three_d:
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            length, component = float(np.linalg.norm(normal)), abs(float(normal[2]))
        else:  # the edge normal's z-component is the tangent's x-component
            tangent = points[1] - points[0]
            length, component = float(np.linalg.norm(tangent)), abs(float(tangent[0]))
        if length > 0.0:
            pairs.append((cells[0], cells[1]))
            normal_z.append(component / length)
    return pairs, np.asarray(normal_z)


def _difference_rows(pairs, weights: np.ndarray, n_cells: int) -> sp.csr_matrix:
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    rows = np.repeat(np.arange(len(pairs)), 2)
    data = np.column_stack((weights, -weights)).reshape(-1)
    return sp.coo_matrix(
        (data, (rows, pairs.reshape(-1))), shape=(len(pairs), n_cells)
    ).tocsr()


def first_order_constraint_matrix(
    mesh: Mesh | Mesh3D, *, z_weight: float = 1.0
) -> sp.csr_matrix:
    """First-order neighbor differences, weighted by ``z_weight`` across horizontal interfaces."""

    pairs, normal_z = _neighbor_pairs(mesh)
    return _difference_rows(
        pairs, 1.0 + normal_z * (float(z_weight) - 1.0), int(mesh.cell_count)
    )


def structure_guided_constraint_matrix(
    mesh: Mesh | Mesh3D,
    structural_ids: np.ndarray,
    *,
    z_weight: float = 1.0,
    cross_structure_weight: float = 0.05,
) -> sp.csr_matrix:
    """First-order constraints with weights reduced across structural-unit boundaries."""

    labels = np.asarray(structural_ids, dtype=np.int32).reshape(-1)
    if labels.shape != (int(mesh.cell_count),):
        raise ValueError(
            f"structural_ids must have one value per regularization cell ({labels.shape} != ({int(mesh.cell_count)},))"
        )
    if cross_structure_weight < 0.0:
        raise ValueError("cross_structure_weight must be non-negative")
    pairs, normal_z = _neighbor_pairs(mesh)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    same = labels[pairs[:, 0]] == labels[pairs[:, 1]]
    weights = (1.0 + normal_z * (float(z_weight) - 1.0)) * np.where(
        same, 1.0, float(cross_structure_weight)
    )
    keep = weights != 0.0
    return _difference_rows(pairs[keep], weights[keep], int(mesh.cell_count))


def _checked(matrix: sp.csr_matrix, n_cells: int) -> sp.csr_matrix:
    if matrix.shape[1] != int(n_cells):
        raise ValueError(
            f"regularization mesh cell count does not match inversion model size ({matrix.shape[1]} != {int(n_cells)})"
        )
    return matrix


class _LinearSpatial:
    def linearized_system(
        self,
        forward,
        current_model,
        n_cells,
        *,
        reference_roughness,
        scale=1.0,
        z_weight=1.0,
    ):
        matrix = float(scale) * self.matrix(forward, n_cells, z_weight=z_weight)
        roughness = matrix @ np.asarray(current_model, dtype=float).reshape(-1)
        return matrix, float(scale) * np.asarray(
            reference_roughness, dtype=float
        ).reshape(-1) - roughness


class _RobustSpatial:
    """IRLS reweighting of first-order neighbor differences."""

    def matrix(self, forward, n_cells: int, *, z_weight: float = 1.0) -> sp.csr_matrix:
        return FirstOrderSpatialRegularization().matrix(
            forward, n_cells, z_weight=z_weight
        )

    def linearized_system(
        self,
        forward,
        current_model,
        n_cells,
        *,
        reference_roughness,
        scale=1.0,
        z_weight=1.0,
    ):
        base = self.matrix(forward, n_cells, z_weight=z_weight)
        current = base @ np.asarray(current_model, dtype=float).reshape(-1)
        reference = np.asarray(reference_roughness, dtype=float).reshape(-1)
        weight = self._irls(np.asarray(current - reference, dtype=float))
        return (sp.diags(float(scale) * weight, format="csr") @ base).tocsr(), float(
            scale
        ) * weight * (reference - current)


@dataclass(frozen=True)
class IdentitySpatialRegularization(_LinearSpatial):
    """Damping regularization in model space."""

    name: str = "identity"

    def matrix(self, forward, n_cells: int, *, z_weight: float = 1.0) -> sp.csr_matrix:
        return sp.eye(int(n_cells), format="csr")


@dataclass(frozen=True)
class FirstOrderSpatialRegularization(_LinearSpatial):
    """First-order neighbor smoothness regularization."""

    name: str = "first_order"

    def matrix(self, forward, n_cells: int, *, z_weight: float = 1.0) -> sp.csr_matrix:
        return _checked(
            first_order_constraint_matrix(
                regularization_mesh(forward), z_weight=z_weight
            ),
            n_cells,
        )


@dataclass(frozen=True)
class StructuralPriorSpatialRegularization(_LinearSpatial):
    """Structure-guided smoothness from ``forward.structural_prior_cell_ids`` (one unit id per cell)."""

    name: str = "structural_prior"

    def matrix(self, forward, n_cells: int, *, z_weight: float = 1.0) -> sp.csr_matrix:
        ids = getattr(forward, "structural_prior_cell_ids", None)
        ids = getattr(forward, "structure_cell_ids", None) if ids is None else ids
        if ids is None:
            raise ValueError(
                "structural_prior regularization requires forward.structural_prior_cell_ids "
                "with one structural-unit id per inversion cell"
            )
        matrix = structure_guided_constraint_matrix(
            regularization_mesh(forward),
            np.asarray(ids, dtype=np.int32),
            z_weight=z_weight,
            cross_structure_weight=float(
                getattr(forward, "structural_cross_weight", 0.05)
            ),
        )
        return _checked(matrix, n_cells)


@dataclass(frozen=True)
class ModelDifferenceSpatialRegularization(FirstOrderSpatialRegularization):
    """Spatial smoothness of time-lapse model differences (assembled in ``core.py``)."""

    name: str = "model_difference_smoothness"


@dataclass(frozen=True)
class FirstOrderSpatialTVRegularization(_RobustSpatial):
    """IRLS-smoothed first-order spatial TV/L1 regularization."""

    epsilon: float = 1.0e-3
    name: str = "first_order_tv"

    def _irls(self, residual):
        return l1_irls_weight(residual, self.epsilon)


@dataclass(frozen=True)
class FirstOrderSpatialHuberRegularization(_RobustSpatial):
    """First-order spatial Huber regularization with IRLS linearization."""

    delta: float = 1.0
    epsilon: float = 1.0e-12
    name: str = "first_order_huber"

    def _irls(self, residual):
        return huber_irls_weight(residual, self.delta, self.epsilon)


def _time_operator(stencil: np.ndarray, n_cells: int, scale: float) -> sp.csr_matrix:
    """Apply a ``(rows, n_times)`` stencil to every cell of a time-major model vector."""

    return sp.kron(
        sp.csr_matrix(stencil * float(scale)),
        sp.eye(int(n_cells), format="csr"),
        format="csr",
    )


def _first_difference(n_times: int) -> np.ndarray:
    return np.eye(n_times)[1:] - np.eye(n_times)[:-1]


class _LinearTemporal:
    def linearized_system(self, current_model, n_cells, n_times, *, scale=1.0):
        matrix = self.matrix(n_cells, n_times, scale=scale)
        return matrix, -np.asarray(
            matrix @ np.asarray(current_model, dtype=float).reshape(-1), dtype=float
        )


class _ReweightedTemporal:
    """Row reweighting of first-order temporal differences."""

    def matrix(
        self, n_cells: int, n_times: int, *, scale: float = 1.0
    ) -> sp.csr_matrix:
        return FirstOrderTemporalRegularization().matrix(n_cells, n_times, scale=scale)

    def linearized_system(
        self, current_model, n_cells, n_times, *, scale=1.0, **options
    ):
        base = self.matrix(n_cells, n_times, scale=1.0)
        change = np.asarray(
            base @ np.asarray(current_model, dtype=float).reshape(-1), dtype=float
        )
        weight = self._row_weights(change, **options)
        return (sp.diags(float(scale) * weight, format="csr") @ base).tocsr(), -float(
            scale
        ) * weight * change


@dataclass(frozen=True)
class FirstOrderTemporalRegularization(_LinearTemporal):
    """First-order L2 temporal smoothness between neighboring time steps."""

    name: str = "first_order_l2"

    def matrix(
        self, n_cells: int, n_times: int, *, scale: float = 1.0
    ) -> sp.csr_matrix:
        return _time_operator(_first_difference(int(n_times)), n_cells, scale)


@dataclass(frozen=True)
class SecondOrderTemporalRegularization(_LinearTemporal):
    """Second-order L2 temporal smoothness of model curvature."""

    name: str = "second_order_l2"

    def matrix(
        self, n_cells: int, n_times: int, *, scale: float = 1.0
    ) -> sp.csr_matrix:
        n_times = int(n_times)
        stencil = np.zeros((max(n_times - 2, 0), n_times))
        for row in range(stencil.shape[0]):
            stencil[row, row : row + 3] = (1.0, -2.0, 1.0)
        return _time_operator(stencil, n_cells, scale)


@dataclass(frozen=True)
class BaselineReferenceTemporalRegularization(_LinearTemporal):
    """Cross-model constraint of every time step to the first (baseline) model."""

    name: str = "baseline_reference"

    def matrix(
        self, n_cells: int, n_times: int, *, scale: float = 1.0
    ) -> sp.csr_matrix:
        n_times = int(n_times)
        stencil = np.eye(n_times)[1:]
        stencil[:, 0] = -1.0
        return _time_operator(stencil, n_cells, scale)


@dataclass(frozen=True)
class FirstOrderTemporalTVRegularization(_ReweightedTemporal):
    """IRLS-smoothed first-order temporal TV/L1 regularization."""

    epsilon: float = 1.0e-3
    name: str = "first_order_tv"

    def _row_weights(self, change):
        return l1_irls_weight(change, self.epsilon)


@dataclass(frozen=True)
class FirstOrderTemporalHuberRegularization(_ReweightedTemporal):
    """First-order temporal Huber regularization with IRLS linearization."""

    delta: float = 1.0
    epsilon: float = 1.0e-12
    name: str = "first_order_huber"

    def _row_weights(self, change):
        return huber_irls_weight(change, self.delta, self.epsilon)


@dataclass(frozen=True)
class ActiveTimeConstraintRegularization(_ReweightedTemporal):
    """Adaptive temporal smoothness that relaxes where the model is actively changing."""

    threshold: float = 0.05
    minimum_weight: float = 0.05
    name: str = "active_time_constraint"

    def _row_weights(
        self,
        change,
        threshold: float | None = None,
        minimum_weight: float | None = None,
    ):
        threshold = self.threshold if threshold is None else float(threshold)
        minimum = (
            self.minimum_weight if minimum_weight is None else float(minimum_weight)
        )
        if threshold <= 0.0:
            raise ValueError("active time threshold must be positive")
        if not 0.0 <= minimum <= 1.0:
            raise ValueError("active time minimum weight must be in [0, 1]")
        return minimum + (1.0 - minimum) / (1.0 + (np.abs(change) / threshold) ** 2)


def _registry(table: dict) -> dict[str, object]:
    return {alias: term for term, aliases in table.items() for alias in aliases}


_SPATIAL_ALIASES = {
    IdentitySpatialRegularization(): ("damping", "identity"),
    FirstOrderSpatialRegularization(): ("first_order_smoothness", "first_order", "smoothness_constrained"),
    StructuralPriorSpatialRegularization(): (
        "structural_prior", "structure_guided", "structure_guided_first_order", "structurally_constrained",
    ),
    ModelDifferenceSpatialRegularization(): ("model_difference_smoothness", "change_smoothness", "change_model_smoothness"),
    FirstOrderSpatialTVRegularization(): (
        "spatial_total_variation", "first_order_tv", "spatial_tv", "tv", "first_order_l1", "spatial_l1",
    ),
    FirstOrderSpatialHuberRegularization(): ("spatial_huber", "first_order_huber", "huber"),
}  # fmt: skip
_TEMPORAL_ALIASES = {
    FirstOrderTemporalRegularization(): ("temporal_smoothness", "first_order", "first_order_l2", "l2"),
    SecondOrderTemporalRegularization(): ("second_order_temporal_smoothness", "second_order", "second_order_l2", "curvature_l2"),
    FirstOrderTemporalTVRegularization(): (
        "temporal_total_variation", "first_order_tv", "temporal_tv", "tv", "first_order_l1", "temporal_l1",
    ),
    FirstOrderTemporalHuberRegularization(): ("temporal_huber", "first_order_huber", "huber"),
    ActiveTimeConstraintRegularization(): ("active_time_constraint", "active_time_constrained", "atc", "4d_atc"),
    BaselineReferenceTemporalRegularization(): ("baseline_reference", "reference_model_constraint", "cross_model_constraint"),
}  # fmt: skip
_SPATIAL_REGULARIZATIONS = _registry(_SPATIAL_ALIASES)
_TEMPORAL_REGULARIZATIONS = _registry(_TEMPORAL_ALIASES)


def available_spatial_regularizations() -> tuple[str, ...]:
    """Return canonical spatial regularization names for user-facing configuration."""

    return tuple(aliases[0] for aliases in _SPATIAL_ALIASES.values())


def available_temporal_regularizations() -> tuple[str, ...]:
    """Return canonical temporal regularization names for user-facing configuration."""

    return tuple(aliases[0] for aliases in _TEMPORAL_ALIASES.values())


def _build(name, registry: dict, choices: tuple[str, ...], label: str):
    if hasattr(name, "matrix"):
        return name
    try:
        return registry[str(name).strip().lower().replace("-", "_")]
    except KeyError as exc:
        raise ValueError(
            f"unknown {label}={name!r}; available choices: {', '.join(choices)}"
        ) from exc


def build_spatial_regularization(
    name: str | SpatialRegularization,
) -> SpatialRegularization:
    """Resolve a spatial regularization object from a registered name."""

    return _build(
        name,
        _SPATIAL_REGULARIZATIONS,
        available_spatial_regularizations(),
        "spatial_regularization",
    )


def build_temporal_regularization(
    name: str | TemporalRegularization,
) -> TemporalRegularization:
    """Resolve a temporal regularization object from a registered name."""

    return _build(
        name,
        _TEMPORAL_REGULARIZATIONS,
        available_temporal_regularizations(),
        "temporal_regularization_type",
    )
