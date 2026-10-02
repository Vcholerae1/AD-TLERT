"""2.5D multi-wavenumber ERT forward operator.

Every cosine-transform wavenumber is solved with the secondary-field formulation

    A(sigma) u_s = A(1) u_p - A(sigma) (rho_src u_p),    u = u_s + rho_src u_p,

where ``u_p`` is the unit-resistivity primary field of each electrode. Flat surfaces use
the analytic half-space primary on the input mesh. Terrain solves on an auxiliary H2/P1
discretization whose primary field is computed numerically on an H2/P2 discretization.
Both cases share one code path parameterized by :class:`_Discretization`.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import logging
import os
from pathlib import Path
import warnings

import numpy as np
from scipy.special import k0 as besselk0
import torch

from adtlert.fem import (
    assemble_local_boundary_mass,
    assemble_local_boundary_mass_p2,
    assemble_local_mass,
    assemble_local_mass_p2,
    assemble_local_stiffness,
    assemble_local_stiffness_p2,
    build_p1_element_data,
    build_p2_element_data,
    robin_boundary_coefficients,
)
from adtlert.forward.integration import (
    build_inverse_cosine_weights,
    survey_wavenumber_bounds,
)
from adtlert.mesh import Mesh
from adtlert.mesh.core import edge_midpoint_builder
from adtlert.survey import Survey
from adtlert.utils.dtypes import FLOAT_DTYPE, NP_FLOAT_DTYPE

Tensor = torch.Tensor

_TERRAIN_CACHE_VERSION = "terrain_auxiliary_v2"
_CUDA = torch.device("cuda")
_CUDSS_LOGGER = logging.getLogger("adtlert.cudss")
_CUDSS_LOGGER.setLevel(logging.ERROR)


@dataclass(frozen=True)
class ForwardResponse:
    """Result of a 2.5D ERT forward solve."""

    apparent_resistivity: Tensor
    resistance: Tensor
    electrode_potentials: Tensor
    integrated_potentials: Tensor
    wavenumbers: Tensor
    weights: Tensor


# ---------------------------------------------------------------------------
# Discretizations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SparsePattern:
    """Fixed CSR structure shared by every assembled operator of one discretization."""

    shape: tuple[int, int]
    volume_inverse: Tensor
    boundary_inverse: Tensor
    indptr: np.ndarray
    indices: np.ndarray
    _device_indices: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def nnz(self) -> int:
        return int(self.indices.size)

    def matvec(self, values: Tensor, vectors: Tensor) -> Tensor:
        """Apply ``A_b`` to ``vectors[b]`` for each operator ``b`` (rows are vectors)."""

        key = str(values.device)
        if key not in self._device_indices:
            self._device_indices[key] = (
                torch.as_tensor(self.indptr, dtype=torch.long, device=values.device),
                torch.as_tensor(self.indices, dtype=torch.long, device=values.device),
            )
        indptr, indices = self._device_indices[key]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Sparse (CSR tensor support|invariant checks)"
            )
            return torch.stack(
                [
                    (
                        torch.sparse_csr_tensor(indptr, indices, value, size=self.shape)
                        @ vector.T
                    ).T
                    for value, vector in zip(values, vectors, strict=True)
                ]
            )


def _sparse_pattern(
    cell_dofs: np.ndarray, boundary_dofs: np.ndarray, dof_count: int
) -> _SparsePattern:
    def local_pairs(connectivity: np.ndarray) -> np.ndarray:
        width = connectivity.shape[1]
        rows = np.repeat(connectivity, width, axis=1).reshape(-1)
        cols = np.tile(connectivity, (1, width)).reshape(-1)
        return np.stack((rows, cols), axis=1)

    volume = local_pairs(cell_dofs)
    unique, inverse = np.unique(
        np.concatenate((volume, local_pairs(boundary_dofs))),
        axis=0,
        return_inverse=True,
    )
    inverse = torch.as_tensor(inverse.reshape(-1), dtype=torch.long)
    indptr = np.concatenate(
        ([0], np.cumsum(np.bincount(unique[:, 0], minlength=dof_count)))
    ).astype(np.int32)
    return _SparsePattern(
        shape=(dof_count, dof_count),
        volume_inverse=inverse[: volume.shape[0]],
        boundary_inverse=inverse[volume.shape[0] :],
        indptr=indptr,
        indices=unique[:, 1].astype(np.int32),
    )


@dataclass(frozen=True)
class _Discretization:
    """Finite-element space used for one family of solves.

    ``parent_cell_ids`` maps each geometry cell to the forward-mesh cell whose
    conductivity it carries. ``boundary_cells`` index geometry cells.
    """

    name: str
    geometry: Mesh
    parent_cell_ids: Tensor
    dof_nodes: Tensor
    cell_dofs: Tensor
    boundary_dofs: Tensor
    stiffness: Tensor
    mass: Tensor
    boundary_mass: Tensor
    boundary_geometries: Tensor
    electrode_matrix: Tensor
    pattern: _SparsePattern
    spd: bool

    @property
    def dof_count(self) -> int:
        return int(self.dof_nodes.shape[0])

    @property
    def boundary_cells(self) -> Tensor:
        return self.geometry.boundary_edge_cells.long()


def _bilinear(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Q1 shape values and reference gradients on ``[-1, 1]^2``."""

    xi, eta = points[:, 0], points[:, 1]
    values = 0.25 * np.stack(
        (
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta),
        ),
        -1,
    )
    dxi = 0.25 * np.stack((-(1 - eta), 1 - eta, 1 + eta, -(1 + eta)), axis=-1)
    deta = 0.25 * np.stack((-(1 - xi), -(1 + xi), 1 + xi, 1 - xi), axis=-1)
    return values, np.stack((dxi, deta), axis=-1)


def _serendipity(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eight-node serendipity shape values and reference gradients (corners, then edge midpoints)."""

    xi, eta = points[:, 0], points[:, 1]
    values = np.stack(
        (
            -0.25 * (1 - xi) * (1 - eta) * (1 + xi + eta),
            -0.25 * (1 + xi) * (1 - eta) * (1 - xi + eta),
            -0.25 * (1 + xi) * (1 + eta) * (1 - xi - eta),
            -0.25 * (1 - xi) * (1 + eta) * (1 + xi - eta),
            0.5 * (1 - xi * xi) * (1 - eta),
            0.5 * (1 + xi) * (1 - eta * eta),
            0.5 * (1 - xi * xi) * (1 + eta),
            0.5 * (1 - xi) * (1 - eta * eta),
        ),
        axis=-1,
    )
    dxi = np.stack(
        (
            0.25 * (1 - eta) * (2 * xi + eta),
            0.25 * (1 - eta) * (2 * xi - eta),
            0.25 * (1 + eta) * (2 * xi + eta),
            0.25 * (1 + eta) * (2 * xi - eta),
            -xi * (1 - eta),
            0.5 * (1 - eta * eta),
            -xi * (1 + eta),
            -0.5 * (1 - eta * eta),
        ),
        axis=-1,
    )
    deta = np.stack(
        (
            0.25 * (1 - xi) * (xi + 2 * eta),
            0.25 * (1 + xi) * (-xi + 2 * eta),
            0.25 * (1 + xi) * (xi + 2 * eta),
            0.25 * (1 - xi) * (-xi + 2 * eta),
            -0.5 * (1 - xi * xi),
            -(1 + xi) * eta,
            0.5 * (1 - xi * xi),
            -(1 - xi) * eta,
        ),
        axis=-1,
    )
    return values, np.stack((dxi, deta), axis=-1)


def _quad_templates(mesh: Mesh, quadratic: bool) -> tuple[Tensor, Tensor, Tensor]:
    """Unit-conductivity stiffness, mass, and boundary-mass templates on bilinear quads."""

    high = quadratic and torch.get_default_dtype() == torch.float64
    dtype = np.float64 if high else NP_FLOAT_DTYPE
    if quadratic:
        gauss, gauss_weights = np.polynomial.legendre.leggauss(3)
        points = np.asarray([[x, y] for y in gauss for x in gauss], dtype=dtype)
        weights = np.asarray(
            [wx * wy for wy in gauss_weights for wx in gauss_weights], dtype=dtype
        )
    else:  # 2x2 Gauss points in counter-clockwise corner order
        g = 1.0 / np.sqrt(3.0)
        points = np.asarray([[-g, -g], [g, -g], [g, g], [-g, g]], dtype=dtype)
        weights = np.ones(4, dtype=dtype)
    _, geometry_gradients = _bilinear(points)
    values, gradients = _serendipity(points) if quadratic else _bilinear(points)

    cell_nodes = np.asarray(mesh.nodes, dtype=dtype)[np.asarray(mesh.cells)]
    jacobians = np.einsum("cid,qia->cqda", cell_nodes, geometry_gradients.astype(dtype))
    physical = np.einsum(
        "qia,cqab->cqib", gradients.astype(dtype), np.linalg.inv(jacobians)
    )
    weighted_det = np.abs(np.linalg.det(jacobians)) * weights
    stiffness = np.einsum("cqid,cqjd,cq->cij", physical, physical, weighted_det)
    values = values.astype(dtype)
    mass = np.einsum("qi,qj,cq->cij", values, values, weighted_det)
    if quadratic:
        reference = (
            np.asarray(
                [[4.0, 2.0, -1.0], [2.0, 16.0, 2.0], [-1.0, 2.0, 4.0]], dtype=dtype
            )
            / 30.0
        )
    else:
        reference = np.asarray([[2.0, 1.0], [1.0, 2.0]], dtype=dtype) / 6.0
    boundary_mass = (
        np.asarray(mesh.boundary_edge_lengths, dtype=dtype)[:, None, None] * reference
    )
    out = torch.float64 if high else FLOAT_DTYPE
    return tuple(
        torch.as_tensor(array, dtype=out) for array in (stiffness, mass, boundary_mass)
    )


def _triangle_templates(
    mesh: Mesh, quadratic: bool, quadrature_order: int
) -> tuple[Tensor, Tensor, Tensor]:
    if quadratic:
        data = build_p2_element_data(mesh)
        return (
            assemble_local_stiffness_p2(data, 1.0),
            assemble_local_mass_p2(data, 1.0),
            assemble_local_boundary_mass_p2(mesh.boundary_edge_lengths, 1.0),
        )
    data = build_p1_element_data(mesh, quadrature_order=quadrature_order)
    return (
        assemble_local_stiffness(data, 1.0),
        assemble_local_mass(data, 1.0),
        assemble_local_boundary_mass(mesh, 1.0),
    )


def _structured_quads(mesh: Mesh) -> Mesh | None:
    """Return the quadrilateral strip mesh behind a quad or columnar half-cell triangle mesh."""

    if mesh.is_quadrilateral_mesh:
        return mesh
    expanded = mesh.expand_columnar_cells()
    if expanded is None:
        return None
    triangles, parents = expanded
    nodes = np.asarray(triangles.nodes, dtype=float)
    pairs: dict[int, list[np.ndarray]] = {}
    for cell, parent in zip(
        np.asarray(triangles.cells), np.asarray(parents), strict=True
    ):
        pairs.setdefault(int(parent), []).append(cell)

    quads = []
    for parent in range(mesh.cell_count):
        pair = pairs.get(parent, [])
        node_ids = (
            np.unique(np.concatenate(pair))
            if len(pair) == 2
            else np.empty(0, dtype=int)
        )
        if node_ids.size != 4:
            return None
        order = np.argsort(nodes[node_ids, 0])
        left, right = node_ids[order[:2]], node_ids[order[2:]]
        if nodes[left, 0].max() > nodes[right, 0].min() + 1e-6:
            return None
        bottom, top = np.argmin, np.argmax
        quads.append(
            [
                left[bottom(nodes[left, 1])],
                right[bottom(nodes[right, 1])],
                right[top(nodes[right, 1])],
                left[top(nodes[left, 1])],
            ]
        )
    return Mesh.from_arrays(
        triangles.nodes, np.asarray(quads), surface_node_ids=triangles.surface_node_ids
    )


def _refine_quads(mesh: Mesh) -> tuple[Mesh, np.ndarray]:
    """Split every quad into four through edge midpoints and the center."""

    nodes, midpoint, _ = edge_midpoint_builder(mesh.nodes)
    cells, parents = [], []
    for parent, (bl, br, tr, tl) in enumerate(np.asarray(mesh.cells).tolist()):
        b, r, t, l = (
            midpoint(bl, br),
            midpoint(br, tr),
            midpoint(tr, tl),
            midpoint(tl, bl),
        )
        center = len(nodes)
        nodes.append(0.25 * (nodes[bl] + nodes[br] + nodes[tr] + nodes[tl]))
        cells += [
            [bl, b, center, l],
            [b, br, r, center],
            [center, r, tr, t],
            [l, center, t, tl],
        ]
        parents += [parent] * 4
    surface = np.asarray(mesh.surface_node_ids).tolist()
    refined_surface = [surface[0]]
    for start, stop in zip(surface[:-1], surface[1:], strict=True):
        refined_surface += [midpoint(start, stop), stop]
    refined = Mesh.from_arrays(
        np.asarray(nodes),
        np.asarray(cells),
        surface_node_ids=np.asarray(refined_surface),
    )
    return refined, np.asarray(parents)


def _serendipity_topology(mesh: Mesh) -> tuple[Tensor, Tensor, Tensor]:
    nodes, midpoint, _ = edge_midpoint_builder(mesh.nodes)
    cells = [
        [
            bl,
            br,
            tr,
            tl,
            midpoint(bl, br),
            midpoint(br, tr),
            midpoint(tr, tl),
            midpoint(tl, bl),
        ]
        for bl, br, tr, tl in np.asarray(mesh.cells).tolist()
    ]
    boundary = [
        [a, midpoint(a, b), b] for a, b in np.asarray(mesh.boundary_edges).tolist()
    ]
    return (
        torch.as_tensor(np.asarray(nodes), dtype=FLOAT_DTYPE),
        torch.as_tensor(cells),
        torch.as_tensor(boundary),
    )


def _points_on_segments(
    points: np.ndarray, segments: np.ndarray, tol: float
) -> np.ndarray:
    """Point-to-segment incidence mask of shape ``(points, segments)``."""

    start, edge = segments[:, 0], segments[:, 1] - segments[:, 0]
    length_sq = np.sum(edge * edge, axis=1)
    projection = np.sum((points[:, None, :] - start) * edge, axis=2) / np.maximum(
        length_sq, tol
    )
    distance = np.linalg.norm(
        points[:, None, :] - (start + projection[:, :, None] * edge), axis=2
    )
    return (
        (projection >= -tol)
        & (projection <= 1.0 + tol)
        & (distance <= tol)
        & (length_sq > tol)
    )


def _surface_interpolation(
    mesh: Mesh, boundary_dofs: Tensor | None, dof_count: int, points, tol: float = 1e-5
) -> Tensor:
    """Interpolate points on the top surface polyline into linear or quadratic edge DOFs."""

    surface_ids = np.asarray(mesh.surface_node_ids)
    surface = np.asarray(mesh.surface_nodes, dtype=float)
    points = np.asarray(points, dtype=float)
    segments = np.stack((surface[:-1], surface[1:]), axis=1)
    hits = _points_on_segments(points, segments, tol)
    midpoints = (
        None
        if boundary_dofs is None
        else {
            tuple(sorted((a, b))): m for a, m, b in np.asarray(boundary_dofs).tolist()
        }
    )
    matrix = np.zeros((points.shape[0], dof_count), dtype=NP_FLOAT_DTYPE)
    for row, point in enumerate(points):
        distances = np.linalg.norm(surface - point, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= tol:
            matrix[row, surface_ids[nearest]] = 1.0
            continue
        segment_ids = np.flatnonzero(hits[row])
        if segment_ids.size == 0:
            raise ValueError(
                "surface point could not be matched to an auxiliary surface edge"
            )
        segment = int(segment_ids[0])
        start, stop = segments[segment]
        edge = stop - start
        length_sq = float(edge @ edge)
        if length_sq <= tol:
            raise ValueError("degenerate surface edge in auxiliary quadrilateral mesh")
        t = min(max(float((point - start) @ edge / length_sq), 0.0), 1.0)
        a, b = int(surface_ids[segment]), int(surface_ids[segment + 1])
        if midpoints is None:
            matrix[row, a], matrix[row, b] = 1.0 - t, t
        else:
            matrix[row, a] = 2.0 * (t - 0.5) * (t - 1.0)
            matrix[row, midpoints[tuple(sorted((a, b)))]] = 4.0 * t * (1.0 - t)
            matrix[row, b] = 2.0 * t * (t - 0.5)
    return torch.as_tensor(matrix)


def _interpolation(mesh: Mesh, cell_dofs: Tensor, dof_count: int, points) -> Tensor:
    """Interpolate points into linear (P1/Q1) or quadratic (P2) DOFs via point location."""

    cell_ids, weights = mesh.locate_points(points)
    weights = np.asarray(weights, dtype=NP_FLOAT_DTYPE)
    if cell_dofs.shape[1] == 6:
        l1, l2, l3 = weights.T
        weights = np.stack(
            (
                l1 * (2 * l1 - 1),
                l2 * (2 * l2 - 1),
                l3 * (2 * l3 - 1),
                4 * l1 * l2,
                4 * l2 * l3,
                4 * l3 * l1,
            ),
            -1,
        )
    dofs = np.asarray(cell_dofs)[np.asarray(cell_ids)]
    matrix = np.zeros((dofs.shape[0], dof_count), dtype=NP_FLOAT_DTYPE)
    np.add.at(
        matrix,
        (np.repeat(np.arange(dofs.shape[0]), dofs.shape[1]), dofs.reshape(-1)),
        weights.reshape(-1),
    )
    return torch.as_tensor(matrix)


def _build_discretization(
    name: str,
    mesh: Mesh,
    survey: Survey,
    wavenumbers: Tensor,
    *,
    auxiliary: bool,
    refine: bool = False,
    quadratic: bool = False,
    quadrature_order: int = 2,
) -> _Discretization:
    """Build the input-mesh discretization (``auxiliary=False``) or a terrain auxiliary one."""

    structured = _structured_quads(mesh) if auxiliary else None
    parents = np.arange(mesh.cell_count)
    if structured is not None:
        # Terrain strips: bilinear geometry, natural (zero Robin) outer boundaries.
        geometry = structured
        if refine:
            geometry, parents = _refine_quads(geometry)
        if quadratic:
            dof_nodes, cell_dofs, boundary_dofs = _serendipity_topology(geometry)
        else:
            dof_nodes, cell_dofs, boundary_dofs = (
                geometry.nodes,
                geometry.cells.long(),
                geometry.boundary_edges.long(),
            )
        stiffness, mass, boundary_mass = _quad_templates(geometry, quadratic)
        electrode_matrix = _surface_interpolation(
            geometry,
            boundary_dofs if quadratic else None,
            int(dof_nodes.shape[0]),
            survey.electrode_positions,
        )
        boundary_geometries = torch.zeros(
            (wavenumbers.shape[0], geometry.boundary_edges.shape[0]), dtype=FLOAT_DTYPE
        )
    else:
        geometry = mesh
        if (
            auxiliary
            and not quadratic
            and (expanded := geometry.expand_columnar_cells()) is not None
        ):
            geometry, expanded_parents = expanded
            parents = parents[np.asarray(expanded_parents)]
        if refine:
            geometry, refined_parents = geometry.refine_uniform()
            parents = parents[np.asarray(refined_parents)]
        if quadratic:
            dof_nodes, cell_dofs, boundary_dofs = geometry.build_quadratic_topology()
            cell_dofs, boundary_dofs = cell_dofs.long(), boundary_dofs.long()
        else:
            dof_nodes, cell_dofs, boundary_dofs = (
                geometry.nodes,
                geometry.cells.long(),
                geometry.boundary_edges.long(),
            )
        if geometry.is_triangle_mesh:
            stiffness, mass, boundary_mass = _triangle_templates(
                geometry, quadratic, quadrature_order
            )
        else:
            stiffness, mass, boundary_mass = _quad_templates(geometry, quadratic)
        electrode_matrix = _interpolation(
            geometry, cell_dofs, int(dof_nodes.shape[0]), survey.electrode_positions
        )
        center = np.mean(
            np.asarray(survey.electrode_positions, dtype=float), axis=0
        ).astype(NP_FLOAT_DTYPE)
        boundary_geometries = torch.stack(
            [
                robin_boundary_coefficients(geometry, 1.0, center, float(k))
                for k in wavenumbers.tolist()
            ]
        )

    return _Discretization(
        name=name,
        geometry=geometry,
        parent_cell_ids=torch.as_tensor(parents, dtype=torch.long),
        dof_nodes=dof_nodes,
        cell_dofs=cell_dofs,
        boundary_dofs=boundary_dofs,
        stiffness=stiffness,
        mass=mass,
        boundary_mass=boundary_mass,
        boundary_geometries=boundary_geometries,
        electrode_matrix=electrode_matrix,
        pattern=_sparse_pattern(
            np.asarray(cell_dofs), np.asarray(boundary_dofs), int(dof_nodes.shape[0])
        ),
        spd=not auxiliary,
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _choice(value: str, choices: tuple[str, ...], name: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in choices:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(choices))}")
    return normalized


def _cell_vector(values, count: int) -> Tensor:
    array = torch.as_tensor(values, dtype=FLOAT_DTYPE)
    if array.ndim == 0:
        return array.expand(count)
    if array.shape != (count,):
        raise ValueError(f"conductivity must be scalar or shape ({count},)")
    return array


def _measurement_vector(values, count: int, dtype) -> Tensor:
    array = torch.as_tensor(values, dtype=FLOAT_DTYPE)
    if array.ndim == 0:
        array = array.expand(count)
    elif array.shape != (count,):
        raise ValueError(f"cotangent must be scalar or shape ({count},)")
    return array.to(dtype)


def _check_finite(values: Tensor, context: str) -> None:
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError(f"{context} produced non-finite values")


def _halfspace_primary(
    nodes: np.ndarray, source: np.ndarray, wavenumber: float, surface: float
) -> np.ndarray:
    """Analytic unit-resistivity half-space primary field (image source when buried)."""

    distance = np.linalg.norm(nodes - source, axis=1)
    values = np.zeros(nodes.shape[0])
    if abs(source[1] - surface) <= 1e-8:
        valid = distance > 1e-12
        values[valid] = besselk0(distance[valid] * wavenumber) / np.pi
        return values
    mirrored = np.linalg.norm(nodes - (source[0], 2.0 * surface - source[1]), axis=1)
    valid = (distance > 1e-12) & (mirrored > 1e-12)
    values[valid] = (
        besselk0(distance[valid] * wavenumber) + besselk0(mirrored[valid] * wavenumber)
    ) / (2.0 * np.pi)
    return values


def _exact_node_indices(dof_nodes, points, tol: float = 1e-6) -> np.ndarray:
    """Indices of DOF nodes coinciding with ``points``."""

    dof_nodes, points = (
        np.asarray(dof_nodes, dtype=float),
        np.asarray(points, dtype=float),
    )
    lookup: dict[tuple[int, ...], int] = {}
    for node_id, node in enumerate(dof_nodes):
        lookup.setdefault(tuple(np.rint(node / tol).astype(np.int64).tolist()), node_id)
    indices = np.empty(points.shape[0], dtype=np.int64)
    for row, point in enumerate(points):
        nearest = lookup.get(tuple(np.rint(point / tol).astype(np.int64).tolist()))
        if nearest is None:
            distances = np.linalg.norm(dof_nodes - point, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] > tol:
                raise ValueError(
                    "point does not coincide with a quadrilateral auxiliary node"
                )
        indices[row] = nearest
    return indices


def _cupy():
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cupy_sparse
    except ImportError as exc:
        raise ImportError(
            "ERTForward2p5D requires CuPy and cuDSS on a CUDA-capable system"
        ) from exc
    return cp, cupy_sparse


def _cudss_solver(matrices, rhs, *, spd: bool):
    try:
        from nvmath.sparse import advanced
    except ImportError as exc:
        raise ImportError(
            "ERTForward2p5D requires nvmath-python to use the cuDSS backend"
        ) from exc
    matrix_type = (
        advanced.DirectSolverMatrixType.SPD
        if spd
        else advanced.DirectSolverMatrixType.SYMMETRIC
    )
    options = advanced.DirectSolverOptions(
        sparse_system_type=matrix_type, logger=_CUDSS_LOGGER, blocking=True
    )
    solver = advanced.DirectSolver(matrices, rhs, options=options)
    if not spd:
        if hasattr(advanced, "DirectSolverAlgType"):  # nvmath-python < 1.0
            solver.plan_config.algorithm = advanced.DirectSolverAlgType.ALG_1
        else:
            solver.plan_config.reordering_algorithm = (
                advanced.DirectSolverReorderingAlg.NESTED_DISSECTION
            )
    solver.plan()
    return solver


_SENSITIVITY_KERNEL = r"""
extern "C" __global__ void normal_sensitivity(
    const SCALAR* __restrict__ phi, const int* __restrict__ a, const int* __restrict__ b,
    const int* __restrict__ m, const int* __restrict__ n, const int* __restrict__ cells,
    const int* __restrict__ targets, const SCALAR* __restrict__ templates, SCALAR* __restrict__ out,
    const int wavenumbers, const int sources, const int nodes, const int measurements,
    const int cell_count, const int target_count) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= measurements * cell_count) return;
    const int cell = id % cell_count, row = id / cell_count, target = targets[cell];
    if (target < 0 || target >= target_count) return;
    const int local[3] = {cells[3 * cell], cells[3 * cell + 1], cells[3 * cell + 2]};
    SCALAR total = 0;
    for (int w = 0; w < wavenumbers; ++w) {
        const SCALAR* field = phi + (long long)w * sources * nodes;
        const SCALAR* tmpl = templates + ((long long)w * cell_count + cell) * 9;
        SCALAR current[3], receiver[3];
        for (int i = 0; i < 3; ++i) {
            current[i] = field[(long long)a[row] * nodes + local[i]] - field[(long long)b[row] * nodes + local[i]];
            receiver[i] = field[(long long)m[row] * nodes + local[i]] - field[(long long)n[row] * nodes + local[i]];
        }
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) total += receiver[i] * tmpl[3 * i + j] * current[j];
    }
    atomicAdd(out + (long long)row * target_count + target, -total);
}
"""


# ---------------------------------------------------------------------------
# Forward operator
# ---------------------------------------------------------------------------


@dataclass
class _Fields:
    """Assembled operator values and total fields for one conductivity model."""

    values: Tensor
    total: Tensor
    _device_copies: dict = field(default_factory=dict)

    def on(self, device: torch.device) -> Tensor:
        if device.type == "cpu":
            return self.total
        if str(device) not in self._device_copies:
            self._device_copies[str(device)] = self.total.to(device)
        return self._device_copies[str(device)]


@dataclass(frozen=True, eq=False)
class ERTForward2p5D:
    """2.5D ERT forward operator with analytic (flat) or numerical (terrain) primary fields."""

    mesh: Mesh
    survey: Survey
    wavenumbers: Tensor
    weights: Tensor
    discretization: _Discretization
    primary_potential_discretization: _Discretization | None
    geometric_discretization: _Discretization | None
    source_cell_ids: Tensor
    source_node_ids: Tensor
    numerical_h2_refined: bool
    numerical_p2_refined: bool
    topographic_geometric_factor_mode: str
    terrain_cache_dir: Path | None
    normal_field_cache_max_entries: int = 8
    _cache: dict = field(default_factory=dict, init=False, repr=False)
    _cudss_state: dict = field(default_factory=dict, init=False, repr=False)
    _field_cache: OrderedDict = field(
        default_factory=OrderedDict, init=False, repr=False
    )

    @classmethod
    def from_mesh_survey(
        cls,
        mesh: Mesh,
        survey: Survey,
        quadrature_order: int = 2,
        numerical_h2_refined: bool = True,
        numerical_p2_refined: bool = True,
        topographic_geometric_factor_mode: str = "analytic",
        terrain_cache_dir: str | Path | None = None,
        normal_field_cache_max_entries: int = 8,
    ) -> ERTForward2p5D:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "ADTLERT requires an NVIDIA GPU with CUDA (cuDSS sparse solves)"
            )
        gf_mode = _choice(
            topographic_geometric_factor_mode,
            ("analytic", "numerical"),
            "topographic_geometric_factor_mode",
        )
        if int(normal_field_cache_max_entries) < 0:
            raise ValueError("normal_field_cache_max_entries must be non-negative")
        quadrature = build_inverse_cosine_weights(*survey_wavenumber_bounds(survey))

        def build(name, **options):
            return _build_discretization(
                name,
                mesh,
                survey,
                quadrature.wavenumbers,
                quadrature_order=quadrature_order,
                **options,
            )

        terrain = not mesh.is_flat_surface
        potential = geometric = None
        if terrain:
            # Terrain solves run in float64; historically this also switches Torch's default dtype.
            torch.set_default_dtype(torch.float64)
            discretization = build(
                "primary", auxiliary=True, refine=numerical_h2_refined
            )
            potential = build(
                "primary_potential",
                auxiliary=True,
                refine=numerical_h2_refined,
                quadratic=True,
            )
            if gf_mode == "numerical":
                geometric = build(
                    "geometric",
                    auxiliary=True,
                    refine=numerical_h2_refined,
                    quadratic=numerical_p2_refined,
                )
        else:
            discretization = build("native", auxiliary=False)

        positions = np.asarray(survey.electrode_positions, dtype=float)
        nodes = np.asarray(mesh.nodes, dtype=float)
        distances = np.linalg.norm(nodes[None, :, :] - positions[:, None, :], axis=2)
        nearest = np.argmin(distances, axis=1)
        source_node_ids = np.where(
            distances[np.arange(len(nearest)), nearest] <= 1e-2, nearest, -1
        )

        return cls(
            mesh=mesh,
            survey=survey,
            wavenumbers=quadrature.wavenumbers,
            weights=quadrature.weights,
            discretization=discretization,
            primary_potential_discretization=potential,
            geometric_discretization=geometric,
            source_cell_ids=mesh.locate_points(survey.electrode_positions)[0].long(),
            source_node_ids=torch.as_tensor(source_node_ids, dtype=torch.long),
            numerical_h2_refined=numerical_h2_refined,
            numerical_p2_refined=numerical_p2_refined,
            topographic_geometric_factor_mode=gf_mode,
            terrain_cache_dir=_terrain_cache_dir(terrain_cache_dir),
            normal_field_cache_max_entries=int(normal_field_cache_max_entries),
        )

    # -- configuration ------------------------------------------------------

    @property
    def use_numerical_primary(self) -> bool:
        return self.primary_potential_discretization is not None

    @property
    def use_numerical_geometric_factors(self) -> bool:
        return self.geometric_discretization is not None

    @property
    def dtype(self) -> torch.dtype:
        """Field precision: float64 on terrain, ``FLOAT_DTYPE`` otherwise."""

        return torch.float64 if self.use_numerical_primary else FLOAT_DTYPE

    @property
    def _device(self) -> torch.device:
        """Device of the dense adjoint contractions (fields are cached on the host)."""

        return _CUDA

    def _cached(self, key, compute):
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    def _on(self, name: str, tensor_fn, device: torch.device) -> Tensor:
        return self._cached((name, str(device)), lambda: tensor_fn().to(device))

    def _abmn(
        self, device: torch.device | str = "cpu"
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        quads = self._on(
            "abmn", lambda: self.survey.measurements.long(), torch.device(device)
        )
        return quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]

    # -- assembly -----------------------------------------------------------

    def _volume_templates(
        self, d: _Discretization, device: torch.device | str = "cpu"
    ) -> Tensor:
        """``K + k_w^2 M`` per wavenumber, shape ``(W, cells, k, k)``."""

        def build():
            wavenumber_sq = torch.square(self.wavenumbers.to(self.dtype))
            return d.stiffness.to(self.dtype) + wavenumber_sq[
                :, None, None, None
            ] * d.mass.to(self.dtype)

        return self._on(f"{d.name}.volume_templates", build, torch.device(device))

    def _boundary_templates(
        self, d: _Discretization, device: torch.device | str = "cpu"
    ) -> Tensor:
        def build():
            return d.boundary_geometries.to(self.dtype)[
                :, :, None, None
            ] * d.boundary_mass.to(self.dtype)

        return self._on(f"{d.name}.boundary_templates", build, torch.device(device))

    def _assemble(self, d: _Discretization, conductivity: Tensor) -> Tensor:
        """CSR values of ``A_w(conductivity)`` for every wavenumber, shape ``(W, nnz)``."""

        sigma = conductivity.to(self.dtype)
        volume = sigma[d.parent_cell_ids][None, :, None, None] * self._volume_templates(
            d
        )
        boundary = (
            sigma[d.parent_cell_ids[d.boundary_cells]][None, :]
            * d.boundary_geometries.to(self.dtype)
        )[:, :, None, None] * d.boundary_mass.to(self.dtype)
        count = self.wavenumbers.shape[0]
        values = torch.zeros((count, d.pattern.nnz), dtype=self.dtype)
        values.index_add_(1, d.pattern.volume_inverse, volume.reshape(count, -1))
        return values.index_add_(
            1, d.pattern.boundary_inverse, boundary.reshape(count, -1)
        )

    def _sigma(self, conductivity) -> Tensor:
        return _cell_vector(conductivity, self.mesh.cell_count)

    # -- linear solves ------------------------------------------------------

    def _solve(
        self,
        d: _Discretization,
        values: Tensor,
        rhs: Tensor,
        *,
        refactorize: bool = True,
    ) -> Tensor:
        """Batched sparse solve ``A_b X_b = rhs_b``; plans are reused per pattern and RHS shape.

        Matrix buffers are refreshed in place, so the symbolic analysis survives across
        models. The numeric factorization is redone unless ``refactorize=False``.
        """

        cp, cupy_sparse = _cupy()
        values_cp, rhs_cp = (
            cp.from_dlpack(tensor.to(_CUDA, self.dtype).contiguous())
            for tensor in (values, rhs)
        )
        key = (d.name, tuple(rhs.shape))
        state = self._cudss_state.get(key)
        if state is None:
            indptr, indices = (
                cp.asarray(d.pattern.indptr),
                cp.asarray(d.pattern.indices),
            )
            data = [cp.array(value, copy=True) for value in values_cp]
            matrices = [
                cupy_sparse.csr_matrix((value, indices, indptr), shape=d.pattern.shape)
                for value in data
            ]
            rhs_buffer = rhs_cp.copy()
            rhs_view = rhs_buffer.transpose((0, 2, 1))
            state = {
                "data": data,
                "matrices": matrices,
                "rhs": rhs_buffer,
                "rhs_view": rhs_view,
            }
            state["solver"] = _cudss_solver(matrices, rhs_view, spd=d.spd)
            self._cudss_state[key] = state
            refactorize = True
        else:
            for buffer, value in zip(state["data"], values_cp, strict=True):
                buffer[...] = value
            state["rhs"][...] = rhs_cp
            state["solver"].reset_operands(b=state["rhs_view"])
        if refactorize:
            state["solver"].factorize()
        solution = cp.ascontiguousarray(state["solver"].solve().transpose((0, 2, 1)))
        return torch.from_dlpack(solution).to(rhs.device, copy=True)

    # -- primary fields and terrain caches ----------------------------------

    def _disk_path(self, name: str, d: _Discretization) -> Path | None:
        if self.terrain_cache_dir is None:
            return None
        digest = hashlib.sha256()
        for label, value in (
            ("version", _TERRAIN_CACHE_VERSION),
            ("name", name),
            ("dtype", str(self.dtype)),
            (
                "options",
                (
                    self.numerical_h2_refined,
                    self.numerical_p2_refined,
                    self.topographic_geometric_factor_mode,
                ),
            ),
        ):
            digest.update(f"{label}={value!r};".encode())
        for array in (
            self.mesh.nodes, self.mesh.cells, self.mesh.surface_node_ids, self.survey.electrode_positions,
            self.survey.measurements, self.wavenumbers, self.weights, d.parent_cell_ids, d.dof_nodes, d.cell_dofs,
            d.boundary_dofs, d.electrode_matrix, d.pattern.indptr, d.pattern.indices, d.stiffness, d.mass,
            d.boundary_mass, d.boundary_geometries,
        ):  # fmt: skip
            array = np.ascontiguousarray(np.asarray(array))
            digest.update(f"{array.dtype}{array.shape}".encode())
            digest.update(array.tobytes())
        return self.terrain_cache_dir / f"{digest.hexdigest()}.npz"

    def _disk_cached(self, name: str, d: _Discretization, compute) -> Tensor:
        """Memory- and (optionally) disk-cached unit potential stack of shape ``(W, E, dofs)``."""

        def load_or_compute():
            path = self._disk_path(name, d)
            shape = (
                self.wavenumbers.shape[0],
                self.survey.electrode_count,
                d.dof_count,
            )
            if path is not None and path.exists():
                try:
                    with np.load(path, allow_pickle=False) as payload:
                        cached = payload["value"]
                    if cached.shape == shape:
                        return torch.as_tensor(cached, dtype=self.dtype)
                except (
                    Exception
                ):  # corrupt or partial cache files are simply recomputed
                    pass
            value = compute()
            if path is not None:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
                    np.savez(tmp, value=value.numpy())
                    tmp.replace(path)
                except OSError:
                    pass
            return value

        return self._cached(name, load_or_compute)

    def _unit_potentials(self, d: _Discretization) -> Tensor:
        """Unit-conductivity potentials of every electrode on ``d``."""

        def compute():
            values = self._assemble(
                d, torch.ones(self.mesh.cell_count, dtype=FLOAT_DTYPE)
            )
            rhs = d.electrode_matrix.to(self.dtype).expand(
                self.wavenumbers.shape[0], -1, -1
            )
            return self._solve(d, values, rhs)

        return self._disk_cached(f"{d.name}_sub_potentials", d, compute)

    def _unit_primary(self) -> Tensor:
        """Unit-resistivity primary field ``u_p`` on the field discretization, ``(W, E, dofs)``."""

        if self.use_numerical_primary:
            potential = self.primary_potential_discretization

            def project():
                selection = _exact_node_indices(
                    potential.dof_nodes, self.discretization.dof_nodes
                )
                return self._unit_potentials(potential)[..., torch.as_tensor(selection)]

            return self._disk_cached(
                "auxiliary_sub_potentials", self.discretization, project
            )
        return self._cached("analytic_primary", self._analytic_primary)

    def _analytic_primary(self) -> Tensor:
        nodes = np.asarray(self.mesh.nodes, dtype=float)
        cells = np.asarray(self.mesh.cells)
        surface = float(np.max(nodes[:, 1]))
        node_ids = self.source_node_ids.tolist()
        radii = {}
        for node in set(node_ids) - {-1}:
            neighbors = np.setdiff1d(
                np.unique(cells[np.any(cells == node, axis=1)]), [node]
            )
            radii[node] = (
                np.min(np.linalg.norm(nodes[neighbors] - nodes[node], axis=1))
                if neighbors.size
                else None
            )
        positions = np.asarray(self.survey.electrode_positions, dtype=float)
        primary = np.zeros((self.wavenumbers.shape[0], len(positions), nodes.shape[0]))
        for w, wavenumber in enumerate(self.wavenumbers.double().tolist()):
            for s, (source, node) in enumerate(zip(positions, node_ids, strict=True)):
                primary[w, s] = _halfspace_primary(nodes, source, wavenumber, surface)
                if (
                    node >= 0
                ):  # pyGIMLi's singular value: K0 at one sixth of the nearest-neighbor distance
                    radius = radii[node]
                    primary[w, s, node] = (
                        0.0
                        if radius is None
                        else besselk0(radius / 6.0 * wavenumber) / np.pi
                    )
        return torch.as_tensor(primary, dtype=FLOAT_DTYPE)

    def _reference_rhs(self) -> Tensor:
        """``A(1) u_p``, the conductivity-independent part of the secondary-field RHS."""

        def compute():
            d = self.discretization
            ones = torch.ones(self.mesh.cell_count, dtype=FLOAT_DTYPE)
            return d.pattern.matvec(self._assemble(d, ones), self._unit_primary())

        return self._cached("reference_rhs", compute)

    def _source_resistivities(self, sigma: Tensor) -> Tensor:
        """Resistivity at each electrode: geometric mean over cells sharing its node, else its cell."""

        def node_cell_weights():
            cells = np.asarray(self.mesh.cells)
            weights = np.zeros(
                (self.survey.electrode_count, self.mesh.cell_count),
                dtype=NP_FLOAT_DTYPE,
            )
            counts = np.ones(self.survey.electrode_count, dtype=NP_FLOAT_DTYPE)
            for source, node in enumerate(self.source_node_ids.tolist()):
                if node >= 0:
                    mask = np.any(cells == node, axis=1)
                    weights[source, mask], counts[source] = 1.0, np.count_nonzero(mask)
            return torch.as_tensor(weights), torch.as_tensor(counts)

        weights, counts = self._cached("source_cell_weights", node_cell_weights)
        node_rho = torch.exp(
            torch.einsum("ec,c->e", weights, -torch.log(sigma)) / counts
        )
        return torch.where(
            self.source_node_ids >= 0, node_rho, 1.0 / sigma[self.source_cell_ids]
        )

    # -- field solves -------------------------------------------------------

    def _field_cache_key(self, sigma: Tensor) -> bytes | None:
        if self.normal_field_cache_max_entries < 1:
            return None
        host = np.ascontiguousarray(sigma.detach().cpu().numpy())
        return (
            hashlib.blake2b(host.view(np.uint8), digest_size=16).digest()
            + host.dtype.str.encode()
        )

    def _fields(self, conductivity) -> _Fields:
        """Total fields for a conductivity model, served from an LRU cache when possible."""

        sigma = self._sigma(conductivity)
        key = self._field_cache_key(sigma)
        if key is not None:
            counter = "hits" if key in self._field_cache else "misses"
            self._cache[counter] = self._cache.get(counter, 0) + 1
            if key in self._field_cache:
                self._field_cache.move_to_end(key)
                return self._field_cache[key]

        d = self.discretization
        values = self._assemble(d, sigma)
        primary = (
            self._unit_primary() * self._source_resistivities(sigma)[None, :, None]
        )
        rhs = self._reference_rhs() - d.pattern.matvec(values, primary)
        fields = _Fields(values, self._solve(d, values, rhs) + primary)
        _check_finite(fields.total, "total fields")
        if key is not None:
            self._field_cache[key] = fields
            while len(self._field_cache) > self.normal_field_cache_max_entries:
                self._field_cache.popitem(last=False)
        return fields

    def normal_field_cache_info(self) -> dict[str, int]:
        """Return LRU field-cache counters for profiling and regression tests."""

        return {
            "entries": len(self._field_cache),
            "max_entries": self.normal_field_cache_max_entries,
            "hits": self._cache.get("hits", 0),
            "misses": self._cache.get("misses", 0),
        }

    # -- measurement maps ---------------------------------------------------

    def _integrate(self, fields: Tensor) -> Tensor:
        return torch.tensordot(self.weights.to(fields.dtype), fields, dims=([0], [0]))

    def _receivers(self) -> tuple[Tensor, Tensor]:
        """Receiver rows ``E[M] - E[N]`` (normal) and ``E[A] - E[B]`` (reciprocal)."""

        def build():
            a, b, m, n = self._abmn()
            electrodes = self.discretization.electrode_matrix
            return (electrodes[m] - electrodes[n]).to(self.dtype), (
                electrodes[a] - electrodes[b]
            ).to(self.dtype)

        return self._cached("receivers", build)

    def _normal_reciprocal(self, integrated: Tensor) -> tuple[Tensor, Tensor]:
        """Normal (AB source, MN receiver) and reciprocal (MN source, AB receiver) resistances."""

        a, b, m, n = self._abmn()
        rows = torch.arange(self.survey.measurement_count)
        normal_receiver, current_receiver = self._receivers()
        normal = integrated @ normal_receiver.T
        reciprocal = integrated @ current_receiver.T
        return normal[a, rows] - normal[b, rows], reciprocal[m, rows] - reciprocal[
            n, rows
        ]

    def _adjoint_rhs(self, cotangent: Tensor, *, reciprocal: bool = False) -> Tensor:
        """Transpose of the (reciprocal) measurement map for ``(Q, D)`` cotangents → ``(W, Q*E, dofs)``."""

        a, b, m, n = self._abmn()
        normal_receiver, current_receiver = self._receivers()
        positive, negative, receiver = (
            (m, n, current_receiver) if reciprocal else (a, b, normal_receiver)
        )
        weighted = cotangent.to(self.dtype)[:, :, None] * receiver[None]
        sources = torch.zeros(
            (cotangent.shape[0], self.survey.electrode_count, receiver.shape[1]),
            dtype=self.dtype,
        )
        sources.index_add_(1, positive, weighted).index_add_(1, negative, -weighted)
        rhs = self.weights.to(self.dtype)[:, None, None, None] * sources[None]
        return rhs.reshape(self.wavenumbers.shape[0], -1, receiver.shape[1])

    @staticmethod
    def _combine(normal: Tensor, reciprocal: Tensor) -> Tensor:
        return torch.sqrt(torch.abs(normal * reciprocal))

    @staticmethod
    def _combine_scale(normal: Tensor, reciprocal: Tensor) -> tuple[Tensor, Tensor]:
        """``sign(n r)`` and the floored combined resistance used by the reciprocal chain rule."""

        combined = torch.sqrt(torch.abs(normal * reciprocal))
        return torch.sign(normal * reciprocal), torch.clamp_min(combined, 1e-30)

    def _geometric_factors(self) -> Tensor:
        if not self.use_numerical_geometric_factors:
            return self.survey.geometric_factors()
        return self._cached("geometric_factors", self._numerical_geometric_factors)

    def _numerical_geometric_factors(self) -> Tensor:
        d = self.geometric_discretization
        integrated = self._integrate(self._unit_potentials(d))
        potentials = integrated @ d.electrode_matrix.to(integrated.dtype).T
        a, b, m, n = self._abmn()
        rows = torch.arange(self.survey.measurement_count)
        sources = potentials[a] - potentials[b]
        return 1.0 / (sources[rows, m] - sources[rows, n])

    def _response(
        self, integrated: Tensor, normal: Tensor, reciprocal: Tensor, currents
    ) -> ForwardResponse:
        resistance = self._combine(normal, reciprocal)
        current = torch.as_tensor(currents, dtype=FLOAT_DTYPE)
        electrode_matrix = self.discretization.electrode_matrix.to(integrated.dtype)
        return ForwardResponse(
            apparent_resistivity=torch.abs(self._geometric_factors())
            * resistance
            / current,
            resistance=resistance,
            electrode_potentials=integrated @ electrode_matrix.T,
            integrated_potentials=integrated,
            wavenumbers=self.wavenumbers,
            weights=self.weights,
        )

    def _measure(self, conductivity) -> tuple[_Fields, Tensor, Tensor, Tensor]:
        fields = self._fields(conductivity)
        integrated = self._integrate(fields.total)
        _check_finite(integrated, "integrated potentials")
        return fields, integrated, *self._normal_reciprocal(integrated)

    # -- public forward API -------------------------------------------------

    def solve(self, conductivity, currents=1.0) -> ForwardResponse:
        """Solve the multi-wavenumber 2.5D forward problem for a conductivity model."""

        _, integrated, normal, reciprocal = self._measure(conductivity)
        return self._response(integrated, normal, reciprocal, currents)

    def resistance(self, conductivity) -> Tensor:
        """Return the reciprocal-averaged resistance of each measurement."""

        _, _, normal, reciprocal = self._measure(conductivity)
        return self._combine(normal, reciprocal)

    def apparent_resistivity_values(self, conductivity, currents=1.0) -> Tensor:
        """Return apparent resistivity values without allocating a ForwardResponse."""

        current = torch.as_tensor(currents, dtype=FLOAT_DTYPE)
        return (
            torch.abs(self._geometric_factors())
            * self.resistance(conductivity)
            / current
        )

    def apparent_resistivity_series(self, conductivities, currents=1.0) -> Tensor:
        """Apparent resistivity for a ``(n_steps, n_cells)`` model series (currents may be per step)."""

        models = self._series(conductivities)
        current = torch.as_tensor(currents, dtype=FLOAT_DTYPE)
        return torch.stack(
            [
                self.apparent_resistivity_values(
                    model, current[step] if current.ndim == 2 else current
                )
                for step, model in enumerate(models)
            ]
        )

    def _series(self, conductivities) -> Tensor:
        models = torch.as_tensor(conductivities, dtype=FLOAT_DTYPE)
        if models.ndim != 2 or models.shape[1] != self.mesh.cell_count:
            raise ValueError(
                f"conductivities must have shape (n_steps, {self.mesh.cell_count})"
            )
        return models

    def prepare(self, conductivity=None, *, include_solver_state: bool = True) -> None:
        """Populate geometry, primary-field, and (optionally) solver/field caches before timed solves."""

        self._unit_primary()
        self._reference_rhs()
        self._receivers()
        self._geometric_factors()
        if include_solver_state:
            self._fields(
                torch.ones(self.mesh.cell_count)
                if conductivity is None
                else conductivity
            )

    # -- derivatives --------------------------------------------------------

    def _target_cells(
        self, cell_parameter_ids, parameter_count
    ) -> tuple[np.ndarray, int]:
        """Map discretization cells to output columns (forward cells or inversion parameters)."""

        parents = self.discretization.parent_cell_ids.numpy()
        if cell_parameter_ids is None:
            if parameter_count is not None:
                raise ValueError("parameter_count requires cell_parameter_ids")
            return parents, self.mesh.cell_count
        ids = np.asarray(
            cell_parameter_ids.cpu()
            if isinstance(cell_parameter_ids, Tensor)
            else cell_parameter_ids
        )
        ids = ids.astype(np.int64).reshape(-1)
        if ids.shape != (self.mesh.cell_count,):
            raise ValueError(
                f"cell_parameter_ids must have shape ({self.mesh.cell_count},)"
            )
        if np.any(ids < -1) or not np.any(ids >= 0):
            raise ValueError(
                "cell_parameter_ids may only contain -1 or non-negative ids, with at least one active"
            )
        count = int(ids.max()) + 1 if parameter_count is None else int(parameter_count)
        if count <= int(ids.max()):
            raise ValueError(
                "parameter_count is smaller than the largest cell parameter id"
            )
        return ids[parents], count

    def _active_targets(self, targets: np.ndarray, device: torch.device):
        """Active discretization cells (``None`` when all are active) and their output column."""

        digest = hashlib.blake2b(targets.tobytes(), digest_size=16).digest()

        def build():
            active = np.flatnonzero(targets >= 0)
            cells = (
                None
                if active.size == targets.size
                else torch.as_tensor(active, device=device)
            )
            return cells, torch.as_tensor(targets[active], device=device)

        return self._cached(("targets", digest, str(device)), build)

    def _cell_gradient(
        self, phi: Tensor, lam: Tensor, *, robin: bool = True, targets=None, count=None
    ) -> Tensor:
        """``-sum_w lam^T dA_w/dsigma phi`` per output column; ``lam`` may carry a leading batch axis.

        The dense contraction runs on the GPU when one is available.
        """

        d, device = self.discretization, self._device
        if targets is None:
            targets, count = d.parent_cell_ids.numpy(), self.mesh.cell_count
        active, columns = self._active_targets(targets, device)
        templates = self._volume_templates(d, device)
        cell_dofs = self._on(f"{d.name}.cell_dofs", lambda: d.cell_dofs, device)
        if active is not None:
            templates, cell_dofs = templates[:, active], cell_dofs[active]
        phi, lam = phi.to(device), lam.to(device)
        batch = "q" if lam.ndim == 4 else ""
        gradient = -torch.einsum(
            f"w{batch}sci,wcij,wscj->{batch}c",
            lam[..., cell_dofs],
            templates,
            phi[..., cell_dofs],
        )
        if robin:
            boundary_dofs = self._on(
                f"{d.name}.boundary_dofs", lambda: d.boundary_dofs, device
            )
            boundary = -torch.einsum(
                f"w{batch}sbi,wbij,wsbj->{batch}b",
                lam[..., boundary_dofs],
                self._boundary_templates(d, device),
                phi[..., boundary_dofs],
            )
            gradient = gradient.index_add(
                gradient.ndim - 1, d.boundary_cells.to(device), boundary
            )
        out = torch.zeros(
            (*gradient.shape[:-1], count), dtype=gradient.dtype, device=device
        )
        return out.index_add_(out.ndim - 1, columns, gradient).cpu()

    def vjp(self, conductivity, cotangent) -> Tensor:
        """Apply the transposed resistance Jacobian to a measurement cotangent."""

        fields, _, normal, reciprocal = self._measure(conductivity)
        sign, scale = self._combine_scale(normal, reciprocal)
        common = (
            0.5
            * _measurement_vector(cotangent, self.survey.measurement_count, self.dtype)
            * sign
            / scale
        )
        rhs = self._adjoint_rhs((common * reciprocal)[None]) + self._adjoint_rhs(
            (common * normal)[None], reciprocal=True
        )
        lam = self._solve(self.discretization, fields.values, rhs)
        _check_finite(lam, "adjoint fields")
        gradient = self._cell_gradient(fields.on(self._device), lam)
        _check_finite(gradient, "conductivity gradient")
        return gradient

    def jvp(self, conductivity, delta_conductivity) -> Tensor:
        """Apply the resistance Jacobian to a conductivity perturbation."""

        d = self.discretization
        fields, _, normal, reciprocal = self._measure(conductivity)
        tangent_rhs = -d.pattern.matvec(
            self._assemble(d, self._sigma(delta_conductivity)), fields.total
        )
        delta = self._solve(d, fields.values, tangent_rhs)
        _check_finite(delta, "tangent fields")
        delta_normal, delta_reciprocal = self._normal_reciprocal(self._integrate(delta))
        sign, scale = self._combine_scale(normal, reciprocal)
        return (
            0.5 * sign * (delta_normal * reciprocal + normal * delta_reciprocal) / scale
        )

    def normal_vjp(
        self, conductivity, cotangent, *, cell_parameter_ids=None, parameter_count=None
    ) -> Tensor:
        """Transpose of the normal-quadrupole sensitivity (see :meth:`normal_jvp`).

        ``cell_parameter_ids`` fuses the aggregation from forward cells into inversion
        parameters; cells with id ``-1`` are ignored.
        """

        targets, count = self._target_cells(cell_parameter_ids, parameter_count)
        device = self._device
        phi = self._fields(conductivity).on(device)
        weights = _measurement_vector(
            cotangent, self.survey.measurement_count, phi.dtype
        ).to(device)
        a, b, m, n = self._abmn(device)
        sources = self.survey.electrode_count
        pairs = torch.zeros(sources * sources, dtype=phi.dtype, device=device).index_add_(
            0, torch.cat((m * sources + a, m * sources + b, n * sources + a, n * sources + b)),
            torch.cat((weights, -weights, -weights, weights)),
        ).reshape(sources, sources)  # fmt: skip
        current = torch.einsum("ef,wfn->wen", pairs, phi)
        receiver = self.weights.to(device, phi.dtype)[:, None, None] * phi
        gradient = self._cell_gradient(
            current, receiver, robin=False, targets=targets, count=count
        )
        _check_finite(gradient, "normal sensitivity vjp")
        return gradient

    def normal_jvp(self, conductivity, delta_conductivity) -> Tensor:
        """Matrix-free normal-quadrupole sensitivity without the Robin boundary derivative.

        This matches ``jacobian(..., normal_sensitivity=True, include_robin_boundary_derivative=False)``,
        the inversion sensitivity convention, rather than differentiating :meth:`resistance`.
        """

        d, device = self.discretization, self._device
        phi = self._fields(conductivity).on(device)
        direction = self._sigma(delta_conductivity).to(device, phi.dtype)[
            d.parent_cell_ids.to(device)
        ]
        volume = direction[None, :, None, None] * self._volume_templates(d, device)
        count = self.wavenumbers.shape[0]
        tangent = torch.zeros(
            (count, d.pattern.nnz), dtype=phi.dtype, device=device
        ).index_add_(1, d.pattern.volume_inverse.to(device), volume.reshape(count, -1))
        gram = torch.einsum(
            "w,wen,wfn->ef",
            self.weights.to(device, phi.dtype),
            phi,
            -d.pattern.matvec(tangent, phi),
        )
        a, b, m, n = self._abmn(device)
        result = (gram[m, a] - gram[m, b] - gram[n, a] + gram[n, b]).cpu()
        _check_finite(result, "normal sensitivity jvp")
        return result

    def normal_vjp_series(
        self,
        conductivities,
        cotangents,
        *,
        cell_parameter_ids=None,
        parameter_count=None,
    ) -> Tensor:
        """Apply :meth:`normal_vjp` to every step of a model series."""

        models = self._series(conductivities)
        cotangents = torch.as_tensor(cotangents, dtype=FLOAT_DTYPE)
        if tuple(cotangents.shape) != (models.shape[0], self.survey.measurement_count):
            raise ValueError(
                f"cotangents must have shape {(models.shape[0], self.survey.measurement_count)}"
            )
        return torch.stack(
            [
                self.normal_vjp(
                    model,
                    cotangent,
                    cell_parameter_ids=cell_parameter_ids,
                    parameter_count=parameter_count,
                )
                for model, cotangent in zip(models, cotangents, strict=True)
            ]
        )

    # -- explicit Jacobians -------------------------------------------------

    def _batch_size(self, batch_size: int | None, direct: bool) -> int:
        if batch_size is None:
            if not direct:
                return 8
            return (
                min(self.survey.measurement_count, 64)
                if self.mesh.cell_count >= 4096
                else self.survey.measurement_count
            )
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        return int(batch_size)

    def _normal_jacobian(
        self, fields: _Fields, batch_size: int, targets: np.ndarray, count: int
    ) -> Tensor:
        """Direct normal sensitivity ``-sum_w w_w (u_M - u_N)^T dA_w (u_A - u_B)`` per measurement chunk."""

        d, device = self.discretization, self._device
        active, columns = self._active_targets(targets, device)
        templates = self.weights.to(device, self.dtype)[
            :, None, None, None
        ] * self._volume_templates(d, device)
        cell_dofs = self._on(f"{d.name}.cell_dofs", lambda: d.cell_dofs, device)
        if active is not None:
            templates, cell_dofs = templates[:, active], cell_dofs[active]
        phi = fields.on(device)
        a, b, m, n = self._abmn(device)
        rows = []
        for start in range(0, self.survey.measurement_count, batch_size):
            chunk = slice(start, start + batch_size)
            if cell_dofs.shape[1] == 3:
                rows.append(
                    self._normal_jacobian_cuda(
                        phi,
                        a[chunk],
                        b[chunk],
                        m[chunk],
                        n[chunk],
                        cell_dofs,
                        columns,
                        templates,
                        count,
                    )
                )
                continue
            current = (phi[:, a[chunk]] - phi[:, b[chunk]])[..., cell_dofs]
            receiver = (phi[:, m[chunk]] - phi[:, n[chunk]])[..., cell_dofs]
            local = -torch.einsum("wbci,wcij,wbcj->bc", receiver, templates, current)
            rows.append(
                torch.zeros(
                    (local.shape[0], count), dtype=local.dtype, device=device
                ).index_add_(1, columns, local)
            )
        return torch.cat(rows).cpu()

    def _normal_jacobian_cuda(
        self, phi, a, b, m, n, cell_dofs, columns, templates, count
    ) -> Tensor:
        """Fused CuPy kernel avoiding ``W x B x C x 3`` gathered field blocks for triangle cells."""

        cp, _ = _cupy()
        scalar = "double" if phi.dtype == torch.float64 else "float"
        kernel = self._cached(
            ("sensitivity_kernel", scalar),
            lambda: cp.RawKernel(
                _SENSITIVITY_KERNEL.replace("SCALAR", scalar), "normal_sensitivity"
            ),
        )
        as_int = [
            cp.from_dlpack(x.to(torch.int32).contiguous())
            for x in (a, b, m, n, cell_dofs.reshape(-1), columns)
        ]
        out = torch.zeros((a.shape[0], count), dtype=phi.dtype, device=phi.device)
        cells = int(cell_dofs.shape[0])
        threads = a.shape[0] * cells
        kernel(
            ((threads + 255) // 256,),
            (256,),
            (cp.from_dlpack(phi.contiguous()), *as_int[:5], as_int[5], cp.from_dlpack(templates.contiguous()),
             cp.from_dlpack(out), np.int32(phi.shape[0]), np.int32(phi.shape[1]), np.int32(phi.shape[2]),
             np.int32(a.shape[0]), np.int32(cells), np.int32(count)),
        )  # fmt: skip
        return out

    def _adjoint_jacobian(
        self,
        fields,
        normal,
        reciprocal,
        batch_size: int,
        robin: bool,
        normal_sensitivity: bool,
    ) -> Tensor:
        """Jacobian rows from batched adjoint solves (identity cotangents, last block zero-padded)."""

        count = self.survey.measurement_count
        sign, scale = self._combine_scale(normal, reciprocal)
        eye = torch.eye(count, dtype=normal.dtype)
        phi = fields.on(self._device)
        rows = []
        for start in range(0, count, batch_size):
            cotangent = torch.zeros((batch_size, count), dtype=normal.dtype)
            cotangent[: min(batch_size, count - start)] = eye[
                start : start + batch_size
            ]
            if normal_sensitivity:
                rhs = self._adjoint_rhs(cotangent)
            else:
                common = 0.5 * cotangent * sign[None, :] / scale[None, :]
                rhs = self._adjoint_rhs(
                    common * reciprocal[None, :]
                ) + self._adjoint_rhs(common * normal[None, :], reciprocal=True)
            lam = self._solve(
                self.discretization, fields.values, rhs, refactorize=start == 0
            )
            lam = lam.reshape(
                self.wavenumbers.shape[0], batch_size, self.survey.electrode_count, -1
            )
            rows.append(self._cell_gradient(phi, lam, robin=robin)[: count - start])
        return torch.cat(rows)

    def solve_with_jacobian(
        self,
        conductivity,
        currents=1.0,
        *,
        batch_size: int | None = None,
        include_robin_boundary_derivative: bool = False,
        normal_sensitivity: bool = True,
        jacobian_cell_parameter_ids=None,
        jacobian_parameter_count: int | None = None,
    ) -> tuple[ForwardResponse, Tensor]:
        """Solve the forward problem and materialize ``d resistance / d conductivity``.

        The default normal-quadrupole sensitivity omits the Robin boundary derivative. Pass
        ``include_robin_boundary_derivative=True, normal_sensitivity=False`` for the exact
        derivative of the reciprocal-averaged response. ``jacobian_cell_parameter_ids``
        accumulates the direct normal sensitivity into parameter columns.
        """

        fields, integrated, normal, reciprocal = self._measure(conductivity)
        direct = normal_sensitivity and not include_robin_boundary_derivative
        batch_size = self._batch_size(batch_size, direct)
        if direct:
            targets, count = self._target_cells(
                jacobian_cell_parameter_ids, jacobian_parameter_count
            )
            jacobian = self._normal_jacobian(fields, batch_size, targets, count)
        else:
            jacobian = self._adjoint_jacobian(
                fields,
                normal,
                reciprocal,
                batch_size,
                include_robin_boundary_derivative,
                normal_sensitivity,
            )
        return self._response(integrated, normal, reciprocal, currents), jacobian

    def jacobian(
        self,
        conductivity,
        *,
        batch_size: int | None = None,
        include_robin_boundary_derivative: bool = False,
        normal_sensitivity: bool = True,
    ) -> Tensor:
        """Materialize the resistance Jacobian (see :meth:`solve_with_jacobian`)."""

        return self.solve_with_jacobian(
            conductivity,
            batch_size=batch_size,
            include_robin_boundary_derivative=include_robin_boundary_derivative,
            normal_sensitivity=normal_sensitivity,
        )[1]

    def jacobian_columnwise(self, conductivity) -> Tensor:
        """Materialize the exact resistance Jacobian with one JVP per cell (for testing)."""

        basis = torch.eye(self.mesh.cell_count, dtype=FLOAT_DTYPE)
        return torch.stack([self.jvp(conductivity, column) for column in basis], dim=1)

    # -- resources ----------------------------------------------------------

    def close(self) -> None:
        """Drop cached fields and release cuDSS solver resources."""

        self._field_cache.clear()
        for key in list(self._cudss_state):
            try:
                self._cudss_state.pop(key)["solver"].free()
            except Exception:  # best-effort release during teardown
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _terrain_cache_dir(cache_dir: str | Path | None) -> Path | None:
    cache_dir = (
        cache_dir
        if cache_dir is not None
        else os.environ.get("ADTLERT_TERRAIN_CACHE_DIR", "").strip()
    )
    return Path(cache_dir).expanduser() if cache_dir else None
