"""Finite-element discretizations used by the 2.5D operator.

A :class:`Discretization` bundles the element space (DOFs, local stiffness/mass/boundary
templates, Robin coefficients), the sparse operator pattern, and the electrode interpolation.
The input mesh is used directly for flat surfaces; terrain solves use refined auxiliary
meshes (bilinear quadrilateral strips or triangles, P1 or quadratic).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
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
from adtlert.forward.kernels import sampled_products
from adtlert.mesh import Mesh
from adtlert.mesh.core import edge_midpoint_builder
from adtlert.survey import Survey
from adtlert.utils.dtypes import FLOAT_DTYPE, NP_FLOAT_DTYPE

Tensor = torch.Tensor


@dataclass(frozen=True)
class SparsePattern:
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

    def _indices(self, device: torch.device) -> tuple[Tensor, Tensor]:
        key = str(device)
        if key not in self._device_indices:
            self._device_indices[key] = tuple(
                torch.as_tensor(x, dtype=torch.long, device=device)
                for x in (self.indptr, self.indices)
            )
        return self._device_indices[key]

    def matvec(self, values: Tensor, vectors: Tensor) -> Tensor:
        """Apply ``A_b`` to ``vectors[b]`` for each operator ``b`` (rows are vectors)."""

        indptr, indices = self._indices(values.device)
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

    def sample(self, left: Tensor, right: Tensor) -> Tensor:
        """``(left @ right)`` at the nonzeros, in CSR order (SDDMM)."""

        return sampled_products(*self._indices(left.device), self.shape, left, right)


def sparse_pattern(
    cell_dofs: np.ndarray, boundary_dofs: np.ndarray, dof_count: int
) -> SparsePattern:
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
    return SparsePattern(
        shape=(dof_count, dof_count),
        volume_inverse=inverse[: volume.shape[0]],
        boundary_inverse=inverse[volume.shape[0] :],
        indptr=indptr,
        indices=unique[:, 1].astype(np.int32),
    )


@dataclass(frozen=True)
class Discretization:
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
    pattern: SparsePattern
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
        b, r, t, left = (
            midpoint(bl, br),
            midpoint(br, tr),
            midpoint(tr, tl),
            midpoint(tl, bl),
        )
        center = len(nodes)
        nodes.append(0.25 * (nodes[bl] + nodes[br] + nodes[tr] + nodes[tl]))
        cells += [
            [bl, b, center, left],
            [b, br, r, center],
            [center, r, tr, t],
            [left, center, t, tl],
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


def build_discretization(
    name: str,
    mesh: Mesh,
    survey: Survey,
    wavenumbers: Tensor,
    *,
    auxiliary: bool,
    refine: bool = False,
    quadratic: bool = False,
    quadrature_order: int = 2,
) -> Discretization:
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

    return Discretization(
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
        pattern=sparse_pattern(
            np.asarray(cell_dofs), np.asarray(boundary_dofs), int(dof_nodes.shape[0])
        ),
        spd=not auxiliary,
    )


def exact_node_indices(dof_nodes, points, tol: float = 1e-6) -> np.ndarray:
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
