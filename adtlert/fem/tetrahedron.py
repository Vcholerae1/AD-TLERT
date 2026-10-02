"""P1/P2 finite-element data for four-node tetrahedra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from adtlert.utils.dtypes import FLOAT_DTYPE, NP_FLOAT_DTYPE

Tensor = torch.Tensor
_EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))


@dataclass(frozen=True)
class TetrahedronP1Data:
    """Cellwise gradients, volumes, and unit-conductivity matrices."""

    gradients: Tensor
    cell_volumes: Tensor
    stiffness_templates: Tensor


@dataclass(frozen=True)
class TetrahedronP2Data:
    """Quadratic tetrahedron topology and cellwise stiffness templates."""

    dof_nodes: Tensor
    cell_connectivity: Tensor
    boundary_connectivity: Tensor
    gradient_values: Tensor
    cell_volumes: Tensor
    stiffness_templates: Tensor


def build_tetrahedron_p1_data(mesh) -> TetrahedronP1Data:
    """Build constant P1 gradients and local stiffness templates."""

    cell_nodes = np.asarray(mesh.nodes, dtype=NP_FLOAT_DTYPE)[np.asarray(mesh.cells)]
    jacobians = np.stack([cell_nodes[:, index] - cell_nodes[:, 0] for index in (1, 2, 3)], axis=-1)
    reference = np.asarray(((-1.0, -1.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), dtype=NP_FLOAT_DTYPE)
    gradients = np.einsum("ij,ejk->eik", reference, np.linalg.inv(jacobians))
    volumes = np.abs(np.linalg.det(jacobians)) / 6.0
    stiffness = volumes[:, None, None] * np.einsum("eid,ejd->eij", gradients, gradients)
    return TetrahedronP1Data(*(torch.as_tensor(array, dtype=FLOAT_DTYPE) for array in (gradients, volumes, stiffness)))


def p2_tetrahedron_shape_values(barycentric: np.ndarray) -> np.ndarray:
    """Evaluate ten-node tetrahedron shape functions (vertices, then edges 01, 12, 20, 03, 13, 23)."""

    coordinates = np.asarray(barycentric, dtype=float)
    edges = np.stack([4.0 * coordinates[..., i] * coordinates[..., j] for i, j in _EDGES], axis=-1)
    return np.concatenate((coordinates * (2.0 * coordinates - 1.0), edges), axis=-1)


def p2_triangle_mass_template(area: float) -> np.ndarray:
    """Six-node triangle mass matrix (vertices, then edges 01, 12, 20) by degree-4 quadrature."""

    a, b, c, d = 0.445948490915965, 0.108103018168070, 0.091576213509771, 0.816847572980459
    barycentric = np.asarray(((a, a, b), (a, b, a), (b, a, a), (c, c, d), (c, d, c), (d, c, c)))
    weights = np.asarray((0.223381589678011,) * 3 + (0.109951743655322,) * 3)
    edges = np.stack([4.0 * barycentric[:, i] * barycentric[:, j] for i, j in ((0, 1), (1, 2), (2, 0))], axis=-1)
    shape = np.concatenate((barycentric * (2.0 * barycentric - 1.0), edges), axis=1)
    return float(area) * np.einsum("q,qi,qj->ij", weights, shape, shape)


def build_tetrahedron_p2_data(mesh) -> TetrahedronP2Data:
    """Build ten-node tetrahedron P2 topology and stiffness matrices (4-point quadrature)."""

    p1 = build_tetrahedron_p1_data(mesh)
    dof_nodes, connectivity, boundary_connectivity = mesh.build_quadratic_topology()
    vertex_gradients = np.asarray(p1.gradients, dtype=NP_FLOAT_DTYPE)
    a, b = 0.585410196624969, 0.138196601125011
    quadrature = np.asarray(((a, b, b, b), (b, a, b, b), (b, b, a, b), (b, b, b, a)))
    vertex_values = (4.0 * quadrature[None, :, :, None] - 1.0) * vertex_gradients[:, None, :, :]
    edge_values = np.stack(
        [
            4.0 * (quadrature[None, :, i, None] * vertex_gradients[:, None, j, :] + quadrature[None, :, j, None] * vertex_gradients[:, None, i, :])
            for i, j in _EDGES
        ],
        axis=2,
    )
    gradients = np.concatenate((vertex_values, edge_values), axis=2)
    volumes = np.asarray(p1.cell_volumes, dtype=NP_FLOAT_DTYPE)
    stiffness = volumes[:, None, None] * np.einsum("eqid,eqjd->eij", gradients, gradients) / 4.0
    return TetrahedronP2Data(
        dof_nodes=dof_nodes,
        cell_connectivity=connectivity,
        boundary_connectivity=boundary_connectivity,
        gradient_values=torch.as_tensor(gradients, dtype=FLOAT_DTYPE),
        cell_volumes=p1.cell_volumes,
        stiffness_templates=torch.as_tensor(stiffness, dtype=FLOAT_DTYPE),
    )
