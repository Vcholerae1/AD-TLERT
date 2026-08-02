"""P1 finite-element data for four-node tetrahedra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adtlert.mesh.core3d import Mesh3D
from adtlert.utils.dtypes import FLOAT_DTYPE, NP_FLOAT_DTYPE
from adtlert.utils.torch_runtime import Array, torch_np


@dataclass(frozen=True)
class TetrahedronP1Data:
    """Cellwise gradients, volumes, and unit-conductivity matrices."""

    gradients: Array
    cell_volumes: Array
    stiffness_templates: Array


@dataclass(frozen=True)
class TetrahedronP2Data:
    """Quadratic tetrahedron topology and cellwise stiffness templates."""

    dof_nodes: Array
    cell_connectivity: Array
    boundary_connectivity: Array
    gradient_values: Array
    cell_volumes: Array
    stiffness_templates: Array


def build_tetrahedron_p1_data(mesh: Mesh3D) -> TetrahedronP1Data:
    """Build constant P1 gradients and local stiffness templates."""

    nodes = np.asarray(mesh.nodes, dtype=NP_FLOAT_DTYPE)
    cells = np.asarray(mesh.cells, dtype=np.int32)
    cell_nodes = nodes[cells]
    jacobians = np.stack(
        (
            cell_nodes[:, 1] - cell_nodes[:, 0],
            cell_nodes[:, 2] - cell_nodes[:, 0],
            cell_nodes[:, 3] - cell_nodes[:, 0],
        ),
        axis=-1,
    )
    inverse_jacobians = np.linalg.inv(jacobians)
    reference_gradients = np.asarray(
        ((-1.0, -1.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=NP_FLOAT_DTYPE,
    )
    gradients = np.einsum("ij,ejk->eik", reference_gradients, inverse_jacobians)
    volumes = np.abs(np.linalg.det(jacobians)) / 6.0
    stiffness = volumes[:, None, None] * np.einsum("eid,ejd->eij", gradients, gradients)
    return TetrahedronP1Data(
        gradients=torch_np.asarray(gradients, dtype=FLOAT_DTYPE),
        cell_volumes=torch_np.asarray(volumes, dtype=FLOAT_DTYPE),
        stiffness_templates=torch_np.asarray(stiffness, dtype=FLOAT_DTYPE),
    )


def _p2_gradient_values(barycentric: np.ndarray, vertex_gradients: np.ndarray) -> np.ndarray:
    """Evaluate ten quadratic shape gradients at barycentric points."""

    values = []
    for coordinates in barycentric:
        vertex_values = [
            (4.0 * coordinates[index] - 1.0) * vertex_gradients[index]
            for index in range(4)
        ]
        edge_values = [
            4.0 * (coordinates[i] * vertex_gradients[j] + coordinates[j] * vertex_gradients[i])
            for i, j in ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
        ]
        values.append(np.asarray((*vertex_values, *edge_values)))
    return np.asarray(values)


def p2_tetrahedron_shape_values(barycentric: np.ndarray) -> np.ndarray:
    """Evaluate ten-node tetrahedron shape functions."""

    coordinates = np.asarray(barycentric, dtype=float)
    vertices = coordinates * (2.0 * coordinates - 1.0)
    edges = np.stack(
        tuple(4.0 * coordinates[..., i] * coordinates[..., j] for i, j in ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))),
        axis=-1,
    )
    return np.concatenate((vertices, edges), axis=-1)


def p2_triangle_mass_template(area: float) -> np.ndarray:
    """Return the six-node triangle mass matrix using degree-four quadrature."""

    a, b = 0.445948490915965, 0.108103018168070
    c, d = 0.091576213509771, 0.816847572980459
    barycentric = np.asarray(((a, a, b), (a, b, a), (b, a, a), (c, c, d), (c, d, c), (d, c, c)))
    weights = np.asarray((0.223381589678011,) * 3 + (0.109951743655322,) * 3)
    vertices = barycentric * (2.0 * barycentric - 1.0)
    edges = np.stack(
        (4.0 * barycentric[:, 0] * barycentric[:, 1], 4.0 * barycentric[:, 1] * barycentric[:, 2], 4.0 * barycentric[:, 2] * barycentric[:, 0]),
        axis=-1,
    )
    shape = np.concatenate((vertices, edges), axis=1)
    return float(area) * np.einsum("q,qi,qj->ij", weights, shape, shape)


def build_tetrahedron_p2_data(mesh: Mesh3D) -> TetrahedronP2Data:
    """Build ten-node tetrahedron P2 topology and stiffness matrices."""

    p1 = build_tetrahedron_p1_data(mesh)
    dof_nodes, connectivity, boundary_connectivity = mesh.build_quadratic_topology()
    vertex_gradients = np.asarray(p1.gradients, dtype=NP_FLOAT_DTYPE)
    a, b = 0.585410196624969, 0.138196601125011
    quadrature = np.asarray(((a, b, b, b), (b, a, b, b), (b, b, a, b), (b, b, b, a)))
    vertex_values = (
        4.0 * quadrature[None, :, :, None] - 1.0
    ) * vertex_gradients[:, None, :, :]
    edge_values = np.stack(
        tuple(
            4.0
            * (
                quadrature[None, :, i, None] * vertex_gradients[:, None, j, :]
                + quadrature[None, :, j, None] * vertex_gradients[:, None, i, :]
            )
            for i, j in ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
        ),
        axis=2,
    )
    gradients = np.concatenate((vertex_values, edge_values), axis=2)
    volumes = np.asarray(p1.cell_volumes, dtype=NP_FLOAT_DTYPE)
    stiffness = volumes[:, None, None] * np.einsum("eqid,eqjd->eij", gradients, gradients) / 4.0
    return TetrahedronP2Data(
        dof_nodes=dof_nodes,
        cell_connectivity=connectivity,
        boundary_connectivity=boundary_connectivity,
        gradient_values=torch_np.asarray(gradients, dtype=FLOAT_DTYPE),
        cell_volumes=p1.cell_volumes,
        stiffness_templates=torch_np.asarray(stiffness, dtype=FLOAT_DTYPE),
    )
