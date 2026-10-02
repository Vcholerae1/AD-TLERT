"""P1/P2 triangle element data and local element matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from adtlert.utils.dtypes import FLOAT_DTYPE, NP_FLOAT_DTYPE

Tensor = torch.Tensor

_P1_RULES = {
    1: ([[1.0 / 3.0, 1.0 / 3.0]], [0.5]),
    2: (
        [[1.0 / 6.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0]],
        [1.0 / 6.0] * 3,
    ),
}
# Degree-4 rule, exact for P2 mass matrices.
_P2_RULE = (
    [
        [0.445948490915965, 0.445948490915965],
        [0.445948490915965, 0.108103018168070],
        [0.108103018168070, 0.445948490915965],
        [0.091576213509771, 0.091576213509771],
        [0.091576213509771, 0.816847572980459],
        [0.816847572980459, 0.091576213509771],
    ],
    [0.1116907948390055] * 3 + [0.0549758718276610] * 3,
)
_P1_REFERENCE_GRADIENTS = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]


@dataclass(frozen=True)
class TriangleQuadrature:
    """Reference triangle quadrature points and weights."""

    points: Tensor
    weights: Tensor


@dataclass(frozen=True)
class P1ElementData:
    """Batched P1 element tensors for tensorized assembly."""

    quadrature_points: Tensor
    quadrature_weights: Tensor
    shape_values: Tensor
    reference_gradients: Tensor
    gradients: Tensor
    cell_quadrature_points: Tensor
    cell_areas: Tensor


@dataclass(frozen=True)
class P2ElementData:
    """Batched P2 element tensors for tensorized assembly."""

    quadrature_points: Tensor
    quadrature_weights: Tensor
    shape_values: Tensor
    reference_gradients: Tensor
    gradients: Tensor
    cell_areas: Tensor


def triangle_quadrature(order: int = 2) -> TriangleQuadrature:
    """Return a one- or three-point Gauss rule on the reference triangle."""

    if order not in _P1_RULES:
        raise ValueError(f"unsupported triangle quadrature order: {order}")
    points, weights = _P1_RULES[order]
    return TriangleQuadrature(
        torch.tensor(points, dtype=FLOAT_DTYPE),
        torch.tensor(weights, dtype=FLOAT_DTYPE),
    )


def p1_shape_functions(points) -> Tensor:
    """Evaluate P1 shape functions at reference coordinates."""

    points = torch.as_tensor(points, dtype=FLOAT_DTYPE)
    xi, eta = points[..., 0], points[..., 1]
    return torch.stack((1.0 - xi - eta, xi, eta), dim=-1)


def reference_shape_gradients() -> Tensor:
    """Return the constant reference gradients of the P1 basis."""

    return torch.tensor(_P1_REFERENCE_GRADIENTS, dtype=FLOAT_DTYPE)


def build_p1_element_data(mesh, quadrature_order: int = 2) -> P1ElementData:
    """Prepare batched P1 element tensors (computed in NumPy at ``NP_FLOAT_DTYPE``)."""

    if quadrature_order not in _P1_RULES:
        raise ValueError(f"unsupported triangle quadrature order: {quadrature_order}")
    points, weights = (
        np.asarray(values, dtype=NP_FLOAT_DTYPE)
        for values in _P1_RULES[quadrature_order]
    )
    xi, eta = points[:, 0], points[:, 1]
    shape_values = np.stack((1.0 - xi - eta, xi, eta), axis=-1)
    reference = np.asarray(_P1_REFERENCE_GRADIENTS, dtype=NP_FLOAT_DTYPE)

    cell_nodes = np.asarray(mesh.nodes, dtype=NP_FLOAT_DTYPE)[np.asarray(mesh.cells)]
    jacobians = np.stack(
        (cell_nodes[:, 1] - cell_nodes[:, 0], cell_nodes[:, 2] - cell_nodes[:, 0]),
        axis=-1,
    )
    cell_gradients = np.einsum(
        "eij,nj->eni", np.linalg.inv(jacobians).transpose((0, 2, 1)), reference
    )
    as_tensor = lambda array: torch.as_tensor(
        np.ascontiguousarray(array), dtype=FLOAT_DTYPE
    )  # noqa: E731
    return P1ElementData(
        quadrature_points=as_tensor(points),
        quadrature_weights=as_tensor(weights),
        shape_values=as_tensor(shape_values),
        reference_gradients=as_tensor(reference),
        gradients=as_tensor(
            np.broadcast_to(
                cell_gradients[:, None], (cell_nodes.shape[0], points.shape[0], 3, 2)
            )
        ),
        cell_quadrature_points=as_tensor(
            np.einsum("qi,eid->eqd", shape_values, cell_nodes)
        ),
        cell_areas=as_tensor(0.5 * np.abs(np.linalg.det(jacobians))),
    )


def triangle_quadrature_p2() -> tuple[Tensor, Tensor]:
    """Return a degree-4 triangle rule exact for P2 mass matrices."""

    return tuple(torch.tensor(values, dtype=FLOAT_DTYPE) for values in _P2_RULE)


def p2_shape_functions(points) -> Tensor:
    """Evaluate P2 shape functions (vertices, then edge midpoints 01, 12, 20)."""

    points = torch.as_tensor(points, dtype=FLOAT_DTYPE)
    l2, l3 = points[..., 0], points[..., 1]
    l1 = 1.0 - l2 - l3
    return torch.stack(
        (
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            l3 * (2.0 * l3 - 1.0),
            4.0 * l1 * l2,
            4.0 * l2 * l3,
            4.0 * l3 * l1,
        ),
        dim=-1,
    )


def reference_shape_gradients_p2(points) -> Tensor:
    """Evaluate P2 reference gradients at reference coordinates."""

    points = torch.as_tensor(points, dtype=FLOAT_DTYPE)
    l2, l3 = points[..., 0, None], points[..., 1, None]
    l1 = 1.0 - l2 - l3
    g1, g2, g3 = torch.tensor(_P1_REFERENCE_GRADIENTS, dtype=FLOAT_DTYPE)
    return torch.stack(
        (
            (4.0 * l1 - 1.0) * g1,
            (4.0 * l2 - 1.0) * g2,
            (4.0 * l3 - 1.0) * g3,
            4.0 * (l1 * g2 + l2 * g1),
            4.0 * (l2 * g3 + l3 * g2),
            4.0 * (l3 * g1 + l1 * g3),
        ),
        dim=-2,
    )


def build_p2_element_data(mesh) -> P2ElementData:
    """Prepare batched P2 element tensors."""

    points, weights = triangle_quadrature_p2()
    reference = reference_shape_gradients_p2(points)
    cell_nodes = mesh.nodes[mesh.cells.long()]
    jacobians = torch.stack(
        (cell_nodes[:, 1] - cell_nodes[:, 0], cell_nodes[:, 2] - cell_nodes[:, 0]),
        dim=-1,
    )
    return P2ElementData(
        quadrature_points=points,
        quadrature_weights=weights,
        shape_values=p2_shape_functions(points),
        reference_gradients=reference,
        gradients=torch.einsum(
            "eij,qnj->eqni", torch.linalg.inv(jacobians).permute(0, 2, 1), reference
        ),
        cell_areas=mesh.cell_areas,
    )


def _scalar(value) -> float | None:
    try:
        array = np.asarray(value)
    except Exception:  # e.g. tensors that require grad
        return None
    return float(array) if array.ndim == 0 else None


def _local(
    spec: str, left, right, data, coefficients, *, numpy_scalars: bool
) -> Tensor:
    """``einsum(spec, left, right, coefficient * 2 |T| w_q)`` over cells and quadrature points.

    Scalar coefficients of P1 data are contracted in NumPy, as the templates always were.
    """

    scalar = _scalar(coefficients) if numpy_scalars else None
    if scalar is not None:
        weights = 2.0 * np.asarray(data.cell_areas, dtype=NP_FLOAT_DTYPE)[:, None]
        weights = (
            weights * np.asarray(data.quadrature_weights, dtype=NP_FLOAT_DTYPE)[None, :]
        )
        operands = (
            np.asarray(left, dtype=NP_FLOAT_DTYPE),
            np.asarray(right, dtype=NP_FLOAT_DTYPE),
        )
        return torch.as_tensor(
            np.einsum(spec, *operands, scalar * weights), dtype=FLOAT_DTYPE
        )
    cells, quadrature = data.cell_areas.shape[0], data.quadrature_weights.shape[0]
    values = torch.as_tensor(coefficients, dtype=FLOAT_DTYPE)
    if values.shape == (cells,):
        values = values[:, None]
    elif values.ndim and values.shape != (cells, quadrature):
        raise ValueError(
            "coefficients must be scalar, shape (num_cells,), or shape (num_cells, num_quadrature)"
        )
    weights = 2.0 * data.cell_areas[:, None] * data.quadrature_weights[None, :]
    return torch.einsum(spec, left, right, values.expand(cells, quadrature) * weights)


def assemble_local_stiffness(element_data: P1ElementData, conductivity) -> Tensor:
    """Elementwise P1 diffusion matrices ``int sigma grad(phi_i) . grad(phi_j)``."""

    gradients = element_data.gradients
    return _local(
        "eqid,eqjd,eq->eij",
        gradients,
        gradients,
        element_data,
        conductivity,
        numpy_scalars=True,
    )


def assemble_local_mass(element_data: P1ElementData, coefficients=1.0) -> Tensor:
    """Elementwise P1 mass matrices ``int c phi_i phi_j``."""

    values = element_data.shape_values
    return _local(
        "qi,qj,eq->eij", values, values, element_data, coefficients, numpy_scalars=True
    )


def assemble_local_stiffness_p2(element_data: P2ElementData, conductivity) -> Tensor:
    """Elementwise P2 diffusion matrices."""

    gradients = element_data.gradients
    return _local(
        "eqid,eqjd,eq->eij",
        gradients,
        gradients,
        element_data,
        conductivity,
        numpy_scalars=False,
    )


def assemble_local_mass_p2(element_data: P2ElementData, coefficients=1.0) -> Tensor:
    """Elementwise P2 mass matrices."""

    values = element_data.shape_values
    return _local(
        "qi,qj,eq->eij", values, values, element_data, coefficients, numpy_scalars=False
    )
