"""Boundary-edge terms of the 2.5D mixed (Robin) boundary condition."""

from __future__ import annotations

import numpy as np
import torch
from scipy.special import k0 as besselk0
from scipy.special import k1 as besselk1

from adtlert.utils.dtypes import FLOAT_DTYPE

_LINEAR_EDGE_MASS = [[2.0, 1.0], [1.0, 2.0]]
_QUADRATIC_EDGE_MASS = [[4.0, 2.0, -1.0], [2.0, 16.0, 2.0], [-1.0, 2.0, 4.0]]


def _edge_mass(
    lengths, coefficients, reference: list[list[float]], denominator: float
) -> torch.Tensor:
    lengths = torch.as_tensor(lengths, dtype=FLOAT_DTYPE)
    values = torch.as_tensor(coefficients, dtype=FLOAT_DTYPE)
    if values.ndim == 0:
        values = values.expand(lengths.shape)
    if values.shape != lengths.shape:
        raise ValueError(
            "boundary coefficients must be scalar or shape (num_boundary_edges,)"
        )
    local = torch.tensor(reference, dtype=FLOAT_DTYPE) / denominator
    return values[:, None, None] * lengths[:, None, None] * local


def assemble_local_boundary_mass(mesh, coefficients) -> torch.Tensor:
    """Linear Robin edge matrices ``c L / 6 [[2, 1], [1, 2]]``."""

    return _edge_mass(mesh.boundary_edge_lengths, coefficients, _LINEAR_EDGE_MASS, 6.0)


def assemble_local_boundary_mass_p2(lengths, coefficients) -> torch.Tensor:
    """Quadratic Robin edge matrices for ``(end, midpoint, end)`` edge DOFs."""

    return _edge_mass(lengths, coefficients, _QUADRATIC_EDGE_MASS, 30.0)


def robin_boundary_coefficients(
    mesh, conductivity, source_center, wavenumber: float
) -> torch.Tensor:
    """2.5D mixed-boundary coefficients ``sigma k (r.n / r) K1(k r) / K0(k r)`` (source plus image).

    The top (ground) surface is a natural boundary and gets zero.
    """

    centers = np.asarray(mesh.boundary_edge_centers, dtype=float)
    normals = np.asarray(mesh.boundary_edge_normals, dtype=float)
    source = np.asarray(source_center, dtype=float)
    r1, r2 = centers - source, centers - np.asarray([source[0], -source[1]])
    r1_abs, r2_abs = np.linalg.norm(r1, axis=1), np.linalg.norm(r2, axis=1)
    geometry = np.zeros(centers.shape[0])
    valid = np.flatnonzero(
        (r1_abs > 1e-12)
        & (r2_abs > 1e-12)
        & ~np.asarray(mesh.surface_edge_mask, dtype=bool)
    )
    denominator = besselk0(r1_abs[valid] * wavenumber) + besselk0(
        r2_abs[valid] * wavenumber
    )
    stable = valid[denominator > 1e-12]
    numerator = np.sum(r1[stable] * normals[stable], axis=1) / r1_abs[
        stable
    ] * besselk1(r1_abs[stable] * wavenumber) + np.sum(
        r2[stable] * normals[stable], axis=1
    ) / r2_abs[stable] * besselk1(r2_abs[stable] * wavenumber)
    geometry[stable] = wavenumber * numerator / denominator[denominator > 1e-12]

    sigma = torch.as_tensor(conductivity, dtype=FLOAT_DTYPE)
    sigma = sigma.expand(mesh.cell_count) if sigma.ndim == 0 else sigma
    if sigma.shape != (mesh.cell_count,):
        raise ValueError("cell coefficients must be scalar or shape (num_cells,)")
    return sigma[mesh.boundary_edge_cells.long()] * torch.as_tensor(
        geometry, dtype=FLOAT_DTYPE
    )
