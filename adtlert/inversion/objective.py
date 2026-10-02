"""Data and regularization pieces of the objective: misfits, coverage, line search.

Internal module: names with a leading underscore are shared inside the ``adtlert.inversion`` package.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from adtlert.inversion.misfit import DataMisfit
from adtlert.inversion.regularization import (
    regularization_mesh,
)
from adtlert.mesh import Mesh, Mesh3D


def _is_difference_misfit(data_misfit: DataMisfit) -> bool:
    return getattr(data_misfit, "name", "") == "log_data_difference_l2"


def _difference_weights(weights: np.ndarray) -> np.ndarray:
    sigma = 1.0 / np.asarray(weights, dtype=float)
    return 1.0 / np.sqrt(sigma[1:] ** 2 + sigma[0][None, :] ** 2)


def _difference_residual(
    predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Baseline residual followed by weighted residuals of changes relative to the first survey."""

    if predicted.shape != observed.shape:
        raise ValueError("predicted_log and observed_log must have the same shape")
    if predicted.ndim != 2 or predicted.shape[0] < 2:
        raise ValueError("log data-difference misfit requires at least two time steps")
    base = ((predicted[0] - observed[0]) * weights[0]).reshape(1, -1)
    change = (predicted[1:] - predicted[0][None, :]) - (
        observed[1:] - observed[0][None, :]
    )
    return np.vstack((base, change * _difference_weights(weights)))


def _data_phi(misfit: DataMisfit, predicted, observed, weights) -> float:
    if _is_difference_misfit(misfit):
        residual = _difference_residual(predicted, observed, weights).reshape(-1)
        return float(np.dot(residual, residual))
    return misfit.phi(predicted, observed, weights)


def _data_chi2(misfit: DataMisfit, predicted, observed, weights) -> float:
    if _is_difference_misfit(misfit):
        return float(np.mean(_difference_residual(predicted, observed, weights) ** 2))
    return misfit.chi2(predicted, observed, weights)


def _data_system(
    misfit: DataMisfit, predicted, observed, weights, jacobians
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Linearized data term over all timesteps (block-coupled for the data-difference misfit)."""

    if not _is_difference_misfit(misfit):
        blocks = [
            misfit.linearized_system(predicted[t], observed[t], weights[t], jac)
            for t, jac in enumerate(jacobians)
        ]
        return sp.block_diag(
            [sp.csr_matrix(matrix) for matrix, _ in blocks], format="csr"
        ), np.concatenate([rhs for _, rhs in blocks])

    n_times = predicted.shape[0]
    if n_times < 2:
        raise ValueError("log data-difference misfit requires at least two time steps")
    zero = sp.csr_matrix((predicted.shape[1], jacobians[0].shape[1]))
    rows = [
        sp.hstack(
            [sp.csr_matrix(jacobians[0] * weights[0][:, None])]
            + [zero] * (n_times - 1),
            format="csr",
        )
    ]
    rhs = [-((predicted[0] - observed[0]) * weights[0])]
    for t, w_t in enumerate(_difference_weights(weights), start=1):
        residual = (predicted[t] - predicted[0]) - (observed[t] - observed[0])
        blocks = [zero] * n_times
        blocks[0] = sp.csr_matrix(-jacobians[0] * w_t[:, None])
        blocks[t] = sp.csr_matrix(jacobians[t] * w_t[:, None])
        rows.append(sp.hstack(blocks, format="csr"))
        rhs.append(-(residual * w_t))
    return sp.vstack(rows, format="csr"), np.concatenate(rhs)


def _data_cotangents(misfit: DataMisfit, predicted, observed, weights) -> np.ndarray:
    """Per-timestep data cotangents whose VJPs give the data-term gradient."""

    if not _is_difference_misfit(misfit):
        return np.vstack(
            [
                misfit.linearized_cotangent(predicted[t], observed[t], weights[t])
                for t in range(predicted.shape[0])
            ]
        )
    cotangents = np.zeros_like(predicted, dtype=float)
    cotangents[0] += weights[0] ** 2 * (predicted[0] - observed[0])
    for t, w_t in enumerate(_difference_weights(weights), start=1):
        contribution = w_t**2 * (
            (predicted[t] - predicted[0]) - (observed[t] - observed[0])
        )
        cotangents[t] += contribution
        cotangents[0] -= contribution
    return cotangents


def _reference_roughness(
    mode: str, current: np.ndarray, matrix, reference, *, identity: bool = False
) -> np.ndarray:
    if mode == "update" or (reference is None and identity):
        return current
    return np.zeros_like(current) if reference is None else matrix @ reference


def _model_difference_operator(base: sp.spmatrix, n_times: int) -> sp.csr_matrix:
    """Spatial roughness of the baseline model and of every change relative to it."""

    base = base.tocsr()
    zero = sp.csr_matrix(base.shape)
    rows = [sp.hstack([base] + [zero] * (n_times - 1), format="csr")]
    for t in range(1, n_times):
        blocks = [zero] * n_times
        blocks[0], blocks[t] = -base, base
        rows.append(sp.hstack(blocks, format="csr"))
    return sp.vstack(rows, format="csr")


def _mesh_cell_areas(mesh: Mesh | Mesh3D) -> np.ndarray:
    if isinstance(mesh, Mesh3D):
        areas = np.asarray(mesh.cell_volumes, dtype=float)
    else:
        cell_nodes = np.asarray(mesh.nodes, dtype=float)[
            np.asarray(mesh.cells, dtype=np.int32)
        ]
        x, y = cell_nodes[:, :, 0], cell_nodes[:, :, 1]
        areas = 0.5 * np.abs(
            np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
        )
    if np.any(areas <= 0.0) or not np.all(np.isfinite(areas)):
        raise ValueError(
            "regularization mesh contains non-positive or non-finite cell areas"
        )
    return areas


def _coverage(forward, sensitivity: np.ndarray) -> np.ndarray:
    """PyGIMLi-style log10 coverage: summed |sensitivity| over data, normalized by cell size.

    ``sensitivity`` is a Jacobian (summed over rows) or, for matrix-free runs, a gradient.
    """

    sensitivity = np.abs(np.asarray(sensitivity, dtype=float))
    total = np.sum(sensitivity, axis=0) if sensitivity.ndim == 2 else sensitivity
    areas = _mesh_cell_areas(regularization_mesh(forward))
    if total.shape != areas.shape:
        raise ValueError(
            f"coverage Jacobian column count does not match regularization mesh cell count ({total.shape[0]} != {areas.shape[0]})"
        )
    return np.log10(np.maximum(total / areas, np.finfo(float).tiny))


def _line_search_tau(
    state,
    step,
    predicted_log,
    candidate_predicted_log,
    objective_matrix,
    objective_reference,
    data_phi,
) -> float:
    """Grid search on ``tau`` in ``[0, 1]`` using a linear model of the data along the step."""

    def phi(tau: float, predicted) -> float:
        roughness = objective_matrix @ (state + tau * step) - objective_reference
        return data_phi(predicted) + float(np.dot(roughness, roughness))

    best_phi = phi(0.0, predicted_log)
    if phi(1.0, candidate_predicted_log) < best_phi:
        return 1.0
    best_tau, direction = 0.0, candidate_predicted_log - predicted_log
    for index in range(1, 101):
        tau = 0.01 * index
        value = phi(tau, predicted_log + tau * direction)
        if value < best_phi:
            best_phi, best_tau = value, tau
    return 0.03 if 0.0 < best_tau < 0.03 else best_tau
