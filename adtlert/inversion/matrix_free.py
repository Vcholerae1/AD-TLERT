"""Matrix-free gradients of the data term through the exact normal-sensitivity VJP.

Internal module: names with a leading underscore are shared inside the ``adtlert.inversion`` package.
"""

from __future__ import annotations

import numpy as np
import torch

from adtlert.forward import ERTForward2p5D, ERTForwardModeling
from adtlert.inversion.config import InversionConfig
from adtlert.inversion.parameterized import ParameterizedERTForward2p5D
from adtlert.utils.dtypes import FLOAT_DTYPE


def _operator(forward) -> ERTForward2p5D | None:
    """2.5D operator whose cells map onto the inversion model (``None`` for other forwards)."""

    if isinstance(forward, ERTForward2p5D):
        return forward
    if isinstance(forward, (ERTForwardModeling, ParameterizedERTForward2p5D)):
        return (
            forward.forward_operator
            if isinstance(forward.forward_operator, ERTForward2p5D)
            else None
        )
    return None


def _supports_normal_matrix_free(forward, config: InversionConfig) -> bool:
    """Whether the exact normal-sensitivity VJP is available (2.5D, no Robin derivative)."""

    return (
        config.normal_sensitivity
        and not config.include_robin_boundary_derivative
        and _operator(forward) is not None
    )


def _normal_log_response_vjp_series(
    forward, log_resistivity, predicted_log, cotangents
) -> np.ndarray:
    """Rows of ``(d log(rhoa) / d log(rho))^T cotangent`` without materializing Jacobians.

    The PyGIMLi-prolongation parameterization follows the fused active-cell aggregation of
    :meth:`ParameterizedERTForward2p5D.forward_and_jacobian`.
    """

    operator = _operator(forward)
    if operator is None:
        raise TypeError(
            "matrix-free normal VJP is currently available for 2.5D ERT only"
        )
    parameter_logs = np.asarray(log_resistivity, dtype=float)
    predicted = np.asarray(predicted_log, dtype=float)
    cotangents = np.asarray(cotangents, dtype=float)
    if (
        parameter_logs.ndim != 2
        or predicted.ndim != 2
        or cotangents.shape != predicted.shape
    ):
        raise ValueError("matrix-free VJP series inputs must be compatible 2D arrays")
    factors = np.abs(np.asarray(operator._geometric_factors(), dtype=float)).reshape(
        1, -1
    )
    resistance_cotangents = cotangents * factors / np.exp(predicted)
    parameterized = isinstance(forward, ParameterizedERTForward2p5D)
    fused = parameterized and forward.background_mode == "pygimli_prolongation"
    rows = []
    for log_model, cotangent in zip(parameter_logs, resistance_cotangents, strict=True):
        full_log, projection = (
            forward._full_log_model_and_projection(log_model)
            if parameterized
            else (log_model, None)
        )
        options = (
            {
                "cell_parameter_ids": forward.forward_cell_parameter_ids,
                "parameter_count": forward.cell_count,
            }
            if fused
            else {}
        )
        gradient_sigma = np.asarray(
            operator.normal_vjp(
                torch.as_tensor(np.exp(-full_log), dtype=FLOAT_DTYPE),
                torch.as_tensor(cotangent, dtype=FLOAT_DTYPE),
                **options,
            ),
            dtype=float,
        ).reshape(-1)
        if fused:
            rows.append(-np.exp(-log_model) * gradient_sigma)
        else:
            gradient = -np.exp(-full_log) * gradient_sigma
            rows.append(
                gradient
                if projection is None
                else np.asarray(projection.T @ gradient, dtype=float).reshape(-1)
            )
    return np.vstack(rows)


def _normal_log_response_vjp(
    forward, log_resistivity, predicted_log, cotangent
) -> np.ndarray:
    return _normal_log_response_vjp_series(
        forward,
        np.reshape(log_resistivity, (1, -1)),
        np.reshape(predicted_log, (1, -1)),
        np.reshape(cotangent, (1, -1)),
    )[0]
