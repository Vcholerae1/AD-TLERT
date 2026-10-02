"""Matrix-free PyTorch autograd bridges for INR-coupled ERT physics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.autograd.function import once_differentiable

from adtlert.forward import ERTForward2p5D, ERTForwardModeling
from adtlert.inversion.core import (
    ParameterizedERTForward2p5D,
    _forward_log_response_series,
)
from adtlert.utils.dtypes import FLOAT_DTYPE


def forward_operator(forward: Any) -> ERTForward2p5D:
    """Resolve the native 2.5D operator used by an inversion facade."""

    if isinstance(forward, ERTForward2p5D):
        return forward
    operator = getattr(forward, "forward_operator", None)
    if isinstance(operator, ERTForward2p5D):
        return operator
    raise TypeError("INR physics requires an ADTLERT ERTForward2p5D operator")


def _full_log_model(forward: Any, parameter_log: np.ndarray) -> np.ndarray:
    if isinstance(forward, ParameterizedERTForward2p5D):
        return np.asarray(forward._full_log_model(parameter_log), dtype=float)
    if isinstance(forward, (ERTForward2p5D, ERTForwardModeling)):
        return np.asarray(parameter_log, dtype=float)
    method = getattr(forward, "_full_log_model", None)
    if callable(method):
        return np.asarray(method(parameter_log), dtype=float)
    return np.asarray(parameter_log, dtype=float)


def prepare_cuda_forward(
    forward: Any,
    parameter_log: np.ndarray,
    *,
    require_cuda: bool,
    device: torch.device | str = "cuda",
) -> dict[str, Any]:
    """Warm solver caches and prove that the requested backend reached CUDA."""

    operator = forward_operator(forward)
    requested_device = torch.device(device)
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for INR inversion, but torch.cuda.is_available() is False")
    if require_cuda and operator.linear_solver_backend == "scipy":
        raise RuntimeError("CUDA is required, but the ERT forward operator uses the SciPy backend")
    if requested_device.type == "cuda":
        device_index = requested_device.index if requested_device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(device_index)
        try:
            import cupy as cp

            cp.cuda.Device(device_index).use()
        except ImportError:
            if require_cuda:
                raise RuntimeError("CUDA INR inversion requires CuPy") from None
    else:
        device_index = None

    full_log = _full_log_model(forward, np.asarray(parameter_log, dtype=float).reshape(-1))
    conductivity = torch.as_tensor(np.exp(-full_log), dtype=FLOAT_DTYPE)
    operator.prepare(conductivity, include_solver_state=True)
    gpu_enabled = bool(operator._cudss_state.get("gpu_enabled", False))
    if require_cuda and not gpu_enabled:
        raise RuntimeError("cuDSS preparation completed without activating the GPU backend")
    return {
        "network_cuda_available": bool(torch.cuda.is_available()),
        "linear_solver_backend": str(operator.linear_solver_backend),
        "cudss_gpu_enabled": gpu_enabled,
        "cudss_zero_copy": bool(operator._cudss_state.get("gpu_zero_copy", False)),
        "gpu_device_index": device_index,
        "gpu_name": torch.cuda.get_device_name(device_index) if device_index is not None else None,
    }


def _exact_log_response_vjp(
    forward: Any,
    log_resistivity: np.ndarray,
    predicted_log: np.ndarray,
    cotangent: np.ndarray,
) -> np.ndarray:
    """Apply the true transpose derivative of the reciprocal-averaged response."""

    parameter_log = np.asarray(log_resistivity, dtype=float).reshape(-1)
    predicted = np.asarray(predicted_log, dtype=float).reshape(-1)
    data_cotangent = np.asarray(cotangent, dtype=float).reshape(-1)
    operator = forward_operator(forward)

    if isinstance(forward, ParameterizedERTForward2p5D):
        full_log, projection = forward._full_log_model_and_projection(parameter_log)
    else:
        full_log = parameter_log
        projection = None

    conductivity = np.exp(-full_log)
    geometric_factors = np.abs(np.asarray(operator._geometric_factors(), dtype=float)).reshape(-1)
    resistance_cotangent = data_cotangent * geometric_factors / np.exp(predicted)
    full_gradient_sigma = np.asarray(
        operator.vjp(
            torch.as_tensor(conductivity, dtype=FLOAT_DTYPE),
            torch.as_tensor(resistance_cotangent, dtype=FLOAT_DTYPE),
        ),
        dtype=float,
    ).reshape(-1)

    if isinstance(forward, ParameterizedERTForward2p5D) and forward.background_mode == "pygimli_prolongation":
        prolongation = forward._resistivity_prolongation_matrix
        if prolongation is None:
            raise ValueError("resistivity prolongation matrix has not been initialized")
        parameter_resistivity = np.exp(parameter_log)
        full_gradient_resistivity = -(conductivity**2) * full_gradient_sigma
        return parameter_resistivity * np.asarray(prolongation.T @ full_gradient_resistivity).reshape(-1)

    full_gradient_log_rho = -conductivity * full_gradient_sigma
    if projection is not None:
        return np.asarray(projection.T @ full_gradient_log_rho, dtype=float).reshape(-1)
    return full_gradient_log_rho


class _MatrixFreeLogRhoaSeries(torch.autograd.Function):
    """``(n_steps, n_cells)`` log-resistivity -> ``(n_steps, n_data)`` log apparent resistivity.

    The backward pass applies the exact per-step VJP, reusing the forward fields cached
    by the operator during the forward pass.
    """

    @staticmethod
    def forward(ctx: Any, log_resistivity: torch.Tensor, forward: Any) -> torch.Tensor:
        parameter_log = log_resistivity.detach().to(device="cpu", dtype=torch.float64).numpy()
        predicted = np.asarray(_forward_log_response_series(forward, parameter_log), dtype=float)
        ctx.forward_model, ctx.parameter_log, ctx.predicted = forward, parameter_log, predicted
        return torch.as_tensor(predicted, device=log_resistivity.device, dtype=log_resistivity.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, output_cotangent: torch.Tensor) -> tuple[torch.Tensor, None]:
        cotangent = output_cotangent.detach().to(device="cpu", dtype=torch.float64).numpy()
        gradient = np.vstack(
            [
                _exact_log_response_vjp(ctx.forward_model, model, response, weight)
                for model, response, weight in zip(ctx.parameter_log, ctx.predicted, cotangent, strict=True)
            ]
        )
        return torch.as_tensor(gradient, device=output_cotangent.device, dtype=output_cotangent.dtype), None


def matrix_free_log_rhoa(log_resistivity: torch.Tensor, forward: Any) -> torch.Tensor:
    """Differentiable log apparent resistivity of one model without a dense Jacobian."""

    return _MatrixFreeLogRhoaSeries.apply(log_resistivity[None, :], forward)[0]


def matrix_free_log_rhoa_series(log_resistivity: torch.Tensor, forward: Any) -> torch.Tensor:
    """Time-series form using cached field solves and exact per-step VJPs."""

    return _MatrixFreeLogRhoaSeries.apply(log_resistivity, forward)


__all__ = [
    "forward_operator",
    "matrix_free_log_rhoa",
    "matrix_free_log_rhoa_series",
    "prepare_cuda_forward",
]
