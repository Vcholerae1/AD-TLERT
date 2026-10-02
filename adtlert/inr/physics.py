"""Differentiable ERT physics for coordinate networks.

The map ``log rho (parameters) -> log rhoa`` is a composition of torch operations: the
background extension of :class:`ParameterizedERTForward2p5D` (when used), the
conductivity, and the exact-VJP autograd function of the 2.5D operator. Autograd chains
them, so no Jacobian is ever formed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from adtlert.forward import ERTForward2p5D, ERTForwardModeling
from adtlert.forward.autograd import apparent_resistivity_autograd
from adtlert.inversion.parameterized import ParameterizedERTForward2p5D


def forward_operator(forward: Any) -> ERTForward2p5D:
    """Resolve the native 2.5D operator used by an inversion facade."""

    if isinstance(forward, ERTForward2p5D):
        return forward
    operator = getattr(forward, "forward_operator", None)
    if isinstance(operator, ERTForward2p5D):
        return operator
    raise TypeError("INR physics requires an ADTLERT ERTForward2p5D operator")


def _conductivity(forward: Any, log_resistivity: torch.Tensor) -> torch.Tensor:
    """Forward-mesh conductivities (float64) of ``(..., n_parameters)`` log-resistivities."""

    if isinstance(forward, ParameterizedERTForward2p5D):
        return torch.exp(-forward.log_model_to_full(log_resistivity))
    if isinstance(forward, (ERTForward2p5D, ERTForwardModeling)):
        return torch.exp(-log_resistivity.to(torch.float64))
    raise TypeError(
        "INR physics requires ERTForward2p5D, ERTForwardModeling, or ParameterizedERTForward2p5D"
    )


def select_cuda_device(device: torch.device | str = "cuda") -> dict[str, Any]:
    """Make ``device`` the current CUDA device; returns its index and name."""

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"INR inversion runs on CUDA devices, got {device}")
    index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(index)
    return {"gpu_device_index": index, "gpu_name": torch.cuda.get_device_name(index)}


def prepare_cuda_forward(
    forward: Any, parameter_log: np.ndarray, *, device: torch.device | str = "cuda"
) -> dict[str, Any]:
    """Select the CUDA device and warm the forward caches and cuDSS plans at ``parameter_log``."""

    report = select_cuda_device(device)
    model = torch.as_tensor(np.asarray(parameter_log, dtype=float).reshape(-1))
    forward_operator(forward).prepare(
        _conductivity(forward, model), include_solver_state=True
    )
    return report


def matrix_free_log_rhoa_series(
    log_resistivity: torch.Tensor, forward: Any
) -> torch.Tensor:
    """``(n_steps, n_parameters)`` log-resistivity -> ``(n_steps, n_data)`` log apparent resistivity.

    Backpropagation applies the exact per-step VJP, reusing the forward fields cached by
    the operator (its field cache must hold at least ``n_steps`` entries).
    """

    operator = forward_operator(forward)
    conductivity = _conductivity(forward, log_resistivity)
    rhoa = torch.stack(
        [apparent_resistivity_autograd(step, operator) for step in conductivity]
    )
    return torch.log(rhoa).to(log_resistivity.dtype)


def matrix_free_log_rhoa(log_resistivity: torch.Tensor, forward: Any) -> torch.Tensor:
    """Differentiable log apparent resistivity of one model without a dense Jacobian."""

    return matrix_free_log_rhoa_series(log_resistivity[None, :], forward)[0]


__all__ = [
    "forward_operator",
    "matrix_free_log_rhoa",
    "matrix_free_log_rhoa_series",
    "prepare_cuda_forward",
    "select_cuda_device",
]
