"""PyTorch automatic-differentiation bridge for the 2.5D ERT solver."""

from __future__ import annotations

from typing import Any

import torch
from torch.autograd.function import once_differentiable

from adtlert.forward.ert2p5d import ERTForward2p5D
from adtlert.utils.dtypes import FLOAT_DTYPE


def _validate_inputs(
    conductivity: torch.Tensor,
    currents: torch.Tensor,
    forward_operator: ERTForward2p5D,
) -> None:
    if not isinstance(forward_operator, ERTForward2p5D):
        raise TypeError("forward_operator must be an ERTForward2p5D instance")
    if not isinstance(conductivity, torch.Tensor) or not conductivity.is_floating_point():
        raise TypeError("conductivity must be a floating-point torch.Tensor")
    if conductivity.ndim != 1 or conductivity.numel() != forward_operator.mesh.cell_count:
        raise ValueError(
            "conductivity must have shape "
            f"({forward_operator.mesh.cell_count},), got {tuple(conductivity.shape)}"
        )
    if not bool(torch.all(torch.isfinite(conductivity))) or bool(torch.any(conductivity <= 0.0)):
        raise ValueError("conductivity must contain finite positive values")
    if not isinstance(currents, torch.Tensor) or not currents.is_floating_point():
        raise TypeError("currents must be a floating-point torch.Tensor")
    valid_current_shapes = {(), (1,), (forward_operator.survey.measurement_count,)}
    if tuple(currents.shape) not in valid_current_shapes:
        raise ValueError(
            "currents must be scalar or have shape "
            f"({forward_operator.survey.measurement_count},), got {tuple(currents.shape)}"
        )
    if not bool(torch.all(torch.isfinite(currents))) or bool(torch.any(currents == 0.0)):
        raise ValueError("currents must contain finite non-zero values")


def _reduce_current_gradient(gradient: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if len(shape) == 0:
        return gradient.sum()
    if tuple(shape) == (1,):
        return gradient.sum().reshape(shape)
    return gradient


class ERT2p5DApparentResistivityFunction(torch.autograd.Function):
    """Expose the matrix-free 2.5D solver to PyTorch forward and reverse AD.

    The primal computation returns apparent resistivity. Reverse-mode AD calls
    :meth:`ERTForward2p5D.vjp`; forward-mode AD calls
    :meth:`ERTForward2p5D.jvp`. Gradients are available for both cell
    conductivity and scalar/per-datum currents. The solver itself remains an
    opaque numerical operation, so second-order derivatives are not supported.
    """

    @staticmethod
    def forward(
        ctx: Any,
        conductivity: torch.Tensor,
        currents: torch.Tensor,
        forward_operator: ERTForward2p5D,
    ) -> torch.Tensor:
        _validate_inputs(conductivity, currents, forward_operator)

        conductivity_work = conductivity.detach().to(device="cpu", dtype=FLOAT_DTYPE)
        currents_work = currents.detach().to(device="cpu", dtype=FLOAT_DTYPE)
        resistance = forward_operator.resistance(conductivity_work).detach()
        geometric_scale = forward_operator._geometric_factors().detach().abs()  # noqa: SLF001
        apparent_resistivity = geometric_scale * resistance / currents_work

        ctx.forward_operator = forward_operator
        ctx.conductivity_device = conductivity.device
        ctx.conductivity_dtype = conductivity.dtype
        ctx.currents_device = currents.device
        ctx.currents_dtype = currents.dtype
        ctx.currents_shape = currents.shape
        saved = (conductivity_work, currents_work, apparent_resistivity, geometric_scale)
        ctx.save_for_backward(*saved)
        ctx.save_for_forward(*saved)
        return apparent_resistivity.to(device=conductivity.device, dtype=conductivity.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        conductivity, currents, apparent_resistivity, geometric_scale = ctx.saved_tensors
        output_cotangent = grad_output.detach().to(device="cpu", dtype=FLOAT_DTYPE)

        conductivity_gradient = None
        if ctx.needs_input_grad[0]:
            resistance_cotangent = output_cotangent * geometric_scale / currents
            conductivity_gradient = ctx.forward_operator.vjp(conductivity, resistance_cotangent)
            conductivity_gradient = conductivity_gradient.to(
                device=ctx.conductivity_device,
                dtype=ctx.conductivity_dtype,
            )

        currents_gradient = None
        if ctx.needs_input_grad[1]:
            expanded_gradient = -output_cotangent * apparent_resistivity / currents
            currents_gradient = _reduce_current_gradient(expanded_gradient, ctx.currents_shape)
            currents_gradient = currents_gradient.to(device=ctx.currents_device, dtype=ctx.currents_dtype)

        return conductivity_gradient, currents_gradient, None

    @staticmethod
    def jvp(
        ctx: Any,
        conductivity_tangent: torch.Tensor | None,
        currents_tangent: torch.Tensor | None,
        forward_operator_tangent: None,
    ) -> torch.Tensor:
        del forward_operator_tangent
        conductivity, currents, apparent_resistivity, geometric_scale = ctx.saved_tensors
        tangent = torch.zeros_like(apparent_resistivity)

        if conductivity_tangent is not None:
            conductivity_direction = conductivity_tangent.detach().to(device="cpu", dtype=FLOAT_DTYPE)
            resistance_tangent = ctx.forward_operator.jvp(conductivity, conductivity_direction)
            tangent = tangent + geometric_scale * resistance_tangent / currents

        if currents_tangent is not None:
            current_direction = currents_tangent.detach().to(device="cpu", dtype=FLOAT_DTYPE)
            tangent = tangent - apparent_resistivity * current_direction / currents

        return tangent.to(device=ctx.conductivity_device, dtype=ctx.conductivity_dtype)


def apparent_resistivity_autograd(
    conductivity: torch.Tensor,
    forward_operator: ERTForward2p5D,
    currents: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Compute differentiable 2.5D apparent resistivity.

    ``conductivity`` determines the output device and dtype. ``currents`` may
    be a scalar, a length-one tensor, or one value per survey datum.
    """

    if not isinstance(conductivity, torch.Tensor):
        raise TypeError("conductivity must be a torch.Tensor")
    current_tensor = torch.as_tensor(currents, device=conductivity.device, dtype=conductivity.dtype)
    return ERT2p5DApparentResistivityFunction.apply(conductivity, current_tensor, forward_operator)


__all__ = ["ERT2p5DApparentResistivityFunction", "apparent_resistivity_autograd"]
