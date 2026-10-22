"""Validation of conductivity models, cotangents and parameter maps shared by the operators."""

from __future__ import annotations

import numpy as np
import torch

from adtlert.utils.dtypes import FLOAT_DTYPE

Tensor = torch.Tensor


def check_finite(values: Tensor, context: str) -> None:
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError(f"{context} produced non-finite values")


def cell_vector(values, count: int, dtype: torch.dtype = FLOAT_DTYPE) -> Tensor:
    """A finite per-cell vector of shape ``(count,)``; scalars are broadcast."""

    vector = torch.as_tensor(values, dtype=dtype)
    if vector.ndim == 0:
        vector = vector.expand(count)
    elif vector.shape != (count,):
        raise ValueError(f"conductivity must be scalar or shape ({count},)")
    if not bool(torch.isfinite(vector).all()):
        raise ValueError("conductivity must contain finite values")
    return vector


def conductivity_model(values, count: int, dtype: torch.dtype = FLOAT_DTYPE) -> Tensor:
    model = cell_vector(values, count, dtype)
    if bool(torch.any(model <= 0.0)):
        raise ValueError("conductivity must contain positive finite values")
    return model


def conductivity_series(values, count: int, dtype: torch.dtype = FLOAT_DTYPE) -> Tensor:
    """Positive finite models of shape ``(n_steps, count)``."""

    models = torch.as_tensor(values, dtype=dtype)
    if models.ndim != 2 or models.shape[1] != count:
        raise ValueError(f"conductivities must have shape (n_steps, {count})")
    if not bool(torch.isfinite(models).all()) or bool(torch.any(models <= 0.0)):
        raise ValueError("conductivity must contain positive finite values")
    return models


def measurement_vector(values, count: int, dtype: torch.dtype) -> Tensor:
    """A per-measurement vector of shape ``(count,)``; scalars are broadcast."""

    vector = torch.as_tensor(values, dtype=dtype)
    if vector.ndim == 0:
        return vector.expand(count)
    if vector.shape != (count,):
        raise ValueError(f"cotangent must be scalar or shape ({count},)")
    return vector


def cotangent_series(values, steps: int, count: int) -> Tensor:
    cotangents = torch.as_tensor(values, dtype=FLOAT_DTYPE)
    if tuple(cotangents.shape) != (steps, count):
        raise ValueError(f"cotangents must have shape {(steps, count)}")
    return cotangents


def parameter_targets(
    cell_parameter_ids, parameter_count: int | None, cell_count: int
) -> tuple[np.ndarray, int]:
    """Output column of every mesh cell (``-1``: dropped) and the number of columns.

    Without ``cell_parameter_ids`` every cell is its own column.
    """

    if cell_parameter_ids is None:
        if parameter_count is not None:
            raise ValueError("parameter_count requires cell_parameter_ids")
        return np.arange(cell_count, dtype=np.int64), cell_count
    if isinstance(cell_parameter_ids, Tensor):
        cell_parameter_ids = cell_parameter_ids.cpu()
    ids = np.asarray(cell_parameter_ids).astype(np.int64).reshape(-1)
    if ids.shape != (cell_count,):
        raise ValueError(f"cell_parameter_ids must have shape ({cell_count},)")
    if np.any(ids < -1) or not np.any(ids >= 0):
        raise ValueError(
            "cell_parameter_ids may only contain -1 or non-negative ids, with at least one active"
        )
    count = int(ids.max()) + 1 if parameter_count is None else int(parameter_count)
    if count <= int(ids.max()):
        raise ValueError(
            "parameter_count is smaller than the largest cell parameter id"
        )
    return ids, count


def option(value: str, choices: tuple[str, ...], name: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in choices:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(choices))}")
    return normalized
