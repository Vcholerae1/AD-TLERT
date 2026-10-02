"""Adapters between log-resistivity vectors and the forward operators.

Internal module: names with a leading underscore are shared inside the ``adtlert.inversion`` package.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from adtlert.forward import ERTForward2p5D
from adtlert.forward.modeling import log_response_and_jacobian
from adtlert.utils.dtypes import FLOAT_DTYPE


def _forward_log_response(forward, log_resistivity: np.ndarray) -> np.ndarray:
    if isinstance(forward, ERTForward2p5D):
        conductivity = torch.as_tensor(1.0 / np.exp(log_resistivity), dtype=FLOAT_DTYPE)
        return np.log(
            np.asarray(forward.apparent_resistivity_values(conductivity), dtype=float)
        )
    if not hasattr(forward, "forward"):
        raise TypeError(
            "forward must be ERTForward2p5D, ERTForwardModeling, or expose forward"
        )
    return np.asarray(forward.forward(log_resistivity, log_transform=True), dtype=float)


def _forward_log_response_series(forward, log_resistivity: np.ndarray) -> np.ndarray:
    """``(n_steps, n_data)`` log responses of a ``(n_steps, n_cells)`` log-model series."""

    log_models = np.asarray(log_resistivity, dtype=float)
    if log_models.ndim != 2:
        raise ValueError("log_resistivity series must be a 2D array")
    return np.vstack([_forward_log_response(forward, row) for row in log_models])


def _forward_and_jacobian_log(
    forward, log_resistivity: np.ndarray, *, robin: bool = False, normal: bool = True
):
    options = {"include_robin_boundary_derivative": robin, "normal_sensitivity": normal}
    if isinstance(forward, ERTForward2p5D):
        return log_response_and_jacobian(
            forward,
            torch.as_tensor(1.0 / np.exp(log_resistivity), dtype=FLOAT_DTYPE),
            **options,
        )
    if not hasattr(forward, "forward_and_jacobian"):
        raise TypeError(
            "forward must be ERTForward2p5D, ERTForwardModeling, or expose forward_and_jacobian"
        )
    return forward.forward_and_jacobian(log_resistivity, log_transform=True, **options)


def _cached_forward_and_jacobian(
    cache: OrderedDict | None, max_entries: int, forward, log_model, robin, normal
):
    """LRU memoization of linearizations shared by overlapping sliding windows."""

    if cache is None or max_entries < 1:
        return _forward_and_jacobian_log(forward, log_model, robin=robin, normal=normal)
    key = (np.ascontiguousarray(log_model, dtype=np.float64).tobytes(), robin, normal)
    if key not in cache:
        cache[key] = _forward_and_jacobian_log(
            forward, log_model, robin=robin, normal=normal
        )
        while len(cache) > max_entries:
            cache.popitem(last=False)
    cache.move_to_end(key)
    return cache[key]
