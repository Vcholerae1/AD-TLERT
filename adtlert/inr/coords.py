"""Coordinate preparation for implicit ERT parameterizations."""

from __future__ import annotations

from typing import Any

import numpy as np


def cell_centers(mesh: Any) -> np.ndarray:
    """Return cell-center coordinates without assuming a structured mesh."""

    nodes = np.asarray(mesh.nodes, dtype=float)
    cells = np.asarray(mesh.cells, dtype=np.int32)
    if nodes.ndim != 2 or nodes.shape[1] < 2:
        raise ValueError("mesh nodes must have shape (n_nodes, >=2)")
    if cells.ndim != 2 or cells.shape[1] < 3:
        raise ValueError("mesh cells must have shape (n_cells, >=3)")
    return np.asarray(nodes[cells].mean(axis=1), dtype=np.float32)


def normalize_coordinates(
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize each coordinate axis to ``[-1, 1]``.

    Constant axes are mapped to zero. The returned center and half-span allow
    the same transform to be applied to new coordinates at inference time.
    """

    values = np.asarray(coordinates, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("coordinates must have shape (n_points, n_dimensions)")
    if not np.all(np.isfinite(values)):
        raise ValueError("coordinates contain non-finite values")
    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    center = 0.5 * (minimum + maximum)
    half_span = 0.5 * (maximum - minimum)
    safe_half_span = np.where(half_span > 0.0, half_span, 1.0).astype(np.float32)
    normalized = (values - center) / safe_half_span
    normalized[:, half_span <= 0.0] = 0.0
    return normalized.astype(np.float32), center.astype(np.float32), safe_half_span


def spatiotemporal_coordinates(
    spatial_coordinates: np.ndarray,
    times: np.ndarray,
    *,
    time_center: float | None = None,
    time_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build normalized ``(time, cell, x/z/t)`` coordinates.

    Spatial normalization is shared by every timestep and time is normalized
    independently. This avoids changing the spatial representation when a
    different time window is selected.
    """

    spatial, spatial_center, spatial_scale = normalize_coordinates(spatial_coordinates)
    time_values = np.asarray(times, dtype=np.float32).reshape(-1, 1)
    if time_values.shape[0] == 0 or not np.all(np.isfinite(time_values)):
        raise ValueError("times must contain at least one finite value")
    if (time_center is None) != (time_scale is None):
        raise ValueError("time_center and time_scale must be supplied together")
    if time_center is None:
        normalized_time, center_array, scale_array = normalize_coordinates(time_values)
    else:
        if (
            not np.isfinite(time_center)
            or not np.isfinite(time_scale)
            or time_scale <= 0.0
        ):
            raise ValueError(
                "time_center must be finite and time_scale must be positive"
            )
        center_array = np.asarray([time_center], dtype=np.float32)
        scale_array = np.asarray([time_scale], dtype=np.float32)
        normalized_time = (time_values - center_array) / scale_array
    tiled_spatial = np.broadcast_to(
        spatial[None, :, :], (time_values.shape[0], *spatial.shape)
    )
    tiled_time = np.broadcast_to(
        normalized_time[:, None, :], (time_values.shape[0], spatial.shape[0], 1)
    )
    combined = np.concatenate((tiled_spatial, tiled_time), axis=-1).astype(np.float32)
    center = np.concatenate((spatial_center, center_array)).astype(np.float32)
    scale = np.concatenate((spatial_scale, scale_array)).astype(np.float32)
    return combined, center, scale


__all__ = ["cell_centers", "normalize_coordinates", "spatiotemporal_coordinates"]
