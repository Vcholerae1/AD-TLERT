"""Normalization and validation of user-supplied models, data, and time-indexed config arrays.

Internal module: names with a leading underscore are shared inside the ``adtlert.inversion`` package.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import scipy.sparse as sp

from adtlert.inversion.config import ArrayLike, InversionConfig


def _model_size(forward) -> int:
    count = getattr(forward, "cell_count", None)
    if count is None:
        count = getattr(getattr(forward, "mesh", None), "cell_count", None)
    if count is None:
        raise TypeError("forward must expose a adtlert-compatible cell count")
    return int(count)


def _measurement_count(forward) -> int:
    survey = getattr(forward, "survey", None) or getattr(
        getattr(forward, "forward_operator", None), "survey", None
    )
    if survey is None:
        raise TypeError("forward must expose a adtlert-compatible measurement count")
    return int(survey.measurement_count)


def _logged(
    values: np.ndarray, *, already_log: bool, name: str, quantity: str
) -> np.ndarray:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    if already_log:
        return values.copy()
    if np.any(values <= 0.0):
        raise ValueError(f"{name} must contain positive {quantity} values")
    return np.log(values)


def _as_log_model(
    model: ArrayLike, expected_size: int, log_model: bool, name: str
) -> np.ndarray:
    values = np.asarray(model, dtype=float).ravel()
    if values.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},)")
    return _logged(values, already_log=log_model, name=name, quantity="resistivity")


def _as_log_models(
    model: ArrayLike, expected_size: int, n_times: int, log_model: bool, name: str
) -> np.ndarray:
    """Return ``(n_cells, n_times)`` log models from a shared or per-time model array."""

    values = np.asarray(model, dtype=float)
    if values.ndim == 1:
        return np.column_stack(
            [_as_log_model(values, expected_size, log_model, name)] * n_times
        )
    if values.shape == (expected_size, n_times):
        matrix = values
    elif values.shape == (n_times, expected_size):
        matrix = values.T
    else:
        raise ValueError(
            f"{name} must have shape ({expected_size},), ({expected_size}, {n_times}), or ({n_times}, {expected_size})"
        )
    return _logged(matrix, already_log=log_model, name=name, quantity="resistivity")


def _as_observed_log(
    data: ArrayLike, log_data: bool, measurement_count: int, *, timelapse: bool
) -> np.ndarray:
    """Return ``(n_times, n_measurements)`` log apparent resistivities (``(n,)`` for single-time)."""

    values = np.asarray(data, dtype=float)
    if not timelapse:
        values = values.ravel()
        if values.shape != (measurement_count,):
            raise ValueError(f"observed_data must have shape ({measurement_count},)")
    elif values.ndim != 2:
        raise ValueError("observed_data must be a 2D array for time-lapse inversion")
    elif values.shape[1] != measurement_count:
        if values.shape[0] != measurement_count:
            raise ValueError(
                "observed_data must have shape (n_times, n_measurements) or (n_measurements, n_times)"
            )
        values = values.T
    return _logged(
        values,
        already_log=log_data,
        name="observed_data",
        quantity="apparent resistivity",
    )


def _weights(data_std: float | ArrayLike, shape: tuple[int, ...]) -> np.ndarray:
    std = np.broadcast_to(np.asarray(data_std, dtype=float), shape)
    if not np.all(np.isfinite(std)):
        raise ValueError("data_std contains non-finite values")
    if np.any(std <= 0.0):
        raise ValueError("data_std must be positive")
    return 1.0 / std


def _config_for_time_index(config: InversionConfig, time_index: int) -> InversionConfig:
    """Slice time-varying ``(n_cells, n_times)`` petrophysical parameters to one timestep."""

    if not config.petrophysical_parameters:
        return config
    sliced, changed = {}, False
    for key, value in config.petrophysical_parameters.items():
        array = None if value is None else np.asarray(value)
        if array is not None and array.ndim >= 2:
            if not 0 <= time_index < array.shape[1]:
                raise IndexError(
                    f"time_index={time_index} outside petrophysical parameter {key!r} shape {array.shape}"
                )
            value, changed = array[:, time_index].copy(), True
        sliced[key] = value
    return replace(config, petrophysical_parameters=sliced) if changed else config


def _config_for_time_window(
    config: InversionConfig, *, observed_shape: tuple[int, int], start: int, end: int
) -> InversionConfig:
    """Slice every time-indexed config array to the window ``[start, end)``."""

    n_times = int(observed_shape[0])
    updates: dict[str, Any] = {}
    if np.ndim(config.data_std):
        updates["data_std"] = np.broadcast_to(
            np.asarray(config.data_std, dtype=float), observed_shape
        )[start:end].copy()
    if config.petrophysical_parameters:
        sliced = {
            key: np.asarray(value)[:, start:end].copy()
            if value is not None
            and np.ndim(value) >= 2
            and np.shape(value)[1] == n_times
            else value
            for key, value in config.petrophysical_parameters.items()
        }
        if any(
            sliced[key] is not value
            for key, value in config.petrophysical_parameters.items()
        ):
            updates["petrophysical_parameters"] = sliced
    for name in ("sensor_constraint_targets", "sensor_constraint_weights"):
        value = getattr(config, name)
        if value is None:
            continue
        array = np.asarray(value, dtype=float)
        if array.ndim >= 2 and array.shape[1] == n_times:
            updates[name] = array[:, start:end].copy()
        elif array.ndim == 1 and array.shape[0] == n_times:
            updates[name] = array[start:end].copy()
    if (
        config.freeze_first_timestep
    ):  # only the window containing the global first timestep is anchored
        updates["freeze_first_timestep"] = start == 0
    return replace(config, **updates) if updates else config


def _sensor_system(
    config: InversionConfig, n_cells: int, n_times: int
) -> tuple[sp.csr_matrix, np.ndarray] | None:
    """Row-scaled ``(kron(I_T, H), targets)`` restricted to valid observations."""

    if config.sensor_constraint <= 0.0:
        return None
    operator = config.sensor_constraint_operator
    if sp.issparse(operator):
        operator = operator.tocsr()
    else:
        operator = np.asarray(operator, dtype=float)
        if operator.ndim != 2:
            raise ValueError("sensor_constraint_operator must be a 2D matrix")
        operator = sp.csr_matrix(operator)
    if operator.nnz and not np.all(np.isfinite(operator.data)):
        raise ValueError("sensor_constraint_operator contains non-finite entries")
    if operator.shape[1] != n_cells:
        raise ValueError(
            f"sensor_constraint_operator second dimension must match n_cells ({operator.shape[1]} != {n_cells})"
        )
    targets = np.asarray(config.sensor_constraint_targets, dtype=float)
    targets = targets.reshape(-1, 1) if targets.ndim == 1 else targets
    if targets.ndim != 2:
        raise ValueError(
            "sensor_constraint_targets must be a 2D matrix [n_constraints, n_times]"
        )
    if targets.shape[0] != operator.shape[0]:
        raise ValueError(
            f"sensor_constraint_targets first dimension must match operator rows ({targets.shape[0]} != {operator.shape[0]})"
        )
    if targets.shape[1] == 1 and n_times > 1:
        targets = np.repeat(targets, n_times, axis=1)
    if targets.shape[1] != n_times:
        raise ValueError(
            f"sensor_constraint_targets second dimension must match n_times ({targets.shape[1]} != {n_times})"
        )

    shape = (operator.shape[0], n_times)
    weights = (
        np.ones(shape)
        if config.sensor_constraint_weights is None
        else np.asarray(config.sensor_constraint_weights, dtype=float)
    )
    if weights.ndim == 1:
        if weights.shape[0] == shape[0]:
            weights = weights[:, None]
        elif weights.shape[0] == shape[1]:
            weights = weights[None, :]
        elif weights.shape[0] == shape[0] * shape[1]:
            weights = weights.reshape(shape, order="F")
        else:
            raise ValueError(
                "sensor_constraint_weights 1D shape must match n_constraints, n_times, or n_constraints*n_times"
            )
    elif weights.ndim > 2:
        raise ValueError("sensor_constraint_weights must be scalar, 1D, or 2D")
    weights = np.broadcast_to(weights, shape)
    if not np.all(np.isfinite(weights)):
        raise ValueError("sensor_constraint_weights contains non-finite values")
    if np.any(weights < 0.0):
        raise ValueError("sensor_constraint_weights must be non-negative")
    valid = (np.isfinite(targets) & (weights > 0.0)).reshape(-1, order="F")
    if not np.any(valid):
        raise ValueError(
            "sensor_constraint has no valid (finite, positive-weight) entries"
        )

    rows = np.flatnonzero(valid)
    matrix = sp.kron(sp.eye(n_times, format="csr"), operator, format="csr")[
        rows
    ].tocsr()
    scale = np.sqrt(weights.reshape(-1, order="F")[rows])
    if scale.size and not np.allclose(scale, 1.0):
        matrix = sp.diags(scale, format="csr") @ matrix
    return matrix.tocsr(), scale * targets.reshape(-1, order="F")[rows]
