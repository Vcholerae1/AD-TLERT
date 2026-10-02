"""Optimizer state <-> log-resistivity maps (petrophysical transforms) and physical-domain regularization.

Internal module: names with a leading underscore are shared inside the ``adtlert.inversion`` package.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from adtlert.inversion.config import ArrayLike, InversionConfig, _physical_quantity
from adtlert.inversion.petrophysics import (
    build_petrophysical_transform,
)


def _petrophysics(config: InversionConfig, values: np.ndarray):
    """Petrophysical transform sized for a ``(n_cells,)`` or ``(n_cells, n_times)`` array."""

    values = np.asarray(values)
    return build_petrophysical_transform(
        config.petrophysical_transform,
        n_cells=int(values.shape[0] if values.ndim == 2 else values.size),
        model_transform=config.model_transform,
        model_bounds=config.model_bounds,
        saturation_floor=float(config.saturation_floor),
        parameters=config.petrophysical_parameters,
    )


def _log_model_to_state(log_model: np.ndarray, config: InversionConfig) -> np.ndarray:
    return _petrophysics(config, log_model).state_from_log_resistivity(
        np.asarray(log_model, dtype=float)
    )


def _state_to_log_model(state: np.ndarray, config: InversionConfig) -> np.ndarray:
    return _petrophysics(config, state).log_resistivity_from_state(
        np.asarray(state, dtype=float)
    )


def _cell_parameter_array(values: ArrayLike, *, n_cells: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    result = np.full(n_cells, float(array)) if array.ndim == 0 else array.reshape(-1)
    if result.shape != (n_cells,):
        raise ValueError(f"{name} must be scalar or have shape ({n_cells},)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _regularization_value_and_derivative(
    state: np.ndarray, config: InversionConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Quantity regularized in the physical domain (parameter or water content) and its state derivative."""

    transform = _petrophysics(config, state)
    parameter, derivative = (
        transform.parameter_from_state(state),
        transform.d_parameter_d_state(state),
    )
    if (
        _physical_quantity(config) == "parameter"
        or transform.parameter_name == "water_content"
    ):
        return np.asarray(parameter, dtype=float), np.asarray(derivative, dtype=float)
    if transform.parameter_name != "saturation":
        raise ValueError(
            "physical_regularization_quantity='theta'/'water_content' requires "
            "petrophysical_transform='saturation' or a transform whose parameter is water_content"
        )
    phi = (config.petrophysical_parameters or {}).get("phi")
    if phi is None:
        raise ValueError(
            "physical_regularization_quantity='theta'/'water_content' requires petrophysical_parameters['phi']"
        )
    phi = _cell_parameter_array(phi, n_cells=int(state.shape[0]), name="phi")
    phi = phi[:, None] if state.ndim == 2 else phi
    return np.asarray(phi * parameter, dtype=float), np.asarray(
        phi * derivative, dtype=float
    )


def _diagonal(derivative: np.ndarray) -> sp.csr_matrix:
    diagonal = np.asarray(derivative, dtype=float).reshape(-1, order="F")
    if not np.all(np.isfinite(diagonal)):
        raise ValueError(
            "regularization projection derivative contains non-finite values"
        )
    return sp.diags(diagonal, format="csr")
