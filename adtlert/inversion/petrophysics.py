"""Petrophysical transforms between the optimizer state and log-resistivity.

The inversion core only needs ``state -> log(rho)`` and its diagonal derivative, so
physical parameterizations (conductivity, saturation, water content) stay independent of
the ERT solver, optimizer, and regularization code. States are ``(n_cells,)`` or
``(n_cells, n_times)``; cell-wise parameters broadcast over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

ArrayLike = object


class PetrophysicalTransform(Protocol):
    """Interface used by the inversion core."""

    name: str
    parameter_name: str

    def state_from_log_resistivity(self, log_resistivity: np.ndarray) -> np.ndarray:
        """Convert a log-resistivity model to optimizer state."""

    def log_resistivity_from_state(self, state: np.ndarray) -> np.ndarray:
        """Map optimizer state to cell log-resistivity."""

    def d_log_resistivity_d_state(self, state: np.ndarray) -> np.ndarray:
        """Return the diagonal chain-rule factor d log(rho) / d state."""

    def clip_state(self, state: np.ndarray) -> np.ndarray:
        """Project state back to admissible values when needed."""

    def parameter_from_state(self, state: np.ndarray) -> np.ndarray:
        """Return the physical parameter represented by the state."""

    def d_parameter_d_state(self, state: np.ndarray) -> np.ndarray:
        """Return the diagonal chain-rule factor d(parameter) / d(state)."""


def _log_bounds(bounds: tuple[float, float] | None) -> tuple[float, float] | None:
    if bounds is None:
        return None
    lo, hi = bounds
    if not 0.0 < lo < hi:
        raise ValueError("model_bounds must be positive and ordered as (min, max)")
    return float(np.log(lo)), float(np.log(hi))


def _clip_log(values: np.ndarray, bounds: tuple[float, float] | None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    log_bounds = _log_bounds(bounds)
    return values if log_bounds is None else np.clip(values, *log_bounds)


def _sigmoid(state: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(state, dtype=float), -60.0, 60.0)))


def _logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1.0e-12, 1.0 - 1.0e-12)
    return np.log(p) - np.log1p(-p)


def _cellwise(value: ArrayLike, like: np.ndarray) -> np.ndarray:
    """Broadcast a per-cell parameter over the time axis of a 2D state."""

    value = np.asarray(value, dtype=float)
    return value.reshape(-1, 1) if np.ndim(like) == 2 else value


def _parameter_array(
    value: ArrayLike, *, n_cells: int, name: str, allow_time: bool = False
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(n_cells, float(array))
    if allow_time and array.ndim > 1:
        if array.shape[0] != n_cells:
            raise ValueError(
                f"{name} first dimension must be {n_cells}, got {array.shape}"
            )
        return array
    array = array.reshape(-1)
    if array.shape != (n_cells,):
        raise ValueError(
            f"{name} must be scalar, cell-wise, or have first dimension {n_cells}"
            if allow_time
            else f"{name} must be scalar or have shape ({n_cells},)"
        )
    return array


@dataclass(frozen=True)
class LogResistivityTransform:
    """State is log-resistivity (``"log"``) or a bounded logit of resistivity (``"log_lu"``)."""

    model_transform: str = "log"
    model_bounds: tuple[float, float] | None = None
    name: str = "log_resistivity"
    parameter_name: str = "resistivity"

    def _bounds(self) -> tuple[float, float]:
        if self.model_transform != "log_lu":
            raise ValueError("model_transform must be 'log' or 'log_lu'")
        if self.model_bounds is None:
            raise ValueError("model_bounds are required for model_transform='log_lu'")
        return self.model_bounds

    def state_from_log_resistivity(self, log_resistivity):
        if self.model_transform == "log":
            return _clip_log(log_resistivity, self.model_bounds)
        lo, hi = self._bounds()
        span = hi - lo
        rho = np.clip(
            np.exp(np.asarray(log_resistivity, dtype=float)),
            lo + span * 1.0e-12,
            hi - span * 1.0e-12,
        )
        return np.log(rho - lo) - np.log(hi - rho)

    def log_resistivity_from_state(self, state):
        if self.model_transform == "log":
            return _clip_log(state, self.model_bounds)
        lo, hi = self._bounds()
        exp_state = np.exp(np.clip(np.asarray(state, dtype=float), -50.0, 50.0))
        return np.log((exp_state * hi + lo) / (exp_state + 1.0))

    def d_log_resistivity_d_state(self, state):
        state = np.asarray(state, dtype=float)
        if self.model_transform == "log":
            return np.ones_like(state)
        if self.model_bounds is None:
            raise ValueError("model_bounds are required for model_transform='log_lu'")
        lo, hi = self.model_bounds
        rho = np.exp(self.log_resistivity_from_state(state))
        return ((rho - lo) * (hi - rho)) / ((hi - lo) * rho)

    def clip_state(self, state):
        return (
            _clip_log(state, self.model_bounds)
            if self.model_transform == "log"
            else np.asarray(state, dtype=float)
        )

    def parameter_from_state(self, state):
        return np.exp(self.log_resistivity_from_state(state))

    def d_parameter_d_state(self, state):
        return self.parameter_from_state(state) * self.d_log_resistivity_d_state(state)


@dataclass(frozen=True)
class LogConductivityTransform:
    """State is log-conductivity."""

    model_bounds: tuple[float, float] | None = None
    name: str = "log_conductivity"
    parameter_name: str = "conductivity"

    def state_from_log_resistivity(self, log_resistivity):
        return self.clip_state(-np.asarray(log_resistivity, dtype=float))

    def log_resistivity_from_state(self, state):
        return -self.clip_state(state)

    def d_log_resistivity_d_state(self, state):
        return -np.ones_like(np.asarray(state, dtype=float))

    def clip_state(self, state):
        state = np.asarray(state, dtype=float)
        log_bounds = _log_bounds(self.model_bounds)
        return (
            state
            if log_bounds is None
            else np.clip(state, -log_bounds[1], -log_bounds[0])
        )

    def parameter_from_state(self, state):
        return np.exp(self.clip_state(state))

    def d_parameter_d_state(self, state):
        return self.parameter_from_state(state)


@dataclass(frozen=True)
class SaturationTransform:
    """Waxman-Smits-style saturation ``sigma = sigma_p S^n + sigma_s S^(n-1)`` with a sigmoid state."""

    rho_sat: np.ndarray
    n: np.ndarray
    rho_sat_s: np.ndarray | None = None
    saturation_floor: float = 1.0e-4
    name: str = "saturation"
    parameter_name: str = "saturation"

    def __post_init__(self) -> None:
        if not 0.0 < float(self.saturation_floor) < 1.0:
            raise ValueError("saturation_floor must be in (0, 1)")
        if np.any(np.asarray(self.rho_sat, dtype=float) <= 0.0):
            raise ValueError("rho_sat must be positive")
        if np.any(np.asarray(self.n, dtype=float) <= 0.0):
            raise ValueError("n must be positive")

    def _conductivities(self) -> tuple[np.ndarray, np.ndarray]:
        """Pore (``sigma_p``) and surface (``sigma_s``) conductivity at full saturation."""

        sigma_sat = 1.0 / np.asarray(self.rho_sat, dtype=float)
        sigma_s = np.zeros_like(sigma_sat)
        if self.rho_sat_s is not None:
            rho_sat_s = np.asarray(self.rho_sat_s, dtype=float)
            surface = np.isfinite(rho_sat_s) & (rho_sat_s > 0.0)
            sigma_s[surface] = 1.0 / rho_sat_s[surface]
        sigma_p = sigma_sat - sigma_s
        if np.any(sigma_p <= 0.0):
            raise ValueError(
                "rho_sat_s must be larger than rho_sat where surface conduction is used"
            )
        return sigma_p, sigma_s

    def _saturation(self, state):
        floor = float(self.saturation_floor)
        return floor + (1.0 - floor) * _sigmoid(state)

    def _sigma(self, saturation):
        saturation = np.asarray(saturation, dtype=float)
        if np.asarray(self.rho_sat).shape[0] != saturation.shape[0]:
            raise ValueError(
                "saturation state first dimension does not match petrophysical parameter count"
            )
        n = _cellwise(self.n, saturation)
        sigma_p, sigma_s = (
            _cellwise(value, saturation) for value in self._conductivities()
        )
        return sigma_p * np.power(saturation, n) + sigma_s * np.power(
            saturation, n - 1.0
        )

    def state_from_log_resistivity(self, log_resistivity):
        """Invert the monotone ``S -> sigma`` map by 60 bisection steps."""

        target = np.exp(-np.asarray(log_resistivity, dtype=float))
        floor = float(self.saturation_floor)
        lo, hi = np.full_like(target, floor), np.ones_like(target)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            below = self._sigma(mid) < target
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        saturation = np.clip(0.5 * (lo + hi), floor, 1.0)
        return _logit((saturation - floor) / (1.0 - floor))

    def log_resistivity_from_state(self, state):
        return -np.log(
            np.maximum(self._sigma(self._saturation(state)), np.finfo(float).tiny)
        )

    def d_log_resistivity_d_state(self, state):
        saturation = self._saturation(state)
        n = _cellwise(self.n, saturation)
        sigma_p, sigma_s = (
            _cellwise(value, saturation) for value in self._conductivities()
        )
        d_sigma = sigma_p * n * np.power(saturation, n - 1.0)
        if np.any(sigma_s != 0.0):
            d_sigma = d_sigma + sigma_s * (n - 1.0) * np.power(saturation, n - 2.0)
        sigma = np.maximum(self._sigma(saturation), np.finfo(float).tiny)
        return -(d_sigma / sigma) * self.d_parameter_d_state(state)

    def clip_state(self, state):
        return np.asarray(state, dtype=float)

    def parameter_from_state(self, state):
        return self._saturation(state)

    def d_parameter_d_state(self, state):
        saturation, floor = self._saturation(state), float(self.saturation_floor)
        return (saturation - floor) * (1.0 - saturation) / (1.0 - floor)


@dataclass(frozen=True)
class RelativeArchieWaterContentTransform:
    """Relative Archie law with bounded volumetric water content.

    ``log(rho_t) = log(rho0) - n log(theta_t / theta0) - log(C_T)``, where ``rho0`` and
    ``theta0`` are baseline models at the reference temperature and ``C_T`` maps field-
    to reference-temperature resistivity (``1`` when omitted). The state is mapped to
    ``theta`` in ``[theta_min, theta_max]`` through a sigmoid.
    """

    rho0: np.ndarray
    theta0: np.ndarray
    n: np.ndarray
    theta_min: np.ndarray
    theta_max: np.ndarray
    temperature_correction_factor: np.ndarray | None = None
    name: str = "relative_archie_water_content"
    parameter_name: str = "water_content"

    def __post_init__(self) -> None:
        rho0, theta0, n = (
            np.asarray(value, dtype=float) for value in (self.rho0, self.theta0, self.n)
        )
        theta_min, theta_max = (
            np.asarray(self.theta_min, dtype=float),
            np.asarray(self.theta_max, dtype=float),
        )
        if np.any(rho0 <= 0.0) or not np.all(np.isfinite(rho0)):
            raise ValueError("rho0 must be positive and finite")
        if np.any(theta0 <= 0.0) or not np.all(np.isfinite(theta0)):
            raise ValueError("theta0 must be positive and finite")
        if np.any(n <= 0.0) or not np.all(np.isfinite(n)):
            raise ValueError("n must be positive and finite")
        if np.any(theta_min <= 0.0) or np.any(theta_max <= theta_min):
            raise ValueError("theta bounds must satisfy 0 < theta_min < theta_max")
        if np.any(theta0 <= theta_min) or np.any(theta0 >= theta_max):
            raise ValueError("theta0 must lie strictly inside [theta_min, theta_max]")
        if self.temperature_correction_factor is not None:
            factor = np.asarray(self.temperature_correction_factor, dtype=float)
            if np.any(factor <= 0.0) or not np.all(np.isfinite(factor)):
                raise ValueError(
                    "temperature_correction_factor must be positive and finite"
                )

    def _temperature_factor(self, state: np.ndarray) -> np.ndarray:
        if self.temperature_correction_factor is None:
            return np.ones_like(state, dtype=float)
        factor = np.asarray(self.temperature_correction_factor, dtype=float)
        if factor.ndim == 0:
            return np.full_like(state, float(factor), dtype=float)
        if factor.ndim == 1:
            return _cellwise(factor, state)
        if factor.shape != state.shape:
            raise ValueError(
                f"temperature_correction_factor must be scalar, cell-wise, or match the state shape ({factor.shape} != {state.shape})"
            )
        return factor

    def _bounds(self, like: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _cellwise(self.theta_min, like), _cellwise(self.theta_max, like)

    def _theta(self, state: np.ndarray) -> np.ndarray:
        theta_min, theta_max = self._bounds(state)
        return theta_min + (theta_max - theta_min) * _sigmoid(state)

    def state_from_log_resistivity(self, log_resistivity):
        log_rho = np.asarray(log_resistivity, dtype=float)
        rho0, theta0, n = (
            _cellwise(value, log_rho) for value in (self.rho0, self.theta0, self.n)
        )
        theta_min, theta_max = self._bounds(log_rho)
        theta = theta0 * np.exp(
            -(log_rho + np.log(self._temperature_factor(log_rho)) - np.log(rho0)) / n
        )
        span = theta_max - theta_min
        theta = np.clip(theta, theta_min + span * 1.0e-12, theta_max - span * 1.0e-12)
        return _logit((theta - theta_min) / span)

    def log_resistivity_from_state(self, state):
        state = np.asarray(state, dtype=float)
        rho0, theta0, n = (
            _cellwise(value, state) for value in (self.rho0, self.theta0, self.n)
        )
        return (
            np.log(rho0)
            - n * (np.log(self._theta(state)) - np.log(theta0))
            - np.log(self._temperature_factor(state))
        )

    def d_log_resistivity_d_state(self, state):
        state = np.asarray(state, dtype=float)
        return (
            -_cellwise(self.n, state)
            * self.d_parameter_d_state(state)
            / self._theta(state)
        )

    def clip_state(self, state):
        return np.asarray(state, dtype=float)

    def parameter_from_state(self, state):
        return self._theta(np.asarray(state, dtype=float))

    def d_parameter_d_state(self, state):
        state = np.asarray(state, dtype=float)
        theta = self._theta(state)
        theta_min, theta_max = self._bounds(state)
        return (theta - theta_min) * (theta_max - theta) / (theta_max - theta_min)


_TRANSFORM_ALIASES = {
    "log_resistivity": ("log_resistivity", "resistivity", "rho"),
    "log_conductivity": ("log_conductivity", "conductivity", "sigma"),
    "saturation": ("saturation", "water_saturation"),
    "relative_archie_water_content": (
        "relative_archie_water_content",
        "relative_archie",
        "water_content",
        "theta",
    ),
}


def available_petrophysical_transforms() -> tuple[str, ...]:
    """Return user-facing petrophysical transform names."""

    return tuple(_TRANSFORM_ALIASES)


def build_petrophysical_transform(
    name: str | PetrophysicalTransform,
    *,
    n_cells: int,
    model_transform: str = "log",
    model_bounds: tuple[float, float] | None = None,
    saturation_floor: float = 1.0e-4,
    parameters: dict[str, ArrayLike] | None = None,
) -> PetrophysicalTransform:
    """Resolve and instantiate a petrophysical transform."""

    if hasattr(name, "log_resistivity_from_state") and hasattr(
        name, "d_log_resistivity_d_state"
    ):
        return name  # type: ignore[return-value]
    key = str(name).strip().lower().replace("-", "_")
    kind = next(
        (
            canonical
            for canonical, aliases in _TRANSFORM_ALIASES.items()
            if key in aliases
        ),
        None,
    )
    if kind is None:
        raise ValueError(
            f"unknown petrophysical_transform={name!r}; available choices: {', '.join(_TRANSFORM_ALIASES)}"
        )
    if kind == "log_resistivity":
        return LogResistivityTransform(
            model_transform=model_transform, model_bounds=model_bounds
        )
    if model_transform != "log":
        raise ValueError(f"{kind} currently supports model_transform='log' only")
    if kind == "log_conductivity":
        return LogConductivityTransform(model_bounds=model_bounds)

    params = parameters or {}
    required = ("rho_sat", "n") if kind == "saturation" else ("rho0", "theta0", "n")
    missing = [param for param in required if param not in params]
    if missing:
        label = "saturation transform" if kind == "saturation" else kind
        raise ValueError(f"{label} requires petrophysical parameters: {missing}")

    def cellwise(param: str, default=None, allow_time: bool = False):
        value = params.get(param, default)
        return (
            None
            if value is None
            else _parameter_array(
                value, n_cells=n_cells, name=param, allow_time=allow_time
            )
        )

    if kind == "saturation":
        return SaturationTransform(
            rho_sat=cellwise("rho_sat"),
            n=cellwise("n"),
            rho_sat_s=cellwise("rho_sat_s"),
            saturation_floor=saturation_floor,
        )
    return RelativeArchieWaterContentTransform(
        rho0=cellwise("rho0"),
        theta0=cellwise("theta0"),
        n=cellwise("n"),
        theta_min=cellwise("theta_min", 0.02),
        theta_max=cellwise("theta_max", 0.5),
        temperature_correction_factor=cellwise(
            "temperature_correction_factor", allow_time=True
        ),
    )
