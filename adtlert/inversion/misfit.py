"""Data-misfit terms in the (already log-transformed) apparent-resistivity domain.

Robust misfits are linearized by iteratively reweighted least squares (IRLS): each
residual is scaled by ``sqrt(w(r))`` so that the Gauss-Newton system minimizes the
robust objective locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class DataMisfit(Protocol):
    """Interface for data-misfit terms used by linearized inversions."""

    name: str

    def residual(
        self, predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Return the weighted residual used for objective reporting."""

    def linearized_rhs(
        self, predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Return the right-hand side for the linearized data equation."""

    def weighted_jacobian(
        self, jacobian: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Apply data weights to a Jacobian block."""

    def linearized_system(
        self, predicted, observed, weights, jacobian
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(A, b)`` for the current linearized data term."""

    def linearized_cotangent(self, predicted, observed, weights) -> np.ndarray:
        """Return the data-space cotangent whose VJP is the model gradient."""

    def phi(
        self, predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray
    ) -> float:
        """Return the unnormalized data objective."""

    def chi2(
        self, predicted: np.ndarray, observed: np.ndarray, weights: np.ndarray
    ) -> float:
        """Return the mean squared weighted residual."""


def l1_irls_weight(residual: np.ndarray, epsilon: float) -> np.ndarray:
    """``sqrt`` of the smoothed-L1 IRLS weight ``(r^2 + eps^2)^(-1/2)``."""

    eps = max(float(epsilon), np.finfo(float).eps)
    return np.power(residual**2 + eps**2, -0.25)


def huber_irls_weight(residual: np.ndarray, delta: float, epsilon: float) -> np.ndarray:
    """``sqrt`` of the Huber IRLS weight ``min(1, delta / |r|)``."""

    delta = max(float(delta), np.finfo(float).eps)
    absolute = np.abs(residual)
    weight_sq = np.ones_like(absolute, dtype=float)
    outliers = absolute > delta
    weight_sq[outliers] = delta / np.maximum(
        absolute[outliers], max(float(epsilon), np.finfo(float).eps)
    )
    return np.sqrt(weight_sq)


class _LogMisfit:
    """Shared weighted-residual machinery; subclasses override ``_irls`` and ``phi``."""

    def _irls(self, residual: np.ndarray) -> np.ndarray | None:
        return None

    def residual(self, predicted, observed, weights) -> np.ndarray:
        return (
            np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
        ) * np.asarray(weights, dtype=float)

    def linearized_rhs(self, predicted, observed, weights) -> np.ndarray:
        return -self.residual(predicted, observed, weights)

    def weighted_jacobian(self, jacobian, weights) -> np.ndarray:
        matrix = np.asarray(jacobian, dtype=float)
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if matrix.ndim != 2:
            raise ValueError("jacobian must be a 2D array")
        if weights.shape != (matrix.shape[0],):
            raise ValueError(
                f"weights must have one value per Jacobian row ({weights.shape} != ({matrix.shape[0]},))"
            )
        return matrix * weights[:, None]

    def linearized_system(
        self, predicted, observed, weights, jacobian
    ) -> tuple[np.ndarray, np.ndarray]:
        residual = self.residual(predicted, observed, weights)
        scale = self._irls(residual)
        if scale is None:
            return self.weighted_jacobian(jacobian, weights), -residual
        return self.weighted_jacobian(jacobian, weights) * scale.reshape(-1)[
            :, None
        ], -scale * residual

    def linearized_cotangent(self, predicted, observed, weights) -> np.ndarray:
        residual = self.residual(predicted, observed, weights)
        scale = self._irls(residual)
        weights = np.asarray(weights, dtype=float)
        return weights * residual if scale is None else weights * (scale**2) * residual

    def phi(self, predicted, observed, weights) -> float:
        residual = self.residual(predicted, observed, weights).reshape(-1)
        return float(np.dot(residual, residual))

    def chi2(self, predicted, observed, weights) -> float:
        return float(np.mean(self.residual(predicted, observed, weights) ** 2))


@dataclass(frozen=True)
class WeightedLogL2Misfit(_LogMisfit):
    """Weighted L2 misfit."""

    name: str = "weighted_log_l2"


@dataclass(frozen=True)
class WeightedLogL1Misfit(_LogMisfit):
    """IRLS-smoothed L1 misfit ``sum 2 (sqrt(r^2 + eps^2) - eps)``."""

    epsilon: float = 1.0e-3
    name: str = "weighted_log_l1"

    def _irls(self, residual):
        return l1_irls_weight(residual, self.epsilon)

    def phi(self, predicted, observed, weights) -> float:
        residual = self.residual(predicted, observed, weights).reshape(-1)
        eps = max(float(self.epsilon), np.finfo(float).eps)
        return float(np.sum(2.0 * (np.sqrt(residual**2 + eps**2) - eps)))


@dataclass(frozen=True)
class WeightedLogHuberMisfit(_LogMisfit):
    """Huber robust misfit."""

    delta: float = 1.0
    epsilon: float = 1.0e-12
    name: str = "weighted_log_huber"

    def _irls(self, residual):
        return huber_irls_weight(residual, self.delta, self.epsilon)

    def phi(self, predicted, observed, weights) -> float:
        residual = np.abs(self.residual(predicted, observed, weights).reshape(-1))
        delta = max(float(self.delta), np.finfo(float).eps)
        return float(
            np.sum(
                np.where(
                    residual <= delta, residual**2, 2.0 * delta * residual - delta**2
                )
            )
        )


@dataclass(frozen=True)
class LogDataDifferenceL2Misfit(WeightedLogL2Misfit):
    """L2 misfit of time-lapse changes ``log(d_t) - log(d_0)``.

    The vector methods are those of the weighted L2 misfit; the coupled time-lapse
    assembly lives in ``core.py``.
    """

    name: str = "log_data_difference_l2"


_ALIASES = {
    WeightedLogL2Misfit(): ("weighted_log_l2", "weighted_l2", "l2", "log_l2"),
    WeightedLogL1Misfit(): ("weighted_log_l1", "log_l1", "l1"),
    WeightedLogHuberMisfit(): ("weighted_log_huber", "log_huber", "huber", "smooth_l1"),
    LogDataDifferenceL2Misfit(): (
        "log_data_difference_l2",
        "data_difference_l2",
        "difference_log_l2",
        "ratio_log_l2",
        "time_lapse_difference_l2",
    ),
}
_DATA_MISFITS: dict[str, DataMisfit] = {
    alias: misfit for misfit, aliases in _ALIASES.items() for alias in aliases
}


def available_data_misfits() -> tuple[str, ...]:
    """Return canonical data-misfit names for user-facing configuration."""

    return tuple(misfit.name for misfit in _ALIASES)


def build_data_misfit(name: str | DataMisfit) -> DataMisfit:
    """Resolve a data-misfit object from a registered name."""

    if hasattr(name, "residual") and hasattr(name, "linearized_rhs"):
        return name  # type: ignore[return-value]
    try:
        return _DATA_MISFITS[str(name).strip().lower().replace("-", "_")]
    except KeyError as exc:
        raise ValueError(
            f"unknown data_misfit={name!r}; available choices: {', '.join(available_data_misfits())}"
        ) from exc
