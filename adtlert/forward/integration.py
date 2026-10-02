"""Inverse cosine-transform quadrature for the 2.5D wavenumber integral."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import roots_laguerre, roots_legendre
import torch

from adtlert.survey import Survey
from adtlert.utils.dtypes import FLOAT_DTYPE


@dataclass(frozen=True)
class CosineTransformWeights:
    """Wavenumbers and weights used in the 2.5D inverse cosine transform."""

    wavenumbers: torch.Tensor
    weights: torch.Tensor


def survey_wavenumber_bounds(survey: Survey) -> tuple[float, float]:
    """Half the smallest and twice the largest electrode separation."""

    positions = np.asarray(survey.electrode_positions, dtype=float)
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    positive = distances[distances > 0.0]
    return float(np.min(positive) / 2.0), float(np.max(positive) * 2.0)


def build_inverse_cosine_weights(r_min: float, r_max: float) -> CosineTransformWeights:
    """Gauss-Legendre rule on ``[0, k0]`` (in ``sqrt(k)``) plus a four-point Gauss-Laguerre tail."""

    k0 = 1.0 / (2.0 * r_min)
    points, weights = roots_legendre(max(int(6.0 * np.log10(r_max / r_min)), 4))
    points, weights = 0.5 * (points + 1.0), 0.5 * weights
    tail_points, tail_weights = roots_laguerre(4)
    wavenumbers = np.concatenate((k0 * points * points, k0 * (tail_points + 1.0)))
    quadrature = np.concatenate(
        (
            2.0 * k0 * points * weights / np.pi,
            k0 * np.exp(tail_points) * tail_weights / np.pi,
        )
    )
    return CosineTransformWeights(
        torch.as_tensor(wavenumbers, dtype=FLOAT_DTYPE),
        torch.as_tensor(quadrature, dtype=FLOAT_DTYPE),
    )
