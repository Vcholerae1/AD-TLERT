"""GPU-first implicit neural representations for ADTLERT inversion."""

from adtlert.inr.coords import (
    cell_centers,
    normalize_coordinates,
    spatiotemporal_coordinates,
)
from adtlert.inr.networks import (
    DualNetworkINR,
    JointSpatioTemporalINR,
    MultiscaleFourierEncoder,
    MultiscaleINR,
    fourier_frequency_bank,
)
from adtlert.inr.physics import (
    matrix_free_log_rhoa,
    matrix_free_log_rhoa_series,
    prepare_cuda_forward,
)
from adtlert.inr.train import (
    History,
    INRConfig,
    INRResult,
    Optimization,
    Progressive,
    Regularization,
    Timing,
    Windows,
    fit_inr,
    fit_timelapse_inr,
)

__all__ = [
    "DualNetworkINR",
    "History",
    "INRConfig",
    "INRResult",
    "JointSpatioTemporalINR",
    "MultiscaleFourierEncoder",
    "MultiscaleINR",
    "Optimization",
    "Progressive",
    "Regularization",
    "Timing",
    "Windows",
    "cell_centers",
    "fit_inr",
    "fit_timelapse_inr",
    "fourier_frequency_bank",
    "matrix_free_log_rhoa",
    "matrix_free_log_rhoa_series",
    "normalize_coordinates",
    "prepare_cuda_forward",
    "spatiotemporal_coordinates",
]
