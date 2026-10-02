"""Shared fixtures. Float64 is selected before adtlert is imported so finite differences are accurate."""

import os

os.environ.setdefault("ADTLERT_ENABLE_FLOAT64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from adtlert.workflows import ParflowGrid, build_terrain_forward_case  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs an NVIDIA GPU (cuDSS, Triton)")


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="requires a CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


def quad_case(
    slope: float = 0.0,
    nx: int = 30,
    nz: int = 12,
    n_electrodes: int = 16,
    anomaly: bool = True,
):
    """Terrain-following quadrilateral strip case; ``slope = 0`` is a flat surface."""

    grid = ParflowGrid(
        dx=2.0,
        dy=1.0,
        dz_base=1.0,
        nx=nx,
        ny=1,
        nz=nz,
        dz_scales=np.linspace(2.0, 0.5, nz),
    )
    slope_x = np.full(nx, slope) + (0.0 if slope == 0 else 0.05 * np.sin(np.arange(nx)))
    rho = np.full((nz, nx), 100.0)
    if anomaly:
        rho[3:7, 10:18] = 30.0
    return build_terrain_forward_case(
        rho, grid, slope_x, y_index=0, n_electrodes=n_electrodes
    )


@pytest.fixture(scope="session")
def flat_case():
    return quad_case(0.0)


@pytest.fixture(scope="session")
def terrain_case():
    return quad_case(0.03)


def conductivity(case) -> torch.Tensor:
    return torch.as_tensor(1.0 / case.resistivity, dtype=torch.float64)
