from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from deepert.inversion.core import InversionConfig, _cupy_timelapse_cgls, _solve_increment


def test_gpu_cgls_matches_lsqr_when_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    try:
        import cupy  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"CuPy is not available: {exc}")

    rng = np.random.default_rng(1234)
    dense = rng.normal(size=(18, 7))
    dense[::3, 2] = 0.0
    matrix = sp.csr_matrix(dense)
    expected = rng.normal(size=7)
    rhs = dense @ expected

    lsqr_solution = _solve_increment(
        matrix,
        rhs,
        InversionConfig(linearized_solver="lsqr", lsqr_atol=1e-10, lsqr_btol=1e-10),
    )
    gpu_solution = _solve_increment(
        matrix,
        rhs,
        InversionConfig(linearized_solver="gpu_cgls", cgls_max_iterations=200, cgls_tolerance=1e-12),
    )

    np.testing.assert_allclose(gpu_solution, lsqr_solution, rtol=1.0e-6, atol=1.0e-7)


def test_gpu_timelapse_cgls_matches_assembled_gpu_cgls_when_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    try:
        import cupy  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"CuPy is not available: {exc}")

    rng = np.random.default_rng(5678)
    n_times = 3
    n_measurements = 5
    n_cells = 4
    jacobians = [rng.normal(size=(n_measurements, n_cells)) for _ in range(n_times)]
    weight = rng.uniform(0.5, 1.5, size=(n_times, n_measurements))
    data_rhs = rng.normal(size=(n_times, n_measurements))
    spatial_regularization = sp.csr_matrix(
        np.asarray(
            [
                [1.0, -1.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 0.0],
                [0.0, 0.0, 1.0, -1.0],
            ]
        )
    )
    spatial_scale = 1.7
    spatial_rhs = rng.normal(size=(n_times, spatial_regularization.shape[0]))
    temporal_scale = 0.8
    temporal_rhs = rng.normal(size=(n_times - 1, n_cells))

    data_blocks = [sp.csr_matrix(jac_t * w_t[:, None]) for jac_t, w_t in zip(jacobians, weight)]
    data_matrix = sp.block_diag(data_blocks, format="csr")
    spatial_matrix = spatial_scale * sp.block_diag([spatial_regularization] * n_times, format="csr")
    rows = []
    cols = []
    values = []
    for time_index in range(1, n_times):
        row_offset = (time_index - 1) * n_cells
        previous_offset = (time_index - 1) * n_cells
        current_offset = time_index * n_cells
        for cell_index in range(n_cells):
            row = row_offset + cell_index
            rows.extend((row, row))
            cols.extend((current_offset + cell_index, previous_offset + cell_index))
            values.extend((temporal_scale, -temporal_scale))
    temporal_matrix = sp.coo_matrix(
        (values, (rows, cols)),
        shape=(n_cells * (n_times - 1), n_cells * n_times),
    ).tocsr()
    matrix = sp.vstack((data_matrix, spatial_matrix, temporal_matrix), format="csr")
    rhs = np.concatenate((data_rhs.reshape(-1), spatial_rhs.reshape(-1), temporal_rhs.reshape(-1)))

    expected = _solve_increment(
        matrix,
        rhs,
        InversionConfig(linearized_solver="gpu_cgls", cgls_max_iterations=120, cgls_tolerance=1e-12),
    )
    actual = _cupy_timelapse_cgls(
        jacobians=jacobians,
        weight=weight,
        data_rhs=data_rhs,
        spatial_regularization=spatial_regularization,
        spatial_scale=spatial_scale,
        spatial_rhs=spatial_rhs,
        temporal_scale=temporal_scale,
        temporal_rhs=temporal_rhs,
        max_iterations=120,
        tolerance=1e-12,
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-7)
