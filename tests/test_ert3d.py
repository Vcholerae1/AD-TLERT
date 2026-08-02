from __future__ import annotations

import numpy as np
import pytest

from adtlert.forward import ERTForward3D, ERTForwardModeling
from adtlert.inversion import ERTInversion, InversionConfig
from adtlert.inversion.regularization import first_order_constraint_matrix
from adtlert.mesh import Mesh3D
from adtlert.survey import Survey


def structured_tetrahedral_mesh() -> Mesh3D:
    x = np.linspace(-10.0, 10.0, 5)
    y = np.linspace(-10.0, 10.0, 5)
    z = np.linspace(-10.0, 0.0, 3)
    nodes = np.asarray(np.meshgrid(x, y, z, indexing="ij")).reshape(3, -1).T

    def node_id(i: int, j: int, k: int) -> int:
        return (i * len(y) + j) * len(z) + k

    cells: list[list[int]] = []
    for i in range(len(x) - 1):
        for j in range(len(y) - 1):
            for k in range(len(z) - 1):
                n000 = node_id(i, j, k)
                n100 = node_id(i + 1, j, k)
                n010 = node_id(i, j + 1, k)
                n110 = node_id(i + 1, j + 1, k)
                n001 = node_id(i, j, k + 1)
                n101 = node_id(i + 1, j, k + 1)
                n011 = node_id(i, j + 1, k + 1)
                n111 = node_id(i + 1, j + 1, k + 1)
                cells.extend(
                    (
                        (n000, n100, n110, n111),
                        (n000, n110, n010, n111),
                        (n000, n010, n011, n111),
                        (n000, n011, n001, n111),
                        (n000, n001, n101, n111),
                        (n000, n101, n100, n111),
                    )
                )
    return Mesh3D.from_arrays(nodes, np.asarray(cells, dtype=np.int32))


def surface_survey() -> Survey:
    electrodes = np.asarray(
        ((-5.0, -5.0, 0.0), (5.0, -5.0, 0.0), (-5.0, 5.0, 0.0), (5.0, 5.0, 0.0)),
        dtype=float,
    )
    return Survey.from_arrays(electrodes, np.asarray(((0, 1, 2, 3),), dtype=np.int32))


def terrain_mesh_and_survey() -> tuple[Mesh3D, Survey]:
    base = structured_tetrahedral_mesh()
    nodes = np.asarray(base.nodes, dtype=float).copy()
    z_min = float(np.min(nodes[:, 2]))
    terrain = 0.8 * np.sin(nodes[:, 0] / 10.0) * np.cos(nodes[:, 1] / 10.0)
    nodes[:, 2] += terrain * (nodes[:, 2] - z_min) / -z_min
    mesh = Mesh3D.from_arrays(nodes, np.asarray(base.cells, dtype=np.int32))
    electrodes = np.asarray(surface_survey().electrode_positions, dtype=float).copy()
    electrodes[:, 2] = 0.8 * np.sin(electrodes[:, 0] / 10.0) * np.cos(electrodes[:, 1] / 10.0)
    survey = Survey.from_arrays(electrodes, np.asarray(((0, 1, 2, 3),), dtype=np.int32))
    return mesh, survey


def cudss_available() -> bool:
    try:
        import cupy as cp
        from nvmath.sparse.advanced import DirectSolver  # noqa: F401

        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def test_mesh3d_locates_surface_points() -> None:
    mesh = structured_tetrahedral_mesh()
    cell_ids, weights = mesh.locate_points(surface_survey().electrode_positions)
    assert np.all(np.asarray(cell_ids) >= 0)
    np.testing.assert_allclose(np.sum(np.asarray(weights), axis=1), 1.0)
    assert mesh.is_flat_surface


def test_uniform_halfspace_returns_model_resistivity() -> None:
    mesh = structured_tetrahedral_mesh()
    forward = ERTForward3D.from_mesh_survey(mesh, surface_survey())
    conductivity = np.full(mesh.cell_count, 1.0 / 100.0)
    response = forward.solve(conductivity)
    np.testing.assert_allclose(np.asarray(response.apparent_resistivity), 100.0, rtol=2.0e-5)


def test_3d_jacobian_matches_directional_finite_difference() -> None:
    mesh = structured_tetrahedral_mesh()
    forward = ERTForward3D.from_mesh_survey(mesh, surface_survey())
    conductivity = np.full(mesh.cell_count, 1.0 / 100.0)
    rng = np.random.default_rng(7)
    direction = conductivity * rng.normal(size=mesh.cell_count)
    epsilon = 1.0e-3

    jacobian = np.asarray(
        forward.jacobian(conductivity, include_robin_boundary_derivative=True),
        dtype=float,
    )
    plus = np.asarray(forward.resistance(conductivity + epsilon * direction), dtype=float)
    minus = np.asarray(forward.resistance(conductivity - epsilon * direction), dtype=float)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    np.testing.assert_allclose(jacobian @ direction, finite_difference, rtol=3.0e-3, atol=1.0e-5)


def test_modeling_facade_routes_3d_mesh() -> None:
    mesh = structured_tetrahedral_mesh()
    modeling = ERTForwardModeling(mesh=mesh, data=surface_survey())
    assert isinstance(modeling.forward_operator, ERTForward3D)
    model = np.full(mesh.cell_count, np.log(100.0))
    response, jacobian = modeling.forward_and_jacobian(
        model,
        include_robin_boundary_derivative=True,
    )
    np.testing.assert_allclose(np.exp(response), 100.0, rtol=2.0e-5)
    assert jacobian.shape == (1, mesh.cell_count)


def test_3d_regularization_and_single_iteration_inversion() -> None:
    mesh = structured_tetrahedral_mesh()
    survey = surface_survey()
    modeling = ERTForwardModeling(mesh=mesh, data=survey)
    constraints = first_order_constraint_matrix(mesh, z_weight=0.5)
    assert constraints.shape[1] == mesh.cell_count
    assert constraints.shape[0] > 0

    observed = np.asarray(
        ERTForward3D.from_mesh_survey(mesh, survey).solve(
            np.full(mesh.cell_count, 1.0 / 80.0)
        ).apparent_resistivity,
        dtype=float,
    )
    result = ERTInversion(
        forward=modeling,
        observed_data=observed,
        config=InversionConfig(
            max_iterations=1,
            spatial_regularization="identity",
            regularization=1.0e-2,
            include_robin_boundary_derivative=True,
        ),
    ).setup().run(np.full(mesh.cell_count, 100.0))
    assert result.final_model.shape == (mesh.cell_count,)
    assert np.all(np.isfinite(result.final_model))


def test_p2_terrain_uses_numerical_geometric_factors() -> None:
    mesh, survey = terrain_mesh_and_survey()
    assert not mesh.is_flat_surface
    forward = ERTForward3D.from_mesh_survey(
        mesh,
        survey,
        element_order=2,
        geometric_factor_mode="auto",
        linear_solver_backend="scipy",
    )
    assert forward._dof_nodes.shape[0] > mesh.node_count
    response = forward.solve(np.full(mesh.cell_count, 1.0 / 100.0))
    np.testing.assert_allclose(np.asarray(response.apparent_resistivity), 100.0, rtol=2.0e-5)


def test_p2_terrain_jacobian_matches_finite_difference() -> None:
    mesh, survey = terrain_mesh_and_survey()
    forward = ERTForward3D.from_mesh_survey(
        mesh,
        survey,
        element_order=2,
        linear_solver_backend="scipy",
    )
    conductivity = np.full(mesh.cell_count, 1.0 / 100.0)
    direction = conductivity * np.random.default_rng(11).normal(size=mesh.cell_count)
    epsilon = 1.0e-3
    jacobian = np.asarray(
        forward.jacobian(conductivity, include_robin_boundary_derivative=True),
        dtype=float,
    )
    plus = np.asarray(forward.resistance(conductivity + epsilon * direction), dtype=float)
    minus = np.asarray(forward.resistance(conductivity - epsilon * direction), dtype=float)
    np.testing.assert_allclose(
        jacobian @ direction,
        (plus - minus) / (2.0 * epsilon),
        rtol=5.0e-3,
        atol=2.0e-5,
    )


@pytest.mark.skipif(not cudss_available(), reason="CUDA/cuDSS is not available")
def test_cudss_matches_scipy_for_p2_terrain() -> None:
    mesh, survey = terrain_mesh_and_survey()
    centers = np.mean(np.asarray(mesh.nodes)[np.asarray(mesh.cells)], axis=1)
    resistivity = np.where(np.linalg.norm(centers - np.asarray((0.0, 0.0, -3.0)), axis=1) < 5.0, 30.0, 100.0)
    scipy_forward = ERTForward3D.from_mesh_survey(
        mesh,
        survey,
        element_order=2,
        linear_solver_backend="scipy",
    )
    cudss_forward = ERTForward3D.from_mesh_survey(
        mesh,
        survey,
        element_order=2,
        linear_solver_backend="cudss",
    )
    scipy_response = np.asarray(scipy_forward.resistance(1.0 / resistivity), dtype=float)
    cudss_response = np.asarray(cudss_forward.resistance(1.0 / resistivity), dtype=float)
    np.testing.assert_allclose(cudss_response, scipy_response, rtol=2.0e-5, atol=1.0e-7)
