"""True 3D DC forward operator on tetrahedral meshes."""

import numpy as np
import pytest
import torch

from adtlert.forward import ERTForward3D, ERTForwardModeling
from adtlert.mesh import Mesh3D
from adtlert.survey import Survey
from adtlert.workflows import build_wenner_alpha_measurements

pytestmark = pytest.mark.gpu


def box_mesh(nx=6, ny=6, nz=4, relief=0.0):
    """Structured box split into six tetrahedra per hexahedron; ``relief`` bends the surface."""

    xs, ys, zs = (
        np.linspace(0, 12, nx + 1),
        np.linspace(0, 12, ny + 1),
        np.linspace(-8, 0, nz + 1),
    )
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    Z = Z + relief * (Z - Z.min()) / (Z.max() - Z.min()) * np.sin(X / 4.0)
    nodes = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    index = np.arange(nodes.shape[0]).reshape(nx + 1, ny + 1, nz + 1)
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                v = [index[i, j, k], index[i + 1, j, k], index[i + 1, j + 1, k], index[i, j + 1, k],
                     index[i, j, k + 1], index[i + 1, j, k + 1], index[i + 1, j + 1, k + 1], index[i, j + 1, k + 1]]  # fmt: skip
                for a, b, c in (
                    (1, 2, 6),
                    (2, 3, 6),
                    (3, 7, 6),
                    (7, 4, 6),
                    (4, 5, 6),
                    (5, 1, 6),
                ):
                    tets.append([v[0], v[a], v[b], v[c]])
    return nodes, np.asarray(tets)


def line_survey(nodes, spacing_ids=None):
    on_line = np.flatnonzero(
        np.isclose(nodes[:, 1], 6.0) & (nodes[:, 2] >= nodes[:, 2].max() - 1e-9)
    )
    on_line = on_line[np.argsort(nodes[on_line, 0])]
    return Survey.from_arrays(
        nodes[on_line], build_wenner_alpha_measurements(len(on_line))
    )


@pytest.fixture(scope="module")
def setup():
    nodes, tets = box_mesh()
    mesh = Mesh3D.from_arrays(nodes, tets)
    sigma = np.full(mesh.cell_count, 0.01)
    sigma[: mesh.cell_count // 3] = 0.03
    return mesh, line_survey(nodes), torch.as_tensor(sigma)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("boundary", ["mixed", "dirichlet"])
def test_homogeneous_half_space_is_exact(setup, order, boundary):
    mesh, survey, _ = setup
    forward = ERTForward3D.from_mesh_survey(
        mesh, survey, element_order=order, boundary_mode=boundary
    )
    rhoa = forward.apparent_resistivity_values(
        torch.full((mesh.cell_count,), 0.01, dtype=torch.float64)
    ).numpy()
    assert np.allclose(rhoa, 100.0, rtol=1e-6)


def test_jacobian_matches_finite_differences(setup):
    mesh, survey, sigma = setup
    forward = ERTForward3D.from_mesh_survey(mesh, survey)
    jacobian = forward.jacobian(sigma, include_robin_boundary_derivative=True).numpy()
    rng = np.random.default_rng(0)
    direction = torch.as_tensor(rng.standard_normal(mesh.cell_count) * 1e-3) * sigma
    h = 1e-3
    numeric = (
        forward.resistance(sigma + h * direction)
        - forward.resistance(sigma - h * direction)
    ).numpy() / (2 * h)
    assert (
        np.abs(jacobian @ direction.numpy() - numeric).max() / np.abs(numeric).max()
        < 1e-5
    )
    batched = forward.jacobian(
        sigma, include_robin_boundary_derivative=True, batch_size=2, cell_batch_size=100
    ).numpy()
    assert np.allclose(batched, jacobian, rtol=1e-10, atol=1e-14)


def test_quadratic_elements_agree_with_linear_on_a_smooth_model(setup):
    mesh, survey, sigma = setup
    linear = (
        ERTForward3D.from_mesh_survey(mesh, survey, element_order=1)
        .apparent_resistivity_values(sigma)
        .numpy()
    )
    quadratic = (
        ERTForward3D.from_mesh_survey(mesh, survey, element_order=2)
        .apparent_resistivity_values(sigma)
        .numpy()
    )
    assert np.allclose(quadratic, linear, rtol=0.15) and not np.allclose(
        quadratic, linear, rtol=1e-9
    )


def test_scaling_series_and_currents(setup):
    mesh, survey, sigma = setup
    forward = ERTForward3D.from_mesh_survey(mesh, survey)
    series = forward.apparent_resistivity_series(torch.stack([sigma, 2 * sigma]))
    assert np.allclose(
        series[1].numpy(), series[0].numpy() / 2, rtol=1e-8
    )  # rho_a ~ 1 / sigma
    response = forward.solve(sigma, currents=2.0)
    assert np.allclose(
        response.apparent_resistivity.numpy(),
        forward.apparent_resistivity_values(sigma).numpy() / 2,
    )
    assert response.node_potentials.shape[1] == mesh.node_count


def test_topography_uses_numerical_geometric_factors():
    nodes, tets = box_mesh(relief=0.8)
    mesh = Mesh3D.from_arrays(nodes, tets)
    top_z = nodes[:, 2].max()
    assert not mesh.is_flat_surface
    on_surface = np.flatnonzero(
        np.isclose(nodes[:, 1], 6.0) & (nodes[:, 2] > top_z - 1.2)
    )
    survey = Survey.from_arrays(
        nodes[on_surface[np.argsort(nodes[on_surface, 0])]][:7],
        build_wenner_alpha_measurements(7)[:3],
    )
    forward = ERTForward3D.from_mesh_survey(mesh, survey, element_order=2)
    assert forward.geometric_factor_mode == "auto"
    sigma = torch.full((mesh.cell_count,), 0.01, dtype=torch.float64)
    assert np.all(np.isfinite(forward.apparent_resistivity_values(sigma).numpy()))


def test_facade_dispatches_to_the_3d_operator_and_validates(setup):
    mesh, survey, sigma = setup
    modeling = ERTForwardModeling(
        mesh=(mesh.nodes.numpy(), mesh.cells.numpy()),
        data=(survey.electrode_positions.numpy(), survey.measurements.numpy()),
    )
    assert isinstance(modeling.forward_operator, ERTForward3D)
    log_rhoa, jacobian = modeling.forward_and_jacobian(np.log(1 / sigma.numpy()))
    assert jacobian.shape == (survey.measurement_count, mesh.cell_count) and np.all(
        np.isfinite(log_rhoa)
    )
    with pytest.raises(ValueError, match="boundary_mode"):
        ERTForward3D.from_mesh_survey(mesh, survey, boundary_mode="x")
    with pytest.raises(ValueError, match="element_order"):
        ERTForward3D.from_mesh_survey(mesh, survey, element_order=3)
    flat_survey = Survey.from_arrays(
        survey.electrode_positions[:, :2], survey.measurements
    )
    with pytest.raises(ValueError, match="three-coordinate"):
        ERTForward3D.from_mesh_survey(mesh, flat_survey)
    with pytest.raises(ValueError, match="shape"):
        Mesh3D.from_arrays(mesh.nodes.numpy(), mesh.cells.numpy()[:, :3])
