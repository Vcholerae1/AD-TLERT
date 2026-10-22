"""True 3D DC forward operator on tetrahedral meshes."""

import numpy as np
import pytest
import torch

from adtlert.forward import (
    ERT3DApparentResistivityFunction,
    ERTForward3D,
    ERTForwardModeling,
    apparent_resistivity_autograd,
)
from adtlert.inr import (
    DualNetworkINR,
    INRConfig,
    MultiscaleINR,
    Optimization,
    cell_centers,
    fit_inr,
    fit_timelapse_inr,
    matrix_free_log_rhoa_series,
    normalize_coordinates,
    spatiotemporal_coordinates,
)
from adtlert.inversion import (
    InversionConfig,
    ParameterizedERTForward2p5D,
    TimeLapseERTInversion,
)
from adtlert.inversion.matrix_free import _normal_log_response_vjp_series
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
        sigma, include_robin_boundary_derivative=True, batch_size=2
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


def assert_close(actual, expected, tol=1e-8):
    actual, expected = (
        np.asarray(actual, dtype=float),
        np.asarray(expected, dtype=float),
    )
    error = np.abs(actual - expected).max() / max(np.abs(expected).max(), 1e-300)
    assert error < tol, f"relative max-norm error {error:.2e} >= {tol:.0e}"


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("boundary", ["mixed", "dirichlet"])
def test_adjoint_dot_test_and_matrix_free_products(setup, order, boundary):
    mesh, survey, sigma = setup
    forward = ERTForward3D.from_mesh_survey(
        mesh, survey, element_order=order, boundary_mode=boundary
    )
    rng = np.random.default_rng(order * 10 + (0 if boundary == "mixed" else 1))
    direction = torch.as_tensor(
        rng.standard_normal(mesh.cell_count) * 1e-3, dtype=torch.float64
    )
    cotangent = torch.as_tensor(
        rng.standard_normal(survey.measurement_count), dtype=torch.float64
    )

    lhs = float((forward.jvp(sigma, direction) * cotangent).sum())
    rhs = float((direction * forward.vjp(sigma, cotangent)).sum())
    assert lhs == pytest.approx(rhs, rel=1e-9)

    lhs_normal = float((forward.normal_jvp(sigma, direction) * cotangent).sum())
    rhs_normal = float((direction * forward.normal_vjp(sigma, cotangent)).sum())
    assert lhs_normal == pytest.approx(rhs_normal, rel=1e-9)

    exact = forward.jacobian(
        sigma, include_robin_boundary_derivative=True, normal_sensitivity=False
    )
    assert_close(exact.T @ cotangent, forward.vjp(sigma, cotangent), tol=1e-10)
    assert_close(exact @ direction, forward.jvp(sigma, direction), tol=1e-10)

    normal = forward.jacobian(sigma)
    assert_close(normal.T @ cotangent, forward.normal_vjp(sigma, cotangent), tol=1e-10)
    assert_close(normal @ direction, forward.normal_jvp(sigma, direction), tol=1e-10)


def test_parameter_aggregation_sums_cells_and_ignores_inactive(setup):
    mesh, survey, sigma = setup
    forward = ERTForward3D.from_mesh_survey(mesh, survey)
    cells = mesh.cell_count
    ids = np.full(cells, -1)
    ids[: cells // 2] = np.arange(cells // 2) // 4
    parameters = int(ids.max()) + 1
    jacobian = forward.jacobian(sigma).numpy()
    expected = np.stack(
        [jacobian[:, ids == p].sum(axis=1) for p in range(parameters)], axis=1
    )
    _, aggregated = forward.solve_with_jacobian(
        sigma,
        include_fields=False,
        jacobian_cell_parameter_ids=ids,
        jacobian_parameter_count=parameters,
    )
    assert_close(aggregated, expected, tol=1e-10)
    cotangent = torch.as_tensor(
        np.random.default_rng(4).standard_normal(survey.measurement_count),
        dtype=torch.float64,
    )
    vjp = forward.normal_vjp(
        sigma, cotangent, cell_parameter_ids=ids, parameter_count=parameters
    )
    assert_close(vjp, expected.T @ cotangent.numpy(), tol=1e-10)


def test_autograd_bridge_matches_vjp_jvp_and_batches_series(setup):
    mesh, survey, sigma = setup
    forward = ERTForward3D.from_mesh_survey(mesh, survey)
    rng = np.random.default_rng(5)
    weights = torch.as_tensor(
        rng.standard_normal(survey.measurement_count), dtype=torch.float64
    )
    sigma_grad = sigma.clone().requires_grad_()
    current = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    rhoa = apparent_resistivity_autograd(sigma_grad, forward, current)
    (rhoa * weights).sum().backward()
    scale = forward._geometric_factors().abs() / 1.5
    assert_close(sigma_grad.grad, forward.vjp(sigma, weights * scale), tol=1e-10)
    assert float(current.grad) == pytest.approx(
        -float((weights * rhoa.detach()).sum()) / 1.5
    )

    direction = torch.as_tensor(
        rng.standard_normal(mesh.cell_count) * 1e-3, dtype=torch.float64
    )
    current_tangent = torch.tensor(0.2, dtype=torch.float64)

    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.forward_operator = forward
    ctx.conductivity_device = sigma.device
    ctx.conductivity_dtype = sigma.dtype
    ctx.saved_tensors = (
        sigma.detach(),
        current.detach(),
        rhoa.detach(),
        forward._geometric_factors().detach().abs(),
    )
    jvp_out = ERT3DApparentResistivityFunction.jvp(
        ctx, direction, current_tangent, None
    )
    expected_jvp = (
        scale * forward.jvp(sigma, direction)
        - rhoa.detach() * current_tangent / current.detach()
    )
    assert_close(jvp_out, expected_jvp, tol=1e-10)

    series = torch.stack([sigma, sigma * 1.3]).requires_grad_()
    series_weights = torch.stack([weights, weights * 0.7])
    series_rhoa = apparent_resistivity_autograd(series, forward)
    (series_rhoa * series_weights).sum().backward()
    for step in range(2):
        single = series[step].detach().clone().requires_grad_()
        value = apparent_resistivity_autograd(single, forward)
        (value * series_weights[step]).sum().backward()
        assert_close(series_rhoa[step].detach(), value.detach(), tol=1e-10)
        assert_close(series.grad[step], single.grad, tol=1e-10)


def test_series_batches_reuse_forward_factorizations_and_field_cache(setup):
    mesh, survey, sigma = setup
    reference = ERTForward3D.from_mesh_survey(mesh, survey)
    forward = ERTForward3D.from_mesh_survey(
        mesh, survey, series_batch_steps=2, normal_field_cache_max_entries=8
    )
    rng = np.random.default_rng(6)
    series = torch.stack([sigma, sigma * 1.2, sigma * 0.8])
    cotangents = torch.as_tensor(
        rng.standard_normal((3, survey.measurement_count)), dtype=torch.float64
    )
    resistance = torch.stack([reference.resistance(model) for model in series])
    gradient = torch.stack(
        [reference.vjp(m, c) for m, c in zip(series, cotangents, strict=True)]
    )

    assert_close(forward.resistance_series(series), resistance, tol=1e-10)
    before = forward.normal_field_cache_info()["hits"]
    normal_grad = forward.normal_vjp_series(series, cotangents)
    assert forward.normal_field_cache_info()["hits"] >= before + 3
    expected_normal = torch.stack(
        [reference.normal_vjp(m, c) for m, c in zip(series, cotangents, strict=True)]
    )
    assert_close(normal_grad, expected_normal, tol=1e-10)

    calls = []
    for plan in forward._solver._plans.values():
        factorize = plan["solver"].factorize
        plan["solver"].factorize = lambda f=factorize: (calls.append(1), f())[1]

    assert_close(forward.vjp_series(series, cotangents), gradient, tol=1e-10)
    assert not calls  # adjoint batches reuse the forward factorizations

    forward.resistance_series(series * 1.7)
    assert len(calls) == 2
    assert_close(forward.vjp_series(series, cotangents), gradient, tol=1e-10)
    assert len(calls) == 4


def test_matrix_free_3d_inversion_and_inr_physics(setup):
    mesh, survey, sigma = setup
    forward = ParameterizedERTForward2p5D.from_mesh_survey(
        mesh, survey, np.arange(mesh.cell_count), regularization_mesh=mesh
    )
    rng = np.random.default_rng(7)
    truth = np.log(1.0 / sigma.numpy())
    model = truth + 0.05 * rng.standard_normal(truth.size)
    predicted, jacobian = forward.forward_and_jacobian(model)
    cotangent = rng.standard_normal(predicted.size)
    matrix_free = _normal_log_response_vjp_series(
        forward, model[None], predicted[None], cotangent[None]
    )[0]
    reference = np.asarray(jacobian).T @ cotangent
    assert_close(matrix_free, reference, tol=1e-8)

    modeling = ERTForwardModeling(mesh=mesh, data=survey)
    series_data = np.stack(
        [modeling.forward(truth + 0.1 * t * (sigma.numpy() > 0.02)) for t in range(2)]
    )
    config = InversionConfig(
        max_iterations=3,
        data_std=0.02,
        optimization_algorithm="lbfgs",
        max_log_step=0.3,
        regularization=0.1,
        temporal_regularization=0.5,
    )
    result = TimeLapseERTInversion(
        modeling, series_data, config, observed_log_data=True
    ).run(np.full(truth.size, np.log(100.0)), initial_log_model=True)
    assert result.iteration_chi2[-1] < result.iteration_chi2[0]

    model_gpu = torch.as_tensor(
        np.vstack([truth, model]), device="cuda", dtype=torch.float64
    ).requires_grad_()
    weights_gpu = torch.as_tensor(
        rng.standard_normal((2, survey.measurement_count)),
        device="cuda",
        dtype=torch.float64,
    )
    (matrix_free_log_rhoa_series(model_gpu, modeling) * weights_gpu).sum().backward()
    h = 1e-5
    plus = model_gpu.detach().clone()
    minus = model_gpu.detach().clone()
    plus[0, 0] += h
    minus[0, 0] -= h
    numeric = float(
        (
            (
                matrix_free_log_rhoa_series(plus, modeling)
                - matrix_free_log_rhoa_series(minus, modeling)
            )
            * weights_gpu
        ).sum()
        / (2 * h)
    )
    assert float(model_gpu.grad[0, 0]) == pytest.approx(numeric, rel=1e-5)

    coords3d = normalize_coordinates(cell_centers(mesh))[0]
    inr_res = fit_inr(
        modeling,
        MultiscaleINR(input_dimensions=3, spatial_dimensions=3),
        coords3d,
        np.exp(series_data[0]),
        0.02,
        config=INRConfig(
            max_iterations=5,
            target_chi2=None,
            optimization=Optimization(learning_rate=1e-2),
        ),
    )
    assert inr_res.history.chi2[-1] < inr_res.history.chi2[0]

    st_coords3d, _, _ = spatiotemporal_coordinates(cell_centers(mesh), np.arange(2.0))
    tl_inr_res = fit_timelapse_inr(
        modeling,
        DualNetworkINR(spatial_dimensions=3),
        st_coords3d,
        np.exp(series_data),
        0.02,
        config=INRConfig(
            max_iterations=5,
            target_chi2=None,
            optimization=Optimization(learning_rate=1e-2),
        ),
    )
    assert tl_inr_res.history.chi2[-1] < tl_inr_res.history.chi2[0]
