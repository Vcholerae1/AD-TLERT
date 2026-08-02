from __future__ import annotations

from unittest.mock import patch

import numpy as np
import torch

from adtlert.forward import ERTForward2p5D, apparent_resistivity_autograd
from adtlert.inversion.core import (
    InversionConfig,
    ParameterizedERTForward2p5D,
    _normal_log_response_vjp,
    invert_single_log_resistivity,
)
from adtlert.mesh import Mesh
from adtlert.survey import Survey
from adtlert.utils.dtypes import FLOAT_DTYPE


def _small_forward_operator() -> ERTForward2p5D:
    x_coordinates = np.linspace(-12.0, 12.0, 7)
    z_coordinates = np.asarray((0.0, -5.0, -12.0))
    nodes = np.asarray([[x_value, z_value] for z_value in z_coordinates for x_value in x_coordinates])
    cells: list[list[int]] = []
    for depth_index in range(len(z_coordinates) - 1):
        for x_index in range(len(x_coordinates) - 1):
            upper_left = depth_index * len(x_coordinates) + x_index
            upper_right = upper_left + 1
            lower_left = (depth_index + 1) * len(x_coordinates) + x_index
            lower_right = lower_left + 1
            cells.append([upper_left, upper_right, lower_right, lower_left])

    mesh = Mesh.from_arrays(
        nodes,
        np.asarray(cells, dtype=np.int32),
        surface_node_ids=np.arange(len(x_coordinates), dtype=np.int32),
    )
    electrodes = np.column_stack((x_coordinates, np.zeros_like(x_coordinates)))
    measurements = np.asarray(
        ((0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)),
        dtype=np.int32,
    )
    survey = Survey.from_arrays(electrodes, measurements)
    return ERTForward2p5D.from_mesh_survey(mesh, survey, linear_solver_backend="scipy")


def _model_and_directions(forward: ERTForward2p5D) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    conductivity = torch.linspace(0.008, 0.014, forward.mesh.cell_count, dtype=FLOAT_DTYPE)
    conductivity_direction = 0.002 * torch.randn(forward.mesh.cell_count, generator=generator, dtype=FLOAT_DTYPE)
    data_cotangent = torch.randn(forward.survey.measurement_count, generator=generator, dtype=FLOAT_DTYPE)
    return conductivity, conductivity_direction, data_cotangent


def test_matrix_free_jvp_and_vjp_satisfy_adjoint_identity() -> None:
    forward = _small_forward_operator()
    conductivity, conductivity_direction, data_cotangent = _model_and_directions(forward)

    jacobian_direction = forward.jvp(conductivity, conductivity_direction)
    transposed_jacobian_cotangent = forward.vjp(conductivity, data_cotangent)

    lhs = torch.dot(data_cotangent, jacobian_direction)
    rhs = torch.dot(conductivity_direction, transposed_jacobian_cotangent)
    torch.testing.assert_close(lhs, rhs, rtol=2.0e-5, atol=2.0e-6)


def test_matrix_free_products_match_exact_materialized_jacobian() -> None:
    forward = _small_forward_operator()
    conductivity, conductivity_direction, data_cotangent = _model_and_directions(forward)
    jacobian = forward.jacobian(
        conductivity,
        batch_size=forward.survey.measurement_count,
        include_robin_boundary_derivative=True,
        normal_sensitivity=False,
    )

    torch.testing.assert_close(
        forward.jvp(conductivity, conductivity_direction),
        jacobian @ conductivity_direction,
        rtol=3.0e-5,
        atol=3.0e-6,
    )
    torch.testing.assert_close(
        forward.vjp(conductivity, data_cotangent),
        jacobian.T @ data_cotangent,
        rtol=3.0e-5,
        atol=3.0e-5,
    )


def test_normal_matrix_free_products_match_materialized_normal_jacobian() -> None:
    forward = _small_forward_operator()
    conductivity, conductivity_direction, data_cotangent = _model_and_directions(forward)
    jacobian = forward.jacobian(
        conductivity,
        batch_size=forward.survey.measurement_count,
        include_robin_boundary_derivative=False,
        normal_sensitivity=True,
    )

    normal_jvp = forward.normal_jvp(conductivity, conductivity_direction)
    normal_vjp = forward.normal_vjp(conductivity, data_cotangent)
    torch.testing.assert_close(normal_jvp, jacobian @ conductivity_direction, rtol=3.0e-5, atol=3.0e-6)
    torch.testing.assert_close(normal_vjp, jacobian.T @ data_cotangent, rtol=3.0e-5, atol=3.0e-5)
    torch.testing.assert_close(
        torch.dot(data_cotangent, normal_jvp),
        torch.dot(conductivity_direction, normal_vjp),
        rtol=3.0e-5,
        atol=3.0e-6,
    )

    changed_conductivity = 1.01 * conductivity
    changed_jacobian = forward.jacobian(
        changed_conductivity,
        batch_size=forward.survey.measurement_count,
        include_robin_boundary_derivative=False,
        normal_sensitivity=True,
    )
    torch.testing.assert_close(
        forward.normal_jvp(changed_conductivity, conductivity_direction),
        changed_jacobian @ conductivity_direction,
        rtol=3.0e-5,
        atol=3.0e-6,
    )


def test_parameterized_log_response_vjp_matches_materialized_jacobian() -> None:
    operator = _small_forward_operator()
    n_forward_cells = operator.mesh.cell_count
    forward_parameter_ids = np.arange(n_forward_cells, dtype=np.int32)
    forward_parameter_ids[-1] = -1
    parameter_cell_ids = np.arange(n_forward_cells - 1, dtype=np.int32)
    forward = ParameterizedERTForward2p5D(
        operator,
        parameter_cell_ids,
        forward_cell_parameter_ids=forward_parameter_ids,
        background_mode="pygimli_prolongation",
    )
    log_model = np.log(np.linspace(70.0, 130.0, forward.cell_count))
    predicted_log, jacobian = forward.forward_and_jacobian(log_model, log_transform=True)
    cotangent = np.linspace(-0.7, 0.9, operator.survey.measurement_count)

    matrix_free_gradient = _normal_log_response_vjp(
        forward,
        log_model,
        predicted_log,
        cotangent,
    )

    np.testing.assert_allclose(matrix_free_gradient, jacobian.T @ cotangent, rtol=3.0e-5, atol=3.0e-6)


def test_adam_inversion_does_not_materialize_jacobian() -> None:
    forward = _small_forward_operator()
    true_model = np.linspace(80.0, 120.0, forward.mesh.cell_count)
    observed = np.asarray(forward.apparent_resistivity_values(torch.as_tensor(1.0 / true_model)), dtype=float)
    initial_model = np.full(forward.mesh.cell_count, 100.0)

    def fail_if_materialized(*args: object, **kwargs: object) -> None:
        raise AssertionError("Adam must not call solve_with_jacobian")

    with patch.object(ERTForward2p5D, "solve_with_jacobian", side_effect=fail_if_materialized):
        result = invert_single_log_resistivity(
            forward,
            observed,
            initial_model,
            config=InversionConfig(
                optimization_algorithm="adam",
                max_iterations=1,
                regularization=0.0,
                line_search=False,
            ),
        )

    assert len(result.iteration_chi2) == 1
    assert np.all(np.isfinite(result.final_model))


def test_response_and_normal_vjp_series_reuse_lru_fields() -> None:
    forward = _small_forward_operator()
    base_conductivity, _, _ = _model_and_directions(forward)
    conductivities = torch.stack((0.98 * base_conductivity, base_conductivity, 1.02 * base_conductivity))
    responses = forward.apparent_resistivity_series(conductivities)
    info_after_response = forward.normal_field_cache_info()

    expected_responses = torch.stack(
        [forward.apparent_resistivity_values(conductivities[index]) for index in range(3)]
    )
    torch.testing.assert_close(responses, expected_responses, rtol=2.0e-5, atol=2.0e-5)
    assert info_after_response["entries"] == 3

    cotangents = torch.stack(
        [torch.linspace(-0.5 + 0.1 * index, 0.8, forward.survey.measurement_count) for index in range(3)]
    ).to(dtype=FLOAT_DTYPE)
    gradients = forward.normal_vjp_series(conductivities, cotangents)
    expected_gradients = torch.stack(
        [
            forward.jacobian(
                conductivities[index],
                batch_size=forward.survey.measurement_count,
                include_robin_boundary_derivative=False,
                normal_sensitivity=True,
            ).T
            @ cotangents[index]
            for index in range(3)
        ]
    )
    torch.testing.assert_close(gradients, expected_gradients, rtol=3.0e-5, atol=3.0e-5)
    assert forward.normal_field_cache_info()["hits"] >= 6


def test_autograd_backward_matches_solver_vjp_and_current_derivative() -> None:
    forward = _small_forward_operator()
    conductivity, _, data_cotangent = _model_and_directions(forward)
    conductivity = conductivity.requires_grad_()
    currents = torch.linspace(0.8, 1.1, forward.survey.measurement_count, dtype=FLOAT_DTYPE).requires_grad_()

    apparent_resistivity = apparent_resistivity_autograd(conductivity, forward, currents)
    loss = torch.dot(apparent_resistivity, data_cotangent)
    loss.backward()

    geometric_scale = forward._geometric_factors().abs()  # noqa: SLF001
    expected_conductivity_gradient = forward.vjp(
        conductivity.detach(),
        data_cotangent * geometric_scale / currents.detach(),
    )
    expected_current_gradient = -data_cotangent * apparent_resistivity.detach() / currents.detach()
    torch.testing.assert_close(conductivity.grad, expected_conductivity_gradient, rtol=3.0e-5, atol=3.0e-5)
    torch.testing.assert_close(currents.grad, expected_current_gradient, rtol=3.0e-5, atol=3.0e-5)


def test_autograd_forward_ad_matches_solver_jvp() -> None:
    forward = _small_forward_operator()
    conductivity, conductivity_direction, _ = _model_and_directions(forward)
    currents = torch.linspace(0.8, 1.1, forward.survey.measurement_count, dtype=FLOAT_DTYPE)
    current_direction = torch.linspace(-0.1, 0.1, forward.survey.measurement_count, dtype=FLOAT_DTYPE)

    with torch.autograd.forward_ad.dual_level():
        dual_conductivity = torch.autograd.forward_ad.make_dual(conductivity, conductivity_direction)
        dual_currents = torch.autograd.forward_ad.make_dual(currents, current_direction)
        dual_response = apparent_resistivity_autograd(dual_conductivity, forward, dual_currents)
        response, response_tangent = torch.autograd.forward_ad.unpack_dual(dual_response)

    geometric_scale = forward._geometric_factors().abs()  # noqa: SLF001
    expected_tangent = (
        geometric_scale * forward.jvp(conductivity, conductivity_direction) / currents
        - response * current_direction / currents
    )
    torch.testing.assert_close(response_tangent, expected_tangent, rtol=3.0e-5, atol=3.0e-5)
