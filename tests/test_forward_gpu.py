"""Physics and derivative checks of the 2.5D operator (cuDSS, Triton)."""

import numpy as np
import pytest
import torch

from adtlert.forward import (
    ERTForward2p5D,
    ERTForwardModeling,
    apparent_resistivity_autograd,
)
from tests.conftest import conductivity

pytestmark = pytest.mark.gpu

RNG = np.random.default_rng(1)


def assert_close(actual, expected, tol=1e-8):
    """Max-norm relative comparison: sums with cancellation are accurate to the largest term."""

    actual, expected = (
        np.asarray(actual, dtype=float),
        np.asarray(expected, dtype=float),
    )
    error = np.abs(actual - expected).max() / max(np.abs(expected).max(), 1e-300)
    assert error < tol, f"relative max-norm error {error:.2e} >= {tol:.0e}"


@pytest.fixture(scope="module")
def flat_forward(flat_case):
    return ERTForward2p5D.from_mesh_survey(
        flat_case.mesh, flat_case.survey, normal_field_cache_max_entries=8
    )


@pytest.fixture(scope="module")
def terrain_forward(terrain_case):
    return ERTForward2p5D.from_mesh_survey(
        terrain_case.mesh, terrain_case.survey, normal_field_cache_max_entries=8
    )


def random_like(count, scale=1.0):
    return torch.as_tensor(RNG.standard_normal(count) * scale, dtype=torch.float64)


def test_homogeneous_half_space_reproduces_resistivity(flat_case, flat_forward):
    sigma = torch.full((flat_case.mesh.cell_count,), 0.01, dtype=torch.float64)
    assert_close(
        flat_forward.apparent_resistivity_values(sigma).numpy(), 100.0, tol=2e-3
    )


def test_homogeneous_terrain_with_numerical_geometric_factors(terrain_case):
    forward = ERTForward2p5D.from_mesh_survey(
        terrain_case.mesh,
        terrain_case.survey,
        topographic_geometric_factor_mode="numerical",
    )
    sigma = torch.full((terrain_case.mesh.cell_count,), 0.01, dtype=torch.float64)
    assert forward.use_numerical_primary and forward.use_numerical_geometric_factors
    assert_close(forward.apparent_resistivity_values(sigma).numpy(), 100.0, tol=2e-2)


def test_response_scales_with_resistivity_and_current(flat_case, flat_forward):
    sigma = conductivity(flat_case)
    base = flat_forward.apparent_resistivity_values(sigma).numpy()
    assert_close(
        flat_forward.apparent_resistivity_values(sigma / 3.0).numpy(),
        3.0 * base,
        tol=1e-9,
    )
    assert np.allclose(
        flat_forward.apparent_resistivity_values(sigma, currents=2.0).numpy(),
        base / 2.0,
    )
    assert np.all(base > 0)


def test_resistive_anomaly_changes_data_in_the_expected_direction(
    flat_case, flat_forward
):
    homogeneous = torch.full((flat_case.mesh.cell_count,), 0.01, dtype=torch.float64)
    conductive = flat_forward.apparent_resistivity_values(
        conductivity(flat_case)
    ).numpy()  # 30 ohm-m body in 100
    assert conductive.min() < 99.0 and conductive.max() <= 100.5
    assert flat_forward.apparent_resistivity_values(homogeneous).numpy().min() > 99.0


@pytest.mark.parametrize("case_name", ["flat", "terrain"])
def test_adjoint_dot_test(
    case_name, flat_case, terrain_case, flat_forward, terrain_forward
):
    case, forward = (
        (flat_case, flat_forward)
        if case_name == "flat"
        else (terrain_case, terrain_forward)
    )
    sigma = conductivity(case)
    direction, cotangent = (
        random_like(sigma.numel(), 1e-3),
        random_like(case.survey.measurement_count),
    )
    lhs = float((forward.jvp(sigma, direction) * cotangent).sum())
    rhs = float((direction * forward.vjp(sigma, cotangent)).sum())
    assert lhs == pytest.approx(rhs, rel=1e-6)
    lhs = float((forward.normal_jvp(sigma, direction) * cotangent).sum())
    rhs = float((direction * forward.normal_vjp(sigma, cotangent)).sum())
    assert lhs == pytest.approx(rhs, rel=1e-6)


def test_jvp_matches_finite_differences_of_the_response(flat_case, flat_forward):
    sigma = conductivity(flat_case)
    direction = random_like(sigma.numel(), 1.0) * sigma
    h = 1e-5
    numeric = (
        flat_forward.resistance(sigma + h * direction)
        - flat_forward.resistance(sigma - h * direction)
    ) / (2 * h)
    assert_close(flat_forward.jvp(sigma, direction).numpy(), numeric.numpy(), tol=1e-5)


def test_exact_jacobian_matches_vjp_and_jvp(flat_case, flat_forward):
    sigma = conductivity(flat_case)
    exact = flat_forward.jacobian(
        sigma, include_robin_boundary_derivative=True, normal_sensitivity=False
    )
    cotangent, direction = random_like(exact.shape[0]), random_like(sigma.numel(), 1e-3)
    assert_close(exact.T @ cotangent, flat_forward.vjp(sigma, cotangent))
    assert_close(exact @ direction, flat_forward.jvp(sigma, direction))


@pytest.mark.parametrize("case_name", ["flat", "terrain"])
def test_normal_jacobian_is_consistent_with_matrix_free_products(
    case_name, flat_case, terrain_case, flat_forward, terrain_forward
):
    case, forward = (
        (flat_case, flat_forward)
        if case_name == "flat"
        else (terrain_case, terrain_forward)
    )
    sigma = conductivity(case)
    jacobian = forward.jacobian(sigma)
    cotangent, direction = (
        random_like(jacobian.shape[0]),
        random_like(sigma.numel(), 1e-3),
    )
    assert_close(jacobian.T @ cotangent, forward.normal_vjp(sigma, cotangent))
    assert_close(jacobian @ direction, forward.normal_jvp(sigma, direction))
    chunked = forward.jacobian(sigma, batch_size=7)
    assert_close(chunked, jacobian)


def test_parameter_aggregation_sums_cells_and_ignores_inactive(flat_case, flat_forward):
    sigma = conductivity(flat_case)
    cells = flat_case.mesh.cell_count
    ids = np.full(cells, -1)
    ids[: cells // 2] = (
        np.arange(cells // 2) // 3
    )  # three cells per parameter, the rest inactive
    parameters = int(ids.max()) + 1
    jacobian = flat_forward.jacobian(sigma).numpy()
    expected = np.stack(
        [jacobian[:, ids == p].sum(axis=1) for p in range(parameters)], axis=1
    )
    _, aggregated = flat_forward.solve_with_jacobian(
        sigma, jacobian_cell_parameter_ids=ids, jacobian_parameter_count=parameters
    )
    assert_close(aggregated, expected)
    cotangent = random_like(jacobian.shape[0])
    vjp = flat_forward.normal_vjp(
        sigma, cotangent, cell_parameter_ids=ids, parameter_count=parameters
    )
    assert_close(vjp, expected.T @ cotangent.numpy())


def test_autograd_bridge_matches_vjp_and_jvp(flat_case, flat_forward):
    sigma = conductivity(flat_case).clone().requires_grad_()
    current = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    weights = random_like(flat_case.survey.measurement_count)
    (
        apparent_resistivity_autograd(sigma, flat_forward, current) * weights
    ).sum().backward()
    scale = flat_forward._geometric_factors().abs() / 1.5
    assert_close(sigma.grad, flat_forward.vjp(sigma.detach(), weights * scale))
    assert float(current.grad) == pytest.approx(
        -float(
            (
                weights
                * apparent_resistivity_autograd(
                    sigma.detach(), flat_forward, current.detach()
                )
            ).sum()
        )
        / 1.5
    )


def test_series_matches_individual_solves_and_uses_the_field_cache(
    flat_case, flat_forward
):
    sigma = conductivity(flat_case)
    series = torch.stack([sigma, sigma * 1.2, sigma * 0.8])
    values = flat_forward.apparent_resistivity_series(series)
    for step in range(3):
        assert_close(
            values[step], flat_forward.apparent_resistivity_values(series[step])
        )
    before = flat_forward.normal_field_cache_info()["hits"]
    flat_forward.normal_vjp_series(
        series, torch.ones(3, flat_case.survey.measurement_count, dtype=torch.float64)
    )
    assert flat_forward.normal_field_cache_info()["hits"] >= before + 3


def count_factorizations(forward) -> list:
    """Record every numeric factorization of the operator's current cuDSS plans."""

    calls = []
    for plan in forward._solver._plans.values():
        factorize = plan["solver"].factorize
        plan["solver"].factorize = lambda f=factorize: (calls.append(1), f())[1]
    return calls


@pytest.mark.parametrize("case_name", ["flat_case", "terrain_case"])
def test_series_batches_reuse_forward_factorizations_only_while_valid(
    case_name, request
):
    case = request.getfixturevalue(case_name)
    reference = ERTForward2p5D.from_mesh_survey(case.mesh, case.survey)
    # Three steps in batches of two and one: two plan shapes, both left holding a factorization.
    forward = ERTForward2p5D.from_mesh_survey(
        case.mesh, case.survey, series_batch_steps=2
    )
    sigma = conductivity(case)
    series = torch.stack([sigma, sigma * 1.2, sigma * 0.8])
    cotangents = torch.stack(
        [random_like(case.survey.measurement_count) for _ in range(3)]
    )
    resistance = torch.stack([reference.resistance(model) for model in series])
    gradient = torch.stack(
        [reference.vjp(m, c) for m, c in zip(series, cotangents, strict=True)]
    )

    assert_close(forward.resistance_series(series), resistance)
    plans = [
        key for key in forward._solver._plans if key[0] == forward.discretization.name
    ]
    assert len(plans) == 2
    calls = count_factorizations(forward)
    assert_close(forward.vjp_series(series, cotangents), gradient)
    assert not calls  # the adjoint batches reuse the forward factorizations

    forward.resistance_series(series * 1.7)  # other models take over both plans
    assert len(calls) == 2
    assert_close(forward.vjp_series(series, cotangents), gradient)
    assert len(calls) == 4  # stale factorizations are not reused
    assert_close(forward.vjp(series[1], cotangents[1]), gradient[1])


def test_autograd_bridge_batches_a_series(flat_case, flat_forward):
    sigma = conductivity(flat_case)
    series = torch.stack([sigma, sigma * 1.3]).requires_grad_()
    weights = torch.stack([random_like(flat_case.survey.measurement_count)] * 2)
    rhoa = apparent_resistivity_autograd(series, flat_forward)
    assert rhoa.shape == (2, flat_case.survey.measurement_count)
    (rhoa * weights).sum().backward()
    for step in range(2):
        single = series[step].detach().clone().requires_grad_()
        value = apparent_resistivity_autograd(single, flat_forward)
        (value * weights[step]).sum().backward()
        assert_close(rhoa[step].detach(), value.detach())
        assert_close(series.grad[step], single.grad)


def test_reciprocity_of_the_normal_response(flat_case, flat_forward):
    sigma = conductivity(flat_case)
    survey = flat_case.survey
    swapped = type(survey).from_arrays(
        survey.electrode_positions, survey.measurements[:, [2, 3, 0, 1]]
    )
    other = ERTForward2p5D.from_mesh_survey(flat_case.mesh, swapped)
    assert_close(
        other.resistance(sigma).numpy(), flat_forward.resistance(sigma).numpy()
    )


def test_constructor_and_input_validation(flat_case, flat_forward):
    with pytest.raises(ValueError, match="topographic_geometric_factor_mode"):
        ERTForward2p5D.from_mesh_survey(
            flat_case.mesh, flat_case.survey, topographic_geometric_factor_mode="x"
        )
    with pytest.raises(ValueError, match="non-negative"):
        ERTForward2p5D.from_mesh_survey(
            flat_case.mesh, flat_case.survey, normal_field_cache_max_entries=-1
        )
    with pytest.raises(ValueError, match="conductivity must be scalar"):
        flat_forward.solve(torch.ones(3, dtype=torch.float64))
    with pytest.raises(ValueError, match="parameter_count requires"):
        flat_forward.normal_vjp(
            conductivity(flat_case),
            torch.ones(flat_case.survey.measurement_count),
            parameter_count=3,
        )


def test_close_releases_solvers_and_the_operator_keeps_working(flat_case):
    forward = ERTForward2p5D.from_mesh_survey(flat_case.mesh, flat_case.survey)
    sigma = conductivity(flat_case)
    first = forward.resistance(sigma).numpy()
    forward.close()
    assert_close(forward.resistance(sigma).numpy(), first)


def test_facade_returns_log_responses_and_jacobians(flat_case):
    modeling = ERTForwardModeling(mesh=flat_case.mesh, data=flat_case.survey)
    log_rho = np.log(flat_case.resistivity)
    log_rhoa, jacobian = modeling.forward_and_jacobian(log_rho)
    assert np.allclose(log_rhoa, modeling.forward(log_rho))
    # Scaling every resistivity scales the data: the exact derivative sums to one over cells
    # (the default "normal" sensitivity omits the Robin boundary term and does not).
    _, exact = modeling.forward_and_jacobian(
        log_rho, include_robin_boundary_derivative=True, normal_sensitivity=False
    )
    assert_close(exact.sum(axis=1), np.ones(exact.shape[0]), tol=1e-6)
    assert not np.allclose(jacobian.sum(axis=1), 1.0, atol=1e-2)
