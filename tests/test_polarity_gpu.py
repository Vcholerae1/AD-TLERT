"""Reversed receiver polarity (a negative geometric factor) describes the same measurement."""

import numpy as np
import pytest
import torch

from adtlert.forward import (
    ERTForward2p5D,
    ERTForward3D,
    ERTForwardModeling,
    apparent_resistivity_autograd,
)
from adtlert.inversion.matrix_free import _normal_log_response_vjp
from adtlert.mesh import Mesh3D
from adtlert.survey import Survey
from tests.conftest import conductivity, quad_case
from tests.test_forward3d_gpu import box_mesh, line_survey

pytestmark = pytest.mark.gpu


def swap_receivers(survey: Survey, rows: np.ndarray) -> Survey:
    quads = np.asarray(survey.measurements).copy()
    quads[rows, 2], quads[rows, 3] = quads[rows, 3].copy(), quads[rows, 2].copy()
    return Survey.from_arrays(np.asarray(survey.electrode_positions), quads)


def case_2p5d(slope):
    case = quad_case(slope)
    return (
        case.mesh,
        case.survey,
        conductivity(case),
        ERTForward2p5D.from_mesh_survey,
        {},
    )


def case_3d():
    nodes, tets = box_mesh()
    mesh = Mesh3D.from_arrays(nodes, tets)
    sigma = np.full(mesh.cell_count, 0.01)
    sigma[: mesh.cell_count // 3] = 0.03
    return (
        mesh,
        line_survey(nodes),
        torch.as_tensor(sigma),
        ERTForward3D,
        {"element_order_3d": 2},
    )


@pytest.mark.parametrize("build", [lambda: case_2p5d(0.0), lambda: case_2p5d(0.03), case_3d],
                         ids=["flat", "terrain", "3d"])  # fmt: skip
def test_swapped_receivers_give_the_same_data_and_derivatives(build):
    mesh, survey, sigma, operator, facade_options = build()
    swapped = np.arange(0, survey.measurement_count, 2)
    flipped = swap_receivers(survey, swapped)
    forwards = [operator(mesh, s) for s in (survey, flipped)]
    assert bool(torch.all(flipped.geometric_factors()[swapped] < 0))

    rhoa = [forward.apparent_resistivity_values(sigma) for forward in forwards]
    assert bool(torch.all(rhoa[1] > 0))
    np.testing.assert_allclose(rhoa[1], rhoa[0], rtol=1e-10)
    resistance = [forward.resistance(sigma) for forward in forwards]
    np.testing.assert_allclose(
        resistance[1][swapped], -resistance[0][swapped], rtol=1e-10
    )

    log_rho = np.log(1.0 / sigma.numpy())
    facades = [
        ERTForwardModeling(mesh=mesh, data=s, **facade_options)
        for s in (survey, flipped)
    ]
    for options in (
        {},
        {"include_robin_boundary_derivative": True, "normal_sensitivity": False},
    ):
        normal, reversed_ = (
            facade.forward_and_jacobian(log_rho, **options)[1] for facade in facades
        )
        np.testing.assert_allclose(
            reversed_, normal, rtol=1e-8, atol=1e-10 * np.abs(normal).max()
        )

    cotangent = np.random.default_rng(0).normal(size=survey.measurement_count)
    predicted = np.log(rhoa[0].numpy())
    vjps = [_normal_log_response_vjp(f, log_rho, predicted, cotangent) for f in facades]
    np.testing.assert_allclose(
        vjps[1], vjps[0], rtol=1e-8, atol=1e-10 * np.abs(vjps[0]).max()
    )

    gradients = []
    for forward in forwards:
        model = sigma.clone().requires_grad_(True)
        torch.log(apparent_resistivity_autograd(model, forward)).sum().backward()
        gradients.append(model.grad)
    np.testing.assert_allclose(gradients[1], gradients[0], rtol=1e-8,
                               atol=1e-10 * float(gradients[0].abs().max()))  # fmt: skip
