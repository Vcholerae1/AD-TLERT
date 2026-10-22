"""Pole configurations: an ABMN index of -1 is an electrode at infinity."""

import numpy as np
import pytest
import torch

from adtlert.forward import ERTForward2p5D, ERTForward3D
from adtlert.mesh import Mesh3D
from adtlert.survey import REMOTE_ELECTRODE, Survey
from adtlert.workflows import build_wenner_alpha_measurements
from tests.conftest import conductivity, quad_case
from tests.test_forward3d_gpu import box_mesh, line_survey
from tests.test_forward_gpu import assert_close

pytestmark = pytest.mark.gpu

R = REMOTE_ELECTRODE


def with_poles(survey: Survey) -> Survey:
    """The survey's own dipole-dipole rows plus pole-dipole, dipole-pole and pole-pole rows."""

    count = survey.electrode_count
    poles = [(a, R, a + 1 + s, a + 2 + s) for a in range(count - 3) for s in (0, 1)
             if a + 2 + s < count]  # fmt: skip
    poles += [(a, a + 1, a + 2 + s, R) for a in range(count - 2) for s in (0, 1)
              if a + 2 + s < count]  # fmt: skip
    poles += [(a, R, a + 1 + s, R) for a in range(count - 1) for s in (0, 1)
              if a + 1 + s < count]  # fmt: skip
    quads = np.vstack((np.asarray(survey.measurements), np.asarray(poles)))
    return Survey.from_arrays(np.asarray(survey.electrode_positions), quads)


def flat():
    case = quad_case(0.0)
    survey = with_poles(case.survey)
    return (
        case.mesh,
        survey,
        conductivity(case),
        ERTForward2p5D.from_mesh_survey(case.mesh, survey),
        2e-3,
    )


def three_d(relief=0.0):
    nodes, tets = box_mesh(relief=relief)
    mesh = Mesh3D.from_arrays(nodes, tets)
    sigma = np.full(mesh.cell_count, 0.01)
    sigma[: mesh.cell_count // 3] = 0.03
    surface = np.isclose(nodes[:, 2], relief * np.sin(nodes[:, 0] / 4.0))
    line = np.flatnonzero(np.isclose(nodes[:, 1], 6.0) & surface)
    line = line[np.argsort(nodes[line, 0])]
    survey = with_poles(
        Survey.from_arrays(nodes[line], build_wenner_alpha_measurements(len(line)))
    )
    return mesh, survey, torch.as_tensor(sigma), ERTForward3D(mesh, survey), 1e-6


CASES = {"flat": flat, "3d": three_d, "3d-terrain": lambda: three_d(relief=0.8)}


@pytest.fixture(scope="module", params=list(CASES))
def case(request):
    return CASES[request.param]()


def random(count, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return torch.as_tensor(rng.standard_normal(count) * scale, dtype=torch.float64)


def test_homogeneous_half_space_is_reproduced_by_every_configuration(case):
    mesh, survey, _, forward, tol = case
    assert bool(torch.any(survey.measurements == R))
    rhoa = forward.apparent_resistivity_values(
        torch.full((mesh.cell_count,), 0.01, dtype=torch.float64)
    )
    assert_close(rhoa.numpy(), 100.0, tol=tol)


def test_derivatives_are_adjoint_consistent_with_remote_electrodes(case):
    _, survey, sigma, forward, _ = case
    direction = random(sigma.numel(), 1e-3) * sigma
    cotangent = random(survey.measurement_count, seed=1)
    for jvp, vjp in (
        (forward.jvp, forward.vjp),
        (forward.normal_jvp, forward.normal_vjp),
    ):
        lhs = float((jvp(sigma, direction) * cotangent).sum())
        rhs = float((direction * vjp(sigma, cotangent)).sum())
        assert lhs == pytest.approx(rhs, rel=1e-6)
    jacobian = forward.jacobian(sigma)
    assert_close(jacobian.T @ cotangent, forward.normal_vjp(sigma, cotangent))
    assert_close(jacobian @ direction, forward.normal_jvp(sigma, direction))


def test_jvp_matches_finite_differences_with_remote_electrodes(case):
    _, _, sigma, forward, _ = case
    direction = random(sigma.numel(), 1.0, seed=2) * sigma
    h = 1e-5
    numeric = (forward.resistance(sigma + h * direction)
               - forward.resistance(sigma - h * direction)) / (2 * h)  # fmt: skip
    assert_close(forward.jvp(sigma, direction).numpy(), numeric.numpy(), tol=1e-5)


def test_terrain_2p5d_rejects_remote_electrodes():
    case = quad_case(0.03)
    with pytest.raises(ValueError, match="remote electrodes need absolute potentials"):
        ERTForward2p5D.from_mesh_survey(case.mesh, with_poles(case.survey))


def test_a_dipole_is_the_difference_of_two_poles_in_3d():
    mesh, _, sigma, _, _ = three_d()
    nodes, _ = box_mesh()
    survey = line_survey(nodes)
    quads = np.asarray(survey.measurements)
    a, b, m, n = quads.T
    poles = np.vstack((np.column_stack((a, np.full_like(a, R), m, n)),
                       np.column_stack((b, np.full_like(b, R), m, n))))  # fmt: skip
    pole_survey = Survey.from_arrays(np.asarray(survey.electrode_positions), poles)
    dipole, pole = ERTForward3D(mesh, survey), ERTForward3D(mesh, pole_survey)
    count = survey.measurement_count
    resistance = pole.resistance(sigma)
    assert_close(
        dipole.resistance(sigma), resistance[:count] - resistance[count:], tol=1e-10
    )
    jacobian = pole.jacobian(sigma)
    assert_close(dipole.jacobian(sigma), jacobian[:count] - jacobian[count:], tol=1e-10)
