"""End-to-end inversion behaviour on a small synthetic problem."""

import numpy as np
import pytest
import torch

from adtlert.forward import ERTForward2p5D, ERTForwardModeling
from adtlert.inversion import (
    ERTInversion,
    InversionConfig,
    ParameterizedERTForward2p5D,
    TimeLapseERTInversion,
    WindowedTimeLapseERTInversion,
)
from adtlert.inversion.core import _normal_log_response_vjp_series
from tests.conftest import quad_case

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def problem():
    case = quad_case(0.0, nx=24, nz=10, n_electrodes=14)
    modeling = ERTForwardModeling(mesh=case.mesh, data=case.survey)
    truth = np.log(case.resistivity)
    rng = np.random.default_rng(0)
    clean = modeling.forward(truth)
    data = clean + 0.01 * rng.standard_normal(clean.shape)
    series = np.stack(
        [modeling.forward(truth + 0.15 * t * (case.resistivity < 50)) for t in range(3)]
    )
    return case, modeling, truth, data, series


def test_single_inversion_reduces_misfit_and_finds_the_conductive_body(problem):
    case, modeling, truth, data, _ = problem
    config = InversionConfig(
        max_iterations=4,
        data_std=0.02,
        spatial_regularization="first_order_smoothness",
        regularization=1.0,
    )
    result = (
        ERTInversion(modeling, data, config, observed_log_data=True)
        .setup()
        .run(np.full(truth.size, np.log(100.0)), initial_log_model=True)
    )
    assert result.iteration_chi2[-1] < 0.5 * result.iteration_chi2[0]
    body = case.resistivity < 50
    gap = result.final_log_model[~body].mean() - result.final_log_model[body].mean()
    assert gap > 0.6  # the truth is log(100 / 30) = 1.2
    assert (
        result.final_model.shape == truth.shape and result.coverage.shape == truth.shape
    )
    assert (
        np.all(np.isfinite(result.predicted_data))
        and result.final_parameter_name == "resistivity"
    )


def test_progress_events_follow_the_documented_sequence(problem):
    _, modeling, truth, data, _ = problem
    events = []
    config = InversionConfig(
        max_iterations=2, data_std=0.02, progress_callback=events.append
    )
    ERTInversion(modeling, data, config, observed_log_data=True).run(
        np.full(truth.size, np.log(100.0)), initial_log_model=True
    )
    names = [event["event"] for event in events]
    assert names[0] == "single_start" and names[-1] == "single_done"
    assert (
        names.count("single_iteration_start")
        == names.count("single_iteration_done")
        == 2
    )
    assert events[-1]["stop_reason"] in {
        "max_iterations",
        "step_tolerance",
        "target_chi2",
    }


@pytest.mark.parametrize("algorithm", ["lbfgs", "adam", "nonlinear_cg"])
def test_first_order_inversions_reduce_misfit(algorithm, problem):
    _, modeling, truth, data, _ = problem
    config = InversionConfig(
        max_iterations=6,
        data_std=0.02,
        optimization_algorithm=algorithm,
        max_log_step=0.3,
        regularization=0.1,
    )
    result = ERTInversion(modeling, data, config, observed_log_data=True).run(
        np.full(truth.size, np.log(100.0)), initial_log_model=True
    )
    chi2 = result.iteration_chi2
    # Adam's trajectory is not monotone; the quasi-Newton and CG methods must clearly converge.
    assert min(chi2) < chi2[0] if algorithm == "adam" else chi2[-1] < 0.1 * chi2[0]


def test_target_chi2_stops_early(problem):
    _, modeling, truth, data, _ = problem
    events = []
    config = InversionConfig(
        max_iterations=8,
        data_std=0.02,
        target_chi2=1e6,
        progress_callback=events.append,
    )
    result = ERTInversion(modeling, data, config, observed_log_data=True).run(
        np.full(truth.size, np.log(100.0)), initial_log_model=True
    )
    assert (
        len(result.iteration_chi2) == 1 and events[-1]["stop_reason"] == "target_chi2"
    )


def identity_parameterization(case, **options):
    forward = ERTForward2p5D.from_mesh_survey(case.mesh, case.survey, **options)
    return ParameterizedERTForward2p5D(
        forward, np.arange(case.mesh.cell_count), regularization_mesh=case.mesh
    )


def test_matrix_free_gradient_equals_the_explicit_jacobian_product(problem):
    case, _, truth, data, _ = problem
    forward = identity_parameterization(case)
    model = truth + 0.1 * np.random.default_rng(2).standard_normal(truth.size)
    predicted, jacobian = forward.forward_and_jacobian(model)
    cotangent = np.random.default_rng(3).standard_normal(predicted.size)
    matrix_free = _normal_log_response_vjp_series(
        forward, model[None], predicted[None], cotangent[None]
    )[0]
    reference = np.asarray(jacobian).T @ cotangent
    assert np.abs(matrix_free - reference).max() / np.abs(reference).max() < 1e-6


def test_time_lapse_and_windowed_inversions(problem):
    case, modeling, truth, _, series = problem
    start = np.full(truth.size, np.log(100.0))
    config = InversionConfig(
        max_iterations=2,
        data_std=0.02,
        temporal_regularization=1.0,
        regularization=1.0,
        spatial_regularization="first_order_smoothness",
    )
    joint = (
        TimeLapseERTInversion(modeling, series, config, observed_log_data=True)
        .setup()
        .run(start, initial_log_model=True)
    )
    assert (
        joint.final_log_models.shape == (truth.size, 3)
        and joint.predicted_log_data.shape == series.shape
    )
    assert (
        joint.iteration_chi2[-1] < joint.iteration_chi2[0]
        and len(joint.all_coverage) == 3
    )
    windowed = (
        WindowedTimeLapseERTInversion(
            modeling, series, config, window_size=2, observed_log_data=True
        )
        .setup()
        .run(start, initial_log_model=True)
    )
    assert len(windowed.window_reports) == 2 and windowed.final_log_models.shape == (
        truth.size,
        3,
    )
    assert [report["start_idx"] for report in windowed.window_reports] == [0, 1]
    with pytest.raises(ValueError, match="at least two"):
        TimeLapseERTInversion(modeling, series[:1], config, observed_log_data=True).run(
            start, initial_log_model=True
        )


def test_freeze_first_timestep_keeps_the_baseline(problem):
    _, modeling, truth, _, series = problem
    start = np.column_stack([np.full(truth.size, np.log(100.0))] * 3)
    start[:, 0] = truth
    config = InversionConfig(
        max_iterations=2,
        data_std=0.02,
        temporal_regularization=1.0,
        freeze_first_timestep=True,
    )
    result = TimeLapseERTInversion(
        modeling, series, config, observed_log_data=True
    ).run(start, initial_log_model=True)
    assert np.allclose(result.final_log_models[:, 0], truth)


def test_time_lapse_with_water_content_parameterization(problem):
    _, modeling, truth, _, series = problem
    n = truth.size
    params = {"rho0": np.exp(truth), "theta0": np.full(n, 0.25), "n": np.full(n, 2.0)}
    config = InversionConfig(
        max_iterations=2, data_std=0.02, temporal_regularization=1.0, regularization=0.5, petrophysical_transform="relative_archie_water_content",
        petrophysical_parameters=params, regularization_domain="physical",
    )  # fmt: skip
    result = TimeLapseERTInversion(
        modeling, series, config, observed_log_data=True
    ).run(np.log(np.exp(truth))[:, None].repeat(3, axis=1), initial_log_model=True)
    assert result.final_parameter_name == "water_content"
    assert np.all(
        (result.final_parameter_models > 0.02) & (result.final_parameter_models < 0.5)
    )


def test_input_validation(problem):
    _, modeling, truth, data, _ = problem
    start = np.full(truth.size, np.log(100.0))
    with pytest.raises(ValueError, match="observed_data must have shape"):
        ERTInversion(
            modeling, data[:-1], InversionConfig(), observed_log_data=True
        ).setup()
    with pytest.raises(ValueError, match="initial_model must have shape"):
        ERTInversion(modeling, data, InversionConfig(), observed_log_data=True).run(
            start[:-1], initial_log_model=True
        )
    with pytest.raises(ValueError, match="positive"):
        ERTInversion(modeling, -np.ones_like(data), InversionConfig()).setup()
    with pytest.raises(ValueError, match="only defined for time-lapse"):
        ERTInversion(
            modeling,
            data,
            InversionConfig(data_misfit="log_data_difference_l2"),
            observed_log_data=True,
        ).run(start, initial_log_model=True)
    assert torch.cuda.is_available()
