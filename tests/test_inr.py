"""Coordinate networks, their physics bridge, and the training loop."""

import numpy as np
import pytest
import torch

from adtlert.forward import ERTForward2p5D, ERTForwardModeling
from adtlert.inr import (
    DualNetworkINR,
    INRConfig,
    JointSpatioTemporalINR,
    MultiscaleFourierEncoder,
    MultiscaleINR,
    Optimization,
    Progressive,
    Regularization,
    Windows,
    cell_centers,
    fit_inr,
    fit_timelapse_inr,
    fourier_frequency_bank,
    normalize_coordinates,
    spatiotemporal_coordinates,
)
from adtlert.inr.physics import matrix_free_log_rhoa_series
from adtlert.inr.train import _huber_mean, _misfit, _Schedule
from adtlert.inversion import ParameterizedERTForward2p5D
from tests.conftest import quad_case

# -- networks and coordinates (no GPU needed) ------------------------------------------------------


def test_frequency_banks():
    octave = fourier_frequency_bank(4, "octave")
    assert torch.allclose(octave, (np.pi * 2.0 ** torch.arange(4)).float())
    log_uniform = fourier_frequency_bank(4, "log_uniform")
    assert log_uniform[0] == pytest.approx(np.pi / 2) and log_uniform[
        -1
    ] == pytest.approx(np.pi * 8)
    assert fourier_frequency_bank(0).numel() == 0
    with pytest.raises(ValueError):
        fourier_frequency_bank(2, "nope")


def test_encoder_features_and_progressive_levels():
    encoder = MultiscaleFourierEncoder(3, spatial_levels=4, temporal_levels=3)
    coordinates = torch.rand(7, 3) * 2 - 1
    features = encoder(coordinates)
    assert features.shape == (7, encoder.output_dimensions) == (7, 3 + 16 + 6)
    assert encoder.feature_mask(
        dtype=torch.float32, device=torch.device("cpu")
    ).shape == (features.shape[1],)
    encoder.set_progressive_levels(0.0, spatial_start_levels=2, temporal_start_levels=1)
    assert torch.equal(
        encoder.spatial_level_weights, torch.tensor([1.0, 1.0, 0.0, 0.0])
    )
    encoder.set_progressive_levels(1.0)
    assert torch.equal(encoder.spatial_level_weights, torch.ones(4)) and torch.equal(
        encoder.temporal_level_weights, torch.ones(3)
    )
    with pytest.raises(ValueError):
        encoder(torch.rand(2, 2))


@pytest.mark.parametrize(
    "network, coordinates",
    [
        (MultiscaleINR(), torch.rand(30, 2)),
        (JointSpatioTemporalINR(), torch.rand(4, 30, 3)),
        (DualNetworkINR(), torch.rand(4, 30, 3)),
    ],
)
def test_networks_start_homogeneous_and_stay_inside_their_bounds(network, coordinates):
    network.eval()
    initial = network(coordinates)
    assert torch.allclose(initial, torch.full_like(initial, np.log(100.0)), atol=1e-5)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.add_(torch.randn_like(parameter) * 5.0)
    low, high = network.resistivity_bounds
    values = torch.exp(network(coordinates))
    assert values.min() >= low * 0.999 and values.max() <= high * 1.001


def test_network_checkpoints_round_trip_with_their_configuration():
    network = DualNetworkINR(
        spatial_levels=3, spatial_hidden_width=16, temporal_hidden_width=8
    )
    clone = DualNetworkINR(**network.configuration())
    clone.load_state_dict(network.state_dict())
    coordinates = torch.rand(3, 20, 3)
    with torch.no_grad():
        network.temporal_mlp[-1].bias.fill_(0.3)
        clone.load_state_dict(network.state_dict())
        assert torch.equal(network(coordinates), clone(coordinates))
    with pytest.raises(ValueError, match="strictly contain"):
        MultiscaleINR(initial_resistivity=5.0, resistivity_bounds=(10.0, 100.0))


def test_dual_network_is_rank_one_in_space_and_time():
    network = DualNetworkINR(initial_resistivity=100.0, resistivity_bounds=None)
    with torch.no_grad():
        network.temporal_mlp[-1].bias.fill_(1.0)
    coordinates = torch.cat([torch.rand(1, 25, 3).expand(5, -1, -1).clone()], dim=0)
    coordinates[..., 2] = torch.linspace(-1, 1, 5)[:, None]
    residual = network(coordinates) - np.log(100.0)
    assert torch.linalg.matrix_rank(residual, atol=1e-5) == 1


def test_coordinate_normalization():
    values = np.array([[0.0, 5.0], [10.0, 5.0], [4.0, 5.0]])
    normalized, center, scale = normalize_coordinates(values)
    assert (
        normalized[:, 0].min() == -1
        and normalized[:, 0].max() == 1
        and np.all(normalized[:, 1] == 0)
    )
    assert np.allclose(
        normalized * scale + center, values, atol=1e-5
    )  # the transform is invertible
    combined, *_ = spatiotemporal_coordinates(values, np.array([0.0, 1.0, 2.0, 3.0]))
    assert combined.shape == (4, 3, 3) and np.allclose(
        combined[:, 0, 2], np.linspace(-1, 1, 4)
    )
    with pytest.raises(ValueError):
        normalize_coordinates(np.array([[np.nan, 1.0]]))


def test_cell_centers():
    case = quad_case(0.0, nx=4, nz=3, n_electrodes=4)
    centers = cell_centers(case.mesh)
    assert centers.shape == (case.mesh.cell_count, 2) and centers.dtype == np.float32


# -- training pieces ---------------------------------------------------------------------------------


def test_window_schedule_covers_every_step_and_alternates_direction():
    schedule = _Schedule(
        7, INRConfig(windows=Windows(size=3, step=2)), torch.device("cpu")
    )
    assert [window.tolist() for window in schedule.windows] == [
        [0, 1, 2],
        [2, 3, 4],
        [4, 5, 6],
    ]
    assert schedule.weights.tolist() == pytest.approx([1, 1, 0.5, 1, 0.5, 1, 1])
    assert (
        schedule.window(1).tolist() == [2, 3, 4]
        and schedule.window(3).tolist() == [4, 5, 6]
        and schedule.window(4).tolist() == [2, 3, 4]
    )
    plain = _Schedule(
        7,
        INRConfig(windows=Windows(size=3, step=2, alternate_direction=False)),
        torch.device("cpu"),
    )
    assert plain.window(3).tolist() == [0, 1, 2]
    full = _Schedule(5, INRConfig(), torch.device("cpu"))
    assert not full.windowed and full.batch_size == 5
    with pytest.raises(ValueError, match="cover every timestep"):
        _Schedule(10, INRConfig(windows=Windows(size=2, step=3)), torch.device("cpu"))


def test_misfit_and_huber_helpers():
    observed, std = torch.zeros(2, 4), torch.full((2, 4), 0.5)
    chi2, rms = _misfit(torch.full((2, 4), 0.1), observed, std, torch.ones(2))
    assert float(chi2) == pytest.approx(0.04) and float(rms) == pytest.approx(
        100 * np.expm1(0.1)
    )
    weighted, _ = _misfit(
        torch.tensor([[0.1] * 4, [0.3] * 4]), observed, std, torch.tensor([3.0, 1.0])
    )
    assert float(weighted) == pytest.approx((3 * 0.04 + 1 * 0.36) / 4)
    assert float(_huber_mean(torch.tensor([0.05, -0.05]), 0.1)) == pytest.approx(
        0.5 * 0.0025 / 0.1
    )
    assert float(_huber_mean(torch.tensor([1.0]), 0.1)) == pytest.approx(1.0 - 0.05)
    assert float(_huber_mean(torch.empty(0), 0.1)) == 0.0


@pytest.mark.parametrize(
    "config, message",
    [(INRConfig(max_iterations=0), "max_iterations"), (INRConfig(optimization=Optimization(optimizer="sgd")), "optimizer"),
     (INRConfig(optimization=Optimization(scheduler="x")), "scheduler"),
     (INRConfig(optimization=Optimization(learning_rate_milestones=(5, 5))), "strictly increasing"),
     (INRConfig(optimization=Optimization(minimum_learning_rate=1.0)), "minimum_learning_rate"),
     (INRConfig(windows=Windows(size=0)), "window size"), (INRConfig(regularization=Regularization(huber_delta=0.0)), "huber"),
     (INRConfig(progressive=Progressive(full_iteration=0)), "full_iteration")],
)  # fmt: skip
def test_config_validation(config, message):
    with pytest.raises(ValueError, match=message):
        config.validate()
    INRConfig().validate()


def test_progressive_schedule_unlocks_levels_of_every_encoder():
    network = DualNetworkINR(spatial_levels=4, temporal_levels=3)
    Progressive(
        full_iteration=10, spatial_start_levels=1, temporal_start_levels=1
    ).apply(network, 0)
    assert network.spatial_encoder.spatial_level_weights.tolist() == [1, 0, 0, 0]
    assert network.temporal_encoder.temporal_level_weights.tolist() == [1, 0, 0]
    Progressive(full_iteration=10).apply(network, 10)
    assert network.spatial_encoder.spatial_level_weights.min() == 1
    assert network.temporal_encoder.temporal_level_weights.min() == 1
    Progressive(enabled=False, full_iteration=10).apply(network, 0)
    assert network.temporal_encoder.temporal_level_weights.min() == 1


@pytest.mark.parametrize(
    "network",
    [MultiscaleINR(input_dimensions=3), JointSpatioTemporalINR(), DualNetworkINR()],
)
def test_networks_round_trip_through_their_configuration(network):  # fmt: skip
    rebuilt = type(network).from_configuration(network.configuration())
    assert rebuilt.configuration() == network.configuration()
    rebuilt.load_state_dict(network.state_dict())
    coordinates = torch.rand(3, 5, 3)
    assert torch.equal(rebuilt(coordinates), network(coordinates))


@pytest.fixture(scope="module")
def small_case():
    return quad_case(0.0, nx=20, nz=8, n_electrodes=12)


def identity_forward(case, entries=8):
    forward = ERTForward2p5D.from_mesh_survey(
        case.mesh, case.survey, normal_field_cache_max_entries=entries
    )
    return ParameterizedERTForward2p5D(
        forward, np.arange(case.mesh.cell_count), regularization_mesh=case.mesh
    )


@pytest.mark.gpu
def test_physics_gradient_matches_finite_differences(small_case):
    forward = identity_forward(small_case)
    rng = np.random.default_rng(0)
    model = torch.as_tensor(
        np.log(100.0) + 0.2 * rng.standard_normal((2, small_case.mesh.cell_count)),
        device="cuda",
    ).requires_grad_()
    weights = torch.as_tensor(
        rng.standard_normal((2, small_case.survey.measurement_count)), device="cuda"
    )
    (matrix_free_log_rhoa_series(model, forward) * weights).sum().backward()
    for step, cell in ((0, 3), (1, small_case.mesh.cell_count - 5)):
        h, values = 1e-5, []
        for sign in (1, -1):
            shifted = model.detach().clone()
            shifted[step, cell] += sign * h
            values.append(
                float((matrix_free_log_rhoa_series(shifted, forward) * weights).sum())
            )
        numeric = (values[0] - values[1]) / (2 * h)
        assert float(model.grad[step, cell]) == pytest.approx(numeric, rel=1e-5)


@pytest.mark.gpu
def test_physics_accepts_facade_and_rejects_unknown_forwards(small_case):
    facade = ERTForwardModeling(mesh=small_case.mesh, data=small_case.survey)
    model = torch.full((2, small_case.mesh.cell_count), np.log(100.0), device="cuda")
    assert matrix_free_log_rhoa_series(model, facade).shape == (
        2,
        small_case.survey.measurement_count,
    )
    with pytest.raises(TypeError):
        matrix_free_log_rhoa_series(model, object())


@pytest.mark.gpu
def test_single_survey_fit_recovers_the_anomaly(small_case):
    forward = identity_forward(small_case)
    truth = np.log(small_case.resistivity)
    observed = np.exp(
        ERTForwardModeling(mesh=small_case.mesh, data=small_case.survey).forward(truth)
    )
    coordinates = normalize_coordinates(cell_centers(small_case.mesh))[0]
    events = []
    config = INRConfig(
        max_iterations=40,
        log_every=10,
        target_chi2=None,
        optimization=Optimization(learning_rate=1e-2),
        regularization=Regularization(spatial=0.01),
        progress_callback=events.append,
    )
    torch.manual_seed(0)
    result = fit_inr(
        forward, MultiscaleINR(), coordinates, observed, 0.02, config=config
    )
    assert (
        result.history.chi2[-1] < 0.2 * result.history.chi2[0]
        and result.iterations == 40
    )
    assert result.best_chi2 == pytest.approx(result.history.chi2.min())
    assert (
        result.log_resistivity.shape == (truth.size,)
        and result.predicted_data.shape == observed.shape
    )
    assert result.stop_reason == "max_iterations" and result.gpu_report["gpu_name"]
    assert result.timing.forward_timesteps == 41 and result.timing.vjp_timesteps == 40
    assert [e["iteration"] for e in events if e["event"] == "iteration"] == [
        0,
        10,
        20,
        30,
        40,
    ]
    assert result.timing.forward > 0 and result.timing.backward > 0


@pytest.mark.gpu
def test_time_lapse_fit_windowed_and_full(small_case):
    forward = identity_forward(small_case, entries=4)
    truth = np.log(small_case.resistivity)
    modeling = ERTForwardModeling(mesh=small_case.mesh, data=small_case.survey)
    observed = np.exp(
        np.stack(
            [
                modeling.forward(truth + 0.1 * t * (small_case.resistivity < 50))
                for t in range(4)
            ]
        )
    )
    coordinates, _, _ = spatiotemporal_coordinates(
        cell_centers(small_case.mesh), np.arange(4.0)
    )
    snapshots = []
    common = {
        "max_iterations": 12,
        "target_chi2": None,
        "optimization": Optimization(learning_rate=1e-2),
        "regularization": Regularization(temporal=0.01),
        "log_every": 100,
    }
    torch.manual_seed(0)
    full = fit_timelapse_inr(
        forward,
        DualNetworkINR(),
        coordinates,
        observed,
        0.02,
        config=INRConfig(**common),
    )
    assert full.log_resistivity.shape == (4, truth.size)
    assert (
        full.timing.forward_timesteps == 13 * 4
        and full.history.full_iterations.tolist() == list(range(13))
    )
    torch.manual_seed(0)
    config = INRConfig(
        **common,
        windows=Windows(size=2, step=1, full_evaluation_interval=5),
        snapshot_interval=6,
        progress_callback=lambda e: (
            snapshots.append(e) if e["event"] == "snapshot" else None
        ),
    )
    windowed = fit_timelapse_inr(
        forward, DualNetworkINR(), coordinates, observed, 0.02, config=config
    )
    assert (
        windowed.config.windows.size == 2
        and windowed.history.full_iterations.tolist() == [0, 5, 6, 10, 12]
    )
    assert [s["iteration"] for s in snapshots] == [6, 12] and snapshots[0][
        "log_resistivity"
    ].shape == (4, truth.size)
    assert windowed.timing.vjp_timesteps == 12 * 2
    assert windowed.best_chi2 == pytest.approx(windowed.history.full_chi2.min())


@pytest.mark.gpu
def test_fit_validates_inputs_and_the_field_cache(small_case):
    forward = identity_forward(small_case, entries=2)
    coordinates = normalize_coordinates(cell_centers(small_case.mesh))[0]
    observed = np.full(small_case.survey.measurement_count, 100.0)
    config = INRConfig(max_iterations=1)
    with pytest.raises(ValueError, match="measurements"):
        fit_inr(
            forward, MultiscaleINR(), coordinates, observed[:-1], 0.02, config=config
        )
    with pytest.raises(ValueError, match="positive"):
        fit_inr(forward, MultiscaleINR(), coordinates, -observed, 0.02, config=config)
    with pytest.raises(ValueError, match="broadcast"):
        fit_inr(
            forward, MultiscaleINR(), coordinates, observed, np.ones(3), config=config
        )
    with pytest.raises(ValueError, match="only valid for time-lapse"):
        fit_inr(
            forward,
            MultiscaleINR(),
            coordinates,
            observed,
            0.02,
            config=INRConfig(windows=Windows(size=2)),
        )
    with pytest.raises(ValueError, match="cells"):
        fit_inr(
            forward, MultiscaleINR(), coordinates[:-1], observed, 0.02, config=config
        )
    series_coordinates, _, _ = spatiotemporal_coordinates(
        cell_centers(small_case.mesh), np.arange(4.0)
    )
    with pytest.raises(ValueError, match="normal_field_cache_max_entries"):
        fit_timelapse_inr(
            forward,
            DualNetworkINR(),
            series_coordinates,
            np.tile(observed, (4, 1)),
            0.02,
            config=config,
        )
    with pytest.raises(ValueError, match="input_dimensions=3"):
        fit_timelapse_inr(
            forward, MultiscaleINR(), coordinates, observed, 0.02, config=config
        )
