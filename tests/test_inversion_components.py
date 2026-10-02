"""CPU-only checks of the inversion building blocks."""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from adtlert.inversion import (
    InversionConfig,
    ParameterizedERTForward2p5D,
    available_data_misfits,
    available_optimization_algorithms,
    available_spatial_regularizations,
    available_temporal_regularizations,
    build_data_misfit,
    build_optimization_algorithm,
    build_petrophysical_transform,
    build_spatial_regularization,
    build_temporal_regularization,
)
from adtlert.inversion.core import _check_config, _window_start_indices
from adtlert.inversion.optimizers import (
    first_order_step,
    linearized_step,
    solve_linearized,
)
from adtlert.inversion.regularization import (
    first_order_constraint_matrix,
    structure_guided_constraint_matrix,
)
from adtlert.mesh import Mesh
from tests.test_mesh_fem import grid_triangles

RNG = np.random.default_rng(0)


# -- petrophysics ---------------------------------------------------------------------------------


def transforms(n):
    saturation = {
        "rho_sat": np.full(n, 20.0),
        "n": np.full(n, 2.0),
        "rho_sat_s": np.full(n, 200.0),
    }
    archie = {
        "rho0": np.full(n, 100.0),
        "theta0": np.full(n, 0.25),
        "n": np.full(n, 2.0),
    }
    return {
        "log": build_petrophysical_transform("log_resistivity", n_cells=n),
        "log_lu": build_petrophysical_transform(
            "log_resistivity",
            n_cells=n,
            model_transform="log_lu",
            model_bounds=(5.0, 500.0),
        ),
        "conductivity": build_petrophysical_transform("log_conductivity", n_cells=n),
        "saturation": build_petrophysical_transform(
            "saturation", n_cells=n, parameters=saturation
        ),
        "archie": build_petrophysical_transform(
            "relative_archie_water_content", n_cells=n, parameters=archie
        ),
    }


@pytest.mark.parametrize(
    "name", ["log", "log_lu", "conductivity", "saturation", "archie"]
)
def test_petrophysical_transforms_round_trip_and_derivatives(name):
    n = 7
    transform = transforms(n)[name]
    log_rho = np.log(RNG.uniform(30.0, 150.0, n))
    state = transform.state_from_log_resistivity(log_rho)
    assert np.allclose(transform.log_resistivity_from_state(state), log_rho, atol=1e-8)
    h = 1e-6
    for derivative, function in (
        (transform.d_log_resistivity_d_state, transform.log_resistivity_from_state),
        (transform.d_parameter_d_state, transform.parameter_from_state),
    ):
        numeric = (function(state + h) - function(state - h)) / (2 * h)
        assert np.allclose(derivative(state), numeric, rtol=1e-5, atol=1e-8)
    # time-lapse states broadcast cell-wise parameters over the time axis
    states = np.column_stack([state, state])
    assert np.allclose(
        transform.log_resistivity_from_state(states)[:, 1], log_rho, atol=1e-8
    )


def test_petrophysical_transform_validation():
    with pytest.raises(ValueError, match="requires petrophysical parameters"):
        build_petrophysical_transform("saturation", n_cells=3)
    with pytest.raises(ValueError, match="unknown petrophysical_transform"):
        build_petrophysical_transform("nonsense", n_cells=3)
    with pytest.raises(ValueError, match="model_transform='log' only"):
        build_petrophysical_transform(
            "log_conductivity", n_cells=3, model_transform="log_lu"
        )


# -- data misfits ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["weighted_log_l2", "weighted_log_l1", "weighted_log_huber"]
)
def test_misfit_cotangent_matches_gradient_of_phi(name):
    misfit = build_data_misfit(name)
    observed, weights = RNG.normal(size=12), RNG.uniform(1.0, 3.0, 12)
    predicted = observed + RNG.normal(scale=0.5, size=12)
    # The IRLS cotangent is the gradient of phi/2 for L2 and a frozen-weight surrogate otherwise;
    # for L2 it must match exactly.
    if name == "weighted_log_l2":
        h = 1e-6
        gradient = np.array(
            [
                (
                    misfit.phi(predicted + h * e, observed, weights)
                    - misfit.phi(predicted - h * e, observed, weights)
                )
                / (4 * h)
                for e in np.eye(12)
            ]
        )
        assert np.allclose(
            misfit.linearized_cotangent(predicted, observed, weights),
            gradient,
            rtol=1e-5,
        )
    jacobian = RNG.normal(size=(12, 5))
    matrix, rhs = misfit.linearized_system(predicted, observed, weights, jacobian)
    assert matrix.shape == (12, 5) and rhs.shape == (12,)
    assert misfit.chi2(observed, observed, weights) == 0.0


def test_misfit_registry():
    assert set(available_data_misfits()) == {
        "weighted_log_l2",
        "weighted_log_l1",
        "weighted_log_huber",
        "log_data_difference_l2",
    }
    assert build_data_misfit("L2") is build_data_misfit("weighted_log_l2")
    with pytest.raises(ValueError, match="unknown data_misfit"):
        build_data_misfit("nope")


# -- regularization -------------------------------------------------------------------------------


def test_first_order_operator_annihilates_constants_and_weights_vertical_jumps():
    mesh = Mesh.from_arrays(*grid_triangles())
    matrix = first_order_constraint_matrix(mesh)
    assert np.allclose(matrix @ np.ones(mesh.cell_count), 0.0)
    assert matrix.shape[1] == mesh.cell_count and np.all(np.abs(matrix).sum(axis=1) > 0)
    scaled = first_order_constraint_matrix(mesh, z_weight=5.0)
    assert scaled.nnz == matrix.nnz and np.abs(scaled).max() > np.abs(matrix).max()


def test_structure_guided_operator_weakens_cross_unit_edges():
    mesh = Mesh.from_arrays(*grid_triangles())
    centers = np.asarray(mesh.nodes)[np.asarray(mesh.cells)].mean(axis=1)
    labels = (centers[:, 1] < -2).astype(int)
    guided = structure_guided_constraint_matrix(
        mesh, labels, cross_structure_weight=0.05
    )
    plain = first_order_constraint_matrix(mesh)
    assert guided.shape == plain.shape
    assert np.abs(guided).sum() < np.abs(plain).sum()


def test_temporal_operators():
    n_cells, n_times = 3, 5
    time = np.arange(n_times, dtype=float)
    linear = np.kron(
        time, np.ones(n_cells)
    )  # time-major state vector: value = t for every cell
    first = build_temporal_regularization("temporal_smoothness").matrix(
        n_cells, n_times
    )
    second = build_temporal_regularization("second_order").matrix(n_cells, n_times)
    baseline = build_temporal_regularization("baseline_reference").matrix(
        n_cells, n_times
    )
    assert first.shape == (n_cells * (n_times - 1), n_cells * n_times)
    assert np.allclose(first @ linear, 1.0) and np.allclose(second @ linear, 0.0)
    assert np.allclose(baseline @ linear, np.repeat(time[1:], n_cells))
    matrix, rhs = build_temporal_regularization(
        "active_time_constraint"
    ).linearized_system(linear, n_cells, n_times, scale=2.0)
    assert matrix.shape == first.shape and np.all(rhs <= 0)
    # Row weights are minimum + (1 - minimum) / (1 + (|change| / threshold)^2): 1 at no change,
    # about the minimum for huge changes, and weighted by 1/5 at twice the threshold.
    atc = build_temporal_regularization("active_time_constraint")
    changes = np.array([0.0, 2 * atc.threshold, 1e6])
    state = np.concatenate([[0.0], np.cumsum(changes)])  # one cell, four times
    weighted, _ = atc.linearized_system(state, 1, 4, scale=1.0)
    expected = atc.minimum_weight + (1 - atc.minimum_weight) / (
        1 + (changes / atc.threshold) ** 2
    )
    assert np.allclose(abs(weighted).max(axis=1).toarray().ravel(), expected)


def test_robust_regularization_downweights_large_changes():
    n_cells, n_times = 2, 4
    state = np.concatenate(
        [
            np.zeros(n_cells),
            np.zeros(n_cells),
            np.full(n_cells, 5.0),
            np.full(n_cells, 5.0),
        ]
    )
    for name in ("temporal_total_variation", "temporal_huber"):
        matrix, _ = build_temporal_regularization(name).linearized_system(
            state, n_cells, n_times
        )
        weights = np.abs(matrix).sum(axis=1).A1
        assert (
            weights[n_cells : 2 * n_cells].max() < weights[:n_cells].min()
        )  # the jump gets a smaller weight


def test_regularization_registries_resolve_aliases():
    assert build_spatial_regularization("damping").name == "identity"
    assert build_spatial_regularization("tv").name == "first_order_tv"
    assert (
        len(available_spatial_regularizations()) == 6
        and len(available_temporal_regularizations()) == 6
    )
    with pytest.raises(ValueError, match="unknown"):
        build_spatial_regularization("nope")


# -- optimizers -----------------------------------------------------------------------------------


def least_squares_problem(n=30, m=60):
    matrix = sp.csr_matrix(RNG.normal(size=(m, n)))
    truth = RNG.normal(size=n)
    return matrix, matrix @ truth, truth


@pytest.mark.parametrize("solver", ["lsqr", "normal_cg", "pyhydro_cgls"])
def test_cpu_linear_solvers_solve_least_squares(solver):
    matrix, rhs, truth = least_squares_problem()
    config = InversionConfig(
        linearized_solver=solver, cgls_tolerance=1e-14, lsqr_atol=1e-12, lsqr_btol=1e-12
    )
    assert np.allclose(solve_linearized(matrix, rhs, config), truth, atol=1e-5)


@pytest.mark.parametrize("algorithm", ["lbfgs", "nonlinear_cg", "adam"])
def test_first_order_optimizers_descend_a_quadratic(algorithm):
    matrix, rhs, _ = least_squares_problem(10, 20)
    config = InversionConfig(
        optimization_algorithm=algorithm,
        max_log_step=None,
        optimizer_max_step=0.05,
        line_search=False,
    )
    x, state = np.zeros(10), {}
    objective = lambda v: float(np.sum((matrix @ v - rhs) ** 2))  # noqa: E731
    start = objective(x)
    for _ in range(25):
        x = x + linearized_step(matrix, rhs - matrix @ x, x, state, config)
    assert objective(x) < 0.5 * start


def test_levenberg_marquardt_damping_shrinks_the_step():
    matrix, rhs, _ = least_squares_problem(10, 20)
    plain = linearized_step(
        matrix, rhs, np.zeros(10), {}, InversionConfig(max_log_step=None)
    )
    damped = linearized_step(
        matrix,
        rhs,
        np.zeros(10),
        {},
        InversionConfig(
            optimization_algorithm="levenberg_marquardt",
            lm_damping=50.0,
            max_log_step=None,
        ),
    )
    assert np.linalg.norm(damped) < np.linalg.norm(plain)


def test_first_order_step_validates_input():
    with pytest.raises(ValueError, match="non-finite"):
        first_order_step(
            np.zeros(3),
            np.array([1.0, np.nan, 0.0]),
            {},
            InversionConfig(optimization_algorithm="adam"),
        )
    assert np.all(
        first_order_step(
            np.zeros(3), np.zeros(3), {}, InversionConfig(optimization_algorithm="adam")
        )
        == 0
    )
    assert (
        build_optimization_algorithm("adam").gradient_based
        and "gauss_newton_cgls" in available_optimization_algorithms()
    )


# -- configuration and windows --------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"max_iterations": 0}, "max_iterations"),
        ({"regularization_mode": "x"}, "regularization_mode"),
        ({"sensor_constraint": 1.0}, "sensor_constraint_operator"),
        ({"model_transform": "log_lu"}, "model_bounds"),
        ({"petrophysical_transform": "nope"}, "unknown petrophysical_transform"),
        ({"data_misfit": "nope"}, "unknown data_misfit"),
        ({"saturation_floor": 1.5}, "saturation_floor"),
    ],
)
def test_inversion_config_validation(overrides, message):
    with pytest.raises(ValueError, match=message):
        _check_config(InversionConfig(**overrides))
    _check_config(InversionConfig())


def test_window_start_indices_cover_the_tail():
    assert _window_start_indices(10, 3, 4) == [0, 4, 7]
    assert _window_start_indices(5, 5, 1) == [0]
    with pytest.raises(ValueError):
        _window_start_indices(2, 3, 1)


# -- parameter mesh -------------------------------------------------------------------------------


def stub_forward(mesh):
    return SimpleNamespace(
        mesh=mesh, survey=SimpleNamespace(measurement_count=1), close=lambda: None
    )


@pytest.mark.parametrize("mode", ["pygimli_prolongation", "nearest", "fixed_mean"])
def test_parameterized_forward_extends_the_background(mode):
    mesh = Mesh.from_arrays(*grid_triangles(8, 5))
    centers = np.asarray(mesh.nodes)[np.asarray(mesh.cells)].mean(axis=1)
    active = np.flatnonzero((centers[:, 0] > 2) & (centers[:, 0] < 6))
    forward = ParameterizedERTForward2p5D(
        stub_forward(mesh), active, background_mode=mode
    )
    assert forward.cell_count == active.size
    parameters = np.log(RNG.uniform(10.0, 100.0, active.size))
    full = forward._full_log_model(parameters)
    assert np.allclose(full[active], parameters)
    assert np.all(np.isfinite(full)) and full.shape == (mesh.cell_count,)
    if (
        mode != "fixed_mean"
    ):  # background values are convex combinations of parameter values
        assert (
            full.min() >= parameters.min() - 1e-9
            and full.max() <= parameters.max() + 1e-9
        )
    with pytest.raises(ValueError):
        ParameterizedERTForward2p5D(stub_forward(mesh), [0, 0])
    with pytest.raises(ValueError):
        ParameterizedERTForward2p5D(stub_forward(mesh), active, background_mode="nope")
