import io
import json

import numpy as np
import pytest
import torch

import adtlert
from adtlert.survey import Survey
from adtlert.utils.progress import InversionProgressPrinter
from adtlert.workflows import (
    ParflowGrid,
    TerrainForwardRunner,
    build_source_position_triangle_inversion_case,
    build_terrain_forward_case,
    build_wenner_alpha_measurements,
    discover_resistivity_slices,
    load_terrain_forward_dat,
    load_terrain_resistivity_slice,
    parse_pftcl,
    parse_resistivity_slice_name,
    run_terrain_forward_file,
    run_terrain_forward_series,
    save_terrain_forward_dat,
    save_terrain_forward_npz,
)
from tests.conftest import quad_case

PFTCL = """
pfset ComputationalGrid.NX "6"
pfset ComputationalGrid.NY "2"
pfset ComputationalGrid.NZ "3"
pfset ComputationalGrid.DX "5.0"
pfset ComputationalGrid.DY "0.2"
pfset ComputationalGrid.DZ "10.0"
pfset Cell.0.dzScale.Value "2.0"
pfset Cell.1.dzScale.Value "0.5"
pfset Cell.2.dzScale.Value "0.1"
"""


def test_public_api_is_importable():
    for name in adtlert.__all__:
        assert hasattr(adtlert, name), name
    assert adtlert.__version__


def test_survey_validation_and_geometric_factors():
    positions = np.column_stack([np.arange(6.0), np.zeros(6)])
    survey = Survey.from_arrays(positions, [[0, 3, 1, 2]])
    # Wenner-alpha: k = 2 pi a
    assert float(survey.geometric_factors()[0]) == pytest.approx(2 * np.pi * 1.0)
    assert float(
        survey.apparent_resistivity(torch.tensor([0.5]), 2.0)[0]
    ) == pytest.approx(np.pi * 1.0 * 0.5 * 2 / 2)
    assert (
        survey.electrode_count == 6
        and survey.measurement_count == 1
        and survey.dimension == 2
    )
    with pytest.raises(ValueError, match="distinct"):
        Survey.from_arrays(positions, [[0, 1, 1, 2]])
    with pytest.raises(ValueError, match="outside"):
        Survey.from_arrays(positions, [[0, 1, 2, 9]])
    with pytest.raises(ValueError, match="shape"):
        Survey.from_arrays(positions, [[0, 1, 2]])


def test_wenner_alpha_measurements():
    rows = build_wenner_alpha_measurements(10)
    assert rows.shape[1] == 4 and len(rows) == sum(10 - 3 * a for a in range(1, 4))
    a, b, m, n = rows.T
    assert np.all(m - a == n - m) and np.all(b - n == n - m) and rows.max() < 10
    with pytest.raises(ValueError):
        build_wenner_alpha_measurements(3)


def test_parflow_parsing(tmp_path):
    path = tmp_path / "model.pftcl"
    path.write_text(PFTCL)
    grid = parse_pftcl(path)
    assert (grid.nx, grid.ny, grid.nz, grid.dx) == (
        6,
        2,
        3,
        5.0,
    ) and grid.dz_scales.tolist() == [2.0, 0.5, 0.1]
    path.write_text(PFTCL.replace('pfset Cell.2.dzScale.Value "0.1"', ""))
    with pytest.raises(ValueError, match="dzScale count"):
        parse_pftcl(path)
    path.write_text("nothing here")
    with pytest.raises(ValueError, match="missing"):
        parse_pftcl(path)


def test_slice_discovery_and_loading(tmp_path):
    grid = ParflowGrid(5.0, 0.2, 10.0, nx=4, ny=2, nz=3, dz_scales=np.ones(3))
    for step in (0, 24, 48, 72):
        np.save(
            tmp_path / f"resistivity2d_y1_t{step:05d}.npy",
            np.full((3, 4), 100.0 + step),
        )
    np.save(tmp_path / "resistivity2d_y0_t00000.npy", np.ones((3, 4)))
    assert parse_resistivity_slice_name("a/resistivity2d_y2_t00048.npy") == (2, 48)
    assert parse_resistivity_slice_name("resistivity_t00012.npy") == (-1, 12)
    with pytest.raises(ValueError):
        parse_resistivity_slice_name("other.npy")
    found = discover_resistivity_slices(tmp_path, y_index=1, file_stride=2, max_steps=2)
    assert [step for step, _ in found] == [0, 48]
    np.save(tmp_path / "volume.npy", np.arange(3 * 2 * 4.0).reshape(3, 2, 4))
    assert load_terrain_resistivity_slice(
        tmp_path / "volume.npy", grid, y_index=1
    ).shape == (3, 4)
    with pytest.raises(ValueError, match="out of range"):
        load_terrain_resistivity_slice(tmp_path / "volume.npy", grid, y_index=5)


def test_terrain_case_geometry_and_validation():
    case = quad_case(0.03, nx=10, nz=4, n_electrodes=6)
    assert (
        case.mesh.cell_count == 40
        and case.resistivity.shape == (40,)
        and case.mesh.is_quadrilateral_mesh
    )
    assert not case.mesh.is_flat_surface and np.allclose(
        case.elec_z, np.interp(case.elec_x, case.x_nodes, case.z_top)
    )
    assert case.with_resistivity(np.full(40, 7.0)).resistivity[0] == 7.0
    with pytest.raises(ValueError, match="positive"):
        case.with_resistivity(np.zeros(40))
    with pytest.raises(ValueError, match="shape"):
        case.with_resistivity(np.ones(3))
    grid = ParflowGrid(2.0, 1.0, 1.0, 10, 1, 4, np.ones(4))
    with pytest.raises(ValueError, match="does not match"):
        build_terrain_forward_case(np.ones((3, 10)), grid, np.zeros(10), y_index=0)
    with pytest.raises(ValueError, match="y_index"):
        build_terrain_forward_case(np.ones((4, 10)), grid, np.zeros(10), y_index=3)


def test_dat_and_npz_round_trip(tmp_path):
    case = quad_case(0.0, nx=10, nz=4, n_electrodes=8)
    rhoa = np.linspace(50.0, 150.0, case.survey.measurement_count)
    save_terrain_forward_dat(tmp_path / "a.dat", case, rhoa, relative_error=0.05)
    data = load_terrain_forward_dat(tmp_path / "a.dat")
    assert np.allclose(data.rhoa, rhoa) and np.array_equal(
        data.measurements, case.survey.measurements.numpy()
    )
    assert np.allclose(data.elec_x, case.elec_x) and np.allclose(data.err, 0.05)
    save_terrain_forward_npz(tmp_path / "a.npz", case, rhoa)
    with np.load(tmp_path / "a.npz") as saved:
        assert set(saved.files) >= {
            "rhoa",
            "a",
            "b",
            "m",
            "n",
            "elec_x",
            "x_nodes",
            "z_top",
            "layer_thickness",
        }
    with pytest.raises(ValueError, match="rhoa must have shape"):
        save_terrain_forward_dat(tmp_path / "b.dat", case, rhoa[:-1])
    (tmp_path / "bad.dat").write_text("1\n# x y z\n0 0 0\n")
    with pytest.raises(ValueError):
        load_terrain_forward_dat(tmp_path / "bad.dat")


@pytest.mark.gpu
def test_runner_and_series_forward(tmp_path):
    case = quad_case(0.0, nx=12, nz=5, n_electrodes=8)
    runner = TerrainForwardRunner.from_case(case, prepare_forward=True)
    first = runner.solve_case(case)
    assert first.shape == (case.survey.measurement_count,) and np.all(first > 0)
    assert np.allclose(
        runner.solve_resistivity(case.resistivity * 2.0), 2.0 * first, rtol=1e-5
    )
    runner.close()

    grid = ParflowGrid(
        2.0, 1.0, 1.0, nx=12, ny=1, nz=5, dz_scales=np.linspace(2.0, 0.5, 5)
    )
    paths = []
    for step in (0, 24):
        paths.append(tmp_path / f"resistivity2d_y0_t{step:05d}.npy")
        np.save(paths[-1], np.full((5, 12), 100.0 * (1 + step / 24)))
    manifest, failures = run_terrain_forward_series(
        paths, grid, np.zeros(12), tmp_path / "out", y_index=0, n_electrodes=8
    )
    assert [record.status for record in manifest] == ["ok", "ok"] and not failures
    assert manifest[1].rhoa_min == pytest.approx(2 * manifest[0].rhoa_min, rel=1e-4)
    again, _ = run_terrain_forward_series(
        paths,
        grid,
        np.zeros(12),
        tmp_path / "out",
        y_index=0,
        n_electrodes=8,
        overwrite=False,
    )
    assert [record.status for record in again] == ["skipped_existing"] * 2
    broken, failed = run_terrain_forward_series(
        [tmp_path / "missing_t00001.npy"],
        grid,
        np.zeros(12),
        tmp_path / "out",
        y_index=0,
    )
    assert not broken and failed[0].status == "failed" and failed[0].step == -1
    record = run_terrain_forward_file(
        paths[0], grid, np.zeros(12), tmp_path / "single", n_electrodes=8
    )
    assert (
        record.status == "ok"
        and (tmp_path / "single" / "synthetic_ert_terrain_vardz_t00000.dat").exists()
    )
    with pytest.raises(ValueError, match="y-index"):
        run_terrain_forward_file(
            paths[0], grid, np.zeros(12), tmp_path / "x", y_index=1
        )


def test_source_position_triangle_case(tmp_path):
    pytest.importorskip("triangle")
    x = np.linspace(0.0, 30.0, 16)
    case = build_source_position_triangle_inversion_case(
        x, np.zeros_like(x), build_wenner_alpha_measurements(16), np.linspace(-10, 40, 11), np.zeros(11), np.ones(5), y_index=0, quality=30.0
    )  # fmt: skip
    assert (
        case.mesh.cell_count
        == len(case.parameter_cell_ids)
        < case.forward_mesh.cell_count
    )
    assert set(np.unique(case.cell_markers)) == {1, 2} and np.all(
        case.cell_markers[case.parameter_cell_ids] == 2
    )
    finer = build_source_position_triangle_inversion_case(
        x, np.zeros_like(x), build_wenner_alpha_measurements(16), np.linspace(-10, 40, 11), np.zeros(11), np.ones(5), y_index=0, quality=30.0, parameter_max_cell_area=2.0
    )  # fmt: skip
    assert finer.mesh.cell_count > case.mesh.cell_count
    with pytest.raises(ValueError, match="strictly increasing"):
        build_source_position_triangle_inversion_case(
            x[::-1],
            np.zeros_like(x),
            build_wenner_alpha_measurements(16),
            np.linspace(-10, 40, 11),
            np.zeros(11),
            np.ones(5),
            y_index=0,
        )
    with pytest.raises(ValueError, match="positive"):
        build_source_position_triangle_inversion_case(
            x,
            np.zeros_like(x),
            build_wenner_alpha_measurements(16),
            np.linspace(-10, 40, 11),
            np.zeros(11),
            np.ones(5),
            y_index=0,
            parameter_max_cell_area=-1.0,
        )


def test_progress_printer_renders_events():
    stream = io.StringIO()
    printer = InversionProgressPrinter(stream=stream)
    events = [
        {"event": "single_start", "n_cells": 5, "n_data": 7, "max_iterations": 3},
        {
            "event": "single_iteration_done",
            "iteration": 1,
            "max_iterations": 3,
            "chi2": 2.5,
            "step_norm": 0.1,
        },
        {
            "event": "single_done",
            "iterations": 1,
            "max_iterations": 3,
            "final_chi2": 2.5,
            "stop_reason": "max_iterations",
        },
        {
            "event": "windowed_start",
            "n_windows": 2,
            "window_size": 3,
            "window_step": 1,
            "max_iterations": 4,
        },
        {
            "event": "window_start",
            "window_index": 1,
            "n_windows": 2,
            "start_idx": 0,
            "end_idx": 2,
        },
        {
            "event": "window_done",
            "window_index": 1,
            "n_windows": 2,
            "start_idx": 0,
            "end_idx": 2,
            "final_chi2": 1.2,
        },
        {"event": "windowed_done", "n_windows": 2, "final_chi2": 1.2},
    ]
    for event in events:
        printer(event)
    printer.finish()
    text = stream.getvalue()
    assert (
        "Single inversion: 5 cells, 7 data" in text
        and "final chi2=2.5" in text
        and "Windowed inversion: 2 windows" in text
    )
    silent = io.StringIO()
    InversionProgressPrinter(enabled=False, stream=silent)(events[0])
    assert silent.getvalue() == "" and json.dumps(events[0])
