#!/usr/bin/env python3
"""Fast native ADTLERT reproduction of ``2_timelapsedERT_inversion.ipynb``."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("ADTLERT_ENABLE_FLOAT64", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from example.shared import grid2d_to_mesh_cells, load_petrophysical_parameters, resolve, write_json
from adtlert.inversion import (
    InversionConfig,
    ParameterizedERTForward2p5D,
    TimeLapseERTInversion,
    WindowedTimeLapseERTInversion,
    available_data_misfits,
    available_linearized_optimizers,
    available_optimization_algorithms,
    available_petrophysical_transforms,
    available_spatial_regularizations,
    available_temporal_regularizations,
)
from adtlert.utils.progress import InversionProgressPrinter
from adtlert.workflows import (
    TerrainForwardData,
    build_source_position_triangle_inversion_case,
    load_terrain_forward_dat,
)

# Switch Torch to float64 after adtlert fixed FLOAT_DTYPE at import, as before.
torch.set_default_dtype(torch.float64)


def _discover_forward_dat(
    input_dir: Path,
    *,
    file_stride: int,
    max_timesteps: int | None,
    selected_steps: list[int] | None = None,
) -> list[tuple[int, Path]]:
    if file_stride < 1:
        raise ValueError("file_stride must be >= 1")
    if max_timesteps is not None and max_timesteps < 2:
        raise ValueError("max_timesteps must be >= 2 when set")

    pattern = re.compile(r"synthetic_ert_terrain_vardz_t(\d+)\.dat$")
    pairs: list[tuple[int, Path]] = []
    for path in input_dir.glob("synthetic_ert_terrain_vardz_t*.dat"):
        match = pattern.search(path.name)
        if match is not None:
            pairs.append((int(match.group(1)), path))

    pairs.sort(key=lambda item: item[0])
    if selected_steps:
        by_step = dict(pairs)
        missing = [step for step in selected_steps if step not in by_step]
        if missing:
            raise ValueError(f"Requested forward steps are unavailable: {missing}")
        pairs = [(step, by_step[step]) for step in selected_steps]
    else:
        pairs = pairs[::file_stride]
        if max_timesteps is not None:
            pairs = pairs[:max_timesteps]
    if len(pairs) < 2:
        raise ValueError(f"Need at least 2 forward .dat files in {input_dir}")
    return pairs


def _load_geometry(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        required = {"x_nodes", "z_top", "layer_thickness"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{path} missing required arrays: {sorted(missing)}")
        return {name: np.asarray(data[name], dtype=float).ravel() for name in required}


def _structural_prior_cell_ids(path: Path, mesh, geometry: dict[str, np.ndarray]) -> np.ndarray:
    """Map a terrain-following 2D structural-unit grid onto inversion mesh cells."""

    class_grid = np.asarray(np.load(path), dtype=np.int32)
    return np.asarray(grid2d_to_mesh_cells(class_grid, mesh, geometry), dtype=np.int32)


def _load_terrain_forward_npz(path: Path) -> TerrainForwardData:
    with np.load(path) as data:
        required = {"rhoa", "a", "b", "m", "n", "elec_x", "elec_z"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{path} missing required arrays: {sorted(missing)}")

        rhoa = np.asarray(data["rhoa"], dtype=float).ravel()
        elec_x = np.asarray(data["elec_x"], dtype=float).ravel()
        elec_z = np.asarray(data["elec_z"], dtype=float).ravel()
        measurements = np.column_stack(
            (
                np.asarray(data["a"], dtype=np.int32).ravel(),
                np.asarray(data["b"], dtype=np.int32).ravel(),
                np.asarray(data["m"], dtype=np.int32).ravel(),
                np.asarray(data["n"], dtype=np.int32).ravel(),
            )
        )
        err = np.asarray(data["err"], dtype=float).ravel() if "err" in data.files else None
        if measurements.shape != (rhoa.size, 4):
            raise ValueError(f"{path}: measurement arrays must have one entry per rhoa value")
        if elec_x.shape != elec_z.shape:
            raise ValueError(f"{path}: elec_x and elec_z must have the same shape")
        if np.any(measurements < 0) or np.any(measurements >= elec_x.size):
            raise ValueError(f"{path}: measurements reference electrodes outside elec_x/elec_z")
        if err is not None and err.shape != rhoa.shape:
            raise ValueError(f"{path}: err must have the same shape as rhoa")
        return TerrainForwardData(
            rhoa=rhoa,
            measurements=measurements,
            elec_x=elec_x,
            elec_z=elec_z,
            err=err,
        )


def _load_forward_file(path: Path, forward_format: str) -> tuple[TerrainForwardData, Path, str]:
    if forward_format not in ("auto", "npz", "dat"):
        raise ValueError("forward_format must be 'auto', 'npz', or 'dat'")

    npz_path = path.with_suffix(".npz")
    if forward_format in ("auto", "npz") and npz_path.exists():
        return _load_terrain_forward_npz(npz_path), npz_path, "npz"
    if forward_format == "npz":
        raise FileNotFoundError(f"{npz_path} does not exist")
    return load_terrain_forward_dat(path), path, "dat"


def _load_forward_series(
    pairs: list[tuple[int, Path]],
    *,
    forward_format: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    list[Path],
    dict[str, int],
]:
    rhoa_rows: list[np.ndarray] = []
    err_rows: list[np.ndarray] = []
    measurements: np.ndarray | None = None
    elec_x: np.ndarray | None = None
    elec_z: np.ndarray | None = None
    have_err = True
    loaded_paths: list[Path] = []
    format_counts: dict[str, int] = {}

    for _, path in pairs:
        data, loaded_path, data_format = _load_forward_file(path, forward_format)
        loaded_paths.append(loaded_path)
        format_counts[data_format] = format_counts.get(data_format, 0) + 1
        if measurements is None:
            measurements = data.measurements
            elec_x = data.elec_x
            elec_z = data.elec_z
        else:
            if not np.array_equal(data.measurements, measurements):
                raise ValueError(f"{path}: measurement layout differs from first timestep")
            if not np.allclose(data.elec_x, elec_x, rtol=0.0, atol=1.0e-10):
                raise ValueError(f"{path}: electrode x positions differ from first timestep")
            if not np.allclose(data.elec_z, elec_z, rtol=0.0, atol=1.0e-10):
                raise ValueError(f"{path}: electrode z positions differ from first timestep")

        rhoa_rows.append(np.asarray(data.rhoa, dtype=float).ravel())
        if data.err is None:
            have_err = False
        else:
            err_rows.append(np.asarray(data.err, dtype=float).ravel())

    if measurements is None or elec_x is None or elec_z is None:
        raise ValueError("no forward data loaded")

    err = np.vstack(err_rows) if have_err and len(err_rows) == len(rhoa_rows) else None
    return np.vstack(rhoa_rows), measurements, elec_x, elec_z, err, loaded_paths, format_counts


def _save_mesh_npz(
    path: Path,
    case,
    *,
    forward_mesh=None,
    parameter_cell_ids: np.ndarray | None = None,
    forward_cell_parameter_ids: np.ndarray | None = None,
    forward_parent_cell_ids: np.ndarray | None = None,
    structural_prior_cell_ids: np.ndarray | None = None,
) -> None:
    actual_forward_mesh = case.forward_mesh if forward_mesh is None else forward_mesh
    actual_parameter_cell_ids = (
        np.asarray(case.parameter_cell_ids, dtype=np.int32)
        if parameter_cell_ids is None
        else np.asarray(parameter_cell_ids, dtype=np.int32)
    )
    payload = {
        "nodes": np.asarray(case.mesh.nodes, dtype=float),
        "cells": np.asarray(case.mesh.cells, dtype=np.int32),
        "forward_nodes": np.asarray(actual_forward_mesh.nodes, dtype=float),
        "forward_cells": np.asarray(actual_forward_mesh.cells, dtype=np.int32),
        "base_forward_nodes": np.asarray(case.forward_mesh.nodes, dtype=float),
        "base_forward_cells": np.asarray(case.forward_mesh.cells, dtype=np.int32),
        "cell_markers": np.asarray(case.cell_markers, dtype=np.int32),
        "parameter_cell_ids": actual_parameter_cell_ids,
        "base_parameter_cell_ids": np.asarray(case.parameter_cell_ids, dtype=np.int32),
    }
    if forward_cell_parameter_ids is not None:
        payload["forward_cell_parameter_ids"] = np.asarray(forward_cell_parameter_ids, dtype=np.int32)
    if forward_parent_cell_ids is not None:
        payload["forward_parent_cell_ids"] = np.asarray(forward_parent_cell_ids, dtype=np.int32)
    if structural_prior_cell_ids is not None:
        payload["structural_prior_cell_ids"] = np.asarray(structural_prior_cell_ids, dtype=np.int32)
    surface_node_ids = getattr(case.mesh, "surface_node_ids", None)
    if surface_node_ids is not None:
        payload["surface_node_ids"] = np.asarray(surface_node_ids, dtype=np.int32)
    np.savez(path, **payload)


def _build_forward_parameterization(
    case,
    refinement: str,
) -> tuple[object, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if refinement == "native":
        return case.forward_mesh, np.asarray(case.parameter_cell_ids, dtype=np.int32), None, None
    if refinement != "h2":
        raise ValueError("forward refinement must be 'h2' or 'native'")

    refined_mesh, parent_cell_ids = case.forward_mesh.refine_uniform()
    parent_cell_ids = np.asarray(parent_cell_ids, dtype=np.int32).ravel()
    parent_parameter_ids = np.full(case.forward_mesh.cell_count, -1, dtype=np.int32)
    parent_parameter_ids[np.asarray(case.parameter_cell_ids, dtype=np.int32)] = np.arange(
        case.mesh.cell_count,
        dtype=np.int32,
    )
    forward_cell_parameter_ids = parent_parameter_ids[parent_cell_ids]

    parameter_cell_ids = np.full(case.mesh.cell_count, -1, dtype=np.int32)
    for cell_id in np.flatnonzero(forward_cell_parameter_ids >= 0):
        parameter_id = int(forward_cell_parameter_ids[cell_id])
        if parameter_cell_ids[parameter_id] < 0:
            parameter_cell_ids[parameter_id] = int(cell_id)
    if np.any(parameter_cell_ids < 0):
        raise ValueError("H2 forward mesh parameterization left some inversion cells unmapped")

    return refined_mesh, parameter_cell_ids, forward_cell_parameter_ids, parent_cell_ids


def _positive_log_limits(*arrays: np.ndarray) -> tuple[float, float]:
    values = np.concatenate(
        [
            np.asarray(array, dtype=float).ravel()
            for array in arrays
            if np.asarray(array, dtype=float).size
        ]
    )
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        raise ValueError("plot values must contain at least one finite positive value")
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmin == vmax:
        vmax = vmin * 1.01
    return vmin, vmax


def _plot_chi2(path: Path, chi2: np.ndarray, *, windowed: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chi2_data = np.asarray(chi2, dtype=float).ravel()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, chi2_data.size + 1), chi2_data, marker="o")
    ax.set_xlabel("Window Index" if windowed else "Iteration")
    ax.set_ylabel("Chi2 (data term)")
    ax.set_yscale("log")
    ax.set_title("Time-Lapse Inversion Convergence")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_true_vs_inverted(
    path: Path,
    *,
    mesh,
    final_models: np.ndarray,
    steps: np.ndarray,
    true_model_dir: Path,
    y_index: int,
    geometry: dict[str, np.ndarray],
    coverage_mask: np.ndarray | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LogNorm

    mid_idx = int(steps.size // 2)
    mid_step = int(steps[mid_idx])
    true_file = true_model_dir / f"resistivity2d_y{y_index}_t{mid_step:05d}.npy"
    if not true_file.exists():
        return

    rho_true = np.asarray(np.load(true_file), dtype=float)
    rho_true_plot = rho_true[::-1, :]
    inv_model = np.asarray(final_models[:, mid_idx], dtype=float)
    if coverage_mask is not None and coverage_mask.shape == inv_model.shape:
        inv_values = inv_model.copy()
        inv_values[coverage_mask] = np.nan
        inv_valid = inv_model[~coverage_mask] if np.any(~coverage_mask) else inv_model
    else:
        inv_values = inv_model
        inv_valid = inv_model

    x_nodes = geometry["x_nodes"]
    z_top = geometry["z_top"]
    layer_thickness = geometry["layer_thickness"]
    cum = np.concatenate(([0.0], np.cumsum(layer_thickness)))
    x_grid = np.tile(x_nodes, (cum.size, 1))
    z_grid = z_top[None, :] - cum[:, None]

    nodes = np.asarray(mesh.nodes, dtype=float)
    cells = np.asarray(mesh.cells, dtype=np.int32)
    visible = np.isfinite(inv_values)
    if not np.any(visible):
        visible = np.ones_like(inv_values, dtype=bool)

    vmin, vmax = _positive_log_limits(rho_true_plot, inv_valid)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = "turbo"

    fig = plt.figure(figsize=(10.0, 8.6), constrained_layout=True)
    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1.0, 0.045],
        height_ratios=[1.0, 1.0],
        wspace=0.06,
        hspace=0.08,
    )
    ax_true = fig.add_subplot(grid[0, 0])
    ax_inv = fig.add_subplot(grid[1, 0], sharex=ax_true, sharey=ax_true)
    cax = fig.add_subplot(grid[:, 1])

    ax_true.pcolormesh(x_grid, z_grid, rho_true_plot, shading="auto", cmap=cmap, norm=norm)
    ax_true.set_title(f"True Model (t{mid_step:05d})")

    collection = PolyCollection(
        nodes[cells][visible],
        array=inv_values[visible],
        cmap=cmap,
        norm=norm,
        edgecolors=(1.0, 1.0, 1.0, 0.25),
        linewidths=0.25,
    )
    ax_inv.add_collection(collection)
    ax_inv.set_title(f"Masked Inverted Model (t{mid_step:05d})")

    y_bottom = float(np.min(z_top - np.sum(layer_thickness)))
    y_top = float(np.max(z_top))
    x_min = float(np.min(x_nodes))
    x_max = float(np.max(x_nodes))
    box_aspect = abs((y_top - y_bottom) / (x_max - x_min)) if x_max > x_min else 1.0
    for ax in (ax_true, ax_inv):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_bottom, y_top)
        ax.set_box_aspect(box_aspect)
        ax.set_ylabel("Elevation (m)")
    ax_true.tick_params(labelbottom=False)
    ax_inv.set_xlabel("X (m)")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(sm, cax=cax)
    colorbar.set_label("Resistivity (ohm-m)")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None, help="Repository root. Auto-detected by default.")
    parser.add_argument("--forward-dir", default="result/1_timelapsedERT_forward_adtlert")
    parser.add_argument("--true-model-dir", default="resistivity_models_2d")
    parser.add_argument("--output-dir", default="result/2_timelapsedERT_inversion_adtlert")
    parser.add_argument("--y-index", type=int, default=2)
    parser.add_argument("--file-stride", type=int, default=1)
    parser.add_argument("--max-timesteps", type=int, default=None)
    parser.add_argument(
        "--steps",
        default=None,
        help="Optional comma-separated explicit timesteps; overrides stride selection.",
    )
    parser.add_argument("--forward-format", choices=("auto", "npz", "dat"), default="auto")
    parser.add_argument("--inversion-mode", choices=("windowed", "full"), default="windowed")
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--window-step", type=int, default=1)
    parser.add_argument("--data-misfit", choices=available_data_misfits(), default="weighted_log_l2")
    parser.add_argument("--regularization", type=float, default=50.0)
    parser.add_argument("--temporal-regularization", type=float, default=10.0)
    parser.add_argument(
        "--temporal-regularization-type",
        choices=available_temporal_regularizations(),
        default="temporal_smoothness",
    )
    parser.add_argument("--regularization-mode", choices=("model", "update"), default="model")
    parser.add_argument(
        "--regularization-domain",
        choices=("state", "physical"),
        default="state",
        help="Apply spatial/temporal regularization in optimizer state domain or mapped physical domain.",
    )
    parser.add_argument(
        "--physical-regularization-quantity",
        choices=("parameter", "theta", "water_content"),
        default="parameter",
        help="When --regularization-domain=physical: regularize transform parameter (e.g., S) or theta.",
    )
    parser.add_argument(
        "--spatial-regularization",
        choices=available_spatial_regularizations(),
        default="first_order_smoothness",
    )
    parser.add_argument(
        "--petrophysical-transform",
        choices=available_petrophysical_transforms(),
        default="log_resistivity",
        help="Inversion parameterization: log-resistivity, log-conductivity, or saturation.",
    )
    parser.add_argument(
        "--petrophysical-parameter-dir",
        default=None,
        help=(
            "Directory containing rho_sat2d/n2d/rho_sat_s2d/phi2d files for saturation inversion. "
            "Defaults to parflow_models/petrophysical_models_2d."
        ),
    )
    parser.add_argument(
        "--petrophysical-preset",
        default="table1_mean",
        help="File suffix preset used for petrophysical parameter grids, e.g. table1_mean.",
    )
    parser.add_argument("--saturation-floor", type=float, default=1.0e-4)
    parser.add_argument(
        "--structural-prior-file",
        default=None,
        help=(
            "Optional class2d .npy file for structurally constrained/structural-prior regularization. "
            "Defaults to parflow_models/petrophysical_models_2d/class2d_y<y-index>.npy when present."
        ),
    )
    parser.add_argument(
        "--structural-cross-weight",
        type=float,
        default=0.05,
        help="Relative smoothness weight across structural-unit boundaries for structural-prior regularization.",
    )
    parser.add_argument(
        "--active-time-threshold",
        type=float,
        default=0.05,
        help="Log-resistivity change threshold used by active-time-constraint regularization.",
    )
    parser.add_argument(
        "--active-time-minimum-weight",
        type=float,
        default=0.05,
        help="Lower bound for adaptive temporal weights in active-time-constraint regularization.",
    )
    parser.add_argument(
        "--optimizer",
        choices=available_optimization_algorithms(),
        default="gauss_newton_cgls",
        help="Outer nonlinear optimizer. Gauss-Newton keeps the previous ADTLERT default behavior.",
    )
    parser.add_argument(
        "--linearized-solver",
        choices=available_linearized_optimizers(),
        default="gpu_cgls",
        help="Backend used by Gauss-Newton/Levenberg-Marquardt linearized updates.",
    )
    parser.add_argument("--lm-damping", type=float, default=1.0e-2)
    parser.add_argument(
        "--optimizer-max-step",
        type=float,
        default=1.0,
        help="Max log-model increment used by first-order optimizers when --max-log-step is not set.",
    )
    parser.add_argument("--lbfgs-history", type=int, default=10)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1.0e-8)
    parser.add_argument("--cgls-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--cgls-max-iterations", type=int, default=60)
    parser.add_argument(
        "--line-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable line search for nonlinear updates.",
    )
    parser.add_argument(
        "--target-chi2",
        type=float,
        default=None,
        help="Optional chi2 stop criterion. Iteration stops once chi2 < target-chi2.",
    )
    parser.add_argument(
        "--step-tolerance",
        type=float,
        default=1.0e-4,
        help="Stop criterion on normalized step length. Use 0.0 to effectively disable this stop.",
    )
    parser.add_argument("--relative-error", type=float, default=0.05)
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--model-min", type=float, default=0.001)
    parser.add_argument("--model-max", type=float, default=1.0e4)
    parser.add_argument("--max-log-step", type=float, default=None)
    parser.add_argument("--coverage-percentile", type=float, default=20.0)
    parser.add_argument("--inversion-mesh-quality", type=float, default=34.0)
    parser.add_argument("--inversion-mesh-smoothing-iterations", type=int, default=10)
    parser.add_argument(
        "--mesh-file",
        default=None,
        help=(
            "Optional mesh npz bootstrap file. If set, bypasses Triangle meshing and "
            "uses saved nodes/cells arrays (see build_source_position_triangle_inversion_case)."
        ),
    )
    parser.add_argument("--forward-refinement", choices=("native", "h2"), default="native")
    parser.add_argument("--linear-solver-backend", default="auto")
    parser.add_argument("--normal-field-cache-max-entries", type=int, default=8)
    parser.add_argument("--terrain-cache-dir", default=None)
    parser.add_argument("--quiet", action="store_true", help="Disable progress output on stderr.")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[2]
    forward_dir = resolve(root, args.forward_dir)
    true_model_dir = resolve(root, args.true_model_dir)
    output_dir = resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = _discover_forward_dat(
        forward_dir,
        file_stride=args.file_stride,
        max_timesteps=args.max_timesteps,
        selected_steps=(
            None
            if not args.steps
            else [int(item) for item in args.steps.split(",") if item.strip()]
        ),
    )
    steps = np.asarray([step for step, _ in pairs], dtype=np.int32)
    measurement_times = np.asarray(steps, dtype=float) / 24.0

    geometry = _load_geometry(forward_dir / "forward_geometry.npz")
    observed_rhoa, measurements, elec_x, elec_z, err, data_files, forward_format_counts = _load_forward_series(
        pairs,
        forward_format=args.forward_format,
    )
    if err is not None:
        data_std = np.log1p(err)
        data_std_source = "forward_err_log1p"
    else:
        data_std = float(np.log1p(args.relative_error))
        data_std_source = "relative_error_log1p"

    mesh_file_path = None if args.mesh_file is None else resolve(root, args.mesh_file)
    case_data_file = pairs[0][1] if mesh_file_path is None else None
    elec_z_case = np.asarray(elec_z, dtype=float)

    case = build_source_position_triangle_inversion_case(
        elec_x,
        elec_z_case,
        measurements,
        geometry["x_nodes"],
        geometry["z_top"],
        geometry["layer_thickness"],
        y_index=args.y_index,
        quality=args.inversion_mesh_quality,
        smoothing_iterations=args.inversion_mesh_smoothing_iterations,
        data_file=case_data_file,
        mesh_file=mesh_file_path,
    )
    if observed_rhoa.shape[1] != case.survey.measurement_count:
        raise ValueError(
            f"observed data has {observed_rhoa.shape[1]} measurements, "
            f"mesh survey has {case.survey.measurement_count}"
        )

    forward_mesh, parameter_cell_ids, forward_cell_parameter_ids, forward_parent_cell_ids = (
        _build_forward_parameterization(case, args.forward_refinement)
    )
    forward = ParameterizedERTForward2p5D.from_mesh_survey(
        forward_mesh,
        case.survey,
        parameter_cell_ids,
        regularization_mesh=case.mesh,
        forward_cell_parameter_ids=forward_cell_parameter_ids,
        linear_solver_backend=args.linear_solver_backend,
        normal_field_cache_max_entries=args.normal_field_cache_max_entries,
        terrain_cache_dir=None if args.terrain_cache_dir is None else resolve(root, args.terrain_cache_dir),
    )
    default_structural_prior = (
        true_model_dir.parent / "parflow_models" / "petrophysical_models_2d" / f"class2d_y{args.y_index}.npy"
    )
    structural_prior_file = (
        resolve(root, args.structural_prior_file)
        if args.structural_prior_file is not None
        else default_structural_prior
    )
    structural_prior_cell_ids = None
    if structural_prior_file.exists():
        structural_prior_cell_ids = _structural_prior_cell_ids(structural_prior_file, case.mesh, geometry)
        forward.structural_prior_cell_ids = structural_prior_cell_ids
        forward.structural_cross_weight = float(args.structural_cross_weight)
    elif str(args.spatial_regularization).replace("-", "_") in {
        "structural_prior",
        "structure_guided",
        "structure_guided_first_order",
        "structurally_constrained",
    }:
        raise FileNotFoundError(
            f"Structural-prior regularization requested but structural prior file does not exist: {structural_prior_file}"
        )

    petrophysical_parameter_dir = (
        resolve(root, args.petrophysical_parameter_dir)
        if args.petrophysical_parameter_dir is not None
        else true_model_dir.parent / "parflow_models" / "petrophysical_models_2d"
    )
    petrophysical_parameters = None
    if str(args.petrophysical_transform).replace("-", "_") == "saturation":
        petrophysical_parameters = load_petrophysical_parameters(
            petrophysical_parameter_dir,
            y_index=args.y_index,
            preset=args.petrophysical_preset,
            mesh=case.mesh,
            geometry=geometry,
        )

    progress = InversionProgressPrinter(enabled=not args.quiet)
    config = InversionConfig(
        max_iterations=args.max_iterations,
        data_std=data_std,
        data_misfit=args.data_misfit,
        regularization=args.regularization,
        regularization_mode=args.regularization_mode,
        regularization_domain=args.regularization_domain,
        physical_regularization_quantity=args.physical_regularization_quantity,
        temporal_regularization=args.temporal_regularization,
        temporal_regularization_type=args.temporal_regularization_type,
        optimization_algorithm=args.optimizer,
        linearized_solver=args.linearized_solver,
        lm_damping=args.lm_damping,
        optimizer_max_step=args.optimizer_max_step,
        lbfgs_history=args.lbfgs_history,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        cgls_tolerance=args.cgls_tolerance,
        cgls_max_iterations=args.cgls_max_iterations,
        target_chi2=args.target_chi2,
        step_tolerance=args.step_tolerance,
        spatial_regularization=args.spatial_regularization,
        petrophysical_transform=args.petrophysical_transform,
        petrophysical_parameters=petrophysical_parameters,
        saturation_floor=args.saturation_floor,
        active_time_threshold=args.active_time_threshold,
        active_time_minimum_weight=args.active_time_minimum_weight,
        z_weight=1.0,
        model_bounds=(args.model_min, args.model_max),
        max_log_step=args.max_log_step,
        line_search=bool(args.line_search),
        progress_callback=progress,
    )

    initial_rhoa = np.median(observed_rhoa, axis=1)
    initial_model = np.tile(initial_rhoa[None, :], (case.mesh.cell_count, 1))
    run_start = time.perf_counter()
    try:
        if args.inversion_mode == "full":
            result = TimeLapseERTInversion(
                forward=forward,
                observed_data=observed_rhoa,
                config=config,
            ).setup().run(initial_model)
            run_meta = {
                "inversion_mode": "full",
                "n_windows": 1,
                "window_size": None,
                "window_step": None,
            }
        else:
            result = WindowedTimeLapseERTInversion(
                forward=forward,
                observed_data=observed_rhoa,
                config=config,
                window_size=args.window_size,
                window_step=args.window_step,
            ).setup().run(initial_model)
            run_meta = {
                "inversion_mode": "windowed",
                "n_windows": len(result.window_reports),
                "window_size": int(args.window_size),
                "window_step": int(args.window_step),
            }
    finally:
        forward.close()
        progress.finish()
    elapsed_sec = time.perf_counter() - run_start

    final_models = np.asarray(result.final_models, dtype=float)
    chi2_all = np.asarray(result.all_chi2, dtype=float)
    coverage = np.asarray(result.coverage, dtype=float).ravel()
    coverage_threshold = float(np.percentile(coverage, args.coverage_percentile))
    coverage_mask = coverage < coverage_threshold

    np.save(output_dir / "final_models.npy", final_models)
    np.save(output_dir / "final_log_models.npy", np.asarray(result.final_log_models, dtype=float))
    np.save(output_dir / "predicted_rhoa.npy", np.asarray(result.predicted_data, dtype=float))
    final_parameter_models = None if result.final_parameter_models is None else np.asarray(result.final_parameter_models, dtype=float)
    if final_parameter_models is not None and result.final_parameter_name != "resistivity":
        np.save(output_dir / f"final_{result.final_parameter_name}_models.npy", final_parameter_models)
        if result.final_parameter_name == "saturation" and petrophysical_parameters is not None and "phi" in petrophysical_parameters:
            water_content = final_parameter_models * np.asarray(petrophysical_parameters["phi"], dtype=float)[:, None]
            np.save(output_dir / "final_water_content_models.npy", water_content)
    np.save(output_dir / "steps.npy", steps)
    np.save(output_dir / "measurement_times_days.npy", measurement_times)
    np.save(output_dir / "chi2_all.npy", chi2_all)
    np.save(output_dir / "coverage.npy", coverage)
    np.save(output_dir / "coverage_mask.npy", coverage_mask.astype(np.uint8))
    _save_mesh_npz(
        output_dir / "timelapse_inversion_mesh.npz",
        case,
        forward_mesh=forward_mesh,
        parameter_cell_ids=parameter_cell_ids,
        forward_cell_parameter_ids=forward_cell_parameter_ids,
        forward_parent_cell_ids=forward_parent_cell_ids,
        structural_prior_cell_ids=structural_prior_cell_ids,
    )

    for column, step in enumerate(steps):
        model = final_models[:, column]
        np.save(output_dir / f"inverted_model_t{int(step):05d}.npy", model)
        masked = model.copy()
        masked[coverage_mask] = np.nan
        np.save(output_dir / f"inverted_model_masked_nan_t{int(step):05d}.npy", masked)
        if final_parameter_models is not None and result.final_parameter_name != "resistivity":
            np.save(output_dir / f"inverted_{result.final_parameter_name}_t{int(step):05d}.npy", final_parameter_models[:, column])

    with (output_dir / "used_data_files.txt").open("w", encoding="utf-8") as stream:
        for path in data_files:
            stream.write(str(path) + "\n")

    if result.window_reports:
        write_json(output_dir / "window_reports.json", result.window_reports)

    plot_files: dict[str, str] = {}
    if not args.no_plot:
        chi2_plot = output_dir / "timelapse_chi2.png"
        _plot_chi2(chi2_plot, chi2_all, windowed=args.inversion_mode == "windowed")
        plot_files["chi2"] = str(chi2_plot)

        mid_plot = output_dir / f"true_vs_inverted_mid_t{int(steps[len(steps) // 2]):05d}.png"
        _plot_true_vs_inverted(
            mid_plot,
            mesh=case.mesh,
            final_models=final_models,
            steps=steps,
            true_model_dir=true_model_dir,
            y_index=args.y_index,
            geometry=geometry,
            coverage_mask=coverage_mask,
        )
        if mid_plot.exists():
            plot_files["true_vs_inverted_mid"] = str(mid_plot)

    summary = {
        "n_timesteps": int(steps.size),
        "first_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "n_cells": int(final_models.shape[0]),
        "mesh_nodes": int(case.mesh.node_count),
        "base_forward_mesh_cells": int(case.forward_mesh.cell_count),
        "base_forward_mesh_nodes": int(case.forward_mesh.node_count),
        "forward_mesh_cells": int(forward_mesh.cell_count),
        "forward_mesh_nodes": int(forward_mesh.node_count),
        "forward_refinement": str(args.forward_refinement),
        "forward_format": str(args.forward_format),
        "forward_format_counts": {key: int(value) for key, value in sorted(forward_format_counts.items())},
        "lambda_val": float(args.regularization),
        "alpha": float(args.temporal_regularization),
        "regularization_domain": str(args.regularization_domain),
        "physical_regularization_quantity": str(args.physical_regularization_quantity),
        "data_misfit": str(args.data_misfit),
        "spatial_regularization": str(args.spatial_regularization),
        "temporal_regularization_type": str(args.temporal_regularization_type),
        "petrophysical_transform": str(args.petrophysical_transform),
        "petrophysical_parameter": str(result.final_parameter_name),
        "petrophysical_parameter_dir": str(petrophysical_parameter_dir),
        "petrophysical_preset": str(args.petrophysical_preset),
        "saturation_floor": float(args.saturation_floor),
        "saved_parameter_models": bool(final_parameter_models is not None and result.final_parameter_name != "resistivity"),
        "saved_water_content_models": bool(
            final_parameter_models is not None
            and result.final_parameter_name == "saturation"
            and petrophysical_parameters is not None
            and "phi" in petrophysical_parameters
        ),
        "structural_prior_file": str(structural_prior_file) if structural_prior_cell_ids is not None else None,
        "structural_cross_weight": float(args.structural_cross_weight),
        "active_time_threshold": float(args.active_time_threshold),
        "active_time_minimum_weight": float(args.active_time_minimum_weight),
        "lambda_rate": 1.0,
        "lambda_min": 1.0,
        "max_iterations": int(args.max_iterations),
        "model_constraints": [float(args.model_min), float(args.model_max)],
        "relative_error": float(args.relative_error),
        "data_std_source": data_std_source,
        "initial_model_source": "per_timestep_median_rhoa",
        "method": str(args.optimizer),
        "optimizer": str(args.optimizer),
        "linearized_solver": str(args.linearized_solver),
        "forward_linear_solver_backend": str(args.linear_solver_backend),
        "normal_field_cache_max_entries": int(args.normal_field_cache_max_entries),
        "lm_damping": float(args.lm_damping),
        "optimizer_max_step": float(args.optimizer_max_step),
        "lbfgs_history": int(args.lbfgs_history),
        "adam_beta1": float(args.adam_beta1),
        "adam_beta2": float(args.adam_beta2),
        "adam_epsilon": float(args.adam_epsilon),
        "cgls_tolerance": float(args.cgls_tolerance),
        "cgls_max_iterations": int(args.cgls_max_iterations),
        "line_search": bool(args.line_search),
        "target_chi2": None if args.target_chi2 is None else float(args.target_chi2),
        "step_tolerance": float(args.step_tolerance),
        "regularization_mode": str(args.regularization_mode),
        "coverage_percentile": float(args.coverage_percentile),
        "coverage_source": (
            "adtlert_pygimli_style_sumabs_area"
            if str(args.optimizer).replace("-", "_") in {"gauss_newton_cgls", "levenberg_marquardt"}
            else "adtlert_normal_vjp_residual_weighted_area"
        ),
        "sensitivity_mode": (
            "explicit_full_jacobian"
            if str(args.optimizer).replace("-", "_") in {"gauss_newton_cgls", "levenberg_marquardt"}
            else "normal_vjp_matrix_free"
        ),
        "coverage_threshold": coverage_threshold,
        "output_dir": str(output_dir),
        "forward_dir": str(forward_dir),
        "mesh_file": str(output_dir / "timelapse_inversion_mesh.npz"),
        "input_mesh_file": None if args.mesh_file is None else str(resolve(root, args.mesh_file)),
        "inversion_mesh_quality": float(args.inversion_mesh_quality),
        "inversion_mesh_smoothing_iterations": int(args.inversion_mesh_smoothing_iterations),
        "torch_enable_float64": torch.get_default_dtype() == torch.float64,
        "elapsed_sec": float(elapsed_sec),
        "elapsed_min": float(elapsed_sec / 60.0),
        "plot_files": plot_files,
    }
    summary.update(run_meta)
    write_json(output_dir / "timelapsed_inversion_summary.json", summary)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
