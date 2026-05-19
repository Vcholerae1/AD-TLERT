#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("DEEPERT_ENABLE_FLOAT64", "1")

from deepert.utils.torch_runtime import torch_runtime
import numpy as np

torch_runtime.config.update("torch_enable_float64", True)

from deepert.inversion import ERTInversion, InversionConfig, ParameterizedERTForward2p5D
from deepert.utils.progress import InversionProgressPrinter
from deepert.workflows import (
    build_source_position_triangle_inversion_case,
    load_terrain_forward_dat,
    parse_resistivity_slice_name,
)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _load_forward_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        required = {"rhoa", "a", "b", "m", "n", "elec_x", "elec_z", "x_nodes", "z_top", "layer_thickness"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{path} missing required arrays: {sorted(missing)}")
        loaded = {
            "rhoa": np.asarray(data["rhoa"], dtype=float).ravel(),
            "measurements": np.column_stack(
                (
                    np.asarray(data["a"], dtype=np.int32).ravel(),
                    np.asarray(data["b"], dtype=np.int32).ravel(),
                    np.asarray(data["m"], dtype=np.int32).ravel(),
                    np.asarray(data["n"], dtype=np.int32).ravel(),
                )
            ),
            "elec_x": np.asarray(data["elec_x"], dtype=float).ravel(),
            "elec_z": np.asarray(data["elec_z"], dtype=float).ravel(),
            "x_nodes": np.asarray(data["x_nodes"], dtype=float).ravel(),
            "z_top": np.asarray(data["z_top"], dtype=float).ravel(),
            "layer_thickness": np.asarray(data["layer_thickness"], dtype=float).ravel(),
        }
        if "err" in data.files:
            loaded["err"] = np.asarray(data["err"], dtype=float).ravel()
        return loaded


def _load_forward_dat(path: Path) -> dict[str, np.ndarray]:
    data = load_terrain_forward_dat(path)
    loaded = {
        "rhoa": data.rhoa,
        "measurements": data.measurements,
        "elec_x": data.elec_x,
        "elec_z": data.elec_z,
    }
    if data.err is not None:
        loaded["err"] = data.err
    return loaded


def _resolve_data_std(args: argparse.Namespace, forward_data: dict[str, np.ndarray]) -> tuple[float | np.ndarray, str]:
    if args.data_std is not None:
        return float(args.data_std), "cli"
    err = forward_data.get("err")
    if err is not None:
        return np.log1p(np.asarray(err, dtype=float).ravel()), "observed_err_log1p"
    return 0.05, "default_0.05"


def _data_std_summary(data_std: float | np.ndarray, shape: tuple[int, ...]) -> dict[str, float]:
    values = np.asarray(data_std, dtype=float)
    if values.ndim == 0:
        values = np.full(shape, float(values), dtype=float)
    else:
        values = np.broadcast_to(values, shape).astype(float, copy=False)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _save_mesh_npz(path: Path, case) -> None:
    np.savez(
        path,
        nodes=np.asarray(case.mesh.nodes, dtype=float),
        cells=np.asarray(case.mesh.cells, dtype=np.int32),
        surface_node_ids=np.asarray(case.mesh.surface_node_ids, dtype=np.int32),
        forward_nodes=np.asarray(case.forward_mesh.nodes, dtype=float),
        forward_cells=np.asarray(case.forward_mesh.cells, dtype=np.int32),
        cell_markers=np.asarray(case.cell_markers, dtype=np.int32),
        parameter_cell_ids=np.asarray(case.parameter_cell_ids, dtype=np.int32),
    )


def _plot_comparison(
    path: Path,
    case,
    true_rho_2d: np.ndarray,
    final_model: np.ndarray,
    coverage: np.ndarray,
    step: int,
    percentile: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm

    nx = len(case.x_nodes) - 1
    nz = len(case.layer_thickness)
    true_top = np.asarray(true_rho_2d, dtype=float)[::-1, :]
    inverted = np.asarray(final_model, dtype=float).ravel()
    coverage_array = np.asarray(coverage, dtype=float).ravel()
    coverage_mask = coverage_array < np.percentile(coverage_array, percentile)

    cum = np.concatenate(([0.0], np.cumsum(case.layer_thickness)))
    x_grid = np.tile(case.x_nodes, (cum.size, 1))
    z_grid = -(case.z_top[None, :] - cum[:, None])
    valid_inv = inverted[~coverage_mask] if np.any(~coverage_mask) else inverted
    vmin = float(max(1.0e-6, min(np.nanmin(true_top), np.nanmin(valid_inv))))
    vmax = float(max(np.nanmax(true_top), np.nanmax(valid_inv)))
    norm = LogNorm(vmin=vmin, vmax=vmax)

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

    ax_true.pcolormesh(x_grid, z_grid, true_top, shading="auto", cmap="turbo", norm=norm)
    ax_true.plot(case.x_nodes, -case.z_top, color="black", linewidth=1.5)
    ax_true.set_title(f"True Model (t{step:05d})")

    nodes = np.asarray(case.mesh.nodes, dtype=float)
    cells = np.asarray(case.mesh.cells, dtype=np.int32)
    plot_nodes = nodes.copy()
    plot_nodes[:, 1] *= -1.0
    visible = ~coverage_mask
    if not np.any(visible):
        visible = np.ones_like(coverage_mask, dtype=bool)
    inverted_mesh = PolyCollection(
        plot_nodes[cells][visible],
        array=inverted[visible],
        cmap="turbo",
        norm=norm,
        edgecolors=(1.0, 1.0, 1.0, 0.25),
        linewidths=0.25,
    )
    ax_inv.add_collection(inverted_mesh)
    ax_inv.plot(case.elec_x, -case.elec_z, color="black", linewidth=1.5)
    ax_inv.set_title(f"Masked Inverted Model (t{step:05d})")

    x_min = float(np.min(case.x_nodes))
    x_max = float(np.max(case.x_nodes))
    y_min = float(np.min(-case.z_top))
    y_max = float(np.max(-(case.z_top - np.sum(case.layer_thickness))))
    box_aspect = abs((y_max - y_min) / (x_max - x_min)) if x_max > x_min else 1.0
    for ax in (ax_true, ax_inv):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)
        ax.set_box_aspect(box_aspect)
        ax.set_ylabel("Elevation (m)")
    ax_true.tick_params(labelbottom=False)
    ax_inv.set_xlabel("X (m)")

    sm = ScalarMappable(norm=norm, cmap="turbo")
    sm.set_array([])
    colorbar = fig.colorbar(sm, cax=cax)
    colorbar.set_label("Resistivity (ohm-m)")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None, help="Repository root. Defaults to this script directory.")
    parser.add_argument(
        "--forward-npz",
        default="2d_resistivity_model/synthetic_ert_terrain_vardz_t04368.npz",
        help="Forward metadata .npz produced by 1_forward_t04368_deepert.py; used for true-model plot geometry.",
    )
    parser.add_argument(
        "--forward-dat",
        default="2d_resistivity_model/synthetic_ert_terrain_vardz_t04368.dat",
        help="Notebook .dat file used for observed data and source-position mesh generation.",
    )
    parser.add_argument("--true-model", default="2d_resistivity_model/resistivity2d_y2_t04368.npy")
    parser.add_argument("--model-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", default="result/deepert_single_inversion_t04368")
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--data-std", type=float, default=None)
    parser.add_argument("--regularization", type=float, default=50.0)
    parser.add_argument("--regularization-mode", choices=("model", "update"), default="update")
    parser.add_argument("--z-weight", type=float, default=1.0)
    parser.add_argument("--model-min", type=float, default=10.0)
    parser.add_argument("--model-max", type=float, default=20000.0)
    parser.add_argument("--inversion-depth-levels", type=int, default=11, help=argparse.SUPPRESS)
    parser.add_argument("--inversion-mesh-quality", type=float, default=34.0)
    parser.add_argument("--inversion-mesh-smoothing-iterations", type=int, default=10)
    parser.add_argument(
        "--inversion-mesh-reference",
        default=None,
        help="Optional saved mesh .npz to reuse as a paraDomain cache instead of regenerating with Triangle.",
    )
    parser.add_argument(
        "--no-mesh-cache",
        action="store_true",
        help="Regenerate the inversion mesh even if output_dir already contains inversion_mesh_t*.npz.",
    )
    parser.add_argument("--target-chi2", type=float, default=1.5)
    parser.add_argument("--max-log-step", type=float, default=None)
    parser.add_argument("--coverage-percentile", type=float, default=20.0)
    parser.add_argument("--linear-solver-backend", default="auto")
    parser.add_argument("--terrain-cache-dir", default=None)
    parser.add_argument("--quiet", action="store_true", help="Disable progress output on stderr.")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parent
    forward_npz = _resolve(root, args.forward_npz)
    forward_dat = _resolve(root, args.forward_dat) if args.forward_dat else None
    true_model_file = _resolve(root, args.true_model)
    mesh_reference = _resolve(root, args.inversion_mesh_reference) if args.inversion_mesh_reference else None
    output_dir = _resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_index, step = parse_resistivity_slice_name(true_model_file)
    mesh_cache = output_dir / f"inversion_mesh_t{step:05d}.npz"
    if mesh_reference is None and not args.no_mesh_cache and mesh_cache.exists():
        mesh_reference = mesh_cache
    forward_meta = _load_forward_npz(forward_npz)
    forward_data = _load_forward_dat(forward_dat) if forward_dat is not None else forward_meta
    forward_data["x_nodes"] = forward_meta["x_nodes"]
    forward_data["z_top"] = forward_meta["z_top"]
    forward_data["layer_thickness"] = forward_meta["layer_thickness"]
    observed_rhoa = forward_data["rhoa"]
    data_std, data_std_source = _resolve_data_std(args, forward_data)
    true_rho_2d = np.asarray(np.load(true_model_file), dtype=float)
    case = build_source_position_triangle_inversion_case(
        forward_data["elec_x"],
        forward_data["elec_z"],
        forward_data["measurements"],
        forward_data["x_nodes"],
        forward_data["z_top"],
        forward_data["layer_thickness"],
        y_index=y_index,
        depth_levels=args.inversion_depth_levels,
        quality=args.inversion_mesh_quality,
        smoothing_iterations=args.inversion_mesh_smoothing_iterations,
        data_file=forward_dat,
        mesh_file=mesh_reference,
    )
    if observed_rhoa.shape != (case.survey.measurement_count,):
        raise ValueError(
            f"Observed rhoa shape {observed_rhoa.shape} does not match survey measurement count "
            f"{case.survey.measurement_count}"
        )

    forward = ParameterizedERTForward2p5D.from_mesh_survey(
        case.forward_mesh,
        case.survey,
        case.parameter_cell_ids,
        regularization_mesh=case.mesh,
        linear_solver_backend=args.linear_solver_backend,
        terrain_cache_dir=None if args.terrain_cache_dir is None else _resolve(root, args.terrain_cache_dir),
    )
    progress = InversionProgressPrinter(enabled=not args.quiet)
    try:
        config = InversionConfig(
            max_iterations=args.max_iterations,
            data_std=data_std,
            regularization=args.regularization,
            regularization_mode=args.regularization_mode,
            spatial_regularization="first_order",
            z_weight=args.z_weight,
            model_bounds=(args.model_min, args.model_max),
            max_log_step=args.max_log_step,
            line_search=True,
            target_chi2=args.target_chi2,
            progress_callback=progress,
        )
        initial_model = np.full(case.mesh.cell_count, float(np.median(observed_rhoa)), dtype=float)
        result = ERTInversion(forward=forward, observed_data=observed_rhoa, config=config).setup().run(initial_model)
    finally:
        forward.close()
        progress.finish()

    plot_coverage = np.asarray(result.coverage, dtype=float).ravel()
    coverage_source = "deepert_pygimli_style_sumabs_area"

    coverage_threshold = float(np.percentile(plot_coverage, args.coverage_percentile))
    coverage_mask = np.asarray(plot_coverage < coverage_threshold, dtype=np.uint8)
    masked_model = np.asarray(result.final_model, dtype=float).copy()
    masked_model[coverage_mask.astype(bool)] = np.nan

    np.save(output_dir / "final_model.npy", result.final_model)
    np.save(output_dir / "final_log_model.npy", result.final_log_model)
    np.save(output_dir / "predicted_rhoa.npy", result.predicted_data)
    np.save(output_dir / "coverage.npy", plot_coverage)
    np.save(output_dir / "coverage_mask.npy", coverage_mask)
    np.save(output_dir / "final_model_masked_nan.npy", masked_model)
    np.save(output_dir / "chi2_history.npy", np.asarray(result.iteration_chi2, dtype=float))
    mesh_file = output_dir / f"inversion_mesh_t{step:05d}.npz"
    _save_mesh_npz(mesh_file, case)

    summary = {
        "forward_npz": str(forward_npz),
        "forward_dat": None if forward_dat is None else str(forward_dat),
        "true_model": str(true_model_file),
        "output_dir": str(output_dir),
        "step": int(step),
        "y_index": int(y_index),
        "mesh_cells": int(case.mesh.cell_count),
        "mesh_nodes": int(case.mesh.node_count),
        "forward_mesh_cells": int(case.forward_mesh.cell_count),
        "forward_mesh_nodes": int(case.forward_mesh.node_count),
        "inversion_mesh": "cached_para_domain" if mesh_reference is not None else "native_source_position_triangle",
        "inversion_mesh_file": str(mesh_file),
        "inversion_mesh_reference": None if mesh_reference is None else str(mesh_reference),
        "inversion_mesh_quality": float(args.inversion_mesh_quality),
        "inversion_mesh_smoothing_iterations": int(args.inversion_mesh_smoothing_iterations),
        "measurements": int(case.survey.measurement_count),
        "max_iterations": int(args.max_iterations),
        "iterations": len(result.iteration_chi2),
        "final_chi2": float(result.iteration_chi2[-1]) if result.iteration_chi2 else None,
        "regularization": float(args.regularization),
        "regularization_mode": args.regularization_mode,
        "data_std": _data_std_summary(data_std, observed_rhoa.shape),
        "data_std_source": data_std_source,
        "line_search": True,
        "max_log_step": None if args.max_log_step is None else float(args.max_log_step),
        "target_chi2": None if args.target_chi2 is None else float(args.target_chi2),
        "torch_enable_float64": bool(torch_runtime.config.torch_enable_float64),
        "coverage_percentile": float(args.coverage_percentile),
        "coverage_source": coverage_source,
        "coverage_threshold": coverage_threshold,
    }
    _write_json(output_dir / "inversion_summary.json", summary)

    if not args.no_plot:
        _plot_comparison(
            output_dir / f"true_vs_inverted_t{step:05d}.png",
            case,
            true_rho_2d,
            result.final_model,
            plot_coverage,
            step,
            args.coverage_percentile,
        )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
