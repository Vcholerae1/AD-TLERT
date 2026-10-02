#!/usr/bin/env python3
"""Single-time ADTLERT inversion for the processed real hillslope ERT data."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("ADTLERT_ENABLE_FLOAT64", "1")

import numpy as np
import torch
from _real_data_common import (
    apply_data_stride,
    build_parameterized_forward,
    build_real_inversion_case,
    data_std_from_err,
    find_project_root,
    plot_chi2,
    plot_resistivity_model,
    quality_mask,
    read_processed_ert,
    resolve_path,
    save_mesh_npz,
    write_json,
)
from adtlert.inversion import (
    ERTInversion,
    InversionConfig,
    available_data_misfits,
    available_linearized_optimizers,
    available_optimization_algorithms,
    available_spatial_regularizations,
)
from adtlert.utils.progress import InversionProgressPrinter

# Switch Torch to float64 after adtlert fixed FLOAT_DTYPE at import, as before.
torch.set_default_dtype(torch.float64)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--input-file", default="ProcessedData/2022-04-20_1230.txt")
    parser.add_argument("--output-dir", default="result/8_real_data/1_single_inversion_real_adtlert")
    parser.add_argument("--mesh-file", default=None, help="Optional saved inversion_mesh.npz to reuse.")
    parser.add_argument("--depth", type=float, default=90.0, help="Displayed/geometric local depth below highest electrode.")
    parser.add_argument("--n-layers", type=int, default=28, help="Stored terrain-layer metadata for this real profile.")
    parser.add_argument("--layer-stretch", type=float, default=1.08)
    parser.add_argument("--data-stride", type=int, default=1, help="Use every Nth valid datum; keep 1 for production.")
    parser.add_argument("--max-error", type=float, default=None, help="Optional maximum reciprocal/data error filter.")
    parser.add_argument("--relative-error", type=float, default=0.05, help="Fallback data std if err column is absent.")
    parser.add_argument("--minimum-log-std", type=float, default=1.0e-3)
    parser.add_argument("--data-misfit", choices=available_data_misfits(), default="weighted_log_l2")
    parser.add_argument("--regularization", type=float, default=50.0)
    parser.add_argument("--regularization-mode", choices=("model", "update"), default="model")
    parser.add_argument("--spatial-regularization", choices=available_spatial_regularizations(), default="first_order_smoothness")
    parser.add_argument("--z-weight", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=available_optimization_algorithms(), default="gauss_newton_cgls")
    parser.add_argument("--linearized-solver", choices=available_linearized_optimizers(), default="gpu_cgls")
    parser.add_argument("--terrain-cache-dir", default=None)
    parser.add_argument("--lm-damping", type=float, default=1.0e-2)
    parser.add_argument("--cgls-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--cgls-max-iterations", type=int, default=2000)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--model-min", type=float, default=1.0)
    parser.add_argument("--model-max", type=float, default=1.0e5)
    parser.add_argument("--max-log-step", type=float, default=1.0)
    parser.add_argument("--target-chi2", type=float, default=None)
    parser.add_argument("--step-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--line-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inversion-mesh-quality", type=float, default=34.0)
    parser.add_argument("--inversion-mesh-smoothing-iterations", type=int, default=10)
    parser.add_argument("--coverage-percentile", type=float, default=20.0)
    parser.add_argument("--plot-vmin", type=float, default=None)
    parser.add_argument("--plot-vmax", type=float, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).resolve() if args.project_root else find_project_root(Path(__file__))
    input_file = resolve_path(root, args.input_file)
    output_dir = resolve_path(root, args.output_dir)
    mesh_file = None if args.mesh_file is None else resolve_path(root, args.mesh_file)
    terrain_cache_dir = None if args.terrain_cache_dir is None else resolve_path(root, args.terrain_cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_data = read_processed_ert(input_file)
    mask = quality_mask(raw_data, max_error=args.max_error)
    mask = apply_data_stride(mask, args.data_stride)
    data = raw_data.with_measurement_mask(mask)
    if data.rhoa.size < 4:
        raise ValueError("Need at least four valid measurements after filtering.")

    case = build_real_inversion_case(
        data,
        depth=args.depth,
        n_layers=args.n_layers,
        layer_stretch=args.layer_stretch,
        inversion_mesh_quality=args.inversion_mesh_quality,
        inversion_mesh_smoothing_iterations=args.inversion_mesh_smoothing_iterations,
        mesh_file=mesh_file,
    )
    if data.rhoa.shape != (case.survey.measurement_count,):
        raise ValueError(f"Observed rhoa shape {data.rhoa.shape} does not match survey count {case.survey.measurement_count}.")

    data_std = data_std_from_err(
        data.err,
        shape=data.rhoa.shape,
        relative_error=args.relative_error,
        minimum_log_std=args.minimum_log_std,
    )
    forward = build_parameterized_forward(
        case,
        terrain_cache_dir=terrain_cache_dir,
    )
    progress = InversionProgressPrinter(enabled=not args.quiet)
    config = InversionConfig(
        max_iterations=args.max_iterations,
        data_std=data_std,
        data_misfit=args.data_misfit,
        regularization=args.regularization,
        regularization_mode=args.regularization_mode,
        spatial_regularization=args.spatial_regularization,
        z_weight=args.z_weight,
        optimization_algorithm=args.optimizer,
        linearized_solver=args.linearized_solver,
        lm_damping=args.lm_damping,
        cgls_tolerance=args.cgls_tolerance,
        cgls_max_iterations=args.cgls_max_iterations,
        model_bounds=(args.model_min, args.model_max),
        max_log_step=args.max_log_step,
        line_search=bool(args.line_search),
        target_chi2=args.target_chi2,
        step_tolerance=args.step_tolerance,
        progress_callback=progress,
    )

    initial_model = np.full(case.mesh.cell_count, float(np.median(data.rhoa)), dtype=float)
    run_start = time.perf_counter()
    try:
        result = ERTInversion(forward=forward, observed_data=data.rhoa, config=config).setup().run(initial_model)
    finally:
        forward.close()
        progress.finish()
    elapsed_sec = time.perf_counter() - run_start

    coverage = np.asarray(result.coverage, dtype=float).ravel()
    coverage_threshold = float(np.percentile(coverage, args.coverage_percentile))
    coverage_mask = coverage < coverage_threshold
    masked_model = np.asarray(result.final_model, dtype=float).copy()
    masked_model[coverage_mask] = np.nan

    np.save(output_dir / "final_model.npy", np.asarray(result.final_model, dtype=float))
    np.save(output_dir / "final_log_model.npy", np.asarray(result.final_log_model, dtype=float))
    np.save(output_dir / "predicted_rhoa.npy", np.asarray(result.predicted_data, dtype=float))
    np.save(output_dir / "observed_rhoa.npy", np.asarray(data.rhoa, dtype=float))
    np.save(output_dir / "data_std.npy", np.asarray(data_std, dtype=float))
    np.save(output_dir / "coverage.npy", coverage)
    np.save(output_dir / "coverage_mask.npy", coverage_mask.astype(np.uint8))
    np.save(output_dir / "final_model_masked_nan.npy", masked_model)
    np.save(output_dir / "chi2_history.npy", np.asarray(result.iteration_chi2, dtype=float))
    save_mesh_npz(output_dir / "inversion_mesh.npz", case)
    np.savez(
        output_dir / "real_data_geometry_and_survey.npz",
        elec_x=data.elec_x,
        elec_z=data.elec_z,
        elec_elevation=data.elec_elevation,
        measurements=data.measurements,
        x_nodes=case.x_nodes,
        z_top=case.z_top,
        layer_thickness=case.layer_thickness,
        used_measurement_mask=mask.astype(np.uint8),
    )

    plot_files: dict[str, str] = {}
    if not args.no_plot:
        model_plot = output_dir / "single_real_inverted_resistivity.png"
        plot_resistivity_model(
            model_plot,
            case=case,
            model=result.final_model,
            coverage_mask=coverage_mask,
            title=f"Real-data inversion: {input_file.stem}",
            depth=args.depth,
            vmin=args.plot_vmin,
            vmax=args.plot_vmax,
        )
        chi2_plot = output_dir / "single_real_chi2.png"
        plot_chi2(chi2_plot, np.asarray(result.iteration_chi2, dtype=float))
        plot_files = {"resistivity": str(model_plot), "chi2": str(chi2_plot)}

    summary = {
        "input_file": str(input_file),
        "timestamp": None if data.timestamp is None else data.timestamp.isoformat(),
        "output_dir": str(output_dir),
        "n_electrodes": int(data.elec_x.size),
        "n_measurements_raw": int(raw_data.rhoa.size),
        "n_measurements_used": int(data.rhoa.size),
        "data_stride": int(args.data_stride),
        "max_error": None if args.max_error is None else float(args.max_error),
        "rhoa_min": float(np.min(data.rhoa)),
        "rhoa_max": float(np.max(data.rhoa)),
        "rhoa_median": float(np.median(data.rhoa)),
        "elevation_reference": float(data.elevation_reference),
        "mesh_cells": int(case.mesh.cell_count),
        "mesh_nodes": int(case.mesh.node_count),
        "forward_mesh_cells": int(case.forward_mesh.cell_count),
        "forward_mesh_nodes": int(case.forward_mesh.node_count),
        "depth": float(args.depth),
        "n_layers": int(args.n_layers),
        "layer_stretch": float(args.layer_stretch),
        "data_std": _data_std_summary(data_std, data.rhoa.shape),
        "data_misfit": str(args.data_misfit),
        "regularization": float(args.regularization),
        "regularization_mode": str(args.regularization_mode),
        "spatial_regularization": str(args.spatial_regularization),
        "optimizer": str(args.optimizer),
        "linearized_solver": str(args.linearized_solver),
        "max_iterations": int(args.max_iterations),
        "iterations": int(len(result.iteration_chi2)),
        "final_chi2": float(result.iteration_chi2[-1]) if result.iteration_chi2 else None,
        "model_bounds": [float(args.model_min), float(args.model_max)],
        "coverage_percentile": float(args.coverage_percentile),
        "coverage_threshold": coverage_threshold,
        "elapsed_sec": float(elapsed_sec),
        "elapsed_min": float(elapsed_sec / 60.0),
        "plot_files": plot_files,
        "torch_enable_float64": torch.get_default_dtype() == torch.float64,
    }
    write_json(output_dir / "single_real_inversion_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
