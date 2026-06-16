#!/usr/bin/env python3
"""Time-lapse Deepert inversion for the processed real hillslope ERT data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("DEEPERT_ENABLE_FLOAT64", "1")

from deepert.utils.torch_runtime import torch_runtime
import numpy as np

torch_runtime.config.update("torch_enable_float64", True)

from deepert.inversion import (
    InversionConfig,
    TimeLapseERTInversion,
    WindowedTimeLapseERTInversion,
    available_data_misfits,
    available_linearized_optimizers,
    available_optimization_algorithms,
    available_spatial_regularizations,
    available_temporal_regularizations,
)
from deepert.utils.progress import InversionProgressPrinter

from _real_data_common import (
    apply_data_stride,
    build_parameterized_forward,
    build_real_inversion_case,
    data_std_from_err,
    discover_processed_files,
    find_project_root,
    plot_chi2,
    plot_timelapse_models,
    quality_mask,
    read_processed_ert,
    resolve_path,
    save_mesh_npz,
    write_json,
)


def _check_same_layout(first, current) -> None:
    if not np.array_equal(current.measurements, first.measurements):
        raise ValueError(f"{current.path}: ABMN layout differs from first timestep {first.path}")
    if not np.allclose(current.elec_x, first.elec_x, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{current.path}: electrode x positions differ from first timestep")
    if not np.allclose(current.elec_z, first.elec_z, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{current.path}: electrode z positions differ from first timestep")


def _selected_plot_indices(n_times: int, max_panels: int) -> np.ndarray:
    if n_times <= max_panels:
        return np.arange(n_times, dtype=np.int32)
    return np.unique(np.round(np.linspace(0, n_times - 1, max_panels)).astype(np.int32))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--input-dir", default="ProcessedData")
    parser.add_argument("--output-dir", default="result/8_real_data/2_timelapse_inversion_real_deepert")
    parser.add_argument("--mesh-file", default=None, help="Optional saved timelapse_inversion_mesh.npz to reuse.")
    parser.add_argument(
        "--start-date",
        default="2022-03-26",
        help="Inclusive start date/time. Default focuses on the spring snowmelt event.",
    )
    parser.add_argument(
        "--end-date",
        default="2022-05-12",
        help="Inclusive date if YYYY-MM-DD is supplied; internally handled as next-day exclusive.",
    )
    parser.add_argument("--file-stride", type=int, default=8, help="Use every Nth processed file after date filtering.")
    parser.add_argument("--max-timesteps", type=int, default=None)
    parser.add_argument("--inversion-mode", choices=("windowed", "full"), default="windowed")
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--window-step", type=int, default=1)
    parser.add_argument("--depth", type=float, default=90.0)
    parser.add_argument("--n-layers", type=int, default=28)
    parser.add_argument("--layer-stretch", type=float, default=1.08)
    parser.add_argument("--data-stride", type=int, default=1, help="Use every Nth common valid datum; keep 1 for production.")
    parser.add_argument("--max-error", type=float, default=None, help="Optional common maximum reciprocal/data error filter.")
    parser.add_argument("--relative-error", type=float, default=0.05)
    parser.add_argument("--minimum-log-std", type=float, default=1.0e-3)
    parser.add_argument("--data-misfit", choices=available_data_misfits(), default="weighted_log_l2")
    parser.add_argument("--regularization", type=float, default=50.0)
    parser.add_argument("--temporal-regularization", type=float, default=10.0)
    parser.add_argument("--temporal-regularization-type", choices=available_temporal_regularizations(), default="temporal_smoothness")
    parser.add_argument("--regularization-mode", choices=("model", "update"), default="model")
    parser.add_argument("--spatial-regularization", choices=available_spatial_regularizations(), default="first_order_smoothness")
    parser.add_argument("--z-weight", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=available_optimization_algorithms(), default="gauss_newton_cgls")
    parser.add_argument("--linearized-solver", choices=available_linearized_optimizers(), default="gpu_cgls")
    parser.add_argument("--linear-solver-backend", default="auto")
    parser.add_argument("--terrain-cache-dir", default=None)
    parser.add_argument("--lm-damping", type=float, default=1.0e-2)
    parser.add_argument("--cgls-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--cgls-max-iterations", type=int, default=2000)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--model-min", type=float, default=1.0)
    parser.add_argument("--model-max", type=float, default=1.0e5)
    parser.add_argument("--max-log-step", type=float, default=1.0)
    parser.add_argument("--target-chi2", type=float, default=None)
    parser.add_argument("--step-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--line-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inversion-mesh-quality", type=float, default=34.0)
    parser.add_argument("--inversion-mesh-smoothing-iterations", type=int, default=10)
    parser.add_argument("--coverage-percentile", type=float, default=20.0)
    parser.add_argument("--plot-count", type=int, default=4)
    parser.add_argument("--plot-vmin", type=float, default=None)
    parser.add_argument("--plot-vmax", type=float, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).resolve() if args.project_root else find_project_root(Path(__file__))
    input_dir = resolve_path(root, args.input_dir)
    output_dir = resolve_path(root, args.output_dir)
    mesh_file = None if args.mesh_file is None else resolve_path(root, args.mesh_file)
    terrain_cache_dir = None if args.terrain_cache_dir is None else resolve_path(root, args.terrain_cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = discover_processed_files(
        input_dir,
        start=args.start_date,
        end=args.end_date,
        file_stride=args.file_stride,
        max_timesteps=args.max_timesteps,
    )
    if len(paths) < 2:
        raise ValueError(f"Need at least two real ERT files after filtering in {input_dir}.")

    first = read_processed_ert(paths[0])
    raw_records = [first]
    for path in paths[1:]:
        current = read_processed_ert(path, elevation_reference=first.elevation_reference)
        _check_same_layout(first, current)
        raw_records.append(current)

    common_mask = np.ones(first.rhoa.shape, dtype=bool)
    for record in raw_records:
        common_mask &= quality_mask(record, max_error=args.max_error)
    common_mask = apply_data_stride(common_mask, args.data_stride)
    if int(common_mask.sum()) < 4:
        raise ValueError("Need at least four common valid measurements after filtering.")

    records = [record.with_measurement_mask(common_mask) for record in raw_records]
    observed_rhoa = np.vstack([record.rhoa for record in records])
    err_matrix = None if any(record.err is None for record in records) else np.vstack([record.err for record in records])
    data_std = data_std_from_err(
        err_matrix,
        shape=observed_rhoa.shape,
        relative_error=args.relative_error,
        minimum_log_std=args.minimum_log_std,
    )
    timestamps = [record.timestamp for record in records]
    if all(timestamp is not None for timestamp in timestamps):
        start_time = timestamps[0]
        measurement_times_days = np.asarray(
            [(timestamp - start_time).total_seconds() / 86400.0 for timestamp in timestamps], dtype=float
        )
        timestamp_labels = [timestamp.strftime("%Y-%m-%d %H:%M") for timestamp in timestamps]
    else:
        measurement_times_days = np.arange(len(records), dtype=float)
        timestamp_labels = [path.stem for path in paths]
    steps = np.arange(len(records), dtype=np.int32)

    case = build_real_inversion_case(
        records[0],
        depth=args.depth,
        n_layers=args.n_layers,
        layer_stretch=args.layer_stretch,
        inversion_mesh_quality=args.inversion_mesh_quality,
        inversion_mesh_smoothing_iterations=args.inversion_mesh_smoothing_iterations,
        mesh_file=mesh_file,
    )
    if observed_rhoa.shape[1] != case.survey.measurement_count:
        raise ValueError(
            f"Observed data has {observed_rhoa.shape[1]} measurements, "
            f"but survey has {case.survey.measurement_count}."
        )

    forward = build_parameterized_forward(
        case,
        linear_solver_backend=args.linear_solver_backend,
        terrain_cache_dir=terrain_cache_dir,
    )
    progress = InversionProgressPrinter(enabled=not args.quiet)
    config = InversionConfig(
        max_iterations=args.max_iterations,
        data_std=data_std,
        data_misfit=args.data_misfit,
        regularization=args.regularization,
        regularization_mode=args.regularization_mode,
        temporal_regularization=args.temporal_regularization,
        temporal_regularization_type=args.temporal_regularization_type,
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
            run_meta = {"inversion_mode": "full", "n_windows": 1, "window_size": None, "window_step": None}
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
                "n_windows": int(len(result.window_reports)),
                "window_size": int(args.window_size),
                "window_step": int(args.window_step),
            }
    finally:
        forward.close()
        progress.finish()
    elapsed_sec = time.perf_counter() - run_start

    final_models = np.asarray(result.final_models, dtype=float)
    coverage = np.asarray(result.coverage, dtype=float).ravel()
    coverage_threshold = float(np.percentile(coverage, args.coverage_percentile))
    coverage_mask = coverage < coverage_threshold

    np.save(output_dir / "final_models.npy", final_models)
    np.save(output_dir / "final_log_models.npy", np.asarray(result.final_log_models, dtype=float))
    np.save(output_dir / "predicted_rhoa.npy", np.asarray(result.predicted_data, dtype=float))
    np.save(output_dir / "observed_rhoa.npy", observed_rhoa)
    np.save(output_dir / "data_std.npy", np.asarray(data_std, dtype=float))
    np.save(output_dir / "steps.npy", steps)
    np.save(output_dir / "measurement_times_days.npy", measurement_times_days)
    np.save(output_dir / "chi2_all.npy", np.asarray(result.all_chi2, dtype=float))
    np.save(output_dir / "coverage.npy", coverage)
    np.save(output_dir / "coverage_mask.npy", coverage_mask.astype(np.uint8))
    save_mesh_npz(output_dir / "timelapse_inversion_mesh.npz", case)
    np.savez(
        output_dir / "real_data_geometry_and_survey.npz",
        elec_x=records[0].elec_x,
        elec_z=records[0].elec_z,
        elec_elevation=records[0].elec_elevation,
        measurements=records[0].measurements,
        x_nodes=case.x_nodes,
        z_top=case.z_top,
        layer_thickness=case.layer_thickness,
        common_measurement_mask=common_mask.astype(np.uint8),
    )

    for column, (step, label) in enumerate(zip(steps, timestamp_labels, strict=True)):
        model = final_models[:, column]
        masked = model.copy()
        masked[coverage_mask] = np.nan
        np.save(output_dir / f"inverted_model_t{int(step):05d}.npy", model)
        np.save(output_dir / f"inverted_model_masked_nan_t{int(step):05d}.npy", masked)
        safe_label = label.replace("-", "").replace(":", "").replace(" ", "_")
        np.save(output_dir / f"inverted_model_{safe_label}.npy", model)

    with (output_dir / "used_data_files.txt").open("w", encoding="utf-8") as stream:
        for path, label in zip(paths, timestamp_labels, strict=True):
            stream.write(f"{label}\t{path}\n")
    if result.window_reports:
        write_json(output_dir / "window_reports.json", result.window_reports)

    plot_files: dict[str, str] = {}
    if not args.no_plot:
        indices = _selected_plot_indices(final_models.shape[1], args.plot_count)
        labels = [timestamp_labels[int(index)] for index in indices]
        model_plot = output_dir / "timelapse_real_inverted_resistivity.png"
        plot_timelapse_models(
            model_plot,
            case=case,
            models=final_models[:, indices],
            labels=labels,
            coverage_mask=coverage_mask,
            depth=args.depth,
            vmin=args.plot_vmin,
            vmax=args.plot_vmax,
        )
        chi2_plot = output_dir / "timelapse_real_chi2.png"
        plot_chi2(chi2_plot, np.asarray(result.all_chi2, dtype=float), xlabel="Window index" if args.inversion_mode == "windowed" else "Iteration")
        plot_files = {"resistivity": str(model_plot), "chi2": str(chi2_plot)}

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "file_stride": int(args.file_stride),
        "max_timesteps": None if args.max_timesteps is None else int(args.max_timesteps),
        "n_timesteps": int(len(records)),
        "first_timestamp": timestamp_labels[0],
        "last_timestamp": timestamp_labels[-1],
        "n_electrodes": int(records[0].elec_x.size),
        "n_measurements_raw": int(first.rhoa.size),
        "n_measurements_used": int(observed_rhoa.shape[1]),
        "data_stride": int(args.data_stride),
        "max_error": None if args.max_error is None else float(args.max_error),
        "rhoa_min": float(np.min(observed_rhoa)),
        "rhoa_max": float(np.max(observed_rhoa)),
        "rhoa_median": float(np.median(observed_rhoa)),
        "elevation_reference": float(records[0].elevation_reference),
        "mesh_cells": int(case.mesh.cell_count),
        "mesh_nodes": int(case.mesh.node_count),
        "forward_mesh_cells": int(case.forward_mesh.cell_count),
        "forward_mesh_nodes": int(case.forward_mesh.node_count),
        "depth": float(args.depth),
        "n_layers": int(args.n_layers),
        "layer_stretch": float(args.layer_stretch),
        "data_misfit": str(args.data_misfit),
        "regularization": float(args.regularization),
        "temporal_regularization": float(args.temporal_regularization),
        "temporal_regularization_type": str(args.temporal_regularization_type),
        "regularization_mode": str(args.regularization_mode),
        "spatial_regularization": str(args.spatial_regularization),
        "optimizer": str(args.optimizer),
        "linearized_solver": str(args.linearized_solver),
        "linear_solver_backend": str(args.linear_solver_backend),
        "max_iterations": int(args.max_iterations),
        "final_chi2": float(result.iteration_chi2[-1]) if result.iteration_chi2 else None,
        "model_bounds": [float(args.model_min), float(args.model_max)],
        "coverage_percentile": float(args.coverage_percentile),
        "coverage_threshold": coverage_threshold,
        "elapsed_sec": float(elapsed_sec),
        "elapsed_min": float(elapsed_sec / 60.0),
        "plot_files": plot_files,
        "torch_enable_float64": bool(torch_runtime.config.torch_enable_float64),
    }
    summary.update(run_meta)
    write_json(output_dir / "timelapse_real_inversion_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
