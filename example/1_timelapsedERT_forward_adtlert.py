#!/usr/bin/env python3
"""Native ADTLERT reproduction of ``1_forward_1year.ipynb``."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("ADTLERT_ENABLE_FLOAT64", "1")

from adtlert.utils.torch_runtime import torch_runtime
import numpy as np

torch_runtime.config.update("torch_enable_float64", True)

from adtlert.workflows import (
    build_terrain_forward_case,
    discover_resistivity_slices,
    load_terrain_resistivity_slice,
    parse_pftcl,
    read_slope_x,
    run_terrain_forward_series,
)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _save_geometry(path: Path, case, grid, *, first_step: int) -> None:
    np.savez(
        path,
        x_nodes=case.x_nodes,
        z_top=case.z_top,
        layer_thickness=case.layer_thickness,
        elec_x=case.elec_x,
        elec_z=case.elec_z,
        y_index=np.asarray([case.y_index], dtype=np.int32),
        first_step=np.asarray([first_step], dtype=np.int32),
        DX=np.asarray([grid.dx], dtype=float),
        DY=np.asarray([grid.dy], dtype=float),
        DZ_BASE=np.asarray([grid.dz_base], dtype=float),
        NX=np.asarray([grid.nx], dtype=np.int32),
        NY=np.asarray([grid.ny], dtype=np.int32),
        NZ=np.asarray([grid.nz], dtype=np.int32),
        scheme_name=np.asarray(["wa"]),
        n_electrodes=np.asarray([len(case.elec_x)], dtype=np.int32),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None, help="Repository root. Defaults to this script directory.")
    parser.add_argument(
        "--input-dir",
        default="../resistivity_models_2d",
        help="Directory containing notebook-style resistivity_t*.npy or resistivity2d_y{y}_t*.npy terrain slices.",
    )
    parser.add_argument("--model-dir", default="../parflow_models")
    parser.add_argument("--output-dir", default="../result/1_timelapsedERT_forward_adtlert")
    parser.add_argument("--y-index", type=int, default=2)
    parser.add_argument("--file-stride", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None, help="Limit selected timesteps for quick checks.")
    parser.add_argument("--n-electrodes", type=int, default=48)
    parser.add_argument("--relative-error", type=float, default=0.03)
    parser.add_argument("--topo-offset", type=float, default=0.0)
    parser.add_argument("--linear-solver-backend", default="auto")
    parser.add_argument("--terrain-cache-dir", default=None)
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing .dat/.npz files.")
    parser.add_argument(
        "--reuse-solver-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse cuDSS symbolic plans and GPU buffers across timesteps. "
            "Use --no-reuse-solver-state to rebuild solver state every timestep."
        ),
    )
    parser.add_argument(
        "--prepare-forward",
        action="store_true",
        help="Warm ADTLERT forward caches with the first selected timestep before the series run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parent
    input_dir = _resolve(root, args.input_dir)
    model_dir = _resolve(root, args.model_dir)
    output_dir = _resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_resistivity_slices(
        input_dir,
        y_index=args.y_index,
        file_stride=args.file_stride,
        max_steps=args.max_steps,
    )
    if not pairs:
        raise FileNotFoundError(
            f"No resistivity2d_y{args.y_index}_t*.npy or resistivity_t*.npy files found in {input_dir}"
        )

    grid = parse_pftcl(model_dir / "sc2d_6.out.pftcl")
    slope_x = read_slope_x(model_dir / "sc2d_6.out.slope_x.pfb", y_index=args.y_index)

    first_step, first_file = pairs[0]
    first_case = build_terrain_forward_case(
        load_terrain_resistivity_slice(first_file, grid, y_index=args.y_index),
        grid,
        slope_x,
        y_index=args.y_index,
        n_electrodes=args.n_electrodes,
        topo_offset=args.topo_offset,
    )
    geometry_file = output_dir / "forward_geometry.npz"
    _save_geometry(geometry_file, first_case, grid, first_step=first_step)

    start = time.perf_counter()
    manifest, failures = run_terrain_forward_series(
        pairs,
        grid,
        slope_x,
        output_dir,
        y_index=args.y_index,
        n_electrodes=args.n_electrodes,
        topo_offset=args.topo_offset,
        relative_error=args.relative_error,
        overwrite=not args.skip_existing,
        linear_solver_backend=args.linear_solver_backend,
        reuse_solver_state=args.reuse_solver_state,
        terrain_cache_dir=None if args.terrain_cache_dir is None else _resolve(root, args.terrain_cache_dir),
        prepare_forward=args.prepare_forward,
    )
    elapsed_sec = time.perf_counter() - start

    manifest_json = [asdict(record) for record in manifest]
    failures_json = [asdict(record) for record in failures]
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_selected": int(len(pairs)),
        "n_ok": int(sum(1 for record in manifest if record.status == "ok")),
        "n_skipped": int(sum(1 for record in manifest if record.status == "skipped_existing")),
        "n_failed": int(len(failures)),
        "first_step": int(pairs[0][0]),
        "last_step": int(pairs[-1][0]),
        "y_index": int(args.y_index),
        "n_electrodes": int(len(first_case.elec_x)),
        "scheme_name": "wa",
        "measurements": int(first_case.survey.measurement_count),
        "mesh_cells": int(first_case.mesh.cell_count),
        "relative_error": float(args.relative_error),
        "geometry_file": str(geometry_file),
        "save_per_step": ".dat and .npz",
        "linear_solver_backend": str(args.linear_solver_backend),
        "reuse_solver_state": bool(args.reuse_solver_state),
        "elapsed_sec": float(elapsed_sec),
        "elapsed_min": float(elapsed_sec / 60.0),
    }

    _write_json(output_dir / "forward_manifest.json", manifest_json)
    _write_json(output_dir / "forward_failures.json", failures_json)
    _write_json(output_dir / "forward_summary.json", summary)

    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
