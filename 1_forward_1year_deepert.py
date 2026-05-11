#!/usr/bin/env python3
"""Deepert reproduction of the 365-timestep terrain ERT forward notebook."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from deepert.workflows import (
    build_terrain_forward_case,
    discover_resistivity_slices,
    parse_pftcl,
    read_slope_x,
    run_terrain_forward_series,
)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None, help="Repository root. Defaults to this script directory.")
    parser.add_argument("--input-dir", default="2d_resistivity_model")
    parser.add_argument("--model-dir", default="models_1year_1day")
    parser.add_argument("--output-dir", default="result/deepert_timelapsedERT_forward")
    parser.add_argument("--y-index", type=int, default=2)
    parser.add_argument("--file-stride", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None, help="Limit selected timesteps for a quick run.")
    parser.add_argument("--n-electrodes", type=int, default=48)
    parser.add_argument("--relative-error", type=float, default=0.03)
    parser.add_argument("--topo-offset", type=float, default=0.0)
    parser.add_argument("--linear-solver-backend", default="auto")
    parser.add_argument("--terrain-cache-dir", default=None)
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing .dat/.npz files.")
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
        raise FileNotFoundError(f"No resistivity2d_y{args.y_index}_t*.npy files found in {input_dir}")

    grid = parse_pftcl(model_dir / "sc2d_6.out.pftcl")
    slope_x = read_slope_x(model_dir / "sc2d_6.out.slope_x.pfb", y_index=args.y_index)

    first_step, first_file = pairs[0]
    first_case = build_terrain_forward_case(
        np.asarray(np.load(first_file), dtype=float),
        grid,
        slope_x,
        y_index=args.y_index,
        n_electrodes=args.n_electrodes,
        topo_offset=args.topo_offset,
    )
    geometry_file = output_dir / "forward_geometry.npz"
    np.savez(
        geometry_file,
        x_nodes=first_case.x_nodes,
        z_top=first_case.z_top,
        layer_thickness=first_case.layer_thickness,
        elec_x=first_case.elec_x,
        elec_z=first_case.elec_z,
        y_index=np.asarray([args.y_index], dtype=np.int32),
        first_step=np.asarray([first_step], dtype=np.int32),
    )

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
        terrain_cache_dir=None if args.terrain_cache_dir is None else _resolve(root, args.terrain_cache_dir),
    )
    elapsed_sec = time.perf_counter() - start

    manifest_json = [asdict(record) for record in manifest]
    failures_json = [asdict(record) for record in failures]
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_selected": len(pairs),
        "n_ok": sum(1 for record in manifest if record.status == "ok"),
        "n_skipped": sum(1 for record in manifest if record.status == "skipped_existing"),
        "n_failed": len(failures),
        "first_step": int(pairs[0][0]),
        "last_step": int(pairs[-1][0]),
        "y_index": int(args.y_index),
        "n_electrodes": int(len(first_case.elec_x)),
        "measurements": int(first_case.survey.measurement_count),
        "mesh_cells": int(first_case.mesh.cell_count),
        "relative_error": float(args.relative_error),
        "geometry_file": str(geometry_file),
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
