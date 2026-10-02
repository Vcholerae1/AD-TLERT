#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("ADTLERT_ENABLE_FLOAT64", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from example.single_time.plotting import plot_case
from example.shared import resolve
from adtlert.workflows import (
    build_terrain_forward_case,
    parse_pftcl,
    parse_resistivity_slice_name,
    read_slope_x,
    run_terrain_forward,
    save_terrain_forward_dat,
    save_terrain_forward_npz,
)

# Switch Torch to float64 after adtlert fixed FLOAT_DTYPE at import, as before.
torch.set_default_dtype(torch.float64)


def _write_summary(path: Path, summary: dict[str, object]) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None, help="Repository root. Auto-detected by default.")
    parser.add_argument(
        "--input-file",
        default="resistivity_models_2d/resistivity2d_y2_t04536.npy",
        help="2D ParFlow resistivity slice in bottom-to-top z ordering.",
    )
    parser.add_argument("--model-dir", default="parflow_models", help="Directory containing pftcl and slope_x files.")
    parser.add_argument("--output-dir", default="result/1_single_forward_adtlert")
    parser.add_argument("--n-electrodes", type=int, default=48)
    parser.add_argument("--relative-error", type=float, default=0.03)
    parser.add_argument("--topo-offset", type=float, default=0.0)
    parser.add_argument("--terrain-cache-dir", default=None)
    parser.add_argument("--no-plot", action="store_true", help="Do not save the PNG preview.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[2]
    input_file = resolve(root, args.input_file)
    model_dir = resolve(root, args.model_dir)
    output_dir = resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_index, step = parse_resistivity_slice_name(input_file)
    grid = parse_pftcl(model_dir / "sc2d_6.out.pftcl")
    slope_x = read_slope_x(model_dir / "sc2d_6.out.slope_x.pfb", y_index=y_index)
    rho_2d = np.asarray(np.load(input_file), dtype=float)

    case = build_terrain_forward_case(
        rho_2d,
        grid,
        slope_x,
        y_index=y_index,
        n_electrodes=args.n_electrodes,
        topo_offset=args.topo_offset,
    )
    rhoa = run_terrain_forward(
        case,
        terrain_cache_dir=None if args.terrain_cache_dir is None else resolve(root, args.terrain_cache_dir),
    )

    dat_file = output_dir / "synthetic_ert_terrain_vardz.dat"
    npz_file = output_dir / "synthetic_ert_terrain_vardz.npz"
    save_terrain_forward_dat(dat_file, case, rhoa, relative_error=args.relative_error)
    save_terrain_forward_npz(npz_file, case, rhoa, relative_error=args.relative_error)

    plot_paths: dict[str, Path] = {}
    if not args.no_plot:
        plot_paths = plot_case(output_dir, case, rhoa, step, "terrain+variableDz")

    summary = {
        "engine": "adtlert",
        "forward_backend": "adtlert.run_terrain_forward",
        "input_file": str(input_file),
        "output_dir": str(output_dir),
        "dat_file": str(dat_file),
        "npz_file": str(npz_file),
        "step": int(step),
        "y_index": int(y_index),
        "scheme_name": "wa",
        "mesh_cells": int(case.mesh.cell_count),
        "measurements": int(case.survey.measurement_count),
        "rhoa_min": float(np.min(rhoa)),
        "rhoa_max": float(np.max(rhoa)),
        "n_electrodes": int(len(case.elec_x)),
        "relative_error": float(args.relative_error),
        "plot_files": {name: str(path) for name, path in plot_paths.items()},
    }
    _write_summary(output_dir / "forward_summary.json", summary)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
