#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from adtlert.workflows import (
    build_terrain_forward_case,
    parse_pftcl,
    parse_resistivity_slice_name,
    read_slope_x,
    save_terrain_forward_dat,
    save_terrain_forward_npz,
)
from example.shared import resolve
from example.single_time.plotting import plot_case


def _write_summary(path: Path, summary: dict[str, object]) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _run_pygimli_forward(
    case, *, relative_error: float, verbose: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import pygimli as pg
        from pygimli.physics import ert
    except ImportError as exc:
        raise RuntimeError(
            "pygimli is not installed in the current environment. "
            "Install it first, e.g. `uv pip install pygimli` or activate an env that contains pygimli."
        ) from exc

    pg.setVerbose(bool(verbose))
    pg.setDebug(bool(verbose))
    pg.core.setDebug(bool(verbose))
    pg.setLogLevel(2 if verbose else 0)

    top_line = pg.meshtools.createPolygon(
        np.c_[case.x_nodes, case.z_top], isClosed=False
    )
    for boundary in top_line.boundaries():
        boundary.setMarker(2)

    y_offsets = np.concatenate(
        ([0.0], -np.cumsum(np.asarray(case.layer_thickness, dtype=float)))
    )
    mesh = pg.meshtools.createMesh2D(top_line, y_offsets, -1, 0, 0, 0, True)

    scheme = ert.createData(elecs=np.c_[case.elec_x, case.elec_z], schemeName="wa")
    fop = ert.ERTModelling()
    fop.setData(scheme)
    fop.setMesh(mesh)

    res_model = np.asarray(case.resistivity, dtype=float)
    rhoa = np.asarray(fop.response(res_model), dtype=float).ravel()
    if rhoa.shape != (case.survey.measurement_count,):
        raise ValueError(
            f"pygimli returned rhoa shape {rhoa.shape}, expected ({case.survey.measurement_count},)"
        )
    if not np.all(np.isfinite(rhoa)) or np.any(rhoa <= 0.0):
        raise ValueError(
            "pygimli forward returned non-finite or non-positive apparent resistivity values"
        )

    scheme["rhoa"] = rhoa
    err = np.asarray(
        ert.ERTManager(scheme).estimateError(
            scheme, absoluteUError=0.0, relativeError=float(relative_error)
        ),
        dtype=float,
    ).ravel()
    if err.shape != rhoa.shape:
        raise ValueError(
            f"pygimli returned err shape {err.shape}, expected {rhoa.shape}"
        )
    scheme["err"] = err

    k_values = np.asarray(scheme["k"], dtype=float).ravel()
    return rhoa, err, k_values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        default=None,
        help="Repository root. Auto-detected by default.",
    )
    parser.add_argument(
        "--input-file",
        default="resistivity_models_2d/resistivity2d_y2_t04536.npy",
        help="2D ParFlow resistivity slice in bottom-to-top z ordering.",
    )
    parser.add_argument(
        "--model-dir",
        default="parflow_models",
        help="Directory containing pftcl and slope_x files.",
    )
    parser.add_argument("--output-dir", default="result/1_single_forward_pygimli")
    parser.add_argument("--n-electrodes", type=int, default=48)
    parser.add_argument("--relative-error", type=float, default=0.03)
    parser.add_argument("--topo-offset", type=float, default=0.0)
    parser.add_argument(
        "--verbose", action="store_true", help="Enable pyGIMLi verbose logs."
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Do not save PNG previews."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = (
        Path(args.project_root).resolve()
        if args.project_root
        else Path(__file__).resolve().parents[2]
    )
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
    rhoa, err, k_values = _run_pygimli_forward(
        case, relative_error=float(args.relative_error), verbose=bool(args.verbose)
    )

    dat_file = output_dir / "synthetic_ert_terrain_vardz.dat"
    npz_file = output_dir / "synthetic_ert_terrain_vardz.npz"
    save_terrain_forward_dat(dat_file, case, rhoa, relative_error=args.relative_error)
    save_terrain_forward_npz(npz_file, case, rhoa, relative_error=args.relative_error)

    # Keep the .npz error consistent with pyGIMLi's estimateError output.
    with np.load(npz_file) as data:
        payload = {name: np.asarray(data[name]) for name in data.files}
    payload["err"] = err
    np.savez(npz_file, **payload)

    plot_paths: dict[str, Path] = {}
    if not args.no_plot:
        plot_paths = plot_case(output_dir, case, rhoa, step, "pygimli")

    summary = {
        "engine": "pygimli",
        "forward_backend": "pygimli.ERTModelling.response",
        "input_file": str(input_file),
        "output_dir": str(output_dir),
        "dat_file": str(dat_file),
        "npz_file": str(npz_file),
        "step": int(step),
        "y_index": int(y_index),
        "mesh_cells": int(case.mesh.cell_count),
        "measurements": int(case.survey.measurement_count),
        "rhoa_min": float(np.min(rhoa)),
        "rhoa_max": float(np.max(rhoa)),
        "n_electrodes": len(case.elec_x),
        "relative_error": float(args.relative_error),
        "k_min": float(np.min(k_values)),
        "k_max": float(np.max(k_values)),
        "plot_files": {name: str(path) for name, path in plot_paths.items()},
    }
    _write_summary(output_dir / "forward_summary.json", summary)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
