#!/usr/bin/env python3
"""Run and time one native-mesh pyGIMLi ERT inversion."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--step", type=int, default=4536)
    parser.add_argument(
        "--data-file",
        default="result/1_single_forward_adtlert/synthetic_ert_terrain_vardz.dat",
        help="Single-time forward data in pyGIMLi .dat format.",
    )
    parser.add_argument(
        "--output-dir",
        default="result/2_single_inversion_pygimli",
    )
    parser.add_argument("--regularization", type=float, default=50.0)
    parser.add_argument("--relative-error", type=float, default=0.03)
    parser.add_argument("--max-iterations", type=int, default=15)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from pygimli.physics import ert
    import pygimli as pg

    root = (
        Path(args.project_root).resolve()
        if args.project_root
        else Path(__file__).resolve().parents[2]
    )
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_file = (root / args.data_file).resolve()

    pg.setVerbose(False)
    pg.setDebug(False)
    pg.core.setDebug(False)
    pg.setLogLevel("WARNING")

    total_start = time.perf_counter()
    load_start = time.perf_counter()
    data = ert.load(str(data_file))
    data["err"] = np.full(
        data.size(), float(args.relative_error), dtype=float
    )
    data_load_sec = time.perf_counter() - load_start

    mesh_start = time.perf_counter()
    manager = ert.ERTManager(data, verbose=False)
    mesh = manager.createMesh(data=data, quality=34)
    mesh_build_sec = time.perf_counter() - mesh_start

    inversion_start = time.perf_counter()
    model = np.asarray(
        manager.invert(
            data=data,
            mesh=mesh,
            lam=float(args.regularization),
            maxIter=int(args.max_iterations),
            limits=[10.0, 20000.0],
            verbose=False,
        ),
        dtype=float,
    ).ravel()
    inversion_sec = time.perf_counter() - inversion_start
    total_sec = time.perf_counter() - total_start

    predicted = np.asarray(manager.inv.response, dtype=float).ravel()
    np.save(output_dir / "final_model.npy", model)
    np.save(output_dir / "predicted_rhoa.npy", predicted)
    mesh.save(str(output_dir / "inversion_mesh.bms"))

    summary = {
        "engine": "pyGIMLi",
        "pygimli_version": str(pg.__version__),
        "mode": "single",
        "step": int(args.step),
        "measurements": int(data.size()),
        "native_mesh_cells": int(mesh.cellCount()),
        "parameter_cells": int(model.size),
        "regularization": float(args.regularization),
        "relative_error": float(args.relative_error),
        "max_iterations": int(args.max_iterations),
        "iterations": int(manager.inv.iter),
        "final_chi2": float(manager.inv.chi2()),
        "data_load_sec": float(data_load_sec),
        "mesh_build_sec": float(mesh_build_sec),
        "inversion_sec": float(inversion_sec),
        "total_sec": float(total_sec),
        "execution": "CPU",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "data_file": str(data_file),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
