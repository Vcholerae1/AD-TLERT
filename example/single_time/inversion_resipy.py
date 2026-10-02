#!/usr/bin/env python3
"""Run reproducible single-time or difference time-lapse inversions with ResIPy/R2."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil
from common import (
    discover_forward_files,
    resipy_parser,
    resistivity_column,
    select_steps,
    write_json,
)


def _parse_steps(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item) for item in value.split(",") if item.strip()]


def _wine_version() -> str | None:
    executable = shutil.which("wine") or shutil.which("wine64")
    if executable is None:
        return None
    completed = subprocess.run(
        [executable, "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    text = (completed.stdout or completed.stderr).strip()
    return text or None


def _peak_rss_sampler(stop: list[bool], values: list[int]) -> None:
    process = psutil.Process()
    while not stop[0]:
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        values.append(total)
        time.sleep(0.05)


def _start_sampler():
    import threading

    stop = [False]
    values: list[int] = []
    thread = threading.Thread(
        target=_peak_rss_sampler, args=(stop, values), daemon=True
    )
    thread.start()
    return stop, values, thread


def _result_payload(project, steps: list[int]) -> dict[str, np.ndarray]:
    if len(project.meshResults) != len(steps):
        raise RuntimeError(
            f"Expected {len(steps)} ResIPy models, got {len(project.meshResults)}"
        )
    centers: list[np.ndarray] = []
    models: list[np.ndarray] = []
    for mesh in project.meshResults:
        center = np.asarray(mesh.elmCentre, dtype=float)
        column = resistivity_column(list(mesh.df.columns))
        centers.append(np.column_stack((center[:, 0], center[:, 2])))
        models.append(np.asarray(mesh.df[column], dtype=float).ravel())
    base = centers[0]
    if any(
        item.shape != base.shape or not np.allclose(item, base) for item in centers[1:]
    ):
        raise RuntimeError("ResIPy result meshes differ across selected steps")
    return {
        "steps": np.asarray(steps, dtype=np.int32),
        "center_x": base[:, 0],
        "center_z": base[:, 1],
        "models": np.column_stack(models),
    }


def _parse_r2_log(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"iterations": None, "final_rms": None}
    text = path.read_text(encoding="utf-8", errors="replace")
    rms = re.findall(r"Final RMS Misfit:\s*([0-9.Ee+-]+)", text)
    iterations = len(re.findall(r"\bIteration\b", text))
    return {
        "iterations": int(iterations),
        "final_rms": float(rms[-1]) if rms else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--forward-dir", default="result/1_single_forward_adtlert")
    parser.add_argument("--output-dir", default="result/2_single_inversion_resipy")
    parser.add_argument("--mode", choices=("single", "timelapse"), default="single")
    parser.add_argument("--steps", default="")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-timesteps", type=int, default=None)
    parser.add_argument("--relative-error", type=float, default=0.03)
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument(
        "--target-rms",
        type=float,
        default=1.0,
        help="R2 RMS stopping threshold; lower it to force a fixed iteration count.",
    )
    parser.add_argument("--mesh-type", choices=("trian", "quad"), default="trian")
    parser.add_argument("--mesh-refine", type=int, default=0)
    parser.add_argument(
        "--mesh-cl",
        type=float,
        default=None,
        help="Characteristic element length near electrodes; ResIPy chooses it when omitted.",
    )
    parser.add_argument("--mesh-cl-factor", type=float, default=2.0)
    parser.add_argument("--rho-min", type=float, default=10.0)
    parser.add_argument("--rho-max", type=float, default=20000.0)
    parser.add_argument(
        "--parallel", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--ncores", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = (
        Path(args.project_root).resolve()
        if args.project_root
        else Path(__file__).resolve().parents[2]
    )
    forward_dir = (root / args.forward_dir).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    available = discover_forward_files(forward_dir)
    steps = select_steps(
        available,
        steps=_parse_steps(args.steps),
        stride=args.stride,
        max_timesteps=args.max_timesteps,
    )
    if args.mode == "timelapse" and len(steps) < 2:
        raise ValueError("Time-lapse mode needs at least two steps")

    from resipy import Project

    project = Project(dirname=str(output_dir / "work"), typ="R2")
    files = [str(available[step]) for step in steps]
    load_start = time.perf_counter()
    if args.mode == "timelapse":
        project.createTimeLapseSurvey(files, parser=resipy_parser, debug=False)
    else:
        if len(files) != 1:
            raise ValueError("Single mode accepts exactly one selected step")
        project.createSurvey(files[0], parser=resipy_parser, debug=False)
    load_sec = time.perf_counter() - load_start

    for survey in project.surveys:
        survey.df["resError"] = float(args.relative_error) * np.maximum(
            np.abs(survey.df["resist"]), 1.0e-12
        )
    project.err = True

    mesh_start = time.perf_counter()
    mesh_kwargs = {
        "typ": args.mesh_type,
        "refine": int(args.mesh_refine),
        "cl_factor": float(args.mesh_cl_factor),
        "res0": float(np.median(project.surveys[0].df["app"])),
        "show_output": True,
    }
    if args.mesh_cl is not None:
        if args.mesh_cl <= 0:
            raise ValueError("--mesh-cl must be positive")
        mesh_kwargs["cl"] = float(args.mesh_cl)
    project.createMesh(
        **mesh_kwargs,
    )
    mesh_sec = time.perf_counter() - mesh_start

    stop, rss_values, thread = _start_sampler()
    inversion_start = time.perf_counter()
    try:
        project.invert(
            param={
                "max_iter": int(args.max_iterations),
                "tolerance": float(args.target_rms),
                "rho_min": float(args.rho_min),
                "rho_max": float(args.rho_max),
            },
            err=True,
            parallel=bool(args.parallel),
            ncores=int(args.ncores),
        )
    finally:
        inversion_sec = time.perf_counter() - inversion_start
        stop[0] = True
        thread.join(timeout=2.0)

    payload = _result_payload(project, steps)
    np.savez(output_dir / "resipy_models.npz", **payload)
    r2_log_path = output_dir / "work" / "R2.out"
    if not r2_log_path.exists():
        r2_log_path = output_dir / "work" / "invdir" / "R2.out"
    log_metrics = _parse_r2_log(r2_log_path)
    summary = {
        "engine": "ResIPy/R2",
        "resipy_version": importlib.metadata.version("resipy"),
        "mode": args.mode,
        "steps": steps,
        "n_timesteps": len(steps),
        "measurements_per_timestep": int(project.surveys[0].df.shape[0]),
        "mesh_elements": int(payload["models"].shape[0]),
        "mesh_type": args.mesh_type,
        "mesh_refine": int(args.mesh_refine),
        "mesh_cl": None if args.mesh_cl is None else float(args.mesh_cl),
        "mesh_cl_factor": float(args.mesh_cl_factor),
        "relative_error": float(args.relative_error),
        "max_iterations": int(args.max_iterations),
        "target_rms": float(args.target_rms),
        "rho_bounds": [float(args.rho_min), float(args.rho_max)],
        "parallel": bool(args.parallel),
        "ncores": int(args.ncores),
        "data_load_sec": float(load_sec),
        "mesh_build_sec": float(mesh_sec),
        "inversion_sec": float(inversion_sec),
        "total_sec": float(load_sec + mesh_sec + inversion_sec),
        "peak_rss_mb": float(max(rss_values, default=0) / (1024.0**2)),
        "wine_version": _wine_version(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "r2_log": log_metrics,
        "output_models": str(output_dir / "resipy_models.npz"),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
