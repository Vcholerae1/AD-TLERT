#!/usr/bin/env python3
"""Run resumable native-mesh ResIPy inversions over sliding time windows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from common import discover_forward_files, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None)
    parser.add_argument(
        "--forward-dir", default="result/1_timelapsedERT_forward_adtlert"
    )
    parser.add_argument(
        "--output-dir",
        default="result/9_resipy_benchmark/resipy_363windows_native_mesh_full_year",
    )
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--window-step", type=int, default=1)
    parser.add_argument("--relative-error", type=float, default=0.05)
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--target-rms", type=float, default=1.0)
    parser.add_argument("--rho-min", type=float, default=0.001)
    parser.add_argument("--rho-max", type=float, default=10000.0)
    return parser


def _aggregate(
    *,
    output_dir: Path,
    steps: list[int],
    window_size: int,
    window_step: int,
    relative_error: float,
    max_iterations: int,
    target_rms: float,
    rho_min: float,
    rho_max: float,
    started_sec: float,
) -> dict[str, object]:
    reports = []
    for path in sorted(output_dir.glob("w*/summary.json")):
        reports.append(json.loads(path.read_text(encoding="utf-8")))

    summary: dict[str, object] = {
        "engine": "ResIPy/R2",
        "mode": "sliding_window_timelapse",
        "n_input_timesteps": len(steps),
        "first_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "window_size": int(window_size),
        "window_step": int(window_step),
        "expected_windows": int((len(steps) - window_size) // window_step + 1),
        "completed_windows": len(reports),
        "relative_error": float(relative_error),
        "max_iterations": int(max_iterations),
        "target_rms": float(target_rms),
        "rho_bounds": [float(rho_min), float(rho_max)],
        "mesh_cl": None,
        "mesh_policy": "ResIPy automatic from electrode spacing",
        "batch_wall_sec": float(time.perf_counter() - started_sec),
    }
    if reports:
        for key in ("data_load_sec", "mesh_build_sec", "inversion_sec", "total_sec"):
            values = np.asarray([report[key] for report in reports], dtype=float)
            summary[key] = {
                "sum": float(np.sum(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        iterations = np.asarray(
            [report["r2_log"]["iterations"] for report in reports], dtype=int
        )
        summary["iterations"] = {
            "mean": float(np.mean(iterations)),
            "median": float(np.median(iterations)),
            "min": int(np.min(iterations)),
            "max": int(np.max(iterations)),
        }
        summary["mesh_elements"] = sorted(
            {int(report["mesh_elements"]) for report in reports}
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = (
        Path(args.project_root).resolve()
        if args.project_root
        else Path(__file__).resolve().parents[2]
    )
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    available = discover_forward_files((root / args.forward_dir).resolve())
    steps = sorted(available)
    if args.window_size < 2 or args.window_size > len(steps):
        raise ValueError("window-size must be between 2 and the number of inputs")
    if args.window_step < 1:
        raise ValueError("window-step must be positive")

    starts = list(
        range(0, len(steps) - int(args.window_size) + 1, int(args.window_step))
    )
    batch_start = time.perf_counter()
    for ordinal, start in enumerate(starts):
        window_steps = steps[start : start + int(args.window_size)]
        case_dir = output_dir / f"w{ordinal:03d}"
        summary_path = case_dir / "summary.json"
        if summary_path.exists():
            print(
                f"[{ordinal + 1}/{len(starts)}] already complete: {window_steps}",
                flush=True,
            )
            continue
        case_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).with_name("resipy_runner.py")),
            "--project-root",
            str(root),
            "--forward-dir",
            str(args.forward_dir),
            "--output-dir",
            str(case_dir),
            "--mode",
            "timelapse",
            "--steps",
            ",".join(str(step) for step in window_steps),
            "--relative-error",
            str(float(args.relative_error)),
            "--max-iterations",
            str(int(args.max_iterations)),
            "--target-rms",
            str(float(args.target_rms)),
            "--rho-min",
            str(float(args.rho_min)),
            "--rho-max",
            str(float(args.rho_max)),
            "--mesh-refine",
            "0",
            "--mesh-cl-factor",
            "2.0",
            "--no-parallel",
            "--ncores",
            "1",
        ]
        print(
            f"[{ordinal + 1}/{len(starts)}] running: {window_steps}",
            flush=True,
        )
        with (output_dir / f"w{ordinal:03d}.log").open(
            "w", encoding="utf-8"
        ) as log:
            subprocess.run(
                command,
                cwd=root,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        checkpoint = _aggregate(
            output_dir=output_dir,
            steps=steps,
            window_size=int(args.window_size),
            window_step=int(args.window_step),
            relative_error=float(args.relative_error),
            max_iterations=int(args.max_iterations),
            target_rms=float(args.target_rms),
            rho_min=float(args.rho_min),
            rho_max=float(args.rho_max),
            started_sec=batch_start,
        )
        write_json(output_dir / "summary.json", checkpoint)
        current = json.loads(summary_path.read_text(encoding="utf-8"))
        print(
            f"[{ordinal + 1}/{len(starts)}] done: "
            f"inversion={current['inversion_sec']:.3f}s, "
            f"total={current['total_sec']:.3f}s",
            flush=True,
        )

    summary = _aggregate(
        output_dir=output_dir,
        steps=steps,
        window_size=int(args.window_size),
        window_step=int(args.window_step),
        relative_error=float(args.relative_error),
        max_iterations=int(args.max_iterations),
        target_rms=float(args.target_rms),
        rho_min=float(args.rho_min),
        rho_max=float(args.rho_max),
        started_sec=batch_start,
    )
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
