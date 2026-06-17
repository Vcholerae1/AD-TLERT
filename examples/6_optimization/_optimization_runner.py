from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


COMMON_ARGS = [
    "--inversion-mode",
    "windowed",
    "--window-size",
    "3",
    "--window-step",
    "1",
    "--data-misfit",
    "weighted_log_l2",
    "--spatial-regularization",
    "first_order_smoothness",
    "--temporal-regularization-type",
    "temporal_smoothness",
    "--temporal-regularization",
    "10.0",
    "--no-plot",
    "--quiet",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _base_script_path(project_root: Path) -> Path:
    return project_root / "examples" / "2_timelapsedERT_inversion_deepert.py"


def _parse_max_rss_kb(time_stderr: str) -> float | None:
    pattern = re.compile(r"Maximum resident set size \(kbytes\):\s*([0-9.]+)")
    match = pattern.search(time_stderr)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _load_json_if_exists(path: Path) -> object | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_npy_if_exists(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return np.asarray(np.load(path), dtype=float)
    except Exception:
        return None


def _iteration_metrics(output_dir: Path) -> dict[str, float | int | None]:
    window_reports = _load_json_if_exists(output_dir / "window_reports.json")
    chi2_all = _load_npy_if_exists(output_dir / "chi2_all.npy")

    total_iterations: int | None = None
    window_count: int | None = None
    avg_iterations_per_window: float | None = None

    if isinstance(window_reports, list) and len(window_reports) > 0:
        iteration_counts = [int(item.get("iterations", 0)) for item in window_reports if isinstance(item, dict)]
        if iteration_counts:
            total_iterations = int(sum(iteration_counts))
            window_count = int(len(iteration_counts))
            avg_iterations_per_window = float(np.mean(iteration_counts))

    if total_iterations is None and chi2_all is not None:
        total_iterations = int(np.asarray(chi2_all).size)

    final_objective_value: float | None = None
    if chi2_all is not None and chi2_all.size > 0:
        final_objective_value = float(np.asarray(chi2_all).ravel()[-1])

    return {
        "total_iterations": total_iterations,
        "window_count": window_count,
        "avg_iterations_per_window": avg_iterations_per_window,
        "final_objective_value": final_objective_value,
    }


def _build_metrics(
    *,
    output_dir: Path,
    optimizer: str,
    command: list[str],
    return_code: int,
    runtime_wall_sec: float,
    max_rss_kb: float | None,
) -> dict[str, object]:
    summary = _load_json_if_exists(output_dir / "timelapsed_inversion_summary.json")
    if not isinstance(summary, dict):
        summary = {}

    iteration_info = _iteration_metrics(output_dir)

    metrics: dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "optimizer": optimizer,
        "command": command,
        "return_code": int(return_code),
        "runtime_wall_sec": float(runtime_wall_sec),
        "runtime_wall_min": float(runtime_wall_sec) / 60.0,
        "max_rss_kb": None if max_rss_kb is None else float(max_rss_kb),
        "max_rss_gb": None if max_rss_kb is None else float(max_rss_kb) / (1024.0 * 1024.0),
        "elapsed_sec_from_summary": summary.get("elapsed_sec"),
        "inversion_mode": summary.get("inversion_mode"),
        "window_size": summary.get("window_size"),
        "window_step": summary.get("window_step"),
        "n_cells": summary.get("n_cells"),
        "n_timesteps": summary.get("n_timesteps"),
        "data_misfit": summary.get("data_misfit"),
        "spatial_regularization": summary.get("spatial_regularization"),
        "temporal_regularization_type": summary.get("temporal_regularization_type"),
        "temporal_regularization": summary.get("alpha"),
        "total_iterations": iteration_info["total_iterations"],
        "window_count": iteration_info["window_count"],
        "avg_iterations_per_window": iteration_info["avg_iterations_per_window"],
        "final_objective_value": iteration_info["final_objective_value"],
        "final_objective_name": "chi2_data",
    }
    return metrics


def _write_case_metrics(output_dir: Path, metrics: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "optimization_benchmark.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def run_optimizer_case(
    *,
    optimizer: str,
    output_name: str,
    extra_cli_args: list[str] | None = None,
) -> int:
    project_root = _project_root()
    base_script = _base_script_path(project_root)
    output_dir = project_root / "result" / "6_optimization" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cli_tail = extra_cli_args if extra_cli_args is not None else sys.argv[1:]

    base_command = [
        sys.executable,
        str(base_script),
        "--project-root",
        str(project_root),
        "--forward-dir",
        "result/1_timelapsedERT_forward_deepert",
        "--true-model-dir",
        "resistivity_models_2d",
        "--optimizer",
        str(optimizer),
        "--output-dir",
        f"result/6_optimization/{output_name}",
        *COMMON_ARGS,
        *cli_tail,
    ]

    timed_command = ["/usr/bin/time", "-v", *base_command]
    print("Running:", " ".join(timed_command))

    start = time.perf_counter()
    completed = subprocess.run(
        timed_command,
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    runtime_wall_sec = time.perf_counter() - start

    (output_dir / "run_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "run_stderr.log").write_text(completed.stderr, encoding="utf-8")

    max_rss_kb = _parse_max_rss_kb(completed.stderr)
    metrics = _build_metrics(
        output_dir=output_dir,
        optimizer=optimizer,
        command=timed_command,
        return_code=int(completed.returncode),
        runtime_wall_sec=float(runtime_wall_sec),
        max_rss_kb=max_rss_kb,
    )
    _write_case_metrics(output_dir, metrics)
    print(json.dumps(metrics, indent=2))

    return int(completed.returncode)


def collect_all_metrics(*, result_root: Path | None = None) -> list[dict[str, object]]:
    root = result_root if result_root is not None else (_project_root() / "result" / "6_optimization")
    if not root.exists():
        raise FileNotFoundError(f"Result root does not exist: {root}")

    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("*/optimization_benchmark.json")):
        value = _load_json_if_exists(path)
        if isinstance(value, dict):
            row = dict(value)
            row["case_dir"] = str(path.parent)
            rows.append(row)
    return rows


def write_aggregate_table(*, rows: list[dict[str, object]], result_root: Path | None = None) -> tuple[Path, Path]:
    root = result_root if result_root is not None else (_project_root() / "result" / "6_optimization")
    root.mkdir(parents=True, exist_ok=True)

    json_path = root / "optimization_benchmark_summary.json"
    csv_path = root / "optimization_benchmark_summary.csv"

    rows_sorted = sorted(rows, key=lambda item: str(item.get("optimizer", "")))
    json_path.write_text(json.dumps(rows_sorted, indent=2), encoding="utf-8")

    fieldnames = [
        "optimizer",
        "return_code",
        "runtime_wall_sec",
        "max_rss_kb",
        "max_rss_gb",
        "total_iterations",
        "window_count",
        "avg_iterations_per_window",
        "final_objective_value",
        "elapsed_sec_from_summary",
        "n_cells",
        "n_timesteps",
        "case_dir",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow({name: row.get(name) for name in fieldnames})

    return json_path, csv_path
