#!/usr/bin/env python3
"""PyGIMLi/PyHydroGeophysX time-lapse ERT inversion for Deepert comparison."""

from __future__ import annotations

import argparse
import builtins
import json
import math
import os
import re
import threading
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pygimli as pg
except ImportError as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "pygimli is not installed. Activate the environment that contains pygimli first."
    ) from exc
from pygimli.physics import ert

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from PyHydroGeophysX.inversion import TimeLapseERTInversion
import PyHydroGeophysX.inversion.time_lapse as pyhgx_time_lapse
import PyHydroGeophysX.solvers.solver as pyhgx_solver


def _safe_float(value: object) -> float:
    array = np.asarray(value)
    if array.ndim == 0:
        return builtins.float(array)
    if array.size == 1:
        return builtins.float(array.reshape(-1)[0])
    return builtins.float(value)


pyhgx_time_lapse.float = _safe_float
pyhgx_solver.float = _safe_float


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _discover_forward_dat(
    input_dir: Path,
    *,
    file_stride: int,
    max_timesteps: int | None,
) -> list[tuple[int, Path]]:
    if file_stride < 1:
        raise ValueError("file_stride must be >= 1")
    if max_timesteps is not None and max_timesteps < 2:
        raise ValueError("max_timesteps must be >= 2 when set")

    pattern = re.compile(r"synthetic_ert_terrain_vardz_t(\d+)\.dat$")
    pairs: list[tuple[int, Path]] = []
    for path in input_dir.glob("synthetic_ert_terrain_vardz_t*.dat"):
        match = pattern.search(path.name)
        if match is not None:
            pairs.append((int(match.group(1)), path))

    pairs.sort(key=lambda item: item[0])
    pairs = pairs[::file_stride]
    if max_timesteps is not None:
        pairs = pairs[:max_timesteps]
    if len(pairs) < 2:
        raise ValueError(f"Need at least 2 forward .dat files in {input_dir}")
    return pairs


def _load_geometry(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        required = {"x_nodes", "z_top", "layer_thickness"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{path} missing required arrays: {sorted(missing)}")
        return {name: np.asarray(data[name], dtype=float).ravel() for name in required}


class _NoOpProgress:
    def update(self, _: int = 1) -> None:
        return

    def close(self) -> None:
        return


class _SimpleProgress:
    def __init__(self, *, total: int, desc: str):
        self.total = max(1, int(total))
        self.desc = str(desc)
        self.count = 0
        self.last_print = 0.0

    def update(self, step: int = 1) -> None:
        self.count += int(step)
        now = time.perf_counter()
        if self.count >= self.total or (now - self.last_print) >= 1.0:
            ratio = min(1.0, self.count / self.total)
            print(f"\r{self.desc}: {self.count}/{self.total} ({ratio * 100.0:5.1f}%)", end="", flush=True)
            self.last_print = now
        if self.count >= self.total:
            print("", flush=True)

    def close(self) -> None:
        if self.count < self.total:
            print("", flush=True)


@contextmanager
def _progress_bar(*, total: int, desc: str, enabled: bool):
    if not enabled:
        yield _NoOpProgress()
        return
    if tqdm is not None:
        bar = tqdm(total=int(total), desc=str(desc), unit="win", dynamic_ncols=True)
        try:
            yield bar
        finally:
            bar.close()
        return
    bar = _SimpleProgress(total=int(total), desc=str(desc))
    try:
        yield bar
    finally:
        bar.close()


@contextmanager
def _elapsed_reporter(*, label: str, enabled: bool, interval_sec: float = 20.0):
    if not enabled:
        yield
        return
    start = time.perf_counter()
    stop_event = threading.Event()

    def _worker() -> None:
        while not stop_event.wait(float(interval_sec)):
            elapsed_min = (time.perf_counter() - start) / 60.0
            print(f"[{label}] elapsed {elapsed_min:.2f} min ...", flush=True)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=0.2)


@contextmanager
def _maybe_silence_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield


def _configure_pygimli_logging(quiet: bool, log_level: str) -> None:
    if quiet:
        pg.setVerbose(False)
        pg.setDebug(False)
        pg.core.setDebug(False)
        pg.setLogLevel(log_level)
    else:
        pg.setVerbose(True)
        pg.setDebug(False)
        pg.core.setDebug(False)


def _extract_window_final_chi2(chi2: np.ndarray) -> float | None:
    if chi2.size == 0:
        return None
    if chi2.ndim == 1:
        return float(chi2[-1])
    return float(np.asarray(chi2[-1]).ravel()[0])


def _build_window_starts(n_steps: int, window_size: int, window_step: int) -> list[int]:
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    if window_size > n_steps:
        raise ValueError(f"window_size={window_size} > n_steps={n_steps}")

    step_stride = max(1, int(window_step))
    starts = list(range(0, n_steps - window_size + 1, step_stride))
    tail_start = n_steps - window_size
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return sorted(set(starts))


def _run_window_inversion_job(
    start_idx: int,
    *,
    data_files: list[str],
    measurement_times: list[float],
    window_size: int,
    inversion_params: dict[str, Any],
    inversion_mesh=None,
    quiet_pygimli: bool,
    pygimli_log_level: str,
    suppress_window_stdout: bool,
) -> dict[str, Any]:
    _configure_pygimli_logging(quiet_pygimli, pygimli_log_level)

    window_files = data_files[start_idx : start_idx + window_size]
    window_times = measurement_times[start_idx : start_idx + window_size]
    mesh_for_window = inversion_mesh
    total_start = time.perf_counter()
    with _maybe_silence_output(suppress_window_stdout):
        setup_start = time.perf_counter()
        inv = TimeLapseERTInversion(
            data_files=window_files,
            measurement_times=window_times,
            mesh=mesh_for_window,
            **inversion_params,
        )
        inv.setup()
        setup_elapsed_sec = float(time.perf_counter() - setup_start)
        run_start = time.perf_counter()
        result = inv.run(initial_model=None)
        run_elapsed_sec = float(time.perf_counter() - run_start)
    elapsed_sec = float(time.perf_counter() - total_start)

    final_models = np.asarray(result.final_models, dtype=float)
    coverage = None
    if len(result.all_coverage) > 0 and result.all_coverage[0] is not None:
        coverage = np.asarray(result.all_coverage[0], dtype=float).ravel()
    chi2 = np.asarray(result.all_chi2, dtype=float)
    iterations = int(chi2.shape[0]) if chi2.ndim > 0 else int(chi2.size > 0)
    return {
        "start_idx": int(start_idx),
        "final_models": final_models,
        "coverage": coverage,
        "chi2": chi2,
        "iterations": int(iterations),
        "setup_elapsed_sec": float(setup_elapsed_sec),
        "run_elapsed_sec": float(run_elapsed_sec),
        "elapsed_sec": float(elapsed_sec),
    }

def _run_windowed_inversion(
    *,
    data_files: list[str],
    measurement_times: list[float],
    steps: np.ndarray,
    inversion_params: dict[str, Any],
    inversion_mesh=None,
    window_size: int,
    window_step: int,
    show_progress: bool,
    quiet_pygimli: bool,
    pygimli_log_level: str,
    suppress_window_stdout: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    n_steps = len(data_files)
    window_starts = _build_window_starts(n_steps, window_size, window_step)

    total_windows = len(window_starts)
    desc = "Window inversion"
    window_results = []
    with _progress_bar(total=total_windows, desc=desc, enabled=bool(show_progress)) as bar:
        for start_idx in window_starts:
            wr = _run_window_inversion_job(
                int(start_idx),
                data_files=data_files,
                measurement_times=measurement_times,
                window_size=int(window_size),
                inversion_params=inversion_params,
                inversion_mesh=inversion_mesh,
                quiet_pygimli=bool(quiet_pygimli),
                pygimli_log_level=str(pygimli_log_level),
                suppress_window_stdout=bool(suppress_window_stdout),
            )
            window_results.append(wr)
            bar.update(1)

    window_results = sorted(window_results, key=lambda r: int(r["start_idx"]))

    first_fm = np.asarray(window_results[0]["final_models"], dtype=float)
    if first_fm.ndim != 2:
        raise ValueError(f"Window {window_results[0]['start_idx']}: final_models not 2D: {first_fm.shape}")
    n_cells = int(first_fm.shape[0])

    contrib_models = [[] for _ in range(n_steps)]
    coverage_bank: list[np.ndarray] = []
    window_final_chi2: list[float] = []
    window_reports: list[dict[str, Any]] = []

    for wr in window_results:
        start_idx = int(wr["start_idx"])
        fm = np.asarray(wr["final_models"], dtype=float)
        if fm.ndim != 2:
            raise ValueError(f"Window {start_idx}: final_models not 2D: {fm.shape}")
        if fm.shape[1] != int(window_size):
            raise ValueError(f"Window {start_idx}: expected width {window_size}, got {fm.shape[1]}")
        if fm.shape[0] != n_cells:
            raise ValueError(f"Window {start_idx}: expected n_cells {n_cells}, got {fm.shape[0]}")

        for local_i in range(fm.shape[1]):
            global_idx = start_idx + local_i
            if 0 <= global_idx < n_steps:
                contrib_models[global_idx].append(fm[:, local_i])

        cov = wr.get("coverage")
        if cov is not None:
            cov = np.asarray(cov, dtype=float).ravel()
            if cov.size == n_cells:
                coverage_bank.append(cov)

        chi = np.asarray(wr.get("chi2", []), dtype=float)
        final_chi = _extract_window_final_chi2(chi)
        if final_chi is not None:
            window_final_chi2.append(final_chi)
        iterations = int(wr.get("iterations", chi.shape[0] if chi.ndim > 0 else int(chi.size > 0)))
        setup_elapsed_sec = float(wr.get("setup_elapsed_sec", np.nan))
        run_elapsed_sec = float(wr.get("run_elapsed_sec", np.nan))
        elapsed_sec = float(wr.get("elapsed_sec", np.nan))
        window_reports.append(
            {
                "start_idx": int(start_idx),
                "end_idx": int(start_idx + window_size - 1),
                "start_step": int(steps[start_idx]),
                "end_step": int(steps[min(start_idx + window_size - 1, len(steps) - 1)]),
                "final_chi2_data": final_chi,
                "iterations": int(iterations),
                "setup_elapsed_sec": setup_elapsed_sec,
                "run_elapsed_sec": run_elapsed_sec,
                "elapsed_sec": elapsed_sec,
            }
        )

    final_cols: list[np.ndarray] = []
    for idx, models in enumerate(contrib_models):
        if len(models) == 0:
            raise ValueError(f"No window contribution for timestep index={idx}, step={int(steps[idx])}")
        stack = np.column_stack(models)
        column = np.exp(np.mean(np.log(np.clip(stack, 1.0e-12, None)), axis=1))
        final_cols.append(column)
    final_models = np.column_stack(final_cols)

    coverage = None
    if coverage_bank:
        coverage = np.nanmedian(np.column_stack(coverage_bank), axis=1)

    chi2_all = np.asarray(window_final_chi2, dtype=float)
    elapsed_values = np.asarray(
        [report["elapsed_sec"] for report in window_reports if np.isfinite(report["elapsed_sec"])],
        dtype=float,
    )
    iteration_values = np.asarray([report["iterations"] for report in window_reports], dtype=float)
    run_meta = {
        "inversion_mode": "windowed",
        "n_windows": int(len(window_starts)),
        "window_size": int(window_size),
        "window_step": int(max(1, window_step)),
        "progress": bool(show_progress),
        "mesh_reused": bool(inversion_mesh is not None),
        "window_elapsed_sec_mean": float(np.mean(elapsed_values)) if elapsed_values.size else None,
        "window_elapsed_sec_median": float(np.median(elapsed_values)) if elapsed_values.size else None,
        "window_elapsed_sec_min": float(np.min(elapsed_values)) if elapsed_values.size else None,
        "window_elapsed_sec_max": float(np.max(elapsed_values)) if elapsed_values.size else None,
        "window_elapsed_sec_sum": float(np.sum(elapsed_values)) if elapsed_values.size else None,
        "window_iterations_mean": float(np.mean(iteration_values)) if iteration_values.size else None,
        "window_iterations_median": float(np.median(iteration_values)) if iteration_values.size else None,
        "window_iterations_min": float(np.min(iteration_values)) if iteration_values.size else None,
        "window_iterations_max": float(np.max(iteration_values)) if iteration_values.size else None,
    }
    return final_models, coverage, chi2_all, run_meta, window_reports


def _build_inversion_mesh_from_data(
    *,
    data_file: str,
    quiet_pygimli: bool,
    pygimli_log_level: str,
    suppress_stdout: bool,
):
    _configure_pygimli_logging(quiet_pygimli, pygimli_log_level)
    with _maybe_silence_output(suppress_stdout):
        # Build the same base mesh used by pyGIMLi's default ERT workflow.
        # Important: do NOT pass paraDomain as the reusable inversion mesh.
        # Reusing paraDomain can change effective parameterization (e.g., 910 vs 911 cells)
        # and produce strongly biased time-lapse solutions.
        data = ert.load(str(data_file))
        manager = ert.ERTManager(data)
        mesh = manager.createMesh(data=data, quality=34)
    return mesh


def _positive_log_limits(*arrays: np.ndarray) -> tuple[float, float]:
    values = np.concatenate(
        [
            np.asarray(array, dtype=float).ravel()
            for array in arrays
            if np.asarray(array, dtype=float).size
        ]
    )
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        raise ValueError("plot values must contain at least one finite positive value")
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if math.isclose(vmin, vmax):
        vmax = vmin * 1.01
    return vmin, vmax


def _plot_chi2(path: Path, chi2: np.ndarray, *, windowed: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chi2_arr = np.asarray(chi2, dtype=float)
    if chi2_arr.size == 0:
        return
    if chi2_arr.ndim == 1:
        chi2_data = chi2_arr
    else:
        chi2_data = np.asarray(chi2_arr[:, 0], dtype=float).ravel()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, chi2_data.size + 1), chi2_data, marker="o")
    ax.set_xlabel("Window Index" if windowed else "Iteration")
    ax.set_ylabel("Chi2 (data term)")
    ax.set_yscale("log")
    ax.set_title("Time-Lapse Inversion Convergence")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_true_vs_inverted(
    path: Path,
    *,
    mesh,
    final_models: np.ndarray,
    steps: np.ndarray,
    true_model_dir: Path,
    y_index: int,
    geometry: dict[str, np.ndarray],
    coverage_mask: np.ndarray | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm

    mid_idx = int(steps.size // 2)
    mid_step = int(steps[mid_idx])
    true_file = true_model_dir / f"resistivity2d_y{y_index}_t{mid_step:05d}.npy"
    if not true_file.exists():
        return

    rho_true = np.asarray(np.load(true_file), dtype=float)
    rho_true_plot = rho_true[::-1, :]
    inv_model = np.asarray(final_models[:, mid_idx], dtype=float).ravel()
    if coverage_mask is not None and coverage_mask.shape == inv_model.shape:
        inv_plot = np.ma.array(inv_model, mask=coverage_mask)
        inv_valid = inv_model[~coverage_mask] if np.any(~coverage_mask) else inv_model
    else:
        inv_plot = inv_model
        inv_valid = inv_model

    x_nodes = geometry["x_nodes"]
    z_top = geometry["z_top"]
    layer_thickness = geometry["layer_thickness"]
    cum = np.concatenate(([0.0], np.cumsum(layer_thickness)))
    x_grid = np.tile(x_nodes, (cum.size, 1))
    z_grid = z_top[None, :] - cum[:, None]

    vmin, vmax = _positive_log_limits(rho_true_plot, inv_valid)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = "turbo"

    fig = plt.figure(figsize=(10.0, 8.6), constrained_layout=True)
    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1.0, 0.045],
        height_ratios=[1.0, 1.0],
        wspace=0.06,
        hspace=0.08,
    )
    ax_true = fig.add_subplot(grid[0, 0])
    ax_inv = fig.add_subplot(grid[1, 0], sharex=ax_true, sharey=ax_true)
    cax = fig.add_subplot(grid[:, 1])

    ax_true.pcolormesh(x_grid, z_grid, rho_true_plot, shading="auto", cmap=cmap, norm=norm)
    ax_true.set_title(f"True Model (t{mid_step:05d})")

    pg.show(
        mesh,
        data=inv_plot,
        ax=ax_inv,
        cMap=cmap,
        colorBar=False,
        logScale=True,
        cMin=vmin,
        cMax=vmax,
    )
    ax_inv.set_title(f"Masked Inverted Model (t{mid_step:05d})")

    y_bottom = float(np.min(z_top - np.sum(layer_thickness)))
    y_top = float(np.max(z_top))
    x_min = float(np.min(x_nodes))
    x_max = float(np.max(x_nodes))
    box_aspect = abs((y_top - y_bottom) / (x_max - x_min)) if x_max > x_min else 1.0
    for ax in (ax_true, ax_inv):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_bottom, y_top)
        ax.set_box_aspect(box_aspect)
        ax.set_ylabel("Elevation (m)")
    ax_true.tick_params(labelbottom=False)
    ax_inv.set_xlabel("X (m)")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(sm, cax=cax)
    colorbar.set_label("Resistivity (ohm-m)")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _predict_rhoa(
    *,
    mesh,
    data_files: list[str],
    final_models: np.ndarray,
    quiet_pygimli: bool,
    pygimli_log_level: str,
    suppress_stdout: bool,
) -> np.ndarray:
    from pygimli.physics import ert

    _configure_pygimli_logging(quiet_pygimli, pygimli_log_level)
    predictions: list[np.ndarray] = []
    with _maybe_silence_output(suppress_stdout):
        for idx, data_file in enumerate(data_files):
            data = ert.load(str(data_file))
            fwd = ert.ERTModelling()
            fwd.setData(data)
            fwd.setMesh(mesh)
            response = np.asarray(fwd.response(pg.Vector(final_models[:, idx])), dtype=float).ravel()
            predictions.append(response)
    return np.vstack(predictions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None, help="Repository root. Auto-detected by default.")
    parser.add_argument("--forward-dir", default="result/1_timelapsedERT_forward_deepert")
    parser.add_argument("--true-model-dir", default="resistivity_models_2d")
    parser.add_argument("--output-dir", default="result/2_timelapsedERT_inversion_pygimli")
    parser.add_argument("--y-index", type=int, default=2)
    parser.add_argument("--file-stride", type=int, default=1)
    parser.add_argument("--max-timesteps", type=int, default=None)
    parser.add_argument("--inversion-mode", choices=("windowed", "full"), default="windowed")
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--window-step", type=int, default=1)
    parser.add_argument(
        "--suppress-window-stdout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress verbose stdout/stderr from each window inversion.",
    )
    parser.add_argument("--regularization", type=float, default=50.0)
    parser.add_argument("--temporal-regularization", type=float, default=10.0)
    parser.add_argument("--lambda-rate", type=float, default=1.0)
    parser.add_argument("--lambda-min", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--model-min", type=float, default=0.001)
    parser.add_argument("--model-max", type=float, default=1.0e4)
    parser.add_argument("--relative-error", type=float, default=0.05)
    parser.add_argument("--method", default="cgls")
    parser.add_argument("--coverage-percentile", type=float, default=20.0)
    parser.add_argument("--quiet-pygimli", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pygimli-log-level", default="WARNING")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress bar and periodic elapsed logs.",
    )
    parser.add_argument(
        "--reuse-mesh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse one inversion mesh across all windows for fair speed benchmarking.",
    )
    parser.add_argument("--save-predicted", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parents[2]
    forward_dir = _resolve(root, args.forward_dir)
    true_model_dir = _resolve(root, args.true_model_dir)
    output_dir = _resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_pygimli_logging(bool(args.quiet_pygimli), str(args.pygimli_log_level))
    pairs = _discover_forward_dat(
        forward_dir,
        file_stride=int(args.file_stride),
        max_timesteps=args.max_timesteps,
    )
    steps = np.asarray([step for step, _ in pairs], dtype=np.int32)
    data_files = [str(path) for _, path in pairs]
    measurement_times = (np.asarray(steps, dtype=float) / 24.0).tolist()

    inversion_params = {
        "lambda_val": float(args.regularization),
        "alpha": float(args.temporal_regularization),
        "lambda_rate": float(args.lambda_rate),
        "lambda_min": float(args.lambda_min),
        "max_iterations": int(args.max_iterations),
        "model_constraints": (float(args.model_min), float(args.model_max)),
        "relativeError": float(args.relative_error),
        "method": str(args.method),
    }

    reuse_mesh_effective = bool(args.reuse_mesh)

    shared_mesh = None
    mesh_build_elapsed_sec = None
    if reuse_mesh_effective:
        mesh_build_start = time.perf_counter()
        shared_mesh = _build_inversion_mesh_from_data(
            data_file=data_files[0],
            quiet_pygimli=bool(args.quiet_pygimli),
            pygimli_log_level=str(args.pygimli_log_level),
            suppress_stdout=bool(args.suppress_window_stdout),
        )
        mesh_build_elapsed_sec = float(time.perf_counter() - mesh_build_start)

    run_start = time.perf_counter()
    window_reports: list[dict[str, Any]] = []
    if args.inversion_mode == "full":
        with _elapsed_reporter(label="full inversion", enabled=bool(args.progress)):
            with _maybe_silence_output(bool(args.suppress_window_stdout)):
                inv = TimeLapseERTInversion(
                    data_files=data_files,
                    measurement_times=measurement_times,
                    mesh=shared_mesh,
                    **inversion_params,
                )
                inv.setup()
                result = inv.run(initial_model=None)

        final_models = np.asarray(result.final_models, dtype=float)
        coverage = None
        if len(result.all_coverage) > 0 and result.all_coverage[0] is not None:
            coverage = np.asarray(result.all_coverage[0], dtype=float).ravel()
        chi2_all = np.asarray(result.all_chi2, dtype=float)
        mesh_for_plot = shared_mesh if shared_mesh is not None else result.mesh
        run_meta = {
            "inversion_mode": "full",
            "n_windows": 1,
            "window_size": None,
            "window_step": None,
            "progress": bool(args.progress),
            "mesh_reused": bool(shared_mesh is not None),
            "mesh_build_elapsed_sec": mesh_build_elapsed_sec,
        }
    else:
        final_models, coverage, chi2_all, run_meta, window_reports = _run_windowed_inversion(
            data_files=data_files,
            measurement_times=measurement_times,
            steps=steps,
            inversion_params=inversion_params,
            inversion_mesh=shared_mesh,
            window_size=int(args.window_size),
            window_step=int(args.window_step),
            show_progress=bool(args.progress),
            quiet_pygimli=bool(args.quiet_pygimli),
            pygimli_log_level=str(args.pygimli_log_level),
            suppress_window_stdout=bool(args.suppress_window_stdout),
        )
        mesh_for_plot = shared_mesh
        if mesh_for_plot is None:
            mesh_for_plot = _build_inversion_mesh_from_data(
                data_file=data_files[len(data_files) // 2],
                quiet_pygimli=bool(args.quiet_pygimli),
                pygimli_log_level=str(args.pygimli_log_level),
                suppress_stdout=bool(args.suppress_window_stdout),
            )
        run_meta["mesh_build_elapsed_sec"] = mesh_build_elapsed_sec

    elapsed_sec = float(time.perf_counter() - run_start)

    coverage_threshold = None
    coverage_mask = None
    if coverage is not None:
        coverage = np.asarray(coverage, dtype=float).ravel()
        coverage_threshold = float(np.percentile(coverage, float(args.coverage_percentile)))
        coverage_mask = coverage < coverage_threshold

    np.save(output_dir / "final_models.npy", np.asarray(final_models, dtype=float))
    np.save(output_dir / "final_log_models.npy", np.log(np.clip(np.asarray(final_models, dtype=float), 1.0e-12, None)))
    np.save(output_dir / "steps.npy", steps)
    np.save(output_dir / "measurement_times_days.npy", np.asarray(measurement_times, dtype=float))
    np.save(output_dir / "chi2_all.npy", np.asarray(chi2_all, dtype=float))
    if coverage is not None:
        np.save(output_dir / "coverage.npy", coverage)
    if coverage_mask is not None:
        np.save(output_dir / "coverage_mask.npy", coverage_mask.astype(np.uint8))

    mesh_file = output_dir / "timelapse_inversion_mesh.bms"
    if mesh_for_plot is not None:
        mesh_for_plot.save(str(mesh_file))

    if args.save_predicted and mesh_for_plot is not None:
        predicted = _predict_rhoa(
            mesh=mesh_for_plot,
            data_files=data_files,
            final_models=np.asarray(final_models, dtype=float),
            quiet_pygimli=bool(args.quiet_pygimli),
            pygimli_log_level=str(args.pygimli_log_level),
            suppress_stdout=bool(args.suppress_window_stdout),
        )
        np.save(output_dir / "predicted_rhoa.npy", predicted)

    for col, step in enumerate(steps):
        model = np.asarray(final_models[:, col], dtype=float).ravel()
        np.save(output_dir / f"inverted_model_t{int(step):05d}.npy", model)
        if coverage_mask is not None:
            masked = model.copy()
            masked[coverage_mask] = np.nan
            np.save(output_dir / f"inverted_model_masked_nan_t{int(step):05d}.npy", masked)

    with (output_dir / "used_data_files.txt").open("w", encoding="utf-8") as stream:
        for path in data_files:
            stream.write(str(path) + "\n")

    if window_reports:
        _write_json(output_dir / "window_reports.json", window_reports)

    plot_files: dict[str, str] = {}
    if not args.no_plot:
        chi2_plot = output_dir / "timelapse_chi2.png"
        _plot_chi2(chi2_plot, np.asarray(chi2_all, dtype=float), windowed=args.inversion_mode == "windowed")
        if chi2_plot.exists():
            plot_files["chi2"] = str(chi2_plot)

        geometry_file = forward_dir / "forward_geometry.npz"
        if geometry_file.exists() and mesh_for_plot is not None:
            geometry = _load_geometry(geometry_file)
            mid_plot = output_dir / f"true_vs_inverted_mid_t{int(steps[len(steps) // 2]):05d}.png"
            _plot_true_vs_inverted(
                mid_plot,
                mesh=mesh_for_plot,
                final_models=np.asarray(final_models, dtype=float),
                steps=steps,
                true_model_dir=true_model_dir,
                y_index=int(args.y_index),
                geometry=geometry,
                coverage_mask=coverage_mask,
            )
            if mid_plot.exists():
                plot_files["true_vs_inverted_mid"] = str(mid_plot)

    chi2_final_data = _extract_window_final_chi2(np.asarray(chi2_all, dtype=float))
    summary = {
        "engine": "pygimli",
        "inversion_backend": "PyHydroGeophysX.TimeLapseERTInversion",
        "n_timesteps": int(steps.size),
        "first_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "n_cells": int(np.asarray(final_models).shape[0]),
        "forward_dir": str(forward_dir),
        "output_dir": str(output_dir),
        "mesh_file": str(mesh_file) if mesh_file.exists() else None,
        "lambda_val": float(args.regularization),
        "alpha": float(args.temporal_regularization),
        "lambda_rate": float(args.lambda_rate),
        "lambda_min": float(args.lambda_min),
        "max_iterations": int(args.max_iterations),
        "model_constraints": [float(args.model_min), float(args.model_max)],
        "relative_error": float(args.relative_error),
        "method": str(args.method),
        "reuse_mesh": bool(reuse_mesh_effective),
        "coverage_percentile": float(args.coverage_percentile),
        "coverage_threshold": coverage_threshold,
        "chi2_final_data": chi2_final_data,
        "elapsed_sec": elapsed_sec,
        "elapsed_min": float(elapsed_sec / 60.0),
        "plot_files": plot_files,
        "save_predicted": bool(args.save_predicted),
    }
    summary.update(run_meta)
    _write_json(output_dir / "timelapsed_inversion_summary.json", summary)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
