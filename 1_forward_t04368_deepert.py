#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("DEEPERT_ENABLE_FLOAT64", "1")

from deepert.utils.torch_runtime import torch_runtime
import numpy as np

torch_runtime.config.update("torch_enable_float64", True)

from deepert.workflows import (
    build_terrain_forward_case,
    parse_pftcl,
    parse_resistivity_slice_name,
    read_slope_x,
    run_terrain_forward,
    save_terrain_forward_dat,
    save_terrain_forward_npz,
)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _write_summary(path: Path, summary: dict[str, object]) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _positive_log_limits(values: np.ndarray) -> tuple[float, float]:
    positive = np.asarray(values, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        raise ValueError("plot values must contain at least one finite positive value")
    vmin = float(np.min(positive))
    vmax = float(np.max(positive))
    if vmax <= vmin:
        vmax = float(vmin * 1.01)
    return vmin, vmax


def _log_ticks(vmin: float, vmax: float, count: int = 5) -> np.ndarray:
    if count < 2:
        return np.asarray([vmin, vmax], dtype=float)
    return np.geomspace(vmin, vmax, count)


def _format_tick(value: float) -> str:
    if abs(value) >= 1.0e4 or (0.0 < abs(value) < 1.0e-2):
        return f"{value:.1e}"
    if abs(value) >= 100.0:
        return f"{value:.0f}"
    return f"{value:g}"


def _mesh_edge_segments(case) -> np.ndarray:
    nodes = np.asarray(case.mesh.nodes, dtype=float)
    cells = np.asarray(case.mesh.cells, dtype=np.int32)
    edges: set[tuple[int, int]] = set()
    for cell in cells:
        for node_id, next_node_id in zip(cell, np.roll(cell, -1), strict=False):
            start = int(node_id)
            stop = int(next_node_id)
            edges.add((start, stop) if start < stop else (stop, start))
    return np.asarray([[nodes[start], nodes[stop]] for start, stop in sorted(edges)], dtype=float)


def _set_terrain_limits(ax, case) -> None:
    nodes = np.asarray(case.mesh.nodes, dtype=float)
    x_min = float(np.min(case.x_nodes))
    x_max = float(np.max(case.x_nodes))
    z_min = float(np.min(nodes[:, 1]))
    z_max = float(np.max(nodes[:, 1]))
    z_pad = max((z_max - z_min) * 0.03, 1.0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min - z_pad, z_max + z_pad)


def _plot_mesh_edges(path: Path, case, step: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.add_collection(LineCollection(_mesh_edge_segments(case), colors="0.5", linewidths=0.6))
    ax.plot(case.x_nodes, case.z_top, "k-", lw=2, label="Topography")
    ax.set_title("X-Z Cross Section with Cell Edges (Terrain + Variable Dz)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Elevation (m)")
    ax.legend()
    ax.grid(alpha=0.25)
    _set_terrain_limits(ax, case)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_resistivity_model(path: Path, case, step: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LogNorm

    nodes = np.asarray(case.mesh.nodes, dtype=float)
    cells = np.asarray(case.mesh.cells, dtype=np.int32)
    resistivity = np.asarray(case.resistivity, dtype=float)
    vmin, vmax = _positive_log_limits(resistivity)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    mesh = PolyCollection(
        nodes[cells],
        array=resistivity,
        cmap="turbo",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        edgecolors=(1.0, 1.0, 1.0, 0.25),
        linewidths=0.25,
    )
    ax.add_collection(mesh)
    ax.plot(case.x_nodes, case.z_top, "k-", lw=2.0, label="Topography")
    ax.scatter(
        case.elec_x,
        case.elec_z,
        s=12,
        c="white",
        edgecolors="k",
        linewidths=0.4,
        zorder=5,
        label="Electrodes",
    )
    ax.set_title(f"Resistivity Model (terrain + variable Dz, step={step})")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Elevation (m)")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    _set_terrain_limits(ax, case)

    colorbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.18)
    ticks = _log_ticks(vmin, vmax)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([_format_tick(tick) for tick in ticks])
    colorbar.set_label("Resistivity (ohm-m)")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_pseudosection(path: Path, case, rhoa: np.ndarray, step: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle

    rhoa_array = np.asarray(rhoa, dtype=float).ravel()
    measurements = np.asarray(case.survey.measurements, dtype=np.int32)
    if rhoa_array.shape != (measurements.shape[0],):
        raise ValueError(f"rhoa must have shape ({measurements.shape[0]},)")

    electrode_x = np.asarray(case.elec_x, dtype=float)
    electrode_spacing = float(np.median(np.diff(electrode_x))) if electrode_x.size > 1 else 1.0
    spacing_ids = measurements[:, 2] - measurements[:, 0]
    x_centers = np.mean(electrode_x[measurements], axis=1)
    max_spacing = int(np.max(spacing_ids))
    vmin, vmax = _positive_log_limits(rhoa_array)

    patches = [
        Rectangle(
            (float(x_center) - 0.51 * electrode_spacing, float(spacing_id) - 0.51),
            1.02 * electrode_spacing,
            1.02,
        )
        for x_center, spacing_id in zip(x_centers, spacing_ids, strict=True)
    ]

    fig, ax = plt.subplots(figsize=(8, 3))
    collection = PatchCollection(
        patches,
        array=rhoa_array,
        cmap="Spectral_r",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        linewidths=0,
        edgecolors="none",
        antialiaseds=False,
    )
    ax.add_collection(collection)
    ax.set_title(f"Synthetic ERT Pseudosection (terrain+variableDz, step={step})")
    ax.set_xlim(
        float(np.min(x_centers) - 0.5 * electrode_spacing),
        float(np.max(x_centers) + 0.5 * electrode_spacing),
    )
    ax.set_ylim(max_spacing + 0.5, 0.5)
    tick_spacing = [spacing for spacing in (1, 6, 11, 16) if spacing <= max_spacing]
    ax.set_yticks(tick_spacing)
    ax.set_yticklabels([f"WA {spacing}" for spacing in tick_spacing])
    ax.grid(axis="x", color="white", linestyle="--", linewidth=0.8, alpha=0.8)

    colorbar = fig.colorbar(collection, ax=ax, orientation="horizontal", pad=0.28)
    ticks = _log_ticks(vmin, vmax)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([_format_tick(tick) for tick in ticks])
    colorbar.set_label("Apparent resistivity (ohm-m)")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_case(output_dir: Path, case, rhoa: np.ndarray, step: int) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {
        "mesh_edges": output_dir / f"mesh_edges_t{step:05d}.png",
        "resistivity_model": output_dir / f"forward_preview_t{step:05d}.png",
        "rhoa_pseudosection": output_dir / f"rhoa_pseudosection_t{step:05d}.png",
    }
    _plot_mesh_edges(plot_paths["mesh_edges"], case, step)
    _plot_resistivity_model(plot_paths["resistivity_model"], case, step)
    _plot_pseudosection(plot_paths["rhoa_pseudosection"], case, rhoa, step)
    return plot_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None, help="Repository root. Defaults to this script directory.")
    parser.add_argument(
        "--input-file",
        default="2d_resistivity_model/resistivity2d_y2_t04368.npy",
        help="2D ParFlow resistivity slice in bottom-to-top z ordering.",
    )
    parser.add_argument("--model-dir", default="models_1year_1day", help="Directory containing pftcl and slope_x files.")
    parser.add_argument("--output-dir", default="result/deepert_single_forward_t04368")
    parser.add_argument("--n-electrodes", type=int, default=48)
    parser.add_argument("--relative-error", type=float, default=0.03)
    parser.add_argument("--topo-offset", type=float, default=0.0)
    parser.add_argument("--linear-solver-backend", default="auto")
    parser.add_argument("--terrain-cache-dir", default=None)
    parser.add_argument("--no-plot", action="store_true", help="Do not save the PNG preview.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).resolve() if args.project_root else Path(__file__).resolve().parent
    input_file = _resolve(root, args.input_file)
    model_dir = _resolve(root, args.model_dir)
    output_dir = _resolve(root, args.output_dir)
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
        linear_solver_backend=args.linear_solver_backend,
        terrain_cache_dir=None if args.terrain_cache_dir is None else _resolve(root, args.terrain_cache_dir),
    )

    dat_file = output_dir / f"synthetic_ert_terrain_vardz_t{step:05d}.dat"
    npz_file = output_dir / f"synthetic_ert_terrain_vardz_t{step:05d}.npz"
    save_terrain_forward_dat(dat_file, case, rhoa, relative_error=args.relative_error)
    save_terrain_forward_npz(npz_file, case, rhoa, relative_error=args.relative_error)

    plot_paths: dict[str, Path] = {}
    if not args.no_plot:
        plot_paths = _plot_case(output_dir, case, rhoa, step)

    summary = {
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
        "n_electrodes": int(len(case.elec_x)),
        "relative_error": float(args.relative_error),
        "plot_files": {name: str(path) for name, path in plot_paths.items()},
    }
    _write_summary(output_dir / "forward_summary.json", summary)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
