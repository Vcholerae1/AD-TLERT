"""Figures shared by the single-time forward examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np


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
    return np.asarray(
        [[nodes[start], nodes[stop]] for start, stop in sorted(edges)], dtype=float
    )


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
    ax.add_collection(
        LineCollection(_mesh_edge_segments(case), colors="0.5", linewidths=0.6)
    )
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


def _plot_pseudosection(
    path: Path, case, rhoa: np.ndarray, step: int, label: str
) -> None:
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
    electrode_spacing = (
        float(np.median(np.diff(electrode_x))) if electrode_x.size > 1 else 1.0
    )
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
    ax.set_title(f"Synthetic ERT Pseudosection ({label}, step={step})")
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


def plot_case(
    output_dir: Path, case, rhoa: np.ndarray, step: int, label: str
) -> dict[str, Path]:
    """Save mesh, resistivity-model, and pseudosection figures; returns their paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {
        "mesh_edges": output_dir / "mesh_edges.png",
        "resistivity_model": output_dir / "forward_preview.png",
        "rhoa_pseudosection": output_dir / "rhoa_pseudosection.png",
    }
    _plot_mesh_edges(plot_paths["mesh_edges"], case, step)
    _plot_resistivity_model(plot_paths["resistivity_model"], case, step)
    _plot_pseudosection(plot_paths["rhoa_pseudosection"], case, rhoa, step, label)
    return plot_paths
