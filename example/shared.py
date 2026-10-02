"""Helpers shared by the ADTLERT example scripts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def grid2d_to_mesh_cells(values_2d: np.ndarray, mesh, geometry: dict[str, np.ndarray]) -> np.ndarray:
    """Map a terrain-following ParFlow 2D grid onto inversion mesh cell centers."""

    grid = np.asarray(values_2d)
    x_nodes = np.asarray(geometry["x_nodes"], dtype=float).ravel()
    z_top = np.asarray(geometry["z_top"], dtype=float).ravel()
    layer_thickness = np.asarray(geometry["layer_thickness"], dtype=float).ravel()
    expected_shape = (layer_thickness.size, x_nodes.size - 1)
    if grid.shape != expected_shape:
        raise ValueError(f"2D grid shape {grid.shape} does not match expected {expected_shape}")

    nodes = np.asarray(mesh.nodes, dtype=float)
    cells = np.asarray(mesh.cells, dtype=np.int32)
    centers = nodes[cells].mean(axis=1)
    x_center = centers[:, 0]
    z_center = centers[:, 1]

    column = np.searchsorted(x_nodes, x_center, side="right") - 1
    column = np.clip(column, 0, x_nodes.size - 2)
    surface_z = np.interp(x_center, x_nodes, z_top)
    depth = np.maximum(surface_z - z_center, 0.0)
    layer_top_to_bottom = np.searchsorted(np.cumsum(layer_thickness), depth, side="right")
    layer_top_to_bottom = np.clip(layer_top_to_bottom, 0, layer_thickness.size - 1)

    grid_top_to_bottom = grid[::-1, :]
    return np.asarray(grid_top_to_bottom[layer_top_to_bottom, column])


def load_petrophysical_parameters(
    parameter_dir: Path,
    *,
    y_index: int,
    preset: str,
    mesh,
    geometry: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Load and map saved 2D petrophysical parameter grids onto inversion cells."""

    files = {
        "rho_sat": parameter_dir / f"rho_sat2d_y{y_index}_base_{preset}.npy",
        "rho_sat_s": parameter_dir / f"rho_sat_s2d_y{y_index}_base_{preset}.npy",
        "n": parameter_dir / f"n2d_y{y_index}_base_{preset}.npy",
        "phi": parameter_dir / f"phi2d_y{y_index}_base_{preset}.npy",
    }
    missing_required = [str(files[name]) for name in ("rho_sat", "n") if not files[name].exists()]
    if missing_required:
        raise FileNotFoundError(f"Missing petrophysical parameter files: {missing_required}")

    mapped: dict[str, np.ndarray] = {}
    for name, path in files.items():
        if not path.exists():
            continue
        mapped[name] = np.asarray(grid2d_to_mesh_cells(np.load(path), mesh, geometry), dtype=float)
    return mapped
