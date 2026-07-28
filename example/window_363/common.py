"""Shared helpers for the Deepert/ResIPy benchmark."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


FORWARD_PATTERN = re.compile(r"synthetic_ert_terrain_vardz_t(\d+)\.npz$")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def discover_forward_files(forward_dir: Path) -> dict[int, Path]:
    pairs: dict[int, Path] = {}
    for path in sorted(forward_dir.glob("synthetic_ert_terrain_vardz_t*.npz")):
        match = FORWARD_PATTERN.fullmatch(path.name)
        if match is not None:
            pairs[int(match.group(1))] = path
    if not pairs:
        raise FileNotFoundError(f"No time-lapse forward files found in {forward_dir}")
    return pairs


def select_steps(
    available: dict[int, Path],
    *,
    steps: list[int] | None,
    stride: int,
    max_timesteps: int | None,
) -> list[int]:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if steps:
        missing = sorted(set(steps).difference(available))
        if missing:
            raise FileNotFoundError(f"Missing requested forward steps: {missing}")
        selected = list(dict.fromkeys(int(step) for step in steps))
    else:
        selected = sorted(available)[::stride]
    if max_timesteps is not None:
        selected = selected[:max_timesteps]
    if not selected:
        raise ValueError("No forward steps selected")
    return selected


def load_forward_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        required = {"rhoa", "a", "b", "m", "n", "elec_x", "elec_z"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{path} missing required arrays: {sorted(missing)}")
        loaded = {
            "rhoa": np.asarray(data["rhoa"], dtype=float).ravel(),
            "a": np.asarray(data["a"], dtype=np.int32).ravel(),
            "b": np.asarray(data["b"], dtype=np.int32).ravel(),
            "m": np.asarray(data["m"], dtype=np.int32).ravel(),
            "n": np.asarray(data["n"], dtype=np.int32).ravel(),
            "elec_x": np.asarray(data["elec_x"], dtype=float).ravel(),
            "elec_z": np.asarray(data["elec_z"], dtype=float).ravel(),
        }
        if "err" in data.files:
            loaded["err"] = np.asarray(data["err"], dtype=float).ravel()
        return loaded


def resipy_parser(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a Deepert forward NPZ through ResIPy's custom-parser interface."""

    data = load_forward_npz(Path(path))
    count = data["elec_x"].size
    elec = pd.DataFrame(
        {
            "x": data["elec_x"],
            "y": np.zeros(count, dtype=float),
            "z": data["elec_z"],
            "buried": np.zeros(count, dtype=bool),
            "remote": np.zeros(count, dtype=bool),
            "label": np.arange(1, count + 1).astype(str),
        }
    )
    measurements = pd.DataFrame(
        {
            # Deepert NPZ files store zero-based electrode indices; ResIPy
            # labels are one-based strings.
            "a": (data["a"] + 1).astype(str),
            "b": (data["b"] + 1).astype(str),
            "m": (data["m"] + 1).astype(str),
            "n": (data["n"] + 1).astype(str),
            "app": data["rhoa"],
            "ip": np.zeros(data["rhoa"].size, dtype=float),
        }
    )
    return elec, measurements


def resistivity_column(columns: list[str]) -> str:
    candidates = (
        "Resistivity",
        "Resistivity(Ohm-m)",
        "Resistivity(ohm.m)",
        "Magnitude(ohm.m)",
    )
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(f"No resistivity column in {columns}")


def grid_to_points(
    grid: np.ndarray,
    *,
    x: np.ndarray,
    z: np.ndarray,
    x_nodes: np.ndarray,
    z_top: np.ndarray,
    layer_thickness: np.ndarray,
) -> np.ndarray:
    """Nearest-cell mapping from the terrain-following truth grid."""

    values = np.asarray(grid, dtype=float)
    expected = (layer_thickness.size, x_nodes.size - 1)
    if values.shape != expected:
        raise ValueError(f"Truth grid shape {values.shape} does not match {expected}")
    column = np.searchsorted(x_nodes, x, side="right") - 1
    column = np.clip(column, 0, x_nodes.size - 2)
    surface = np.interp(x, x_nodes, z_top)
    depth = np.maximum(surface - z, 0.0)
    layer = np.searchsorted(np.cumsum(layer_thickness), depth, side="right")
    layer = np.clip(layer, 0, layer_thickness.size - 1)
    return values[::-1, :][layer, column]


def model_metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    valid = np.isfinite(estimate) & np.isfinite(truth) & (estimate > 0) & (truth > 0)
    if not np.any(valid):
        raise ValueError("No finite positive model values for metrics")
    estimate = estimate[valid]
    truth = truth[valid]
    log_error = np.log(estimate) - np.log(truth)
    return {
        "n": int(estimate.size),
        "log_rmse": float(np.sqrt(np.mean(log_error**2))),
        "log_mae": float(np.mean(np.abs(log_error))),
        "relative_l2": float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth)),
        "median_absolute_percent_error": float(
            100.0 * np.median(np.abs(estimate - truth) / np.maximum(np.abs(truth), 1.0e-12))
        ),
        "log_correlation": float(np.corrcoef(np.log(estimate), np.log(truth))[0, 1]),
    }
