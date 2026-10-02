"""Shared experiment plumbing for the flat-model time-lapse INR notebooks.

The notebooks keep the scientific steps visible, while this module centralizes
data validation, output naming, and comparison plots so the three methods
cannot silently drift apart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from adtlert.inr.coords import cell_centers, spatiotemporal_coordinates
from adtlert.inversion import ParameterizedERTForward2p5D
from adtlert.workflows import build_source_position_triangle_inversion_case
from example.window_363.inversion_adtlert import (
    _build_forward_parameterization,
    _discover_forward_dat,
    _load_forward_series,
    _load_geometry,
    _save_mesh_npz,
)


@dataclass(frozen=True)
class FlatTimelapseData:
    steps: np.ndarray
    geometry: dict[str, np.ndarray]
    observed_rhoa: np.ndarray
    measurements: np.ndarray
    elec_x: np.ndarray
    elec_z: np.ndarray
    err: np.ndarray
    data_std: np.ndarray
    data_files: tuple[Path, ...]
    truth_models: np.ndarray
    metadata: dict[str, Any]
    truth_anomaly_curve: np.ndarray


@dataclass
class INRSnapshotRecorder:
    """Persist periodic INR models in the same cell-by-time orientation as final_models.npy."""

    directory: Path
    records: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: dict[str, Any]) -> Path | None:
        if event.get("event") != "snapshot":
            return None
        iteration = int(event["iteration"])
        resistivity = np.exp(np.asarray(event["log_resistivity"], dtype=np.float64)).T
        path = self.directory / f"resistivity_iter_{iteration:04d}.npy"
        np.save(path, resistivity)
        self.records.append(
            {
                "iteration": iteration,
                "chi2": event.get("chi2"),
                "rms": event.get("rms"),
                "learning_rate": float(event["learning_rate"]),
                "file": path.name,
            }
        )
        return path

    def save_summary(self) -> Path:
        path = self.directory / "snapshot_summary.json"
        path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")
        return path


@dataclass(frozen=True)
class INRForwardBundle:
    case: Any
    forward: ParameterizedERTForward2p5D
    forward_mesh: Any
    parameter_cell_ids: np.ndarray
    forward_cell_parameter_ids: np.ndarray | None
    forward_parent_cell_ids: np.ndarray | None


def load_flat_timelapse_data(forward_dir: Path, truth_dir: Path) -> FlatTimelapseData:
    """Load the 49 DAT surveys and their matching structured-grid truth."""

    pairs = _discover_forward_dat(forward_dir, file_stride=1, max_timesteps=None, selected_steps=None)
    steps = np.asarray([step for step, _ in pairs], dtype=np.int32)
    geometry = _load_geometry(forward_dir / "forward_geometry.npz")
    observed, measurements, elec_x, elec_z, err, data_files, _ = _load_forward_series(
        pairs,
        forward_format="dat",
    )
    truth_models = np.asarray(np.load(truth_dir / "resistivity.npy", allow_pickle=False), dtype=np.float64)
    metadata = json.loads((truth_dir / "metadata.json").read_text(encoding="utf-8"))
    if err is None:
        raise ValueError("time-lapse INR experiments require per-datum relative errors")
    if observed.shape != (steps.size, measurements.shape[0]):
        raise ValueError("forward DAT files do not form a consistent time series")
    if truth_models.shape[0] != steps.size:
        raise ValueError("truth and observed time series have different lengths")
    if not np.all(np.isfinite(observed)) or np.any(observed <= 0.0):
        raise ValueError("observed apparent resistivities must be finite and positive")

    x_nodes = geometry["x_nodes"]
    layer_thickness = geometry["layer_thickness"]
    x_centers = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    depth_edges = np.concatenate(([0.0], np.cumsum(layer_thickness)))
    depth_centers = 0.5 * (depth_edges[:-1] + depth_edges[1:])
    bounds = metadata["anomaly_bounds"]
    truth_mask = (
        (x_centers[None, :] >= bounds["x_min"])
        & (x_centers[None, :] <= bounds["x_max"])
        & (depth_centers[:, None] >= bounds["depth_min"])
        & (depth_centers[:, None] <= bounds["depth_max"])
    )
    truth_curve = np.median(truth_models[:, truth_mask], axis=1)
    return FlatTimelapseData(
        steps=steps,
        geometry=geometry,
        observed_rhoa=np.asarray(observed, dtype=np.float64),
        measurements=np.asarray(measurements, dtype=np.int32),
        elec_x=np.asarray(elec_x, dtype=np.float64),
        elec_z=np.asarray(elec_z, dtype=np.float64),
        err=np.asarray(err, dtype=np.float64),
        data_std=np.log1p(np.asarray(err, dtype=np.float64)),
        data_files=tuple(Path(path) for path in data_files),
        truth_models=truth_models,
        metadata=metadata,
        truth_anomaly_curve=np.asarray(truth_curve, dtype=np.float64),
    )


def build_inr_forward(
    data: FlatTimelapseData,
    *,
    mesh_quality: float = 34.0,
    mesh_max_cell_area: float = 100.0,
    mesh_smoothing_iterations: int = 10,
    forward_refinement: str = "native",
    linear_solver_backend: str = "cudss",
    normal_field_cache_max_entries: int = 8,
) -> INRForwardBundle:
    """Build the shared independent inversion mesh and GPU forward facade."""

    case = build_source_position_triangle_inversion_case(
        data.elec_x,
        data.elec_z,
        data.measurements,
        data.geometry["x_nodes"],
        data.geometry["z_top"],
        data.geometry["layer_thickness"],
        y_index=0,
        quality=mesh_quality,
        parameter_max_cell_area=mesh_max_cell_area,
        smoothing_iterations=mesh_smoothing_iterations,
        data_file=data.data_files[0],
    )
    if case.mesh.cell_count == data.truth_models.shape[1] * data.truth_models.shape[2]:
        raise AssertionError("the inversion parameter mesh must differ from the synthetic forward grid")
    forward_mesh, parameter_ids, forward_parameter_ids, parent_ids = _build_forward_parameterization(
        case,
        forward_refinement,
    )
    forward = ParameterizedERTForward2p5D.from_mesh_survey(
        forward_mesh,
        case.survey,
        parameter_ids,
        regularization_mesh=case.mesh,
        forward_cell_parameter_ids=forward_parameter_ids,
        linear_solver_backend=linear_solver_backend,
        normal_field_cache_max_entries=normal_field_cache_max_entries,
    )
    return INRForwardBundle(
        case=case,
        forward=forward,
        forward_mesh=forward_mesh,
        parameter_cell_ids=np.asarray(parameter_ids),
        forward_cell_parameter_ids=None if forward_parameter_ids is None else np.asarray(forward_parameter_ids),
        forward_parent_cell_ids=None if parent_ids is None else np.asarray(parent_ids),
    )


def make_global_coordinates(bundle: INRForwardBundle, steps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create one global normalized coordinate system shared by every method/window."""

    return spatiotemporal_coordinates(cell_centers(bundle.case.mesh), np.asarray(steps, dtype=np.float32))


def _jsonable_config(config: Any) -> dict[str, Any]:
    values = asdict(config)
    values.pop("progress_callback", None)
    values.pop("extra_penalty", None)
    return values


def save_inr_result(
    output_dir: Path,
    *,
    method: str,
    data: FlatTimelapseData,
    bundle: INRForwardBundle,
    network: torch.nn.Module,
    config: Any,
    result: Any,
    coordinate_center: np.ndarray,
    coordinate_scale: np.ndarray,
) -> dict[str, Any]:
    """Save every INR method in the same orientation and file layout."""

    output_dir.mkdir(parents=True, exist_ok=True)
    final_models = np.asarray(result.resistivity, dtype=np.float64).T
    final_log_models = np.asarray(result.log_resistivity, dtype=np.float64).T
    predicted = np.asarray(result.predicted_data, dtype=np.float64)
    weighted_residual = (np.log(predicted) - np.log(data.observed_rhoa)) / data.data_std
    chi2_by_time = np.mean(weighted_residual**2, axis=1)

    np.save(output_dir / "final_models.npy", final_models)
    np.save(output_dir / "final_log_models.npy", final_log_models)
    np.save(output_dir / "predicted_rhoa.npy", predicted)
    np.save(output_dir / "steps.npy", data.steps)
    np.save(output_dir / "chi2_by_time.npy", chi2_by_time)
    np.savez_compressed(
        output_dir / "training_history.npz",
        chi2=result.chi2_history,
        rms=result.rms_history,
        objective=result.objective_history,
        spatial_penalty=result.spatial_penalty_history,
        temporal_penalty=result.temporal_penalty_history,
        extra_penalty=result.extra_penalty_history,
        full_chi2_iterations=result.full_chi2_iterations,
        full_chi2=result.full_chi2_history,
    )
    np.savez_compressed(
        output_dir / "coordinate_transform.npz",
        center=np.asarray(coordinate_center),
        scale=np.asarray(coordinate_scale),
    )
    _save_mesh_npz(
        output_dir / "timelapse_inversion_mesh.npz",
        bundle.case,
        forward_mesh=bundle.forward_mesh,
        parameter_cell_ids=bundle.parameter_cell_ids,
        forward_cell_parameter_ids=bundle.forward_cell_parameter_ids,
        forward_parent_cell_ids=bundle.forward_parent_cell_ids,
    )
    checkpoint = {
        "network_class": type(network).__name__,
        "network_configuration": network.configuration(),
        "state_dict": {name: value.detach().cpu() for name, value in network.state_dict().items()},
        "coordinate_center": np.asarray(coordinate_center),
        "coordinate_scale": np.asarray(coordinate_scale),
    }
    torch.save(checkpoint, output_dir / "network.pt")
    (output_dir / "used_data_files.txt").write_text(
        "".join(f"{path}\n" for path in data.data_files),
        encoding="utf-8",
    )

    parameter_count = int(sum(parameter.numel() for parameter in network.parameters()))
    summary = {
        "engine": "ADTLERT matrix-free GPU INR",
        "method": method,
        "network_class": type(network).__name__,
        "network_configuration": network.configuration(),
        "network_parameter_count": parameter_count,
        "n_timesteps": int(data.steps.size),
        "n_measurements_per_timestep": int(data.observed_rhoa.shape[1]),
        "n_inversion_cells": int(bundle.case.mesh.cell_count),
        "n_forward_cells": int(bundle.forward_mesh.cell_count),
        "mesh_is_independent_from_synthetic_grid": True,
        "initial_resistivity_ohm_m": 100.0,
        "explicit_spatial_regularization": float(config.spatial_regularization),
        "explicit_temporal_regularization": float(config.temporal_regularization),
        "config": _jsonable_config(config),
        "iterations": int(result.iterations),
        "best_iteration": int(result.best_iteration),
        "best_chi2": float(result.best_chi2),
        "mean_chi2": float(np.mean(chi2_by_time)),
        "median_chi2": float(np.median(chi2_by_time)),
        "max_chi2": float(np.max(chi2_by_time)),
        "elapsed_seconds": float(result.elapsed_seconds),
        "seconds_per_timestep": float(result.elapsed_seconds / data.steps.size),
        "physics_forward_timesteps": int(result.physics_forward_timesteps),
        "physics_vjp_timesteps": int(result.physics_vjp_timesteps),
        "stop_reason": result.stop_reason,
        "gpu": result.gpu_report,
    }
    (output_dir / "inversion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot_inr_recovery(
    output_dir: Path,
    *,
    method_label: str,
    data: FlatTimelapseData,
    bundle: INRForwardBundle,
    result: Any,
    selected_steps: tuple[int, ...] = (0, 12, 24, 33, 39, 48),
) -> np.ndarray:
    """Draw the same truth/inversion snapshots and anomaly curve for every method."""

    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LogNorm

    final_models = np.asarray(result.resistivity, dtype=np.float64).T
    nodes = np.asarray(bundle.case.mesh.nodes, dtype=float)
    cells = np.asarray(bundle.case.mesh.cells, dtype=np.int32)
    centers = nodes[cells].mean(axis=1)
    surface = np.interp(centers[:, 0], data.geometry["x_nodes"], data.geometry["z_top"])
    depth = surface - centers[:, 1]
    bounds = data.metadata["anomaly_bounds"]
    anomaly_mask = (
        (centers[:, 0] >= bounds["x_min"])
        & (centers[:, 0] <= bounds["x_max"])
        & (depth >= bounds["depth_min"])
        & (depth <= bounds["depth_max"])
    )
    if not np.any(anomaly_mask):
        raise AssertionError("the inversion mesh contains no cells in the anomaly rectangle")
    inverted_curve = np.median(final_models[anomaly_mask], axis=0)

    indices = [int(np.flatnonzero(data.steps == step)[0]) for step in selected_steps]
    norm = LogNorm(vmin=30.0, vmax=600.0)
    cmap = "turbo"
    cumulative_depth = np.concatenate(([0.0], np.cumsum(data.geometry["layer_thickness"])))
    x_grid = np.tile(data.geometry["x_nodes"], (cumulative_depth.size, 1))
    z_grid = data.geometry["z_top"][None, :] - cumulative_depth[:, None]

    fig = plt.figure(figsize=(18, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(3, len(selected_steps), height_ratios=[1.0, 1.0, 0.78])
    for column, (step, time_index) in enumerate(zip(selected_steps, indices, strict=True)):
        truth_axis = fig.add_subplot(grid[0, column])
        inversion_axis = fig.add_subplot(grid[1, column], sharex=truth_axis, sharey=truth_axis)
        truth_axis.pcolormesh(
            x_grid,
            z_grid,
            data.truth_models[time_index],
            shading="auto",
            cmap=cmap,
            norm=norm,
        )
        collection = PolyCollection(
            nodes[cells],
            array=final_models[:, time_index],
            cmap=cmap,
            norm=norm,
            edgecolors=(1, 1, 1, 0.20),
            linewidths=0.18,
        )
        inversion_axis.add_collection(collection)
        truth_axis.set_title(f"t={step}")
        if column == 0:
            truth_axis.set_ylabel("Truth\nElevation (m)")
            inversion_axis.set_ylabel(f"{method_label}\nElevation (m)")
        else:
            truth_axis.tick_params(labelleft=False)
            inversion_axis.tick_params(labelleft=False)
        truth_axis.tick_params(labelbottom=False)
        inversion_axis.set_xlabel("X (m)")
        for axis in (truth_axis, inversion_axis):
            axis.set_xlim(float(data.geometry["x_nodes"].min()), float(data.geometry["x_nodes"].max()))
            axis.set_ylim(float(z_grid.min()), float(z_grid.max()))

    curve_axis = fig.add_subplot(grid[2, :])
    curve_axis.plot(data.steps, data.truth_anomaly_curve, color="black", linewidth=2.4, label="True anomaly")
    curve_axis.plot(data.steps, inverted_curve, color="#d62728", linewidth=2.1, label=method_label)
    for step in selected_steps:
        curve_axis.axvline(step, color="0.75", linewidth=0.7, zorder=0)
    curve_axis.set(
        xlabel="Time step",
        ylabel="Median resistivity (ohm-m)",
        title="Temporal recovery inside the anomaly rectangle",
    )
    curve_axis.grid(alpha=0.22)
    curve_axis.legend(frameon=False, ncol=2)
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=fig.axes[:-1], shrink=0.72, pad=0.012)
    colorbar.set_label("Resistivity (ohm-m)")
    fig.savefig(output_dir / "model_and_temporal_recovery.png", dpi=180, bbox_inches="tight")
    plt.show()
    return inverted_curve


__all__ = [
    "FlatTimelapseData",
    "INRForwardBundle",
    "INRSnapshotRecorder",
    "build_inr_forward",
    "load_flat_timelapse_data",
    "make_global_coordinates",
    "plot_inr_recovery",
    "save_inr_result",
]
