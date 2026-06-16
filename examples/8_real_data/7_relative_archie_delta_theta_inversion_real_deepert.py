#!/usr/bin/env python3
"""Direct AD inversion of water-content change for the real hillslope ERT data.

This script parameterizes the ERT model with bounded volumetric water content
through a relative Archie relation:

    log(rho_t) = log(rho0) - n * log(theta_t / theta0)

The unknown optimized by Deepert is theta(t).  Delta theta is saved as
``theta(t) - theta0`` and can be compared directly with TMC soil-moisture
sensor changes. Optionally, a soft sensor-constraint term can be added so that
local cell-averaged theta follows TEROS observations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("DEEPERT_ENABLE_FLOAT64", "1")

from deepert.utils.torch_runtime import torch_runtime
import numpy as np
import pandas as pd
import scipy.sparse as sp

torch_runtime.config.update("torch_enable_float64", True)

from deepert.inversion import (  # noqa: E402
    InversionConfig,
    TimeLapseERTInversion,
    WindowedTimeLapseERTInversion,
    available_data_misfits,
    available_linearized_optimizers,
    available_optimization_algorithms,
    available_spatial_regularizations,
    available_temporal_regularizations,
)
from deepert.utils.progress import InversionProgressPrinter  # noqa: E402

from _real_data_common import (  # noqa: E402
    apply_data_stride,
    build_parameterized_forward,
    build_real_inversion_case,
    data_std_from_err,
    discover_processed_files,
    find_project_root,
    plot_chi2,
    quality_mask,
    read_processed_ert,
    resolve_path,
    save_mesh_npz,
    write_json,
)


def _check_same_layout(first, current) -> None:
    if not np.array_equal(current.measurements, first.measurements):
        raise ValueError(f"{current.path}: ABMN layout differs from first timestep {first.path}")
    if not np.allclose(current.elec_x, first.elec_x, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{current.path}: electrode x positions differ from first timestep")
    if not np.allclose(current.elec_z, first.elec_z, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{current.path}: electrode z positions differ from first timestep")


def _selected_plot_indices(n_times: int, max_panels: int) -> np.ndarray:
    if n_times <= max_panels:
        return np.arange(n_times, dtype=np.int32)
    return np.unique(np.round(np.linspace(0, n_times - 1, max_panels)).astype(np.int32))


def _cell_geometry(case) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nodes = np.asarray(case.mesh.nodes, dtype=float)
    cells = np.asarray(case.mesh.cells, dtype=np.int32)
    centers = nodes[cells].mean(axis=1)
    cell_x = centers[:, 0]
    surface_at_center = np.interp(cell_x, np.asarray(case.x_nodes, dtype=float), np.asarray(case.z_top, dtype=float))
    cell_depth = np.maximum(surface_at_center - centers[:, 1], 0.0)
    return nodes, cells, cell_x, cell_depth


def _load_projected_tmc_tables(sensor_reference_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    locations_file = sensor_reference_dir / "real_sensor_locations_projected.csv"
    timeseries_file = sensor_reference_dir / "real_sensor_timeseries_at_ert_times.csv"
    if not locations_file.exists() or not timeseries_file.exists():
        raise FileNotFoundError(
            "Missing projected TMC sensor tables. Run examples/8_real_data/4_temperature_and_soil_moisture_QA.ipynb first."
        )
    sensor_locations = pd.read_csv(locations_file)
    sensor_ts = pd.read_csv(timeseries_file)
    sensor_ts["DateTime"] = pd.to_datetime(sensor_ts["DateTime"])
    return sensor_locations, sensor_ts


def _sensor_row_weights(
    *,
    cell_x: np.ndarray,
    cell_depth: np.ndarray,
    sensor_x: float,
    sensor_depth: float,
    horizontal_radius: float,
    vertical_radius: float,
    min_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    rx = max(float(horizontal_radius), 1.0e-6)
    rz = max(float(vertical_radius), 1.0e-6)
    selected = (np.abs(cell_x - float(sensor_x)) <= rx) & (np.abs(cell_depth - float(sensor_depth)) <= rz)
    ids = np.flatnonzero(selected)
    min_count = max(int(min_cells), 1)
    if ids.size < min_count:
        scaled_dist2 = ((cell_x - float(sensor_x)) / rx) ** 2 + ((cell_depth - float(sensor_depth)) / rz) ** 2
        ids = np.argpartition(scaled_dist2, min(min_count, scaled_dist2.size) - 1)[: min(min_count, scaled_dist2.size)]
    dx = (cell_x[ids] - float(sensor_x)) / rx
    dz = (cell_depth[ids] - float(sensor_depth)) / rz
    weights = np.exp(-0.5 * (dx**2 + dz**2))
    if not np.any(np.isfinite(weights)) or np.all(weights <= 0.0):
        weights = np.ones_like(weights, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    return ids.astype(np.int32), weights


def _build_sensor_constraint_from_tmc(
    *,
    sensor_reference_dir: Path,
    case,
    ert_times: pd.DatetimeIndex,
    theta0_model: np.ndarray,
    theta_min: float,
    theta_max: float,
    sensor_sigma: float,
    horizontal_radius: float,
    vertical_radius: float,
    min_cells: int,
    target_mode: str,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    sensor_locations, sensor_ts = _load_projected_tmc_tables(sensor_reference_dir)
    _, _, cell_x, cell_depth = _cell_geometry(case)
    sensor_depths = np.array([0.1, 0.3, 0.6, 0.8], dtype=float)

    row_sensor_ids: list[np.ndarray] = []
    row_sensor_weights: list[np.ndarray] = []
    targets_rows: list[np.ndarray] = []
    metadata_rows: list[dict[str, float | str | int]] = []

    for station in sorted(sensor_ts["station"].unique()):
        loc = sensor_locations.loc[sensor_locations["ID"].eq(station)]
        if loc.empty:
            continue
        station_x = float(loc["profile_x_m"].iloc[0])
        station_table = (
            sensor_ts.loc[sensor_ts["station"].eq(station), ["DateTime", *[f"MC_{depth:.1f}m" for depth in sensor_depths]]]
            .set_index("DateTime")
            .reindex(ert_times)
        )
        if station_table.empty:
            continue
        for depth in sensor_depths:
            column = f"MC_{depth:.1f}m"
            if column not in station_table.columns:
                continue
            values = station_table[column].to_numpy(dtype=float)
            values = np.where(values > 0.0, values, np.nan)
            if np.count_nonzero(np.isfinite(values)) < 2:
                continue
            sensor_ids, sensor_weights = _sensor_row_weights(
                cell_x=cell_x,
                cell_depth=cell_depth,
                sensor_x=station_x,
                sensor_depth=float(depth),
                horizontal_radius=float(horizontal_radius),
                vertical_radius=float(vertical_radius),
                min_cells=int(min_cells),
            )
            row_sensor_ids.append(sensor_ids)
            row_sensor_weights.append(sensor_weights)
            targets_rows.append(np.clip(values, float(theta_min), float(theta_max)))
            metadata_rows.append(
                {
                    "station": str(station),
                    "depth_m": float(depth),
                    "profile_x_m": float(station_x),
                    "n_support_cells": int(sensor_ids.size),
                }
            )

    if not targets_rows:
        raise ValueError("No valid sensor rows available to build sensor constraint.")

    n_rows = len(targets_rows)
    n_cells = int(case.mesh.cell_count)
    n_times = int(len(ert_times))
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for ids, weights in zip(row_sensor_ids, row_sensor_weights, strict=True):
        indices.extend(int(i) for i in ids)
        data.extend(float(w) for w in weights)
        indptr.append(len(indices))
    operator = sp.csr_matrix(
        (
            np.asarray(data, dtype=float),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(n_rows, n_cells),
        dtype=float,
    )
    observed_targets = np.vstack(targets_rows).astype(float, copy=False)
    targets = observed_targets.copy()
    if target_mode == "delta_from_model_baseline":
        theta0_support = np.asarray(operator @ np.asarray(theta0_model, dtype=float), dtype=float).reshape(-1)
        sensor_delta = observed_targets - observed_targets[:, [0]]
        targets = theta0_support[:, None] + sensor_delta
        targets = np.clip(targets, float(theta_min), float(theta_max))
        metadata = pd.DataFrame(metadata_rows)
        metadata["theta0_sensor"] = observed_targets[:, 0]
        metadata["theta0_model_support"] = theta0_support
    elif target_mode == "absolute":
        metadata = pd.DataFrame(metadata_rows)
        metadata["theta0_sensor"] = observed_targets[:, 0]
        metadata["theta0_model_support"] = np.asarray(operator @ np.asarray(theta0_model, dtype=float), dtype=float).reshape(-1)
    else:
        raise ValueError(f"Unknown sensor constraint target mode: {target_mode!r}")
    weights = np.full((n_rows, n_times), 1.0 / max(float(sensor_sigma), 1.0e-6) ** 2, dtype=float)
    weights[~np.isfinite(observed_targets)] = 0.0
    metadata["target_mode"] = str(target_mode)
    metadata["row_index"] = np.arange(n_rows, dtype=int)
    metadata["valid_points"] = np.sum(np.isfinite(observed_targets), axis=1).astype(int)
    return operator, targets, observed_targets, weights, metadata


def _evaluate_sensor_constraint_fit(
    *,
    final_theta: np.ndarray,
    sensor_operator: sp.csr_matrix,
    sensor_targets: np.ndarray,
    sensor_observed_targets: np.ndarray | None,
    sensor_metadata: pd.DataFrame,
    measurement_times_days: np.ndarray,
    timestamp_labels: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    theta_series = np.asarray(sensor_operator @ np.asarray(final_theta, dtype=float), dtype=float)
    target_series = (
        np.asarray(sensor_observed_targets, dtype=float)
        if sensor_observed_targets is not None
        else np.asarray(sensor_targets, dtype=float)
    )
    model_delta = theta_series - theta_series[:, [0]]
    target_delta = target_series - target_series[:, [0]]
    valid = np.isfinite(target_delta)
    residual = (model_delta - target_delta)[valid]
    if residual.size == 0:
        metrics = {
            "n_points": 0,
            "rmse_delta_theta": float("nan"),
            "mae_delta_theta": float("nan"),
            "bias_delta_theta": float("nan"),
            "correlation": float("nan"),
            "ert_final_mean": float("nan"),
            "sensor_final_mean": float("nan"),
        }
    else:
        model_valid = model_delta[valid]
        target_valid = target_delta[valid]
        corr = float("nan")
        if model_valid.size > 2 and np.nanstd(model_valid) > 0 and np.nanstd(target_valid) > 0:
            corr = float(np.corrcoef(target_valid, model_valid)[0, 1])
        metrics = {
            "n_points": int(residual.size),
            "rmse_delta_theta": float(np.sqrt(np.mean(residual**2))),
            "mae_delta_theta": float(np.mean(np.abs(residual))),
            "bias_delta_theta": float(np.mean(residual)),
            "correlation": corr,
            "ert_final_mean": float(np.nanmean(model_delta[:, -1])),
            "sensor_final_mean": float(np.nanmean(target_delta[:, -1])),
        }

    records: list[dict[str, float | str | int]] = []
    for row_idx in range(target_series.shape[0]):
        station = str(sensor_metadata.loc[row_idx, "station"])
        depth = float(sensor_metadata.loc[row_idx, "depth_m"])
        for time_idx in range(target_series.shape[1]):
            records.append(
                {
                    "row_index": int(row_idx),
                    "station": station,
                    "depth_m": depth,
                    "time_index": int(time_idx),
                    "day_since_baseline": float(measurement_times_days[time_idx]),
                    "timestamp": str(timestamp_labels[time_idx]),
                    "theta_sensor": float(target_series[row_idx, time_idx]) if np.isfinite(target_series[row_idx, time_idx]) else np.nan,
                    "theta_ert": float(theta_series[row_idx, time_idx]),
                    "delta_theta_sensor": float(target_delta[row_idx, time_idx]) if np.isfinite(target_delta[row_idx, time_idx]) else np.nan,
                    "delta_theta_ert": float(model_delta[row_idx, time_idx]),
                }
            )
    return pd.DataFrame(records), metrics


def _build_theta0_from_tmc(
    *,
    sensor_reference_dir: Path,
    case,
    ert_times: pd.DatetimeIndex,
    theta_min: float,
    theta_max: float,
    deep_theta_background: float,
    deep_anchor_depth_m: float,
) -> np.ndarray:
    sensor_locations, sensor_ts = _load_projected_tmc_tables(sensor_reference_dir)

    baseline_time = pd.Timestamp(ert_times[0])
    sensor_depths = np.array([0.1, 0.3, 0.6, 0.8], dtype=float)
    station_names: list[str] = []
    station_x: list[float] = []
    station_theta0_profiles: list[np.ndarray] = []

    for station in sorted(sensor_ts["station"].unique()):
        loc = sensor_locations.loc[sensor_locations["ID"].eq(station)]
        if loc.empty:
            continue
        row0 = sensor_ts.loc[(sensor_ts["station"].eq(station)) & (sensor_ts["DateTime"].eq(baseline_time))]
        if row0.empty:
            continue
        values = np.asarray([float(row0[f"MC_{depth:.1f}m"].iloc[0]) for depth in sensor_depths], dtype=float)
        # Zero/negative MC values in this data set indicate sensor dropouts.
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            continue
        station_names.append(str(station))
        station_x.append(float(loc["profile_x_m"].iloc[0]))
        station_theta0_profiles.append(values)

    if len(station_x) < 2:
        raise ValueError("Need at least two valid TMC stations to interpolate theta0")

    station_x_array = np.asarray(station_x, dtype=float)
    station_profiles = np.asarray(station_theta0_profiles, dtype=float)
    order = np.argsort(station_x_array)
    station_x_array = station_x_array[order]
    station_profiles = station_profiles[order]

    theta_depths = np.r_[sensor_depths, float(deep_anchor_depth_m)]
    station_profiles = np.column_stack(
        [station_profiles, np.full(station_profiles.shape[0], float(deep_theta_background), dtype=float)]
    )

    _, _, cell_x, cell_depth = _cell_geometry(case)
    clipped_depth = np.minimum(cell_depth, float(deep_anchor_depth_m))
    station_cell_theta = np.vstack(
        [
            np.interp(clipped_depth, theta_depths, profile, left=profile[0], right=float(deep_theta_background))
            for profile in station_profiles
        ]
    )

    theta0 = np.empty(case.mesh.cell_count, dtype=float)
    for cell_idx in range(case.mesh.cell_count):
        theta0[cell_idx] = np.interp(
            cell_x[cell_idx],
            station_x_array,
            station_cell_theta[:, cell_idx],
            left=station_cell_theta[0, cell_idx],
            right=station_cell_theta[-1, cell_idx],
        )
    return np.clip(theta0, float(theta_min), float(theta_max))


def _load_baseline_resistivity(path: Path, *, column: int, n_cells: int) -> np.ndarray:
    values = np.asarray(np.load(path), dtype=float)
    if values.ndim == 2:
        if not (0 <= int(column) < values.shape[1]):
            raise ValueError(f"baseline column {column} outside file with shape {values.shape}")
        values = values[:, int(column)]
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.shape != (int(n_cells),):
        raise ValueError(f"baseline rho0 has shape {values.shape}; expected ({int(n_cells)},). Use a matching mesh.")
    if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("baseline rho0 must be positive and finite")
    return values


def _load_temperature_correction_factor(path: Path, *, n_cells: int, n_times: int) -> np.ndarray:
    values = np.asarray(np.load(path), dtype=float)
    if values.shape != (int(n_cells), int(n_times)):
        raise ValueError(
            f"temperature correction factor has shape {values.shape}; expected ({int(n_cells)}, {int(n_times)})."
        )
    if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("temperature correction factor must be positive and finite")
    return values


def _plot_delta_theta_models(
    path: Path,
    *,
    case,
    delta_theta: np.ndarray,
    labels: list[str],
    coverage_mask: np.ndarray | None,
    depth: float,
    clim: tuple[float, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import TwoSlopeNorm

    nodes = np.asarray(case.mesh.nodes, dtype=float)
    cells = np.asarray(case.mesh.cells, dtype=np.int32)
    models = np.asarray(delta_theta, dtype=float)
    if models.ndim == 1:
        models = models[:, None]

    n_panels = models.shape[1]
    fig, axes = plt.subplots(1, n_panels, figsize=(4.1 * n_panels, 3.4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    norm = TwoSlopeNorm(vmin=float(clim[0]), vcenter=0.0, vmax=float(clim[1]))
    last = None
    for index, ax in enumerate(axes):
        values = models[:, index]
        visible = np.isfinite(values)
        if coverage_mask is not None:
            visible &= ~np.asarray(coverage_mask, dtype=bool).ravel()
        last = PolyCollection(nodes[cells][visible], array=values[visible], cmap="BrBG", norm=norm, edgecolors="none")
        ax.add_collection(last)
        ax.plot(case.elec_x, case.elec_z, color="black", lw=1.2)
        ax.set_xlim(float(np.min(case.elec_x)), float(np.max(case.elec_x)))
        z_top = float(np.max(case.elec_z))
        ax.set_ylim(z_top - float(depth), z_top + 3.0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
        ax.set_title(labels[index])
        ax.set_xlabel("Distance (m)")
        if index == 0:
            ax.set_ylabel("Relative elevation (m)")
        else:
            ax.tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.18, top=0.86, wspace=0.08)
    if last is not None:
        cax = fig.add_axes([0.92, 0.25, 0.012, 0.52])
        cbar = fig.colorbar(last, cax=cax)
        cbar.set_label("Delta theta (V/V)")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--input-dir", default="ProcessedData")
    parser.add_argument("--output-dir", default="result/8_real_data/7_relative_archie_delta_theta_inversion_real_deepert")
    parser.add_argument(
        "--sensor-reference-dir",
        default="result/8_real_data/2_timelapse_inversion_real_deepert",
        help="Directory containing projected TMC sensor tables from notebook 4.",
    )
    parser.add_argument(
        "--mesh-file",
        default="result/8_real_data/2_timelapse_inversion_real_deepert/timelapse_inversion_mesh.npz",
        help="Saved inversion mesh to reuse. Reusing the resistivity mesh keeps comparisons clean.",
    )
    parser.add_argument(
        "--baseline-resistivity-file",
        default="result/8_real_data/2_timelapse_inversion_real_deepert/final_models_temperature_corrected.npy",
        help="Baseline resistivity model used as rho0 at the reference temperature.",
    )
    parser.add_argument("--baseline-column", type=int, default=0)
    parser.add_argument(
        "--temperature-correction-factor-file",
        default="result/8_real_data/2_timelapse_inversion_real_deepert/temperature_correction_factor.npy",
        help=(
            "Cell/time correction factor C_T where rho_Tref = rho_field * C_T. "
            "Run 5_temperature_correction_real_resistivity.ipynb first."
        ),
    )
    parser.add_argument(
        "--use-temperature-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include temperature correction in the AD petrophysical chain.",
    )
    parser.add_argument("--start-date", default="2022-03-26")
    parser.add_argument("--end-date", default="2022-05-12")
    parser.add_argument("--file-stride", type=int, default=8)
    parser.add_argument("--max-timesteps", type=int, default=None)
    parser.add_argument("--inversion-mode", choices=("windowed", "full"), default="windowed")
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--window-step", type=int, default=1)
    parser.add_argument("--depth", type=float, default=90.0)
    parser.add_argument("--n-layers", type=int, default=28)
    parser.add_argument("--layer-stretch", type=float, default=1.08)
    parser.add_argument("--data-stride", type=int, default=1)
    parser.add_argument("--max-error", type=float, default=None)
    parser.add_argument("--relative-error", type=float, default=0.05)
    parser.add_argument("--minimum-log-std", type=float, default=1.0e-3)
    parser.add_argument("--data-misfit", choices=available_data_misfits(), default="weighted_log_l2")
    parser.add_argument("--regularization", type=float, default=50.0)
    parser.add_argument("--temporal-regularization", type=float, default=10.0)
    parser.add_argument("--temporal-regularization-type", choices=available_temporal_regularizations(), default="temporal_smoothness")
    parser.add_argument(
        "--freeze-baseline-theta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Freeze theta at the first survey to theta0 (baseline anchor). "
            "In windowed mode this applies only to windows starting at global t0."
        ),
    )
    parser.add_argument("--regularization-mode", choices=("model", "update"), default="model")
    parser.add_argument("--regularization-domain", choices=("state", "physical"), default="physical")
    parser.add_argument(
        "--physical-regularization-quantity",
        choices=("parameter", "theta", "water_content"),
        default="parameter",
        help="For this transform, parameter and theta/water_content are the same physical quantity.",
    )
    parser.add_argument("--spatial-regularization", choices=available_spatial_regularizations(), default="first_order_smoothness")
    parser.add_argument("--z-weight", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=available_optimization_algorithms(), default="gauss_newton_cgls")
    parser.add_argument("--linearized-solver", choices=available_linearized_optimizers(), default="gpu_cgls")
    parser.add_argument("--linear-solver-backend", default="auto")
    parser.add_argument("--terrain-cache-dir", default=None)
    parser.add_argument("--lm-damping", type=float, default=1.0e-2)
    parser.add_argument("--cgls-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--cgls-max-iterations", type=int, default=2000)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--max-state-step", type=float, default=1.0, help="Maximum optimizer-state step per iteration.")
    parser.add_argument("--target-chi2", type=float, default=None)
    parser.add_argument("--step-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--line-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inversion-mesh-quality", type=float, default=34.0)
    parser.add_argument("--inversion-mesh-smoothing-iterations", type=int, default=10)
    parser.add_argument("--coverage-percentile", type=float, default=20.0)
    parser.add_argument("--saturation-exponent", type=float, default=1.6)
    parser.add_argument("--theta-min", type=float, default=0.02)
    parser.add_argument("--theta-max", type=float, default=0.50)
    parser.add_argument("--deep-theta-background", type=float, default=0.12)
    parser.add_argument("--deep-anchor-depth", type=float, default=5.0)
    parser.add_argument(
        "--sensor-constraint-lambda",
        type=float,
        default=0.0,
        help="Soft sensor constraint weight lambda_s. Set >0 to enforce H*theta against the selected sensor target.",
    )
    parser.add_argument(
        "--sensor-constraint-target-mode",
        choices=("delta_from_model_baseline", "absolute"),
        default="delta_from_model_baseline",
        help=(
            "How TMC observations enter the soft constraint. "
            "'delta_from_model_baseline' enforces H*theta ~= H*theta0 + "
            "(theta_sensor(t)-theta_sensor(t0)), which is usually better for "
            "comparing ERT-scale delta theta with point sensors. "
            "'absolute' enforces H*theta ~= theta_sensor directly."
        ),
    )
    parser.add_argument(
        "--sensor-constraint-sigma",
        type=float,
        default=0.03,
        help="Assumed theta uncertainty for sensor constraint (V/V).",
    )
    parser.add_argument(
        "--sensor-constraint-horizontal-radius",
        type=float,
        default=8.0,
        help="Horizontal support radius (m) for mapping sensor points to cell averages.",
    )
    parser.add_argument(
        "--sensor-constraint-vertical-radius",
        type=float,
        default=0.8,
        help="Vertical support radius (m) for mapping sensor points to cell averages.",
    )
    parser.add_argument(
        "--sensor-constraint-min-cells",
        type=int,
        default=6,
        help="Minimum number of mesh cells used for each sensor-support averaging row.",
    )
    parser.add_argument("--plot-count", type=int, default=4)
    parser.add_argument("--delta-theta-vmin", type=float, default=-0.08)
    parser.add_argument("--delta-theta-vmax", type=float, default=0.08)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).resolve() if args.project_root else find_project_root(Path(__file__))
    input_dir = resolve_path(root, args.input_dir)
    output_dir = resolve_path(root, args.output_dir)
    sensor_reference_dir = resolve_path(root, args.sensor_reference_dir)
    mesh_file = None if args.mesh_file is None else resolve_path(root, args.mesh_file)
    baseline_resistivity_file = resolve_path(root, args.baseline_resistivity_file)
    temperature_correction_factor_file = (
        None
        if args.temperature_correction_factor_file is None
        else resolve_path(root, args.temperature_correction_factor_file)
    )
    terrain_cache_dir = None if args.terrain_cache_dir is None else resolve_path(root, args.terrain_cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = discover_processed_files(
        input_dir,
        start=args.start_date,
        end=args.end_date,
        file_stride=args.file_stride,
        max_timesteps=args.max_timesteps,
    )
    if len(paths) < 2:
        raise ValueError(f"Need at least two real ERT files after filtering in {input_dir}.")

    first = read_processed_ert(paths[0])
    raw_records = [first]
    for path in paths[1:]:
        current = read_processed_ert(path, elevation_reference=first.elevation_reference)
        _check_same_layout(first, current)
        raw_records.append(current)

    common_mask = np.ones(first.rhoa.shape, dtype=bool)
    for record in raw_records:
        common_mask &= quality_mask(record, max_error=args.max_error)
    common_mask = apply_data_stride(common_mask, args.data_stride)
    if int(common_mask.sum()) < 4:
        raise ValueError("Need at least four common valid measurements after filtering.")

    records = [record.with_measurement_mask(common_mask) for record in raw_records]
    observed_rhoa = np.vstack([record.rhoa for record in records])
    err_matrix = None if any(record.err is None for record in records) else np.vstack([record.err for record in records])
    data_std = data_std_from_err(
        err_matrix,
        shape=observed_rhoa.shape,
        relative_error=args.relative_error,
        minimum_log_std=args.minimum_log_std,
    )
    timestamps = [record.timestamp for record in records]
    if all(timestamp is not None for timestamp in timestamps):
        start_time = timestamps[0]
        measurement_times_days = np.asarray(
            [(timestamp - start_time).total_seconds() / 86400.0 for timestamp in timestamps], dtype=float
        )
        timestamp_labels = [timestamp.strftime("%Y-%m-%d %H:%M") for timestamp in timestamps]
        ert_times = pd.DatetimeIndex(timestamps)
    else:
        measurement_times_days = np.arange(len(records), dtype=float)
        timestamp_labels = [path.stem for path in paths]
        ert_times = pd.to_datetime(timestamp_labels)
    steps = np.arange(len(records), dtype=np.int32)

    case = build_real_inversion_case(
        records[0],
        depth=args.depth,
        n_layers=args.n_layers,
        layer_stretch=args.layer_stretch,
        inversion_mesh_quality=args.inversion_mesh_quality,
        inversion_mesh_smoothing_iterations=args.inversion_mesh_smoothing_iterations,
        mesh_file=mesh_file,
    )
    if observed_rhoa.shape[1] != case.survey.measurement_count:
        raise ValueError(
            f"Observed data has {observed_rhoa.shape[1]} measurements, but survey has {case.survey.measurement_count}."
        )

    rho0 = _load_baseline_resistivity(
        baseline_resistivity_file,
        column=args.baseline_column,
        n_cells=case.mesh.cell_count,
    )
    theta0 = _build_theta0_from_tmc(
        sensor_reference_dir=sensor_reference_dir,
        case=case,
        ert_times=ert_times,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        deep_theta_background=args.deep_theta_background,
        deep_anchor_depth_m=args.deep_anchor_depth,
    )
    sensor_constraint_operator = None
    sensor_constraint_targets = None
    sensor_constraint_observed_targets = None
    sensor_constraint_weights = None
    sensor_constraint_metadata = None
    if float(args.sensor_constraint_lambda) > 0.0:
        (
            sensor_constraint_operator,
            sensor_constraint_targets,
            sensor_constraint_observed_targets,
            sensor_constraint_weights,
            sensor_constraint_metadata,
        ) = _build_sensor_constraint_from_tmc(
            sensor_reference_dir=sensor_reference_dir,
            case=case,
            ert_times=ert_times,
            theta0_model=theta0,
            theta_min=float(args.theta_min),
            theta_max=float(args.theta_max),
            sensor_sigma=float(args.sensor_constraint_sigma),
            horizontal_radius=float(args.sensor_constraint_horizontal_radius),
            vertical_radius=float(args.sensor_constraint_vertical_radius),
            min_cells=int(args.sensor_constraint_min_cells),
            target_mode=str(args.sensor_constraint_target_mode),
        )
    if args.use_temperature_correction:
        if temperature_correction_factor_file is None or not temperature_correction_factor_file.exists():
            raise FileNotFoundError(
                "Missing temperature correction factor. Run "
                "examples/8_real_data/5_temperature_correction_real_resistivity.ipynb first, "
                "or pass --no-use-temperature-correction."
            )
        temperature_correction_factor = _load_temperature_correction_factor(
            temperature_correction_factor_file,
            n_cells=case.mesh.cell_count,
            n_times=len(records),
        )
    else:
        temperature_correction_factor = np.ones((case.mesh.cell_count, len(records)), dtype=float)

    forward = build_parameterized_forward(
        case,
        linear_solver_backend=args.linear_solver_backend,
        terrain_cache_dir=terrain_cache_dir,
    )
    progress = InversionProgressPrinter(enabled=not args.quiet)
    config = InversionConfig(
        max_iterations=args.max_iterations,
        data_std=data_std,
        data_misfit=args.data_misfit,
        regularization=args.regularization,
        regularization_mode=args.regularization_mode,
        regularization_domain=args.regularization_domain,
        physical_regularization_quantity=args.physical_regularization_quantity,
        temporal_regularization=args.temporal_regularization,
        temporal_regularization_type=args.temporal_regularization_type,
        spatial_regularization=args.spatial_regularization,
        z_weight=args.z_weight,
        optimization_algorithm=args.optimizer,
        linearized_solver=args.linearized_solver,
        lm_damping=args.lm_damping,
        cgls_tolerance=args.cgls_tolerance,
        cgls_max_iterations=args.cgls_max_iterations,
        model_bounds=None,
        petrophysical_transform="relative_archie_water_content",
        petrophysical_parameters={
            "rho0": rho0,
            "theta0": theta0,
            "n": float(args.saturation_exponent),
            "theta_min": float(args.theta_min),
            "theta_max": float(args.theta_max),
            "temperature_correction_factor": temperature_correction_factor
            if args.use_temperature_correction
            else None,
        },
        max_log_step=args.max_state_step,
        line_search=bool(args.line_search),
        target_chi2=args.target_chi2,
        step_tolerance=args.step_tolerance,
        freeze_first_timestep=bool(args.freeze_baseline_theta),
        sensor_constraint=float(args.sensor_constraint_lambda),
        sensor_constraint_operator=sensor_constraint_operator,
        sensor_constraint_targets=sensor_constraint_targets,
        sensor_constraint_weights=sensor_constraint_weights,
        progress_callback=progress,
    )

    # Starting with theta(t)=theta0 at all timesteps corresponds to reference-
    # temperature rho0 mapped back to field-temperature resistivity.
    initial_model = rho0[:, None] / temperature_correction_factor
    run_start = time.perf_counter()
    try:
        if args.inversion_mode == "full":
            result = TimeLapseERTInversion(
                forward=forward,
                observed_data=observed_rhoa,
                config=config,
            ).setup().run(initial_model)
            run_meta = {"inversion_mode": "full", "n_windows": 1, "window_size": None, "window_step": None}
        else:
            result = WindowedTimeLapseERTInversion(
                forward=forward,
                observed_data=observed_rhoa,
                config=config,
                window_size=args.window_size,
                window_step=args.window_step,
            ).setup().run(initial_model)
            run_meta = {
                "inversion_mode": "windowed",
                "n_windows": int(len(result.window_reports)),
                "window_size": int(args.window_size),
                "window_step": int(args.window_step),
            }
    finally:
        forward.close()
        progress.finish()
    elapsed_sec = time.perf_counter() - run_start

    final_models = np.asarray(result.final_models, dtype=float)
    final_theta = np.asarray(result.final_parameter_models, dtype=float)
    if result.final_parameter_name != "water_content":
        raise RuntimeError(f"Expected water_content parameter output, got {result.final_parameter_name!r}")
    # Time-lapse change should be zero at the reference survey.  We therefore
    # report Delta theta relative to the inverted first-time water-content model,
    # while also saving the offset from the prescribed TMC-interpolated theta0.
    final_delta_theta_from_theta0 = final_theta - theta0[:, None]
    final_delta_theta = final_theta - final_theta[:, [0]]

    coverage = np.asarray(result.coverage, dtype=float).ravel()
    coverage_threshold = float(np.percentile(coverage, args.coverage_percentile))
    coverage_mask = coverage < coverage_threshold

    np.save(output_dir / "rho0_baseline_resistivity.npy", rho0)
    np.save(output_dir / "theta0_sensor_interpolated.npy", theta0)
    if args.use_temperature_correction:
        np.save(output_dir / "temperature_correction_factor.npy", temperature_correction_factor)
    np.save(output_dir / "final_models.npy", final_models)
    if args.use_temperature_correction:
        np.save(output_dir / "final_models_reference_temperature.npy", final_models * temperature_correction_factor)
    np.save(output_dir / "final_log_models.npy", np.asarray(result.final_log_models, dtype=float))
    np.save(output_dir / "final_water_content_models.npy", final_theta)
    np.save(output_dir / "final_delta_theta_models.npy", final_delta_theta)
    np.save(output_dir / "final_water_content_offset_from_theta0_models.npy", final_delta_theta_from_theta0)
    np.save(output_dir / "predicted_rhoa.npy", np.asarray(result.predicted_data, dtype=float))
    np.save(output_dir / "observed_rhoa.npy", observed_rhoa)
    np.save(output_dir / "data_std.npy", np.asarray(data_std, dtype=float))
    np.save(output_dir / "steps.npy", steps)
    np.save(output_dir / "measurement_times_days.npy", measurement_times_days)
    np.save(output_dir / "chi2_all.npy", np.asarray(result.all_chi2, dtype=float))
    np.save(output_dir / "coverage.npy", coverage)
    np.save(output_dir / "coverage_mask.npy", coverage_mask.astype(np.uint8))
    save_mesh_npz(output_dir / "timelapse_inversion_mesh.npz", case)
    np.savez(
        output_dir / "real_data_geometry_and_survey.npz",
        elec_x=records[0].elec_x,
        elec_z=records[0].elec_z,
        elec_elevation=records[0].elec_elevation,
        measurements=records[0].measurements,
        x_nodes=case.x_nodes,
        z_top=case.z_top,
        layer_thickness=case.layer_thickness,
        common_measurement_mask=common_mask.astype(np.uint8),
    )
    sensor_metrics: dict[str, float] = {}
    if sensor_constraint_operator is not None and sensor_constraint_targets is not None and sensor_constraint_metadata is not None:
        sp.save_npz(output_dir / "sensor_constraint_operator.npz", sensor_constraint_operator)
        np.save(output_dir / "sensor_constraint_targets.npy", np.asarray(sensor_constraint_targets, dtype=float))
        if sensor_constraint_observed_targets is not None:
            np.save(
                output_dir / "sensor_constraint_observed_targets.npy",
                np.asarray(sensor_constraint_observed_targets, dtype=float),
            )
        np.save(output_dir / "sensor_constraint_weights.npy", np.asarray(sensor_constraint_weights, dtype=float))
        sensor_constraint_metadata.to_csv(output_dir / "sensor_constraint_metadata.csv", index=False)
        comparison, sensor_metrics = _evaluate_sensor_constraint_fit(
            final_theta=final_theta,
            sensor_operator=sensor_constraint_operator,
            sensor_targets=np.asarray(sensor_constraint_targets, dtype=float),
            sensor_observed_targets=(
                None
                if sensor_constraint_observed_targets is None
                else np.asarray(sensor_constraint_observed_targets, dtype=float)
            ),
            sensor_metadata=sensor_constraint_metadata,
            measurement_times_days=measurement_times_days,
            timestamp_labels=timestamp_labels,
        )
        comparison.to_csv(output_dir / "ad_delta_theta_sensor_comparison.csv", index=False)
        valid = comparison.replace([np.inf, -np.inf], np.nan).dropna(subset=["delta_theta_sensor", "delta_theta_ert"])
        if not valid.empty:
            by_row = []
            for (row_index, station, depth_m), sub in valid.groupby(["row_index", "station", "depth_m"], sort=True):
                residual = sub["delta_theta_ert"].to_numpy(dtype=float) - sub["delta_theta_sensor"].to_numpy(dtype=float)
                corr = float("nan")
                if residual.size > 2 and np.std(sub["delta_theta_sensor"]) > 0 and np.std(sub["delta_theta_ert"]) > 0:
                    corr = float(np.corrcoef(sub["delta_theta_sensor"], sub["delta_theta_ert"])[0, 1])
                by_row.append(
                    {
                        "row_index": int(row_index),
                        "station": str(station),
                        "depth_m": float(depth_m),
                        "n_points": int(residual.size),
                        "rmse_delta_theta": float(np.sqrt(np.mean(residual**2))),
                        "mae_delta_theta": float(np.mean(np.abs(residual))),
                        "bias_delta_theta": float(np.mean(residual)),
                        "correlation": corr,
                    }
                )
            pd.DataFrame(by_row).to_csv(output_dir / "ad_delta_theta_sensor_metrics_by_depth.csv", index=False)
        sensor_metrics_with_case = {"case": output_dir.name, **sensor_metrics}
        write_json(output_dir / "ad_delta_theta_sensor_metrics.json", sensor_metrics_with_case)

    for column, (step, label) in enumerate(zip(steps, timestamp_labels, strict=True)):
        rho_model = final_models[:, column]
        theta_model = final_theta[:, column]
        delta_theta_model = final_delta_theta[:, column]
        np.save(output_dir / f"inverted_resistivity_t{int(step):05d}.npy", rho_model)
        np.save(output_dir / f"inverted_water_content_t{int(step):05d}.npy", theta_model)
        np.save(output_dir / f"inverted_delta_theta_t{int(step):05d}.npy", delta_theta_model)
        masked_delta = delta_theta_model.copy()
        masked_delta[coverage_mask] = np.nan
        np.save(output_dir / f"inverted_delta_theta_masked_nan_t{int(step):05d}.npy", masked_delta)
        safe_label = label.replace("-", "").replace(":", "").replace(" ", "_")
        np.save(output_dir / f"inverted_delta_theta_{safe_label}.npy", delta_theta_model)

    with (output_dir / "used_data_files.txt").open("w", encoding="utf-8") as stream:
        for path, label in zip(paths, timestamp_labels, strict=True):
            stream.write(f"{label}\t{path}\n")
    if result.window_reports:
        write_json(output_dir / "window_reports.json", result.window_reports)

    plot_files: dict[str, str] = {}
    if not args.no_plot:
        indices = _selected_plot_indices(final_delta_theta.shape[1], args.plot_count)
        labels = [timestamp_labels[int(index)] for index in indices]
        delta_plot = output_dir / "timelapse_real_ad_delta_theta.png"
        _plot_delta_theta_models(
            delta_plot,
            case=case,
            delta_theta=final_delta_theta[:, indices],
            labels=labels,
            coverage_mask=coverage_mask,
            depth=args.depth,
            clim=(args.delta_theta_vmin, args.delta_theta_vmax),
        )
        chi2_plot = output_dir / "timelapse_real_ad_delta_theta_chi2.png"
        plot_chi2(
            chi2_plot,
            np.asarray(result.all_chi2, dtype=float),
            xlabel="Window index" if args.inversion_mode == "windowed" else "Iteration",
        )
        plot_files = {"delta_theta": str(delta_plot), "chi2": str(chi2_plot)}

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "sensor_reference_dir": str(sensor_reference_dir),
        "mesh_file": None if mesh_file is None else str(mesh_file),
        "baseline_resistivity_file": str(baseline_resistivity_file),
        "baseline_column": int(args.baseline_column),
        "use_temperature_correction": bool(args.use_temperature_correction),
        "temperature_correction_factor_file": None
        if temperature_correction_factor_file is None
        else str(temperature_correction_factor_file),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "file_stride": int(args.file_stride),
        "max_timesteps": None if args.max_timesteps is None else int(args.max_timesteps),
        "n_timesteps": int(len(records)),
        "first_timestamp": timestamp_labels[0],
        "last_timestamp": timestamp_labels[-1],
        "n_electrodes": int(records[0].elec_x.size),
        "n_measurements_raw": int(first.rhoa.size),
        "n_measurements_used": int(observed_rhoa.shape[1]),
        "data_stride": int(args.data_stride),
        "max_error": None if args.max_error is None else float(args.max_error),
        "rhoa_min": float(np.min(observed_rhoa)),
        "rhoa_max": float(np.max(observed_rhoa)),
        "rhoa_median": float(np.median(observed_rhoa)),
        "elevation_reference": float(records[0].elevation_reference),
        "mesh_cells": int(case.mesh.cell_count),
        "mesh_nodes": int(case.mesh.node_count),
        "forward_mesh_cells": int(case.forward_mesh.cell_count),
        "forward_mesh_nodes": int(case.forward_mesh.node_count),
        "depth": float(args.depth),
        "n_layers": int(args.n_layers),
        "layer_stretch": float(args.layer_stretch),
        "data_misfit": str(args.data_misfit),
        "regularization": float(args.regularization),
        "regularization_domain": str(args.regularization_domain),
        "physical_regularization_quantity": str(args.physical_regularization_quantity),
        "temporal_regularization": float(args.temporal_regularization),
        "temporal_regularization_type": str(args.temporal_regularization_type),
        "freeze_baseline_theta": bool(args.freeze_baseline_theta),
        "regularization_mode": str(args.regularization_mode),
        "spatial_regularization": str(args.spatial_regularization),
        "optimizer": str(args.optimizer),
        "linearized_solver": str(args.linearized_solver),
        "linear_solver_backend": str(args.linear_solver_backend),
        "max_iterations": int(args.max_iterations),
        "final_chi2": float(result.iteration_chi2[-1]) if result.iteration_chi2 else None,
        "coverage_percentile": float(args.coverage_percentile),
        "coverage_threshold": coverage_threshold,
        "saturation_exponent_n": float(args.saturation_exponent),
        "theta_min": float(args.theta_min),
        "theta_max": float(args.theta_max),
        "theta0_min": float(np.min(theta0)),
        "theta0_max": float(np.max(theta0)),
        "sensor_constraint_lambda": float(args.sensor_constraint_lambda),
        "sensor_constraint_target_mode": str(args.sensor_constraint_target_mode),
        "sensor_constraint_sigma": float(args.sensor_constraint_sigma),
        "sensor_constraint_horizontal_radius": float(args.sensor_constraint_horizontal_radius),
        "sensor_constraint_vertical_radius": float(args.sensor_constraint_vertical_radius),
        "sensor_constraint_min_cells": int(args.sensor_constraint_min_cells),
        "sensor_constraint_rows": 0 if sensor_constraint_metadata is None else int(len(sensor_constraint_metadata)),
        "sensor_constraint_valid_points": 0
        if sensor_constraint_metadata is None
        else int(sensor_constraint_metadata["valid_points"].sum()),
        "delta_theta_min": float(np.min(final_delta_theta)),
        "delta_theta_max": float(np.max(final_delta_theta)),
        "delta_theta_definition": "final_water_content_models[:, t] - final_water_content_models[:, 0]",
        "water_content_offset_from_theta0_min": float(np.min(final_delta_theta_from_theta0)),
        "water_content_offset_from_theta0_max": float(np.max(final_delta_theta_from_theta0)),
        "temperature_correction_factor_min": float(np.min(temperature_correction_factor)),
        "temperature_correction_factor_max": float(np.max(temperature_correction_factor)),
        "elapsed_sec": float(elapsed_sec),
        "elapsed_min": float(elapsed_sec / 60.0),
        "plot_files": plot_files,
        "torch_enable_float64": bool(torch_runtime.config.torch_enable_float64),
        "temperature_note": (
            "Temperature correction is included in the petrophysical chain: "
            "relative Archie maps theta to rho at Tref, then rho_field = rho_Tref / C_T for forward prediction."
            if args.use_temperature_correction
            else "Temperature correction disabled; relative Archie maps theta directly to field-temperature rho."
        ),
        "sensor_constraint_metrics": sensor_metrics,
    }
    summary.update(run_meta)
    write_json(output_dir / "timelapse_real_ad_delta_theta_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
