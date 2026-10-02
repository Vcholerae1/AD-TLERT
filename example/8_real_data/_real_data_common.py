from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from adtlert.inversion import ParameterizedERTForward2p5D
from adtlert.workflows import build_source_position_triangle_inversion_case


@dataclass(frozen=True)
class RealERTData:
    """Parsed real ERT file in ADTLERT-ready local coordinates."""

    path: Path
    rhoa: np.ndarray
    measurements: np.ndarray
    elec_x: np.ndarray
    elec_z: np.ndarray
    elec_elevation: np.ndarray
    err: np.ndarray | None
    valid: np.ndarray
    timestamp: datetime | None
    elevation_reference: float

    def with_measurement_mask(self, mask: np.ndarray) -> "RealERTData":
        mask_array = np.asarray(mask, dtype=bool).ravel()
        if mask_array.shape != self.rhoa.shape:
            raise ValueError(
                f"mask shape {mask_array.shape} does not match rhoa shape {self.rhoa.shape}"
            )
        return replace(
            self,
            rhoa=self.rhoa[mask_array],
            measurements=self.measurements[mask_array],
            err=None if self.err is None else self.err[mask_array],
            valid=self.valid[mask_array],
        )


def find_project_root(start: Path) -> Path:
    """Find the repository root from a script path or current working directory."""

    start_path = start.resolve()
    candidates = [start_path] if start_path.is_dir() else [start_path.parent]
    candidates.extend(candidates[0].parents)
    for candidate in candidates:
        if (candidate / "adtlert").exists() and (candidate / "ProcessedData").exists():
            return candidate
    raise FileNotFoundError(
        "Cannot locate project root containing adtlert/ and ProcessedData/."
    )


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def parse_processed_timestamp(path: Path) -> datetime | None:
    match = re.match(r"(\d{4}-\d{2}-\d{2})_(\d{4})\.txt$", path.name)
    if match is None:
        return None
    return datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%Y-%m-%d_%H%M")


def _parse_bound(value: str | None, *, end: bool) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        day = datetime.strptime(text, "%Y-%m-%d")
        return day + timedelta(days=1) if end else day
    for fmt in ("%Y-%m-%d_%H%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return datetime.fromisoformat(text)


def discover_processed_files(
    input_dir: Path,
    *,
    start: str | None = None,
    end: str | None = None,
    file_stride: int = 1,
    max_timesteps: int | None = None,
) -> list[Path]:
    if file_stride < 1:
        raise ValueError("file_stride must be >= 1")
    if max_timesteps is not None and max_timesteps < 2:
        raise ValueError("max_timesteps must be >= 2 when set")

    start_dt = _parse_bound(start, end=False)
    end_dt = _parse_bound(end, end=True)
    records: list[tuple[datetime, Path]] = []
    for path in input_dir.glob("*.txt"):
        timestamp = parse_processed_timestamp(path)
        if timestamp is None:
            continue
        if start_dt is not None and timestamp < start_dt:
            continue
        if end_dt is not None and timestamp >= end_dt:
            continue
        records.append((timestamp, path))

    records.sort(key=lambda item: item[0])
    paths = [path for _, path in records][::file_stride]
    if max_timesteps is not None:
        paths = paths[:max_timesteps]
    return paths


def read_processed_ert(
    path: Path, *, elevation_reference: float | None = None
) -> RealERTData:
    """Read the ProcessedData text format used by the real hillslope dataset.

    The file stores electrodes as x, y, elevation. ADTLERT uses a 2D x-z
    coordinate system, so z is converted to relative elevation by subtracting
    the profile maximum elevation unless an explicit reference is supplied.
    """

    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    if len(lines) < 5:
        raise ValueError(f"{path}: file is too short")

    electrode_count = int(lines[0].split()[0])
    electrode_start = 2
    electrode_stop = electrode_start + electrode_count
    electrodes = np.loadtxt(lines[electrode_start:electrode_stop], dtype=float)
    if electrodes.shape != (electrode_count, 3):
        raise ValueError(
            f"{path}: expected {electrode_count} electrode rows with x y z columns"
        )

    data_count_index = electrode_stop
    measurement_count = int(lines[data_count_index].split()[0])
    header_index = data_count_index + 1
    columns = lines[header_index].lstrip("#").split()
    data_start = header_index + 1
    data_stop = data_start + measurement_count
    if data_stop > len(lines):
        raise ValueError(f"{path}: expected {measurement_count} measurement rows")
    table = np.loadtxt(lines[data_start:data_stop], dtype=float)
    if table.ndim == 1:
        table = table[None, :]
    if table.shape[0] != measurement_count:
        raise ValueError(
            f"{path}: parsed measurement count {table.shape[0]} != {measurement_count}"
        )

    column_index = {name: index for index, name in enumerate(columns)}
    required = {"a", "b", "m", "n", "rhoa"}
    missing = required.difference(column_index)
    if missing:
        raise KeyError(
            f"{path}: missing required measurement columns: {sorted(missing)}"
        )

    measurements = (
        table[:, [column_index[name] for name in ("a", "b", "m", "n")]].astype(np.int32)
        - 1
    )
    if np.any(measurements < 0) or np.any(measurements >= electrode_count):
        raise ValueError(
            f"{path}: ABMN indices must be one-based electrode ids in [1, {electrode_count}]"
        )

    rhoa = np.asarray(table[:, column_index["rhoa"]], dtype=float)
    err = (
        np.asarray(table[:, column_index["err"]], dtype=float)
        if "err" in column_index
        else None
    )
    valid = (
        np.asarray(table[:, column_index["valid"]] > 0.0, dtype=bool)
        if "valid" in column_index
        else np.ones_like(rhoa, dtype=bool)
    )

    elec_x = np.asarray(electrodes[:, 0], dtype=float)
    elec_elevation = np.asarray(electrodes[:, 2], dtype=float)
    if np.any(np.diff(elec_x) <= 0.0):
        raise ValueError(f"{path}: electrode x positions must be strictly increasing")
    z_reference = float(
        np.nanmax(elec_elevation)
        if elevation_reference is None
        else elevation_reference
    )
    elec_z = elec_elevation - z_reference

    return RealERTData(
        path=path,
        rhoa=rhoa,
        measurements=measurements,
        elec_x=elec_x,
        elec_z=elec_z,
        elec_elevation=elec_elevation,
        err=err,
        valid=valid,
        timestamp=parse_processed_timestamp(path),
        elevation_reference=z_reference,
    )


def quality_mask(data: RealERTData, *, max_error: float | None = None) -> np.ndarray:
    mask = np.asarray(data.valid, dtype=bool).copy()
    mask &= np.isfinite(data.rhoa) & (data.rhoa > 0.0)
    if data.err is not None:
        mask &= np.isfinite(data.err)
        if max_error is not None:
            mask &= data.err <= float(max_error)
    return mask


def apply_data_stride(mask: np.ndarray, stride: int) -> np.ndarray:
    if stride < 1:
        raise ValueError("data_stride must be >= 1")
    mask_array = np.asarray(mask, dtype=bool).copy()
    if stride == 1:
        return mask_array
    selected = np.zeros_like(mask_array, dtype=bool)
    selected[np.flatnonzero(mask_array)[::stride]] = True
    return selected


def data_std_from_err(
    err: np.ndarray | None,
    *,
    shape: tuple[int, ...],
    relative_error: float,
    minimum_log_std: float = 1.0e-3,
) -> float | np.ndarray:
    if err is None:
        return float(max(np.log1p(float(relative_error)), minimum_log_std))
    std = np.log1p(np.asarray(err, dtype=float))
    std = np.broadcast_to(std, shape).astype(float, copy=True)
    return np.maximum(std, float(minimum_log_std))


def stretched_layer_thickness(
    depth: float, n_layers: int, stretch: float
) -> np.ndarray:
    if depth <= 0.0:
        raise ValueError("depth must be positive")
    if n_layers < 2:
        raise ValueError("n_layers must be >= 2")
    if stretch <= 0.0:
        raise ValueError("layer_stretch must be positive")
    if abs(stretch - 1.0) < 1.0e-12:
        return np.full(int(n_layers), float(depth) / int(n_layers), dtype=float)
    raw = float(stretch) ** np.arange(int(n_layers), dtype=float)
    return raw * (float(depth) / float(raw.sum()))


def build_real_geometry(
    data: RealERTData, *, depth: float, n_layers: int, layer_stretch: float
) -> dict[str, np.ndarray]:
    return {
        "x_nodes": np.asarray(data.elec_x, dtype=float).copy(),
        "z_top": np.asarray(data.elec_z, dtype=float).copy(),
        "layer_thickness": stretched_layer_thickness(depth, n_layers, layer_stretch),
    }


def build_real_inversion_case(
    data: RealERTData,
    *,
    depth: float,
    n_layers: int,
    layer_stretch: float,
    inversion_mesh_quality: float,
    inversion_mesh_smoothing_iterations: int,
    mesh_file: Path | None = None,
):
    geometry = build_real_geometry(
        data, depth=depth, n_layers=n_layers, layer_stretch=layer_stretch
    )
    return build_source_position_triangle_inversion_case(
        data.elec_x,
        data.elec_z,
        data.measurements,
        geometry["x_nodes"],
        geometry["z_top"],
        geometry["layer_thickness"],
        y_index=0,
        quality=float(inversion_mesh_quality),
        smoothing_iterations=int(inversion_mesh_smoothing_iterations),
        data_file=None,
        mesh_file=mesh_file,
    )


def build_parameterized_forward(case, *, terrain_cache_dir: Path | None = None):
    return ParameterizedERTForward2p5D.from_mesh_survey(
        case.forward_mesh,
        case.survey,
        case.parameter_cell_ids,
        regularization_mesh=case.mesh,
        terrain_cache_dir=terrain_cache_dir,
    )


def save_mesh_npz(path: Path, case) -> None:
    np.savez(
        path,
        nodes=np.asarray(case.mesh.nodes, dtype=float),
        cells=np.asarray(case.mesh.cells, dtype=np.int32),
        forward_nodes=np.asarray(case.forward_mesh.nodes, dtype=float),
        forward_cells=np.asarray(case.forward_mesh.cells, dtype=np.int32),
        cell_markers=np.asarray(case.cell_markers, dtype=np.int32),
        parameter_cell_ids=np.asarray(case.parameter_cell_ids, dtype=np.int32),
        elec_x=np.asarray(case.elec_x, dtype=float),
        elec_z=np.asarray(case.elec_z, dtype=float),
        x_nodes=np.asarray(case.x_nodes, dtype=float),
        z_top=np.asarray(case.z_top, dtype=float),
        layer_thickness=np.asarray(case.layer_thickness, dtype=float),
    )


def _positive_limits(
    values: np.ndarray, *, vmin: float | None = None, vmax: float | None = None
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    if finite.size == 0:
        raise ValueError("plot values must contain at least one finite positive value")
    lower = float(np.nanpercentile(finite, 2.0) if vmin is None else vmin)
    upper = float(np.nanpercentile(finite, 98.0) if vmax is None else vmax)
    lower = max(lower, np.finfo(float).tiny)
    if upper <= lower:
        upper = lower * 1.01
    return lower, upper


def plot_resistivity_model(
    path: Path,
    *,
    case,
    model: np.ndarray,
    coverage_mask: np.ndarray | None = None,
    title: str = "",
    depth: float,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "turbo",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LogNorm

    values = np.asarray(model, dtype=float).ravel()
    nodes = np.asarray(case.mesh.nodes, dtype=float)
    cells = np.asarray(case.mesh.cells, dtype=np.int32)
    visible = np.isfinite(values) & (values > 0.0)
    if coverage_mask is not None:
        visible &= ~np.asarray(coverage_mask, dtype=bool).ravel()
    if not np.any(visible):
        visible = np.isfinite(values) & (values > 0.0)

    lower, upper = _positive_limits(values[visible], vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(11.0, 4.4))
    collection = PolyCollection(
        nodes[cells][visible],
        array=values[visible],
        cmap=cmap,
        norm=LogNorm(vmin=lower, vmax=upper),
        edgecolors="none",
    )
    ax.add_collection(collection)
    ax.plot(case.elec_x, case.elec_z, color="black", lw=1.4)
    ax.scatter(case.elec_x, case.elec_z, s=6, color="black", zorder=3)
    ax.set_xlim(float(np.min(case.elec_x)), float(np.max(case.elec_x)))
    z_top = float(np.max(case.elec_z))
    ax.set_ylim(z_top - float(depth), z_top + 3.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Relative elevation (m)")
    ax.set_title(title)
    cbar = fig.colorbar(collection, ax=ax, fraction=0.030, pad=0.02)
    cbar.ax.set_title(r"$\rho$ [$\Omega$m]")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_chi2(path: Path, chi2: np.ndarray, *, xlabel: str = "Iteration") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chi2_values = np.asarray(chi2, dtype=float).ravel()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(np.arange(1, chi2_values.size + 1), chi2_values, marker="o", lw=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\chi^2$")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_timelapse_models(
    path: Path,
    *,
    case,
    models: np.ndarray,
    labels: list[str],
    coverage_mask: np.ndarray | None,
    depth: float,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "turbo",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LogNorm

    model_matrix = np.asarray(models, dtype=float)
    n_panels = model_matrix.shape[1]
    visible_base = np.ones(model_matrix.shape[0], dtype=bool)
    if coverage_mask is not None:
        visible_base &= ~np.asarray(coverage_mask, dtype=bool).ravel()
    plot_values = model_matrix[visible_base, :].reshape(-1)
    lower, upper = _positive_limits(plot_values, vmin=vmin, vmax=vmax)
    norm = LogNorm(vmin=lower, vmax=upper)

    nodes = np.asarray(case.mesh.nodes, dtype=float)
    cells = np.asarray(case.mesh.cells, dtype=np.int32)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.4 * n_panels, 4.0), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    last = None
    for ax, column, label in zip(axes, range(n_panels), labels, strict=False):
        values = model_matrix[:, column]
        visible = visible_base & np.isfinite(values) & (values > 0.0)
        if not np.any(visible):
            visible = np.isfinite(values) & (values > 0.0)
        last = PolyCollection(
            nodes[cells][visible],
            array=values[visible],
            cmap=cmap,
            norm=norm,
            edgecolors="none",
        )
        ax.add_collection(last)
        ax.plot(case.elec_x, case.elec_z, color="black", lw=1.2)
        ax.set_title(label)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(float(np.min(case.elec_x)), float(np.max(case.elec_x)))
        z_top = float(np.max(case.elec_z))
        ax.set_ylim(z_top - float(depth), z_top + 3.0)
        ax.set_xlabel("Distance (m)")
    axes[0].set_ylabel("Relative elevation (m)")
    if last is not None:
        cbar = fig.colorbar(last, ax=axes, fraction=0.018, pad=0.02)
        cbar.ax.set_title(r"$\rho$ [$\Omega$m]")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def check_same_layout(first, current) -> None:
    if not np.array_equal(current.measurements, first.measurements):
        raise ValueError(
            f"{current.path}: ABMN layout differs from first timestep {first.path}"
        )
    if not np.allclose(current.elec_x, first.elec_x, rtol=0.0, atol=1.0e-10):
        raise ValueError(
            f"{current.path}: electrode x positions differ from first timestep"
        )
    if not np.allclose(current.elec_z, first.elec_z, rtol=0.0, atol=1.0e-10):
        raise ValueError(
            f"{current.path}: electrode z positions differ from first timestep"
        )


def selected_plot_indices(n_times: int, max_panels: int) -> np.ndarray:
    if n_times <= max_panels:
        return np.arange(n_times, dtype=np.int32)
    return np.unique(np.round(np.linspace(0, n_times - 1, max_panels)).astype(np.int32))
