"""Terrain-following ERT forward helpers used by the ParFlow notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import jax.numpy as jnp
import numpy as np

from deepert.forward import ERTForward2p5D
from deepert.mesh import Mesh
from deepert.survey import Survey


@dataclass(frozen=True)
class ParflowGrid:
    """ParFlow grid settings needed to rebuild a 2D terrain slice."""

    dx: float
    dy: float
    dz_base: float
    nx: int
    ny: int
    nz: int
    dz_scales: np.ndarray


@dataclass(frozen=True)
class TerrainForwardCase:
    """Native forward inputs and metadata for one terrain-following timestep."""

    mesh: Mesh
    survey: Survey
    resistivity: np.ndarray
    elec_x: np.ndarray
    elec_z: np.ndarray
    x_nodes: np.ndarray
    z_top: np.ndarray
    layer_thickness: np.ndarray
    y_index: int

    def with_resistivity(self, resistivity: np.ndarray) -> "TerrainForwardCase":
        """Return the same terrain geometry with a different cell resistivity vector."""

        resistivity_array = np.asarray(resistivity, dtype=float)
        if resistivity_array.shape != self.resistivity.shape:
            raise ValueError(f"resistivity must have shape {self.resistivity.shape}")
        if not np.all(np.isfinite(resistivity_array)):
            raise ValueError("resistivity contains non-finite values")
        if np.any(resistivity_array <= 0.0):
            raise ValueError("resistivity must contain positive values")

        return TerrainForwardCase(
            mesh=self.mesh,
            survey=self.survey,
            resistivity=resistivity_array,
            elec_x=self.elec_x,
            elec_z=self.elec_z,
            x_nodes=self.x_nodes,
            z_top=self.z_top,
            layer_thickness=self.layer_thickness,
            y_index=self.y_index,
        )


@dataclass(frozen=True)
class SourcePositionInversionCase:
    """Triangle inversion mesh generated from source/electrode positions."""

    mesh: Mesh
    survey: Survey
    elec_x: np.ndarray
    elec_z: np.ndarray
    x_nodes: np.ndarray
    z_top: np.ndarray
    layer_thickness: np.ndarray
    y_index: int


@dataclass(frozen=True)
class TerrainForwardRecord:
    """Manifest row for one terrain-forward timestep."""

    step: int
    input_file: str
    dat_file: str
    npz_file: str
    status: str
    rhoa_min: float | None = None
    rhoa_max: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class TerrainForwardRunner:
    """Reusable terrain-forward operator for a fixed mesh, survey, and topography."""

    case_template: TerrainForwardCase
    forward: ERTForward2p5D
    reuse_solver_state: bool = False

    @classmethod
    def from_case(
        cls,
        case: TerrainForwardCase,
        *,
        linear_solver_backend: str = "auto",
        reuse_solver_state: bool = False,
        terrain_cache_dir: str | Path | None = None,
        prepare_forward: bool = False,
    ) -> "TerrainForwardRunner":
        """Build a reusable forward operator from one terrain case.

        By default each solve rebuilds cuDSS solver state while retaining the
        geometry, JAX kernels, and auxiliary-field caches. Reusing cuDSS solver
        state keeps only symbolic/plan/buffer state across models; each changed
        conductivity still updates matrix values and refactorizes before solve.
        """

        forward_kwargs = {"linear_solver_backend": linear_solver_backend}
        if terrain_cache_dir is not None:
            forward_kwargs["terrain_cache_dir"] = terrain_cache_dir
        forward = ERTForward2p5D.from_mesh_survey(case.mesh, case.survey, **forward_kwargs)
        runner = cls(case_template=case, forward=forward, reuse_solver_state=bool(reuse_solver_state))
        if prepare_forward:
            runner.prepare_resistivity(case.resistivity)
        return runner

    def case_with_resistivity(self, resistivity: np.ndarray) -> TerrainForwardCase:
        """Return a case for this fixed geometry and a new resistivity vector."""

        return self.case_template.with_resistivity(resistivity)

    def solve_resistivity(self, resistivity: np.ndarray) -> np.ndarray:
        """Compute apparent resistivity for a cell resistivity vector."""

        if not self.reuse_solver_state:
            self.forward.close()
        conductivity = jnp.asarray(1.0 / np.asarray(resistivity, dtype=float), dtype=jnp.float32)
        rhoa = np.asarray(self.forward.solve(conductivity=conductivity).apparent_resistivity, dtype=float)
        if not np.isfinite(rhoa).all():
            raise ValueError("forward returned non-finite apparent resistivity values")
        return rhoa

    def prepare_resistivity(self, resistivity: np.ndarray) -> None:
        """Pre-populate caches for a representative terrain resistivity vector."""

        conductivity = jnp.asarray(1.0 / np.asarray(resistivity, dtype=float), dtype=jnp.float32)
        self.forward.prepare(conductivity, include_solver_state=self.reuse_solver_state)

    def solve_case(self, case: TerrainForwardCase) -> np.ndarray:
        """Compute apparent resistivity for a terrain case sharing this geometry."""

        return self.solve_resistivity(case.resistivity)

    def close(self) -> None:
        """Release cached GPU solver resources held by the reusable operator."""

        self.forward.close()


def parse_resistivity_slice_name(path: str | Path) -> tuple[int, int]:
    """Parse ``(y_index, timestep)`` from ``resistivity2d_y{y}_t{step}.npy``."""

    name = Path(path).name
    match = re.search(r"resistivity2d_y(\d+)_t(\d+)\.npy$", name)
    if match is None:
        raise ValueError(f"cannot parse y-index and timestep from filename: {name}")
    return int(match.group(1)), int(match.group(2))


def discover_resistivity_slices(
    input_dir: str | Path,
    *,
    y_index: int | None = None,
    file_stride: int = 1,
    max_steps: int | None = None,
) -> list[tuple[int, Path]]:
    """Return sorted ``(step, path)`` pairs for terrain resistivity slices."""

    if file_stride < 1:
        raise ValueError("file_stride must be >= 1")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be >= 1 when set")

    root = Path(input_dir)
    pairs: list[tuple[int, Path]] = []
    for path in root.glob("resistivity2d_y*_t*.npy"):
        try:
            found_y_index, step = parse_resistivity_slice_name(path)
        except ValueError:
            continue
        if y_index is not None and found_y_index != y_index:
            continue
        pairs.append((step, path))

    pairs.sort(key=lambda item: item[0])
    pairs = pairs[::file_stride]
    if max_steps is not None:
        pairs = pairs[:max_steps]
    return pairs


def parse_pftcl(path: str | Path) -> ParflowGrid:
    """Parse ParFlow grid dimensions and ``dzScale`` values from a pftcl file."""

    pftcl_path = Path(path)
    values: dict[str, float | int] = {}
    dz_scales: dict[int, float] = {}

    for line in pftcl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        for key in ("DX", "DY", "DZ"):
            match = re.search(rf'ComputationalGrid\.{key}\s+"([0-9eE+\-.]+)"', line)
            if match is not None:
                values[key.lower()] = float(match.group(1))
        for key in ("NX", "NY", "NZ"):
            match = re.search(rf'ComputationalGrid\.{key}\s+"(\d+)"', line)
            if match is not None:
                values[key.lower()] = int(match.group(1))
        match = re.search(r'Cell\.(\d+)\.dzScale\.Value\s+"([0-9eE+\-.]+)"', line)
        if match is not None:
            dz_scales[int(match.group(1))] = float(match.group(2))

    missing = [key for key in ("dx", "dy", "dz", "nx", "ny", "nz") if key not in values]
    if missing:
        raise ValueError(f"failed to parse ComputationalGrid settings from {pftcl_path}: missing {missing}")

    nz = int(values["nz"])
    if len(dz_scales) != nz:
        raise ValueError(f"dzScale count {len(dz_scales)} does not match NZ={nz}")

    return ParflowGrid(
        dx=float(values["dx"]),
        dy=float(values["dy"]),
        dz_base=float(values["dz"]),
        nx=int(values["nx"]),
        ny=int(values["ny"]),
        nz=nz,
        dz_scales=np.asarray([dz_scales[idx] for idx in range(nz)], dtype=float),
    )


def read_slope_x(path: str | Path, y_index: int) -> np.ndarray:
    """Read one ``slope_x`` y-slice from a ParFlow PFB file."""

    try:
        from parflow.tools.io import read_pfb
    except ImportError as exc:
        raise ImportError(
            "Reading ParFlow PFB files requires the examples extra: "
            "`uv sync --extra examples`."
        ) from exc

    slope_x_3d = np.asarray(read_pfb(str(path)), dtype=float)
    if slope_x_3d.ndim != 3:
        raise ValueError(f"expected slope_x PFB to load as a 3D array, got shape={slope_x_3d.shape}")
    if not 0 <= y_index < slope_x_3d.shape[1]:
        raise ValueError(f"y_index={y_index} out of range for slope_x shape={slope_x_3d.shape}")
    return slope_x_3d[0, y_index, :]


def build_wenner_alpha_measurements(electrode_count: int) -> np.ndarray:
    """Return reference ``schemeName='wa'`` ABMN ordering for a linear line."""

    if electrode_count < 4:
        raise ValueError("Wenner-alpha surveys need at least four electrodes")

    measurements: list[list[int]] = []
    for spacing in range(1, electrode_count // 3 + 1):
        for start in range(electrode_count - 3 * spacing):
            measurements.append([start, start + 3 * spacing, start + spacing, start + 2 * spacing])
    return np.asarray(measurements, dtype=np.int32)


def _source_depth_levels(max_depth: float, electrode_spacing: float, depth_levels: int) -> np.ndarray:
    if depth_levels < 2:
        raise ValueError("depth_levels must be >= 2")
    if not np.isfinite(max_depth) or max_depth <= 0.0:
        raise ValueError("max_depth must be positive")
    if not np.isfinite(electrode_spacing) or electrode_spacing <= 0.0:
        raise ValueError("electrode_spacing must be positive")

    if depth_levels == 2:
        return np.asarray([0.0, max_depth], dtype=float)

    first_depth = max(electrode_spacing * 0.5, max_depth * 0.02)
    if first_depth >= max_depth:
        return np.linspace(0.0, max_depth, depth_levels, dtype=float)

    levels = np.concatenate(([0.0], np.geomspace(first_depth, max_depth, depth_levels - 1)))
    levels[-1] = max_depth
    return levels.astype(float, copy=False)


def build_source_position_triangle_inversion_case(
    elec_x: np.ndarray,
    elec_z: np.ndarray,
    measurements: np.ndarray,
    x_nodes: np.ndarray,
    z_top: np.ndarray,
    layer_thickness: np.ndarray,
    *,
    y_index: int,
    depth_levels: int = 11,
) -> SourcePositionInversionCase:
    """Build a source-position driven triangular inversion mesh.

    The forward notebooks let pyGIMLi create ``paraDomain`` from the ERT source
    positions instead of reusing the structured ParFlow forward grid. This helper
    mirrors that modelling choice in native deepert: electrodes define the top
    mesh row, depth levels are generated automatically from electrode spacing and
    model depth, and each strip is split into triangles.
    """

    elec_x_array = np.asarray(elec_x, dtype=float).ravel()
    elec_z_array = np.asarray(elec_z, dtype=float).ravel()
    if elec_x_array.shape != elec_z_array.shape:
        raise ValueError("elec_x and elec_z must have the same shape")
    if elec_x_array.size < 4:
        raise ValueError("at least four source/electrode positions are required")
    if not np.all(np.isfinite(elec_x_array)) or not np.all(np.isfinite(elec_z_array)):
        raise ValueError("electrode positions contain non-finite values")
    if np.any(np.diff(elec_x_array) <= 0.0):
        raise ValueError("elec_x must be strictly increasing")

    measurement_array = np.asarray(measurements, dtype=np.int32)
    if measurement_array.ndim != 2 or measurement_array.shape[1] != 4:
        raise ValueError("measurements must have shape (n_measurements, 4)")
    if np.any(measurement_array < 0) or np.any(measurement_array >= elec_x_array.size):
        raise ValueError("measurements reference electrodes outside the source positions")

    x_node_array = np.asarray(x_nodes, dtype=float).ravel()
    z_top_array = np.asarray(z_top, dtype=float).ravel()
    thickness_array = np.asarray(layer_thickness, dtype=float).ravel()
    if x_node_array.shape != z_top_array.shape:
        raise ValueError("x_nodes and z_top must have the same shape")
    if x_node_array.size < 2:
        raise ValueError("x_nodes must contain at least two nodes")
    if np.any(np.diff(x_node_array) <= 0.0):
        raise ValueError("x_nodes must be strictly increasing")
    if not np.all(np.isfinite(thickness_array)) or np.any(thickness_array <= 0.0):
        raise ValueError("layer_thickness must contain positive finite values")

    electrode_spacing = float(np.median(np.diff(elec_x_array)))
    depths = _source_depth_levels(float(np.sum(thickness_array)), electrode_spacing, int(depth_levels))

    rows = []
    for row_index, depth in enumerate(depths):
        if row_index == 0:
            row_x = elec_x_array
            row_z_top = elec_z_array
        else:
            offset = (0.35 * electrode_spacing) * (1.0 if row_index % 2 else -1.0)
            row_x = np.clip(elec_x_array + offset, elec_x_array[0], elec_x_array[-1])
            row_x[0] = elec_x_array[0]
            row_x[-1] = elec_x_array[-1]
            row_z_top = np.interp(row_x, x_node_array, z_top_array)
        rows.append(np.column_stack((row_x, row_z_top - depth)))
    nodes = np.vstack(rows)

    electrode_count = int(elec_x_array.size)

    def node_id(row_index: int, electrode_index: int) -> int:
        return row_index * electrode_count + electrode_index

    cells: list[list[int]] = []
    for row_index in range(len(depths) - 1):
        for electrode_index in range(electrode_count - 1):
            top_left = node_id(row_index, electrode_index)
            top_right = node_id(row_index, electrode_index + 1)
            bottom_left = node_id(row_index + 1, electrode_index)
            bottom_right = node_id(row_index + 1, electrode_index + 1)
            if (row_index + electrode_index) % 2 == 0:
                cells.append([top_left, top_right, bottom_right])
                cells.append([top_left, bottom_right, bottom_left])
            else:
                cells.append([top_left, top_right, bottom_left])
                cells.append([top_right, bottom_right, bottom_left])

    mesh = Mesh.from_arrays(
        jnp.asarray(nodes),
        jnp.asarray(cells, dtype=jnp.int32),
        surface_node_ids=jnp.arange(electrode_count, dtype=jnp.int32),
    )
    survey = Survey.from_arrays(
        jnp.asarray(np.column_stack((elec_x_array, elec_z_array))),
        jnp.asarray(measurement_array),
    )
    return SourcePositionInversionCase(
        mesh=mesh,
        survey=survey,
        elec_x=elec_x_array,
        elec_z=elec_z_array,
        x_nodes=x_node_array,
        z_top=z_top_array,
        layer_thickness=thickness_array,
        y_index=int(y_index),
    )


def _terrain_resistivity_vector(rho_2d: np.ndarray, grid: ParflowGrid) -> np.ndarray:
    """Validate and flatten a ParFlow bottom-to-top resistivity slice."""

    rho_array = np.asarray(rho_2d, dtype=float)
    if rho_array.shape != (grid.nz, grid.nx):
        raise ValueError(f"resistivity shape {rho_array.shape} does not match parsed grid {(grid.nz, grid.nx)}")
    if not np.all(np.isfinite(rho_array)):
        raise ValueError("resistivity contains non-finite values")
    if np.any(rho_array <= 0.0):
        raise ValueError("resistivity must contain positive values")

    return np.asarray(rho_array[::-1, :].reshape(-1), dtype=float)


def build_terrain_forward_case(
    rho_2d: np.ndarray,
    grid: ParflowGrid,
    slope_x: np.ndarray,
    *,
    y_index: int,
    n_electrodes: int = 48,
    topo_offset: float = 0.0,
) -> TerrainForwardCase:
    """Build native deepert mesh/survey/model arrays for one ParFlow slice.

    ``rho_2d`` is expected in ParFlow z-order, bottom-to-top. The returned
    resistivity vector is top-to-bottom and cell-aligned with the generated
    terrain-following mesh.
    """

    resistivity = _terrain_resistivity_vector(rho_2d, grid)
    slope_array = np.asarray(slope_x, dtype=float).ravel()
    if not 0 <= y_index < grid.ny:
        raise ValueError(f"y_index={y_index} out of range [0, {grid.ny - 1}]")
    if slope_array.shape != (grid.nx,):
        raise ValueError(f"slope_x shape {slope_array.shape} does not match NX={grid.nx}")

    layer_thickness = (grid.dz_base * grid.dz_scales)[::-1]
    y_offsets = np.concatenate(([0.0], -np.cumsum(layer_thickness)))

    x_nodes = np.arange(grid.nx + 1, dtype=float) * grid.dx
    z_top = np.zeros(grid.nx + 1, dtype=float)
    z_top[1:] = np.cumsum(slope_array * grid.dx)
    z_top = z_top + topo_offset

    nodes = np.asarray(
        [[x_coord, z_coord + offset] for offset in y_offsets for x_coord, z_coord in zip(x_nodes, z_top, strict=False)],
        dtype=float,
    )

    def node_id(layer_index: int, column_index: int) -> int:
        return layer_index * (grid.nx + 1) + column_index

    cells: list[list[int]] = []
    for layer_index in range(grid.nz):
        for column_index in range(grid.nx):
            cells.append(
                [
                    node_id(layer_index, column_index),
                    node_id(layer_index, column_index + 1),
                    node_id(layer_index + 1, column_index + 1),
                    node_id(layer_index + 1, column_index),
                ]
            )

    electrode_count = min(int(n_electrodes), grid.nx + 1)
    elec_x = np.linspace(float(x_nodes.min()), float(x_nodes.max()), electrode_count)
    elec_z = np.interp(elec_x, x_nodes, z_top)
    measurements = build_wenner_alpha_measurements(electrode_count)

    mesh = Mesh.from_arrays(
        jnp.asarray(nodes),
        jnp.asarray(cells),
        surface_node_ids=jnp.arange(grid.nx + 1, dtype=jnp.int32),
    )
    survey = Survey.from_arrays(
        jnp.asarray(np.column_stack((elec_x, elec_z))),
        jnp.asarray(measurements),
    )
    return TerrainForwardCase(
        mesh=mesh,
        survey=survey,
        resistivity=resistivity,
        elec_x=elec_x,
        elec_z=elec_z,
        x_nodes=x_nodes,
        z_top=z_top,
        layer_thickness=layer_thickness,
        y_index=int(y_index),
    )


def run_terrain_forward(
    case: TerrainForwardCase,
    *,
    linear_solver_backend: str = "auto",
    reuse_solver_state: bool = False,
    terrain_cache_dir: str | Path | None = None,
    prepare_forward: bool = False,
) -> np.ndarray:
    """Compute apparent resistivity for a terrain case."""

    runner = TerrainForwardRunner.from_case(
        case,
        linear_solver_backend=linear_solver_backend,
        reuse_solver_state=reuse_solver_state,
        terrain_cache_dir=terrain_cache_dir,
        prepare_forward=prepare_forward,
    )
    try:
        return runner.solve_case(case)
    finally:
        runner.close()


def save_terrain_forward_dat(
    path: str | Path,
    case: TerrainForwardCase,
    rhoa: np.ndarray,
    *,
    relative_error: float = 0.03,
) -> None:
    """Save a reference-style ERT ``.dat`` file matching the notebooks."""

    output_path = Path(path)
    rhoa_array = np.asarray(rhoa, dtype=float).ravel()
    if rhoa_array.shape != (case.survey.measurement_count,):
        raise ValueError(f"rhoa must have shape ({case.survey.measurement_count},)")

    electrodes = np.asarray(case.survey.electrode_positions, dtype=float)
    measurements = np.asarray(case.survey.measurements, dtype=np.int32)
    geometric_factors = np.asarray(case.survey.geometric_factors(), dtype=float)
    err = np.full(case.survey.measurement_count, float(relative_error), dtype=float)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        stream.write(f"{electrodes.shape[0]}\n")
        stream.write("# x y z\n")
        for x_coord, z_coord in electrodes:
            stream.write(f"{x_coord:.14g}\t{z_coord:.14g}\t0\n")
        stream.write(f"{measurements.shape[0]}\n")
        stream.write("# a b m n err i ip iperr k r rhoa u valid \n")
        for abmn, error_value, k_value, rhoa_value in zip(
            measurements,
            err,
            geometric_factors,
            rhoa_array,
            strict=True,
        ):
            a_idx, b_idx, m_idx, n_idx = abmn + 1
            stream.write(
                f"{a_idx}\t{b_idx}\t{m_idx}\t{n_idx}\t"
                f"{error_value:.14e}\t0.00000000000000e+00\t0.00000000000000e+00\t"
                f"0.00000000000000e+00\t{k_value:.14e}\t0.00000000000000e+00\t"
                f"{rhoa_value:.14e}\t0.00000000000000e+00\t1\n"
            )


def save_terrain_forward_npz(
    path: str | Path,
    case: TerrainForwardCase,
    rhoa: np.ndarray,
    *,
    relative_error: float = 0.03,
) -> None:
    """Save compact forward artifacts and terrain metadata for plotting."""

    output_path = Path(path)
    rhoa_array = np.asarray(rhoa, dtype=float).ravel()
    if rhoa_array.shape != (case.survey.measurement_count,):
        raise ValueError(f"rhoa must have shape ({case.survey.measurement_count},)")

    measurements = np.asarray(case.survey.measurements, dtype=np.int32)
    err = np.full(case.survey.measurement_count, float(relative_error), dtype=float)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        rhoa=rhoa_array,
        err=err,
        a=measurements[:, 0],
        b=measurements[:, 1],
        m=measurements[:, 2],
        n=measurements[:, 3],
        elec_x=case.elec_x,
        elec_z=case.elec_z,
        x_nodes=case.x_nodes,
        z_top=case.z_top,
        layer_thickness=case.layer_thickness,
        y_index=np.asarray([case.y_index], dtype=np.int32),
    )


def run_terrain_forward_file(
    input_file: str | Path,
    grid: ParflowGrid,
    slope_x: np.ndarray,
    output_dir: str | Path,
    *,
    y_index: int | None = None,
    n_electrodes: int = 48,
    topo_offset: float = 0.0,
    relative_error: float = 0.03,
    overwrite: bool = True,
    linear_solver_backend: str = "auto",
    reuse_solver_state: bool = False,
    terrain_cache_dir: str | Path | None = None,
    prepare_forward: bool = False,
) -> TerrainForwardRecord:
    """Run and save one terrain-forward timestep from a resistivity ``.npy`` file."""

    path = Path(input_file)
    parsed_y_index, step = parse_resistivity_slice_name(path)
    if y_index is None:
        y_index = parsed_y_index
    elif parsed_y_index != y_index:
        raise ValueError(f"input file y-index {parsed_y_index} does not match requested y_index={y_index}")

    output_root = Path(output_dir)
    dat_file = output_root / f"synthetic_ert_terrain_vardz_t{step:05d}.dat"
    npz_file = output_root / f"synthetic_ert_terrain_vardz_t{step:05d}.npz"
    if not overwrite and dat_file.exists() and npz_file.exists():
        return TerrainForwardRecord(
            step=step,
            input_file=str(path),
            dat_file=str(dat_file),
            npz_file=str(npz_file),
            status="skipped_existing",
        )

    rho_2d = np.asarray(np.load(path), dtype=float)
    case = build_terrain_forward_case(
        rho_2d,
        grid,
        slope_x,
        y_index=y_index,
        n_electrodes=n_electrodes,
        topo_offset=topo_offset,
    )
    rhoa = run_terrain_forward(
        case,
        linear_solver_backend=linear_solver_backend,
        reuse_solver_state=reuse_solver_state,
        terrain_cache_dir=terrain_cache_dir,
        prepare_forward=prepare_forward,
    )
    save_terrain_forward_dat(dat_file, case, rhoa, relative_error=relative_error)
    save_terrain_forward_npz(npz_file, case, rhoa, relative_error=relative_error)
    return TerrainForwardRecord(
        step=step,
        input_file=str(path),
        dat_file=str(dat_file),
        npz_file=str(npz_file),
        status="ok",
        rhoa_min=float(np.min(rhoa)),
        rhoa_max=float(np.max(rhoa)),
    )


def run_terrain_forward_series(
    input_files: list[str | Path] | list[tuple[int, str | Path]],
    grid: ParflowGrid,
    slope_x: np.ndarray,
    output_dir: str | Path,
    *,
    y_index: int | None = None,
    n_electrodes: int = 48,
    topo_offset: float = 0.0,
    relative_error: float = 0.03,
    overwrite: bool = True,
    linear_solver_backend: str = "auto",
    reuse_solver_state: bool = False,
    terrain_cache_dir: str | Path | None = None,
    prepare_forward: bool = False,
) -> tuple[list[TerrainForwardRecord], list[TerrainForwardRecord]]:
    """Run a sequential terrain-forward series and return ``(manifest, failures)``.

    The terrain mesh, survey, sparse pattern, auxiliary discretization, JAX
    kernels, and auxiliary-field caches are reused across successful timesteps
    with the same y-index. By default cuDSS solver state is rebuilt per solve to
    avoid non-physical negative apparent resistivities observed when reusing it
    across terrain timesteps. Set ``reuse_solver_state=True`` only for
    experimental timing runs.
    """

    normalized_files: list[Path] = []
    for item in input_files:
        if isinstance(item, tuple):
            _, path = item
            normalized_files.append(Path(path))
        else:
            normalized_files.append(Path(item))

    manifest: list[TerrainForwardRecord] = []
    failures: list[TerrainForwardRecord] = []
    runners: dict[int, TerrainForwardRunner] = {}
    output_root = Path(output_dir)
    try:
        for path in normalized_files:
            try:
                parsed_y_index, step = parse_resistivity_slice_name(path)
                resolved_y_index = parsed_y_index
                if y_index is not None:
                    if parsed_y_index != y_index:
                        raise ValueError(f"input file y-index {parsed_y_index} does not match requested y_index={y_index}")
                    resolved_y_index = int(y_index)

                dat_file = output_root / f"synthetic_ert_terrain_vardz_t{step:05d}.dat"
                npz_file = output_root / f"synthetic_ert_terrain_vardz_t{step:05d}.npz"
                if not overwrite and dat_file.exists() and npz_file.exists():
                    manifest.append(
                        TerrainForwardRecord(
                            step=step,
                            input_file=str(path),
                            dat_file=str(dat_file),
                            npz_file=str(npz_file),
                            status="skipped_existing",
                        )
                    )
                    continue

                rho_2d = np.asarray(np.load(path), dtype=float)
                runner = runners.get(resolved_y_index)
                if runner is None:
                    case = build_terrain_forward_case(
                        rho_2d,
                        grid,
                        slope_x,
                        y_index=resolved_y_index,
                        n_electrodes=n_electrodes,
                        topo_offset=topo_offset,
                    )
                    runner = TerrainForwardRunner.from_case(
                        case,
                        linear_solver_backend=linear_solver_backend,
                        reuse_solver_state=reuse_solver_state,
                        terrain_cache_dir=terrain_cache_dir,
                        prepare_forward=prepare_forward,
                    )
                    runners[resolved_y_index] = runner
                else:
                    case = runner.case_with_resistivity(_terrain_resistivity_vector(rho_2d, grid))

                rhoa = runner.solve_case(case)
                save_terrain_forward_dat(dat_file, case, rhoa, relative_error=relative_error)
                save_terrain_forward_npz(npz_file, case, rhoa, relative_error=relative_error)
                manifest.append(
                    TerrainForwardRecord(
                        step=step,
                        input_file=str(path),
                        dat_file=str(dat_file),
                        npz_file=str(npz_file),
                        status="ok",
                        rhoa_min=float(np.min(rhoa)),
                        rhoa_max=float(np.max(rhoa)),
                    )
                )
            except Exception as exc:
                step = -1
                try:
                    _, step = parse_resistivity_slice_name(path)
                except ValueError:
                    pass
                failures.append(
                    TerrainForwardRecord(
                        step=step,
                        input_file=str(path),
                        dat_file="",
                        npz_file="",
                        status="failed",
                        error=str(exc),
                    )
                )
    finally:
        for runner in runners.values():
            runner.close()

    manifest.sort(key=lambda record: record.step)
    failures.sort(key=lambda record: record.step)
    return manifest, failures
