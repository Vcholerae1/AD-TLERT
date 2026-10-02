"""Terrain-following ERT workflows for ParFlow 2D slices."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from adtlert.forward import ERTForward2p5D
from adtlert.mesh import Mesh
from adtlert.survey import Survey
from adtlert.utils.dtypes import FLOAT_DTYPE


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


def _checked_resistivity(values, shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != shape:
        raise ValueError(f"resistivity must have shape {shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("resistivity contains non-finite values")
    if np.any(values <= 0.0):
        raise ValueError("resistivity must contain positive values")
    return values


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

    def with_resistivity(self, resistivity: np.ndarray) -> TerrainForwardCase:
        """Return the same terrain geometry with a different cell resistivity vector."""

        return replace(
            self, resistivity=_checked_resistivity(resistivity, self.resistivity.shape)
        )


@dataclass(frozen=True)
class SourcePositionInversionCase:
    """Triangle inversion mesh generated from source/electrode positions."""

    mesh: Mesh
    forward_mesh: Mesh
    survey: Survey
    parameter_cell_ids: np.ndarray
    cell_markers: np.ndarray
    elec_x: np.ndarray
    elec_z: np.ndarray
    x_nodes: np.ndarray
    z_top: np.ndarray
    layer_thickness: np.ndarray
    y_index: int


@dataclass(frozen=True)
class TerrainForwardData:
    """ERT data parsed from a terrain forward ``.dat`` file."""

    rhoa: np.ndarray
    measurements: np.ndarray
    elec_x: np.ndarray
    elec_z: np.ndarray
    err: np.ndarray | None = None


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
    """Reusable forward operator for a fixed terrain mesh, survey, and topography.

    cuDSS plans are kept across models by default; every changed conductivity is still
    refactorized. If that fast path returns invalid values, solver state is rebuilt once.
    """

    case_template: TerrainForwardCase
    forward: ERTForward2p5D
    reuse_solver_state: bool = True

    @classmethod
    def from_case(
        cls,
        case: TerrainForwardCase,
        *,
        reuse_solver_state: bool = True,
        terrain_cache_dir: str | Path | None = None,
        prepare_forward: bool = False,
    ) -> TerrainForwardRunner:
        forward = ERTForward2p5D.from_mesh_survey(
            case.mesh, case.survey, terrain_cache_dir=terrain_cache_dir
        )
        runner = cls(
            case_template=case,
            forward=forward,
            reuse_solver_state=bool(reuse_solver_state),
        )
        if prepare_forward:
            runner.prepare_resistivity(case.resistivity)
        return runner

    def case_with_resistivity(self, resistivity: np.ndarray) -> TerrainForwardCase:
        return self.case_template.with_resistivity(resistivity)

    def solve_resistivity(self, resistivity: np.ndarray) -> np.ndarray:
        """Apparent resistivity for a cell resistivity vector."""

        conductivity = torch.as_tensor(
            1.0 / np.asarray(resistivity, dtype=float), dtype=FLOAT_DTYPE
        )
        if self.reuse_solver_state:
            rhoa = np.asarray(
                self.forward.apparent_resistivity_values(conductivity), dtype=float
            )
            if np.all(np.isfinite(rhoa)) and np.all(rhoa > 0.0):
                return rhoa
        self.forward.close()
        rhoa = np.asarray(
            self.forward.apparent_resistivity_values(conductivity), dtype=float
        )
        if not np.all(np.isfinite(rhoa)) or np.any(rhoa <= 0.0):
            raise ValueError(
                "forward returned non-finite or non-positive apparent resistivity values"
            )
        return rhoa

    def prepare_resistivity(self, resistivity: np.ndarray) -> None:
        """Pre-populate caches for a representative terrain resistivity vector."""

        conductivity = torch.as_tensor(
            1.0 / np.asarray(resistivity, dtype=float), dtype=FLOAT_DTYPE
        )
        self.forward.prepare(conductivity, include_solver_state=self.reuse_solver_state)

    def solve_case(self, case: TerrainForwardCase) -> np.ndarray:
        return self.solve_resistivity(case.resistivity)

    def close(self) -> None:
        self.forward.close()


# ---------------------------------------------------------------------------
# ParFlow inputs
# ---------------------------------------------------------------------------


def parse_resistivity_slice_name(path: str | Path) -> tuple[int, int]:
    """Parse ``(y_index, timestep)`` from ``resistivity2d_y{y}_t{step}.npy`` (or ``resistivity_t{step}.npy`` → ``y=-1``)."""

    name = Path(path).name
    if match := re.search(r"resistivity2d_y(\d+)_t(\d+)\.npy$", name):
        return int(match.group(1)), int(match.group(2))
    if match := re.search(r"resistivity_t(\d+)\.npy$", name):
        return -1, int(match.group(1))
    raise ValueError(f"cannot parse y-index and timestep from filename: {name}")


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
    pairs = set()
    for pattern in ("resistivity2d_y*_t*.npy", "resistivity_t*.npy"):
        for path in Path(input_dir).glob(pattern):
            try:
                found_y, step = parse_resistivity_slice_name(path)
            except ValueError:
                continue
            if y_index is None or found_y < 0 or found_y == y_index:
                pairs.add((step, path))
    return sorted(pairs, key=lambda item: item[0])[::file_stride][:max_steps]


def load_terrain_resistivity_slice(
    path: str | Path, grid: ParflowGrid, *, y_index: int
) -> np.ndarray:
    """Load a 2D slice from a 2D or 3D (any of the ParFlow axis orders) resistivity file."""

    values = np.asarray(np.load(path), dtype=float)
    if values.ndim == 2:
        return values
    if values.ndim != 3:
        raise ValueError(
            f"{path}: resistivity array must be 2D or 3D, got shape={values.shape}"
        )
    if not 0 <= y_index < grid.ny:
        raise ValueError(f"y_index={y_index} out of range for NY={grid.ny}")
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    if values.shape == (nz, ny, nx):
        return values[:, y_index, :]
    if values.shape == (ny, nz, nx):
        return values[y_index, :, :]
    if values.shape == (nx, ny, nz):
        return values[:, y_index, :].T
    raise ValueError(
        f"{path}: cannot infer 3D resistivity axis order from shape={values.shape}; "
        f"expected ({nz}, {ny}, {nx}), ({ny}, {nz}, {nx}), or ({nx}, {ny}, {nz})"
    )


def parse_pftcl(path: str | Path) -> ParflowGrid:
    """Parse ParFlow grid dimensions and ``dzScale`` values from a pftcl file."""

    values: dict[str, float] = {}
    dz_scales: dict[int, float] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        for key, pattern in (
            ("DX", r"[0-9eE+\-.]+"),
            ("DY", r"[0-9eE+\-.]+"),
            ("DZ", r"[0-9eE+\-.]+"),
            ("NX", r"\d+"),
            ("NY", r"\d+"),
            ("NZ", r"\d+"),
        ):
            if match := re.search(rf'ComputationalGrid\.{key}\s+"({pattern})"', line):
                values[key.lower()] = float(match.group(1))
        if match := re.search(r'Cell\.(\d+)\.dzScale\.Value\s+"([0-9eE+\-.]+)"', line):
            dz_scales[int(match.group(1))] = float(match.group(2))
    missing = [key for key in ("dx", "dy", "dz", "nx", "ny", "nz") if key not in values]
    if missing:
        raise ValueError(
            f"failed to parse ComputationalGrid settings from {path}: missing {missing}"
        )
    nz = int(values["nz"])
    if len(dz_scales) != nz:
        raise ValueError(f"dzScale count {len(dz_scales)} does not match NZ={nz}")
    return ParflowGrid(
        dx=values["dx"],
        dy=values["dy"],
        dz_base=values["dz"],
        nx=int(values["nx"]),
        ny=int(values["ny"]),
        nz=nz,
        dz_scales=np.asarray([dz_scales[index] for index in range(nz)], dtype=float),
    )


def read_slope_x(path: str | Path, y_index: int) -> np.ndarray:
    """Read one ``slope_x`` y-slice from a ParFlow PFB file."""

    try:
        from parflow.tools.io import read_pfb
    except ImportError as exc:
        raise ImportError(
            "Reading ParFlow PFB files requires the examples extra: `uv sync --extra examples`."
        ) from exc
    slope_x = np.asarray(read_pfb(str(path)), dtype=float)
    if slope_x.ndim != 3:
        raise ValueError(
            f"expected slope_x PFB to load as a 3D array, got shape={slope_x.shape}"
        )
    if not 0 <= y_index < slope_x.shape[1]:
        raise ValueError(
            f"y_index={y_index} out of range for slope_x shape={slope_x.shape}"
        )
    return slope_x[0, y_index, :]


def build_wenner_alpha_measurements(electrode_count: int) -> np.ndarray:
    """Wenner-alpha ABMN rows (``schemeName='wa'`` ordering) for a linear electrode line."""

    if electrode_count < 4:
        raise ValueError("Wenner-alpha surveys need at least four electrodes")
    return np.asarray(
        [
            [start, start + 3 * spacing, start + spacing, start + 2 * spacing]
            for spacing in range(1, electrode_count // 3 + 1)
            for start in range(electrode_count - 3 * spacing)
        ],
        dtype=np.int32,
    )


# ---------------------------------------------------------------------------
# ERT .dat files
# ---------------------------------------------------------------------------


def load_terrain_forward_dat(path: str | Path) -> TerrainForwardData:
    """Read the unified ERT ``.dat`` format (sensors, then ``a b m n ... rhoa`` rows, 1-based)."""

    path = Path(path)
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) < 4:
        raise ValueError(f"{path} is too short to be an ERT .dat file")

    def skip_comments(cursor: int) -> int:
        while cursor < len(lines) and lines[cursor].startswith("#"):
            cursor += 1
        return cursor

    try:
        sensor_count = int(lines[0])
    except ValueError as exc:
        raise ValueError(f"{path}: first line must be sensor count") from exc
    cursor = skip_comments(1)
    if cursor + sensor_count > len(lines):
        raise ValueError(f"{path}: sensor table is truncated")
    sensors = np.asarray(
        [
            [float(part) for part in line.split()[:3]]
            for line in lines[cursor : cursor + sensor_count]
        ]
    )
    sensors = np.rint(sensors / 1.0e-12) * 1.0e-12  # GIMLi's tolerance rounding
    cursor = skip_comments(cursor + sensor_count)
    if cursor >= len(lines):
        raise ValueError(f"{path}: missing measurement count")
    try:
        data_count = int(lines[cursor])
    except ValueError as exc:
        raise ValueError(f"{path}: measurement count must be an integer") from exc
    cursor += 1
    header = None
    while cursor < len(lines) and lines[cursor].startswith("#"):
        header = lines[cursor].lstrip("#").split()
        cursor += 1
    if header is None:
        raise ValueError(f"{path}: missing measurement header")
    if cursor + data_count > len(lines):
        raise ValueError(f"{path}: measurement table is truncated")
    columns = {name: index for index, name in enumerate(header)}
    missing = [name for name in ("a", "b", "m", "n", "rhoa") if name not in columns]
    if missing:
        raise ValueError(f"{path}: missing required measurement columns {missing}")
    values = np.asarray(
        [
            [float(part) for part in line.split()]
            for line in lines[cursor : cursor + data_count]
        ]
    )
    if values.shape[1] < len(header):
        raise ValueError(f"{path}: measurement rows have fewer columns than the header")
    return TerrainForwardData(
        rhoa=values[:, columns["rhoa"]],
        measurements=values[:, [columns[key] for key in "abmn"]].astype(np.int32) - 1,
        elec_x=sensors[:, 0],
        elec_z=sensors[:, 1],
        err=values[:, columns["err"]] if "err" in columns else None,
    )


def _geometric_factors(electrodes: np.ndarray, measurements: np.ndarray) -> np.ndarray:
    a, b, m, n = (electrodes[measurements[:, index]] for index in range(4))

    def distance(p, q):
        return np.linalg.norm(p - q, axis=-1)

    return (
        2.0
        * np.pi
        / (
            1.0 / distance(a, m)
            - 1.0 / distance(a, n)
            - 1.0 / distance(b, m)
            + 1.0 / distance(b, n)
        )
    )


def _checked_rhoa(case: TerrainForwardCase, rhoa) -> np.ndarray:
    rhoa = np.asarray(rhoa, dtype=float).ravel()
    if rhoa.shape != (case.survey.measurement_count,):
        raise ValueError(f"rhoa must have shape ({case.survey.measurement_count},)")
    return rhoa


def save_terrain_forward_dat(
    path: str | Path,
    case: TerrainForwardCase,
    rhoa: np.ndarray,
    *,
    relative_error: float = 0.03,
) -> None:
    """Save a pyGIMLi-style ERT ``.dat`` file."""

    rhoa = _checked_rhoa(case, rhoa)
    electrodes = np.column_stack(
        (np.asarray(case.elec_x, dtype=float), np.asarray(case.elec_z, dtype=float))
    )
    measurements = np.asarray(case.survey.measurements, dtype=np.int32)
    zero = "0.00000000000000e+00"
    lines = [f"{electrodes.shape[0]}\n", "# x y z\n"]
    lines += [f"{x:.14g}\t{z:.14g}\t0\n" for x, z in electrodes]
    lines += [
        f"{measurements.shape[0]}\n",
        "# a b m n err i ip iperr k r rhoa u valid \n",
    ]
    lines += [
        f"{a + 1}\t{b + 1}\t{m + 1}\t{n + 1}\t{float(relative_error):.14e}\t{zero}\t{zero}\t{zero}\t{k:.14e}\t{zero}\t{value:.14e}\t{zero}\t1\n"
        for (a, b, m, n), k, value in zip(
            measurements.tolist(),
            _geometric_factors(electrodes, measurements),
            rhoa,
            strict=True,
        )
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def save_terrain_forward_npz(
    path: str | Path,
    case: TerrainForwardCase,
    rhoa: np.ndarray,
    *,
    relative_error: float = 0.03,
) -> None:
    """Save compact forward data and terrain metadata for plotting."""

    rhoa = _checked_rhoa(case, rhoa)
    measurements = np.asarray(case.survey.measurements, dtype=np.int32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        rhoa=rhoa,
        err=np.full(rhoa.shape, float(relative_error)),
        **{key: measurements[:, index] for index, key in enumerate("abmn")},
        elec_x=case.elec_x,
        elec_z=case.elec_z,
        x_nodes=case.x_nodes,
        z_top=case.z_top,
        layer_thickness=case.layer_thickness,
        y_index=np.asarray([case.y_index], dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Source-position triangle inversion mesh
# ---------------------------------------------------------------------------


def _smooth_triangle_nodes(
    nodes: np.ndarray,
    cells: np.ndarray,
    markers: np.ndarray,
    *,
    plc_node_count: int,
    iterations: int,
) -> np.ndarray:
    """pyGIMLi's Laplacian smoothing of Triangle meshes, keeping PLC, boundary, and region-interface nodes fixed."""

    if iterations < 0:
        raise ValueError("smoothing iterations must be non-negative")
    smoothed = np.asarray(nodes, dtype=float).copy()
    if iterations == 0:
        return smoothed
    cells = np.asarray(cells, dtype=np.int32)
    markers = np.asarray(markers, dtype=np.int32).reshape(-1)
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError("triangle cells must have shape (n_cells, 3)")
    if markers.shape[0] != cells.shape[0]:
        raise ValueError("triangle markers must have one value per cell")

    neighbors = [set() for _ in range(smoothed.shape[0])]
    edge_markers: dict[tuple[int, int], list[int]] = {}
    for (a, b, c), marker in zip(cells.tolist(), markers.tolist(), strict=True):
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
        for edge in ((a, b), (b, c), (c, a)):
            edge_markers.setdefault(tuple(sorted(edge)), []).append(marker)
    fixed = np.zeros(smoothed.shape[0], dtype=bool)
    fixed[: min(max(plc_node_count, 0), smoothed.shape[0])] = True
    for edge, owners in edge_markers.items():
        if len(owners) == 1 or len(set(owners)) > 1:
            fixed[list(edge)] = True
    adjacency = [np.asarray(sorted(items), dtype=np.int32) for items in neighbors]
    for _ in range(iterations):
        for node, adjacent in enumerate(adjacency):
            if not fixed[node] and adjacent.size:
                smoothed[node] = (smoothed[node] + smoothed[adjacent].sum(axis=0)) / (
                    adjacent.size + 1
                )
    return smoothed


def _source_position_triangle_arrays(
    elec_x: np.ndarray,
    elec_z: np.ndarray,
    *,
    quality: float,
    parameter_max_cell_area: float | None = None,
    smoothing_iterations: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """pyGIMLi-style PLC (parameter box plus outer boundary) meshed with Triangle.

    Returns parameter nodes and cells, full nodes and cells, cell markers (2 = parameter
    domain), and the full-mesh ids of the parameter cells.
    """

    try:
        import triangle
    except ImportError as exc:
        raise ImportError(
            "Building the source-position inversion mesh requires the optional `triangle` package. "
            "Install the example dependencies with `uv sync --extra examples`."
        ) from exc
    if quality <= 0.0:
        raise ValueError("quality must be positive")
    if parameter_max_cell_area is not None and parameter_max_cell_area <= 0.0:
        raise ValueError("parameter_max_cell_area must be positive when provided")

    sensors = np.column_stack((elec_x, elec_z))
    x_min, x_max = float(np.min(elec_x)), float(np.max(elec_x))
    para_bound = 2.0 * float(np.linalg.norm(sensors[1] - sensors[0]))
    para_depth = 0.4 * (x_max - x_min)
    outer_bound = 4.0 * abs(x_max - x_min)
    bottom = min(float(elec_z[0] - para_depth), float(elec_z[-1] - para_depth))
    vertices: list[list[float]] = []
    segments: list[list[int]] = []
    regions: list[list[float]] = []

    def node(x: float, y: float) -> int:
        vertices.append([float(x), float(y)])
        return len(vertices) - 1

    def chain(*nodes: int) -> None:
        segments.extend([a, b] for a, b in zip(nodes[:-1], nodes[1:], strict=True))

    n1 = node(x_min - para_bound, float(elec_z[0]))
    n2 = node(x_min - para_bound, bottom)
    n3 = node(x_max + para_bound, bottom)
    n4 = node(x_max + para_bound, float(elec_z[-1]))
    if outer_bound > para_bound:
        n11 = node(vertices[n1][0] - outer_bound, vertices[n1][1])
        n12 = node(vertices[n11][0], vertices[n11][1] - (outer_bound + para_depth))
        n14 = node(vertices[n4][0] + outer_bound, vertices[n4][1])
        n13 = node(vertices[n14][0], vertices[n14][1] - (outer_bound + para_depth))
        chain(n1, n11, n12, n13, n14, n4)
        regions.append([vertices[n12][0] + 1.0e-3, vertices[n12][1] + 1.0e-3, 1.0, 0.0])
    chain(n1, n2, n3, n4)
    regions.append(
        [
            vertices[n2][0] + 1.0e-3,
            vertices[n2][1] + 1.0e-3,
            2.0,
            0.0 if parameter_max_cell_area is None else float(parameter_max_cell_area),
        ]
    )

    surface = [n1]
    for index, (x, z) in enumerate(sensors):
        surface.append(node(x, z))
        if index < len(sensors) - 1:
            surface.append(
                node(
                    0.5 * (x + sensors[index + 1][0]), 0.5 * (z + sensors[index + 1][1])
                )
            )
    surface.append(n4)
    surface = sorted(dict.fromkeys(surface), key=lambda node_id: vertices[node_id][0])
    chain(*surface[::-1])

    mesh = triangle.triangulate(
        {
            "vertices": np.asarray(vertices),
            "segments": np.asarray(segments, dtype=np.int32),
            "regions": np.asarray(regions),
        },
        f"pzeAq{quality:g}aQ",
    )
    cells = np.asarray(mesh["triangles"], dtype=np.int32)
    markers = np.rint(
        np.asarray(mesh["triangle_attributes"], dtype=float).reshape(-1)
    ).astype(np.int32)
    nodes = _smooth_triangle_nodes(
        mesh["vertices"],
        cells,
        markers,
        plc_node_count=len(vertices),
        iterations=int(smoothing_iterations),
    )
    parameter_ids = np.flatnonzero(markers == 2).astype(np.int32)
    if parameter_ids.size == 0:
        raise ValueError("triangle did not produce any parameter-domain cells")
    used = np.asarray(
        list(dict.fromkeys(cells[parameter_ids].reshape(-1).tolist())), dtype=np.int32
    )
    remap = np.full(nodes.shape[0], -1, dtype=np.int32)
    remap[used] = np.arange(used.size, dtype=np.int32)
    return (
        nodes[used],
        remap[cells[parameter_ids]],
        nodes,
        cells,
        markers,
        parameter_ids,
    )


def _load_mesh_npz(path: str | Path):
    with np.load(path) as data:
        missing = {"nodes", "cells"}.difference(data.files)
        if missing:
            raise KeyError(f"{path} missing required mesh arrays: {sorted(missing)}")
        nodes = np.asarray(data["nodes"], dtype=float)
        cells = np.asarray(data["cells"], dtype=np.int32)

        def get(key, default, dtype):
            return np.asarray(data[key], dtype=dtype) if key in data.files else default

        full_cells = get("forward_cells", cells, np.int32)
        return (
            nodes,
            cells,
            None
            if "surface_node_ids" not in data.files
            else get("surface_node_ids", None, np.int32).ravel(),
            get("forward_nodes", nodes, float),
            full_cells,
            get(
                "cell_markers",
                np.full(full_cells.shape[0], 2, dtype=np.int32),
                np.int32,
            ).ravel(),
            get(
                "parameter_cell_ids",
                np.arange(cells.shape[0], dtype=np.int32),
                np.int32,
            ).ravel(),
        )


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
    quality: float = 34.0,
    parameter_max_cell_area: float | None = None,
    smoothing_iterations: int = 10,
    data_file: str | Path | None = None,
    mesh_file: str | Path | None = None,
) -> SourcePositionInversionCase:
    """Build a pyGIMLi-style ``paraDomain`` triangle mesh from the electrode positions.

    ``data_file`` replaces the electrodes and measurements with those of a ``.dat`` file;
    ``mesh_file`` loads saved mesh arrays instead of running Triangle.
    """

    elec_x, elec_z = (
        np.asarray(elec_x, dtype=float).ravel(),
        np.asarray(elec_z, dtype=float).ravel(),
    )
    if elec_x.shape != elec_z.shape:
        raise ValueError("elec_x and elec_z must have the same shape")
    if elec_x.size < 4:
        raise ValueError("at least four source/electrode positions are required")
    if not np.all(np.isfinite(elec_x)) or not np.all(np.isfinite(elec_z)):
        raise ValueError("electrode positions contain non-finite values")
    if np.any(np.diff(elec_x) <= 0.0):
        raise ValueError("elec_x must be strictly increasing")
    measurements = np.asarray(measurements, dtype=np.int32)
    if measurements.ndim != 2 or measurements.shape[1] != 4:
        raise ValueError("measurements must have shape (n_measurements, 4)")
    if np.any(measurements < 0) or np.any(measurements >= elec_x.size):
        raise ValueError(
            "measurements reference electrodes outside the source positions"
        )
    x_nodes, z_top = (
        np.asarray(x_nodes, dtype=float).ravel(),
        np.asarray(z_top, dtype=float).ravel(),
    )
    layer_thickness = np.asarray(layer_thickness, dtype=float).ravel()
    if x_nodes.shape != z_top.shape:
        raise ValueError("x_nodes and z_top must have the same shape")
    if x_nodes.size < 2:
        raise ValueError("x_nodes must contain at least two nodes")
    if np.any(np.diff(x_nodes) <= 0.0):
        raise ValueError("x_nodes must be strictly increasing")
    if not np.all(np.isfinite(layer_thickness)) or np.any(layer_thickness <= 0.0):
        raise ValueError("layer_thickness must contain positive finite values")

    if data_file is not None:
        data = load_terrain_forward_dat(data_file)
        elec_x, elec_z, measurements = data.elec_x, data.elec_z, data.measurements
    if mesh_file is not None:
        nodes, cells, surface_ids, full_nodes, full_cells, markers, parameter_ids = (
            _load_mesh_npz(mesh_file)
        )
    else:
        if depth_levels < 2:
            raise ValueError("depth_levels must be >= 2")
        nodes, cells, full_nodes, full_cells, markers, parameter_ids = (
            _source_position_triangle_arrays(
                elec_x,
                elec_z,
                quality=float(quality),
                parameter_max_cell_area=parameter_max_cell_area,
                smoothing_iterations=int(smoothing_iterations),
            )
        )
        surface_ids = None
    return SourcePositionInversionCase(
        mesh=Mesh.from_arrays(nodes, cells, surface_node_ids=surface_ids),
        forward_mesh=Mesh.from_arrays(full_nodes, full_cells),
        survey=Survey.from_arrays(np.column_stack((elec_x, elec_z)), measurements),
        parameter_cell_ids=np.asarray(parameter_ids, dtype=np.int32),
        cell_markers=np.asarray(markers, dtype=np.int32),
        elec_x=elec_x,
        elec_z=elec_z,
        x_nodes=x_nodes,
        z_top=z_top,
        layer_thickness=layer_thickness,
        y_index=int(y_index),
    )


# ---------------------------------------------------------------------------
# Terrain forward modelling
# ---------------------------------------------------------------------------


def _terrain_resistivity_vector(rho_2d: np.ndarray, grid: ParflowGrid) -> np.ndarray:
    """Flatten a bottom-to-top ParFlow slice into top-to-bottom cell order."""

    rho = np.asarray(rho_2d, dtype=float)
    if rho.shape != (grid.nz, grid.nx):
        raise ValueError(
            f"resistivity shape {rho.shape} does not match parsed grid {(grid.nz, grid.nx)}"
        )
    return _checked_resistivity(rho, rho.shape)[::-1, :].reshape(-1)


def build_terrain_forward_case(
    rho_2d: np.ndarray,
    grid: ParflowGrid,
    slope_x: np.ndarray,
    *,
    y_index: int,
    n_electrodes: int = 48,
    topo_offset: float = 0.0,
) -> TerrainForwardCase:
    """Terrain-following quad mesh, Wenner-alpha survey, and cell resistivities for one ParFlow slice.

    ``rho_2d`` is in ParFlow z-order (bottom-to-top); the returned resistivity is top-to-bottom.
    """

    resistivity = _terrain_resistivity_vector(rho_2d, grid)
    slope = np.asarray(slope_x, dtype=float).ravel()
    if not 0 <= y_index < grid.ny:
        raise ValueError(f"y_index={y_index} out of range [0, {grid.ny - 1}]")
    if slope.shape != (grid.nx,):
        raise ValueError(f"slope_x shape {slope.shape} does not match NX={grid.nx}")

    layer_thickness = (grid.dz_base * grid.dz_scales)[::-1]
    depth_offsets = np.concatenate(([0.0], -np.cumsum(layer_thickness)))
    x_nodes = np.arange(grid.nx + 1, dtype=float) * grid.dx
    z_top = np.concatenate(([0.0], np.cumsum(slope * grid.dx))) + topo_offset
    nodes = np.asarray(
        [
            [x, z + offset]
            for offset in depth_offsets
            for x, z in zip(x_nodes, z_top, strict=True)
        ]
    )
    width = grid.nx + 1
    cells = [
        [
            layer * width + column,
            layer * width + column + 1,
            (layer + 1) * width + column + 1,
            (layer + 1) * width + column,
        ]
        for layer in range(grid.nz)
        for column in range(grid.nx)
    ]
    electrode_count = min(int(n_electrodes), width)
    elec_x = np.linspace(float(x_nodes.min()), float(x_nodes.max()), electrode_count)
    elec_z = np.interp(elec_x, x_nodes, z_top)
    return TerrainForwardCase(
        mesh=Mesh.from_arrays(nodes, cells, surface_node_ids=np.arange(width)),
        survey=Survey.from_arrays(
            np.column_stack((elec_x, elec_z)),
            build_wenner_alpha_measurements(electrode_count),
        ),
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
    reuse_solver_state: bool = True,
    terrain_cache_dir: str | Path | None = None,
    prepare_forward: bool = False,
) -> np.ndarray:
    """Compute apparent resistivity for a terrain case."""

    runner = TerrainForwardRunner.from_case(
        case,
        reuse_solver_state=reuse_solver_state,
        terrain_cache_dir=terrain_cache_dir,
        prepare_forward=prepare_forward,
    )
    try:
        return runner.solve_case(case)
    finally:
        runner.close()


def _resolve_slice(path: Path, y_index: int | None) -> tuple[int, int]:
    parsed_y, step = parse_resistivity_slice_name(path)
    if y_index is None:
        if parsed_y < 0:
            raise ValueError(
                f"{path}: y_index is required for notebook-style resistivity_t*.npy files"
            )
        return parsed_y, step
    if parsed_y >= 0 and parsed_y != y_index:
        raise ValueError(
            f"input file y-index {parsed_y} does not match requested y_index={y_index}"
        )
    return int(y_index), step


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
    reuse_solver_state: bool = True,
    terrain_cache_dir: str | Path | None = None,
    prepare_forward: bool = False,
    _raise_errors: bool = False,
) -> tuple[list[TerrainForwardRecord], list[TerrainForwardRecord]]:
    """Run and save a terrain-forward series; returns ``(manifest, failures)``.

    One forward operator (mesh, survey, auxiliary discretizations, caches, cuDSS plans)
    is reused per y-index. Failed timesteps are reported, not raised.
    """

    paths = [Path(item[1] if isinstance(item, tuple) else item) for item in input_files]
    manifest, failures, runners = [], [], {}
    output_root = Path(output_dir)
    try:
        for path in paths:
            try:
                slice_y, step = _resolve_slice(path, y_index)
                files = {
                    kind: output_root
                    / f"synthetic_ert_terrain_vardz_t{step:05d}.{kind}"
                    for kind in ("dat", "npz")
                }
                record = TerrainForwardRecord(
                    step,
                    str(path),
                    str(files["dat"]),
                    str(files["npz"]),
                    "skipped_existing",
                )
                if not overwrite and all(file.exists() for file in files.values()):
                    manifest.append(record)
                    continue
                rho_2d = load_terrain_resistivity_slice(path, grid, y_index=slice_y)
                if slice_y not in runners:
                    case = build_terrain_forward_case(
                        rho_2d,
                        grid,
                        slope_x,
                        y_index=slice_y,
                        n_electrodes=n_electrodes,
                        topo_offset=topo_offset,
                    )
                    runners[slice_y] = TerrainForwardRunner.from_case(
                        case,
                        reuse_solver_state=reuse_solver_state,
                        terrain_cache_dir=terrain_cache_dir,
                        prepare_forward=prepare_forward,
                    )
                runner = runners[slice_y]
                case = runner.case_template
                rhoa = runner.solve_resistivity(
                    _terrain_resistivity_vector(rho_2d, grid)
                )
                save_terrain_forward_dat(
                    files["dat"], case, rhoa, relative_error=relative_error
                )
                save_terrain_forward_npz(
                    files["npz"], case, rhoa, relative_error=relative_error
                )
                manifest.append(
                    replace(
                        record,
                        status="ok",
                        rhoa_min=float(np.min(rhoa)),
                        rhoa_max=float(np.max(rhoa)),
                    )
                )
            except Exception as exc:
                if _raise_errors:
                    raise
                try:
                    step = parse_resistivity_slice_name(path)[1]
                except ValueError:
                    step = -1
                failures.append(
                    TerrainForwardRecord(
                        step, str(path), "", "", "failed", error=str(exc)
                    )
                )
    finally:
        for runner in runners.values():
            runner.close()
    return sorted(manifest, key=lambda record: record.step), sorted(
        failures, key=lambda record: record.step
    )


def run_terrain_forward_file(
    input_file: str | Path,
    grid: ParflowGrid,
    slope_x: np.ndarray,
    output_dir: str | Path,
    *,
    y_index: int | None = None,
    **options,
) -> TerrainForwardRecord:
    """Run and save one terrain-forward timestep (errors are raised)."""

    manifest, _ = run_terrain_forward_series(
        [input_file],
        grid,
        slope_x,
        output_dir,
        y_index=y_index,
        _raise_errors=True,
        **options,
    )
    return manifest[0]
