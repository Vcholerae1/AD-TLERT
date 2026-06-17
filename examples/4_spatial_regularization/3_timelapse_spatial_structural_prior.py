from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from deepert.workflows import load_terrain_forward_dat

from _spatial_regularization_runner import run_spatial_regularization_case


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pick_first_forward_dat(forward_dir: Path) -> Path:
    pattern = re.compile(r"synthetic_ert_terrain_vardz_t(\d+)\.dat$")
    pairs: list[tuple[int, Path]] = []
    for path in forward_dir.glob("synthetic_ert_terrain_vardz_t*.dat"):
        match = pattern.search(path.name)
        if match is not None:
            pairs.append((int(match.group(1)), path))
    if not pairs:
        raise FileNotFoundError(f"No forward .dat files found in {forward_dir}")
    pairs.sort(key=lambda item: item[0])
    return pairs[0][1]


def _block_edges(n_cells: int, stride: int) -> np.ndarray:
    if n_cells < 1:
        raise ValueError("n_cells must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    edges = list(range(0, n_cells + 1, stride))
    if edges[-1] != n_cells:
        edges.append(n_cells)
    return np.asarray(edges, dtype=np.int32)


def _unique_sorted_with_tol(values: np.ndarray, tol: float = 1.0e-8) -> np.ndarray:
    arr = np.sort(np.asarray(values, dtype=float).ravel())
    if arr.size == 0:
        return arr
    keep = [arr[0]]
    for value in arr[1:]:
        if abs(value - keep[-1]) > tol:
            keep.append(value)
    return np.asarray(keep, dtype=float)


def _triangle_node_adjacency(cells: np.ndarray, node_count: int) -> list[np.ndarray]:
    adjacency = [set() for _ in range(node_count)]
    for cell in cells:
        a, b, c = (int(cell[0]), int(cell[1]), int(cell[2]))
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))
    return [np.asarray(sorted(neighbors), dtype=np.int32) for neighbors in adjacency]


def _triangle_smoothing_fixed_mask(
    cells: np.ndarray,
    markers: np.ndarray,
    node_count: int,
    plc_node_count: int,
) -> np.ndarray:
    fixed = np.zeros(node_count, dtype=bool)
    fixed[: min(max(int(plc_node_count), 0), node_count)] = True

    edge_counts: dict[tuple[int, int], int] = {}
    edge_markers: dict[tuple[int, int], set[int]] = {}
    for cell, marker in zip(cells, markers, strict=True):
        a, b, c = (int(cell[0]), int(cell[1]), int(cell[2]))
        cell_edges = (
            (min(a, b), max(a, b)),
            (min(b, c), max(b, c)),
            (min(c, a), max(c, a)),
        )
        marker_id = int(marker)
        for edge in cell_edges:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
            edge_markers.setdefault(edge, set()).add(marker_id)

    for edge, count in edge_counts.items():
        if count == 1 or len(edge_markers[edge]) > 1:
            fixed[list(edge)] = True
    return fixed


def _smooth_triangle_nodes(
    nodes: np.ndarray,
    cells: np.ndarray,
    markers: np.ndarray,
    *,
    plc_node_count: int,
    iterations: int,
) -> np.ndarray:
    if iterations < 0:
        raise ValueError("smoothing iterations must be non-negative")
    nodes_array = np.asarray(nodes, dtype=float)
    if iterations == 0:
        return nodes_array.copy()

    cells_array = np.asarray(cells, dtype=np.int32)
    markers_array = np.asarray(markers, dtype=np.int32).reshape(-1)
    if cells_array.ndim != 2 or cells_array.shape[1] != 3:
        raise ValueError("triangle cells must have shape (n_cells, 3)")
    if markers_array.shape[0] != cells_array.shape[0]:
        raise ValueError("triangle markers must have one value per cell")

    smoothed = nodes_array.copy()
    adjacency = _triangle_node_adjacency(cells_array, smoothed.shape[0])
    fixed = _triangle_smoothing_fixed_mask(cells_array, markers_array, smoothed.shape[0], plc_node_count)
    for _ in range(iterations):
        for node_id, neighbors in enumerate(adjacency):
            if fixed[node_id] or neighbors.size == 0:
                continue
            smoothed[node_id] = (smoothed[node_id] + smoothed[neighbors].sum(axis=0)) / (neighbors.size + 1)
    return smoothed


def _fill_missing_transition_indices(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    x = np.arange(arr.size, dtype=float)
    valid = np.isfinite(arr) & (arr >= 0.0)
    if not np.any(valid):
        raise ValueError(f"No valid {name} transitions found in structural class map")
    filled = np.interp(x, x[valid], arr[valid])
    return np.rint(filled).astype(np.int32)


def _structural_interfaces_from_class_map(
    *,
    class_map_file: Path,
    x_nodes: np.ndarray,
    z_top: np.ndarray,
    layer_thickness: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_raw = np.asarray(np.load(class_map_file), dtype=np.int32)
    nx = x_nodes.size - 1
    nz = layer_thickness.size
    expected_shape = (nz, nx)
    if class_raw.shape != expected_shape:
        raise ValueError(
            f"Structural class map shape {class_raw.shape} does not match expected {expected_shape}"
        )

    # class2d is stored bottom->top; flip to top->bottom for transition search.
    class_top_to_bottom = class_raw[::-1, :]

    b12_idx = np.full(nx, -1.0, dtype=float)
    b23_idx = np.full(nx, -1.0, dtype=float)
    for col in range(nx):
        profile = class_top_to_bottom[:, col]
        idx12 = np.flatnonzero((profile[:-1] == 1) & (profile[1:] == 2))
        idx23 = np.flatnonzero((profile[:-1] == 2) & (profile[1:] == 3))
        if idx12.size:
            b12_idx[col] = float(idx12[0])
        if idx23.size:
            b23_idx[col] = float(idx23[0])

    b12 = _fill_missing_transition_indices(b12_idx, name="regolith->fractured")
    b23 = _fill_missing_transition_indices(b23_idx, name="fractured->fresh")
    b23 = np.maximum(b23, b12 + 1)

    depth_edges = np.concatenate(([0.0], np.cumsum(np.asarray(layer_thickness, dtype=float))))
    x_centers = 0.5 * (np.asarray(x_nodes[:-1], dtype=float) + np.asarray(x_nodes[1:], dtype=float))
    z_surface_center = np.interp(x_centers, np.asarray(x_nodes, dtype=float), np.asarray(z_top, dtype=float))

    z_b12_center = z_surface_center - depth_edges[b12 + 1]
    z_b23_center = z_surface_center - depth_edges[b23 + 1]
    return x_centers, z_b12_center, z_b23_center


def _build_layered_irregular_triangle_mesh(
    *,
    x_nodes: np.ndarray,
    z_top: np.ndarray,
    z_layer_12: np.ndarray,
    z_layer_23: np.ndarray,
    z_bottom: np.ndarray,
    quality: float,
    smoothing_iterations: int,
    area_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        import triangle as triangle_lib
    except ImportError as exc:
        raise ImportError(
            "Building layered irregular mesh requires the optional `triangle` package. "
            "Install example dependencies with `uv sync --extra examples`."
        ) from exc

    if quality <= 0.0:
        raise ValueError("quality must be positive")
    if area_scale <= 0.0:
        raise ValueError("area_scale must be positive")

    x_arr = np.asarray(x_nodes, dtype=float).ravel()
    z_arr = np.asarray(z_top, dtype=float).ravel()
    z12 = np.asarray(z_layer_12, dtype=float).ravel()
    z23 = np.asarray(z_layer_23, dtype=float).ravel()
    zb = np.asarray(z_bottom, dtype=float).ravel()
    if x_arr.size < 2 or z_arr.shape != x_arr.shape:
        raise ValueError("invalid x_nodes/z_top")
    if z12.shape != z_arr.shape or z23.shape != z_arr.shape or zb.shape != z_arr.shape:
        raise ValueError("layer interface arrays must match x_nodes shape")
    if np.any(z_arr < z12) or np.any(z12 < z23) or np.any(z23 < zb):
        raise ValueError("invalid structural layering order (expected z_top >= z12 >= z23 >= z_bottom)")

    nx = x_arr.size - 1
    nz = 3
    z_interfaces = np.vstack([z_arr, z12, z23, zb])

    node_ids = np.arange((nz + 1) * (nx + 1), dtype=np.int32).reshape(nz + 1, nx + 1)
    vertices = np.column_stack((np.tile(x_arr, nz + 1), z_interfaces.reshape(-1)))

    segments_set: set[tuple[int, int]] = set()

    def add_segment(a: int, b: int) -> None:
        if a == b:
            return
        key = (a, b) if a < b else (b, a)
        segments_set.add(key)

    # Horizontal segments on every layer interface (top, internal boundaries, bottom).
    for layer_idx in range(nz + 1):
        for x_idx in range(nx):
            add_segment(int(node_ids[layer_idx, x_idx]), int(node_ids[layer_idx, x_idx + 1]))

    # Vertical side boundaries (left/right) to close each layer region.
    for layer_idx in range(nz):
        add_segment(int(node_ids[layer_idx, 0]), int(node_ids[layer_idx + 1, 0]))
        add_segment(int(node_ids[layer_idx, nx]), int(node_ids[layer_idx + 1, nx]))

    x_mid = float(0.5 * (x_arr[0] + x_arr[-1]))
    ix_mid = int(np.searchsorted(x_arr, x_mid, side="right") - 1)
    ix_mid = int(np.clip(ix_mid, 0, nx - 1))

    regions: list[list[float]] = []
    dx_ref = float(np.mean(np.diff(x_arr)))
    for layer_idx in range(nz):
        z_top_mid = float(0.5 * (z_interfaces[layer_idx, ix_mid] + z_interfaces[layer_idx, ix_mid + 1]))
        z_bot_mid = float(0.5 * (z_interfaces[layer_idx + 1, ix_mid] + z_interfaces[layer_idx + 1, ix_mid + 1]))
        z_seed = 0.5 * (z_top_mid + z_bot_mid)
        marker = float(layer_idx + 2)
        dz_mid = max(z_top_mid - z_bot_mid, 1.0e-6)
        max_area = float(area_scale * dx_ref * dz_mid)
        regions.append([x_mid, z_seed, marker, max_area])

    triangle_input = {
        "vertices": np.asarray(vertices, dtype=float),
        "segments": np.asarray(sorted(segments_set), dtype=np.int32),
        "regions": np.asarray(regions, dtype=float),
    }
    triangle_mesh = triangle_lib.triangulate(triangle_input, f"pzeAq{float(quality):g}aQ")
    mesh_vertices = np.asarray(triangle_mesh["vertices"], dtype=float)
    mesh_cells = np.asarray(triangle_mesh["triangles"], dtype=np.int32)
    cell_markers = np.rint(np.asarray(triangle_mesh["triangle_attributes"], dtype=float).reshape(-1)).astype(np.int32)

    mesh_vertices = _smooth_triangle_nodes(
        mesh_vertices,
        mesh_cells,
        cell_markers,
        plc_node_count=vertices.shape[0],
        iterations=int(smoothing_iterations),
    )

    # Rebuild surface node ids on the terrain polyline after smoothing.
    z_surface_interp = np.interp(mesh_vertices[:, 0], x_arr, z_arr)
    tol = max(1.0e-6, 1.0e-6 * float(np.max(np.abs(z_arr)) + 1.0))
    surface_mask = np.abs(mesh_vertices[:, 1] - z_surface_interp) <= tol
    surface_node_ids = np.flatnonzero(surface_mask)
    if surface_node_ids.size == 0:
        raise RuntimeError("Failed to detect surface nodes on generated layered triangular mesh")
    order = np.argsort(mesh_vertices[surface_node_ids, 0])
    surface_node_ids = surface_node_ids[order].astype(np.int32)

    return mesh_vertices, mesh_cells, cell_markers, surface_node_ids


def _build_layered_mesh_npz(
    *,
    project_root: Path,
    y_index: int = 2,
    x_coarsen: int = 2,
    z_coarsen: int = 2,
    electrode_refine_stride: int = 0,
    mesh_quality: float = 34.0,
    smoothing_iterations: int = 10,
    area_scale: float = 1.0,
    class_map_rel: str | None = None,
    forward_dir_rel: str = "result/1_timelapsedERT_forward_deepert",
    mesh_out_rel: str | None = None,
) -> Path:
    forward_dir = (project_root / forward_dir_rel).resolve()
    geometry_path = forward_dir / "forward_geometry.npz"
    if not geometry_path.exists():
        raise FileNotFoundError(f"Missing geometry file: {geometry_path}")

    with np.load(geometry_path) as data:
        x_nodes = np.asarray(data["x_nodes"], dtype=float).ravel()
        z_top = np.asarray(data["z_top"], dtype=float).ravel()
        layer_thickness = np.asarray(data["layer_thickness"], dtype=float).ravel()

    if x_nodes.size < 2 or z_top.shape != x_nodes.shape:
        raise ValueError("Invalid forward geometry: x_nodes/z_top mismatch")
    if layer_thickness.size < 1 or np.any(layer_thickness <= 0.0):
        raise ValueError("Invalid forward geometry: layer_thickness must be positive")

    dat_path = _pick_first_forward_dat(forward_dir)
    forward_dat = load_terrain_forward_dat(dat_path)
    elec_x = np.asarray(forward_dat.elec_x, dtype=float).ravel()
    elec_z = np.asarray(forward_dat.elec_z, dtype=float).ravel()
    measurements = np.asarray(forward_dat.measurements, dtype=np.int32)

    if class_map_rel is None:
        class_map_path = project_root / "parflow_models" / "petrophysical_models_2d" / f"class2d_y{int(y_index)}.npy"
    else:
        class_map_path = Path(class_map_rel)
        if not class_map_path.is_absolute():
            class_map_path = project_root / class_map_path
    class_map_path = class_map_path.resolve()
    if not class_map_path.exists():
        raise FileNotFoundError(f"Missing structural class map file: {class_map_path}")

    nx_native = x_nodes.size - 1
    x_edges = _block_edges(nx_native, int(x_coarsen))

    x_nodes_seed = x_nodes[x_edges]
    electrode_anchor_x = np.asarray([], dtype=float)
    if int(electrode_refine_stride) > 0:
        stride = int(electrode_refine_stride)
        electrode_anchor_x = np.asarray(elec_x[::stride], dtype=float)
        if electrode_anchor_x.size == 0 or abs(float(electrode_anchor_x[-1]) - float(elec_x[-1])) > 1.0e-9:
            electrode_anchor_x = np.concatenate((electrode_anchor_x, np.asarray([float(elec_x[-1])], dtype=float)))
    x_nodes_coarse = _unique_sorted_with_tol(
        np.concatenate((x_nodes_seed, electrode_anchor_x, np.asarray([x_nodes[0], x_nodes[-1]], dtype=float))),
        tol=1.0e-9,
    )
    z_top_coarse = np.interp(x_nodes_coarse, x_nodes, z_top)

    x_centers_native, z_b12_center, z_b23_center = _structural_interfaces_from_class_map(
        class_map_file=class_map_path,
        x_nodes=x_nodes,
        z_top=z_top,
        layer_thickness=layer_thickness,
    )
    z_b12_coarse = np.interp(x_nodes_coarse, x_centers_native, z_b12_center)
    z_b23_coarse = np.interp(x_nodes_coarse, x_centers_native, z_b23_center)
    z_bottom_native = np.asarray(z_top, dtype=float) - float(np.sum(layer_thickness))
    z_bottom_coarse = np.interp(x_nodes_coarse, x_nodes, z_bottom_native)

    # Enforce strict layer ordering after interpolation.
    eps = 1.0e-4
    z_b12_coarse = np.minimum(z_b12_coarse, z_top_coarse - eps)
    z_b23_coarse = np.minimum(z_b23_coarse, z_b12_coarse - eps)
    z_bottom_coarse = np.minimum(z_bottom_coarse, z_b23_coarse - eps)

    # Keep all electrodes inside/on the mesh surface after coarsening.
    # In practice x_nodes_coarse includes elec_x so this is usually zero.
    z_interp_elec = np.interp(elec_x, x_nodes_coarse, z_top_coarse)
    max_positive_gap = float(np.max(elec_z - z_interp_elec))
    surface_shift = 0.0
    if max_positive_gap > 0.0:
        surface_shift = max_positive_gap + 1.0e-4
        z_top_coarse = z_top_coarse + surface_shift
        z_b12_coarse = z_b12_coarse + surface_shift
        z_b23_coarse = z_b23_coarse + surface_shift
        z_bottom_coarse = z_bottom_coarse + surface_shift

    nodes, cells_array, cell_markers, surface_node_ids = _build_layered_irregular_triangle_mesh(
        x_nodes=x_nodes_coarse,
        z_top=z_top_coarse,
        z_layer_12=z_b12_coarse,
        z_layer_23=z_b23_coarse,
        z_bottom=z_bottom_coarse,
        quality=float(mesh_quality),
        smoothing_iterations=int(smoothing_iterations),
        area_scale=float(area_scale),
    )

    layer_thickness_structural = np.asarray(
        [
            float(np.mean(z_top_coarse - z_b12_coarse)),
            float(np.mean(z_b12_coarse - z_b23_coarse)),
            float(np.mean(z_b23_coarse - z_bottom_coarse)),
        ],
        dtype=float,
    )

    parameter_cell_ids = np.arange(cells_array.shape[0], dtype=np.int32)

    if mesh_out_rel is None:
        mesh_out_rel = (
            f"result/4_spatial_regularization/meshes/"
            f"structural_prior_layered_irregular_mesh_y{int(y_index)}"
            f"_x{int(x_coarsen)}_r{int(electrode_refine_stride)}"
            f"_q{float(mesh_quality):g}_s{int(smoothing_iterations)}_a{float(area_scale):g}.npz"
        )
    mesh_path = (project_root / mesh_out_rel).resolve()
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        mesh_path,
        nodes=nodes,
        cells=cells_array,
        surface_node_ids=surface_node_ids,
        forward_nodes=nodes,
        forward_cells=cells_array,
        cell_markers=cell_markers,
        parameter_cell_ids=parameter_cell_ids,
        x_nodes=x_nodes_coarse,
        z_top=z_top_coarse,
        layer_thickness=layer_thickness_structural,
        z_layer_12=z_b12_coarse,
        z_layer_23=z_b23_coarse,
        z_bottom=z_bottom_coarse,
        x_nodes_native=x_nodes,
        z_top_native=z_top,
        layer_thickness_native=layer_thickness,
        y_index=np.asarray([int(y_index)], dtype=np.int32),
        x_coarsen=np.asarray([int(x_coarsen)], dtype=np.int32),
        z_coarsen=np.asarray([int(z_coarsen)], dtype=np.int32),
        electrode_refine_stride=np.asarray([int(electrode_refine_stride)], dtype=np.int32),
        structural_class_map_file=np.asarray([str(class_map_path)], dtype="U512"),
        mesh_quality=np.asarray([float(mesh_quality)], dtype=float),
        smoothing_iterations=np.asarray([int(smoothing_iterations)], dtype=np.int32),
        area_scale=np.asarray([float(area_scale)], dtype=float),
        applied_surface_shift=np.asarray([float(surface_shift)], dtype=float),
        elec_x=elec_x,
        elec_z=elec_z,
        measurements=measurements,
    )
    return mesh_path


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--y-index", type=int, default=2)
    parser.add_argument(
        "--x-coarsen",
        type=int,
        default=2,
        help="Coarsening stride along X columns relative to forward geometry.",
    )
    parser.add_argument(
        "--z-coarsen",
        type=int,
        default=2,
        help="Compatibility argument (unused when mesh is built from structural class boundaries).",
    )
    parser.add_argument(
        "--electrode-refine-stride",
        type=int,
        default=0,
        help=(
            "Inject every k-th electrode X as an extra surface anchor (k>0). "
            "Use 0 to disable electrode-anchor refinement."
        ),
    )
    parser.add_argument(
        "--mesh-out-rel",
        default=None,
        help="Optional mesh output path relative to project root.",
    )
    parser.add_argument(
        "--class-map-rel",
        default=None,
        help=(
            "Structural class map path relative to project root, e.g. "
            "parflow_models/petrophysical_models_2d/class2d_y2.npy. "
            "Defaults to class2d_y<y-index>.npy under petrophysical_models_2d."
        ),
    )
    parser.add_argument(
        "--mesh-quality",
        type=float,
        default=34.0,
        help="Triangle quality parameter (q). Keep <= 34 to avoid non-termination.",
    )
    parser.add_argument(
        "--mesh-smoothing-iterations",
        type=int,
        default=10,
        help="Laplacian smoothing iterations for the generated triangular mesh.",
    )
    parser.add_argument(
        "--mesh-area-scale",
        type=float,
        default=1.0,
        help="Per-layer target maximum-area scale relative to dx*dz after coarsening.",
    )
    return parser


if __name__ == "__main__":
    cli_args, passthrough = _build_cli_parser().parse_known_args(sys.argv[1:])
    root = _project_root()
    mesh_file = _build_layered_mesh_npz(
        project_root=root,
        y_index=int(cli_args.y_index),
        x_coarsen=int(cli_args.x_coarsen),
        z_coarsen=int(cli_args.z_coarsen),
        electrode_refine_stride=int(cli_args.electrode_refine_stride),
        mesh_quality=float(cli_args.mesh_quality),
        smoothing_iterations=int(cli_args.mesh_smoothing_iterations),
        area_scale=float(cli_args.mesh_area_scale),
        class_map_rel=cli_args.class_map_rel,
        mesh_out_rel=cli_args.mesh_out_rel,
    )
    raise SystemExit(
        run_spatial_regularization_case(
            spatial_regularization="structural_prior",
            output_name="structural_prior",
            extra_cli_args=["--mesh-file", str(mesh_file), *passthrough],
        )
    )
