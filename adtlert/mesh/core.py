"""2D triangle/quadrilateral meshes with the boundary and surface metadata used by the solver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import heapq
from pathlib import Path

import meshio
import numpy as np
import torch

from adtlert.utils.dtypes import FLOAT_DTYPE, INT_DTYPE, NP_FLOAT_DTYPE

Tensor = torch.Tensor


def _float(array) -> Tensor:
    return torch.as_tensor(np.asarray(array, dtype=NP_FLOAT_DTYPE), dtype=FLOAT_DTYPE)


def _int(array) -> Tensor:
    return torch.as_tensor(np.asarray(array, dtype=np.int32).copy(), dtype=INT_DTYPE)


def edge_midpoint_builder(nodes) -> tuple[list[np.ndarray], Callable[[int, int], int], dict[tuple[int, int], int]]:
    """Return a growing node list, ``midpoint(a, b)`` (adds each edge midpoint once), and its index."""

    points = [np.asarray(point, dtype=float) for point in np.asarray(nodes, dtype=float)]
    index: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key not in index:
            index[key] = len(points)
            points.append(0.5 * (points[key[0]] + points[key[1]]))
        return index[key]

    return points, midpoint, index


def triangle_areas(nodes, cells) -> Tensor:
    """Compute the area of each triangle cell."""

    cell_nodes = np.asarray(nodes, dtype=NP_FLOAT_DTYPE)[np.asarray(cells, dtype=np.int32)]
    e1, e2 = cell_nodes[:, 1] - cell_nodes[:, 0], cell_nodes[:, 2] - cell_nodes[:, 0]
    return _float(0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]))


def cell_areas_2d(nodes, cells) -> Tensor:
    """Shoelace areas of triangle or quadrilateral cells."""

    cells = np.asarray(cells, dtype=np.int32)
    if cells.ndim != 2 or cells.shape[1] not in (3, 4):
        raise ValueError("cells must have shape (num_cells, 3) or (num_cells, 4)")
    cell_nodes = np.asarray(nodes, dtype=NP_FLOAT_DTYPE)[cells]
    x, y = cell_nodes[:, :, 0], cell_nodes[:, :, 1]
    return _float(0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)))


def _cell_edges(cells: np.ndarray) -> np.ndarray:
    """``(cells, width, 2)`` sorted node pairs of every polygon edge."""

    if cells.ndim != 2 or cells.shape[1] not in (3, 4):
        raise ValueError("cells must have shape (num_cells, 3) or (num_cells, 4)")
    return np.sort(np.stack((cells, np.roll(cells, -1, axis=1)), axis=-1), axis=-1)


def extract_boundary_edges(cells) -> Tensor:
    """Return sorted edges that belong to exactly one 2D cell."""

    edges, counts = np.unique(_cell_edges(np.asarray(cells, dtype=np.int32)).reshape(-1, 2), axis=0, return_counts=True)
    return _int(edges[counts == 1])


def _boundary_topology(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lexicographically sorted boundary edges and the cell owning each."""

    edges = _cell_edges(cells)
    unique, first, counts = np.unique(edges.reshape(-1, 2), axis=0, return_index=True, return_counts=True)
    boundary = counts == 1
    return unique[boundary], first[boundary] // cells.shape[1]


def _boundary_geometry(nodes: np.ndarray, cells: np.ndarray, edges: np.ndarray, edge_cells: np.ndarray):
    """Centers, lengths, and outward unit normals of boundary edges."""

    edge_nodes = nodes[edges]
    centers = np.mean(edge_nodes, axis=1)
    vectors = edge_nodes[:, 1] - edge_nodes[:, 0]
    lengths = np.linalg.norm(vectors, axis=1)
    normals = np.stack((vectors[:, 1], -vectors[:, 0]), axis=1) / lengths[:, None]
    outward = np.sum(normals * (centers - np.mean(nodes[cells[edge_cells]], axis=1)), axis=1) >= 0.0
    return centers, lengths, normals * np.where(outward, 1.0, -1.0).astype(NP_FLOAT_DTYPE)[:, None]


def _infer_surface_path(nodes: np.ndarray, edges: np.ndarray, tol: float = 1e-8) -> list[int]:
    """Top boundary path between the upper-left and upper-right corners.

    Dijkstra over boundary edges, with edge costs that grow with depth below the
    highest boundary edge so the path hugs the top.
    """

    boundary_nodes = np.unique(edges)
    x = nodes[boundary_nodes, 0]

    def upper_corner(extreme: float) -> int:
        candidates = boundary_nodes[np.abs(x - extreme) <= tol].tolist()
        return int(max(candidates, key=lambda node: (nodes[node, 1], -nodes[node, 0])))

    start, stop = upper_corner(float(x.min())), upper_corner(float(x.max()))
    centers_y = np.mean(nodes[edges], axis=1)[:, 1]
    lengths = np.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1)
    span = max(float(centers_y.max() - centers_y.min()), tol)
    adjacency: dict[int, list[tuple[int, float]]] = {int(node): [] for node in boundary_nodes}
    for (a, b), center_y, length in zip(edges.tolist(), centers_y, lengths, strict=True):
        cost = float(length * (1.0 + 100.0 * (centers_y.max() - center_y) / span))
        adjacency[a].append((b, cost))
        adjacency[b].append((a, cost))

    distances, previous, visited, heap = {start: 0.0}, {}, set(), [(0.0, start)]
    while heap:
        distance, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == stop:
            break
        for neighbor, cost in adjacency[node]:
            if distance + cost < distances.get(neighbor, float("inf")):
                distances[neighbor], previous[neighbor] = distance + cost, node
                heapq.heappush(heap, (distance + cost, neighbor))

    if start == stop:
        return [start]
    if stop not in previous:
        return [start, stop]
    path = [stop]
    while path[-1] != start:
        path.append(previous[path[-1]])
    return path[::-1]


def _surface_metadata(nodes: np.ndarray, edges: np.ndarray, surface_ids, tol: float = 1e-8):
    """Surface-edge mask, surface nodes, mean surface level, and flatness of an ordered surface path."""

    surface_ids = np.asarray(surface_ids, dtype=np.int32)
    surface_edges = {tuple(sorted(pair)) for pair in zip(surface_ids[:-1].tolist(), surface_ids[1:].tolist(), strict=True)}
    mask = np.asarray([tuple(edge) in surface_edges for edge in edges.tolist()], dtype=bool)
    surface = nodes[surface_ids]
    if surface.size == 0:
        return surface_ids, mask, surface, 0.0, True
    flat = bool(np.max(np.abs(surface[:, 1] - surface[0, 1])) <= tol)
    return surface_ids, mask, surface, float(np.mean(surface[:, 1])), flat


def _first_hit(points: np.ndarray, nodes: np.ndarray, cells: np.ndarray, tol: float, inside) -> tuple[Tensor, Tensor]:
    """Locate points: a node within ``tol`` wins (weight 1), else the first cell ``inside`` accepts."""

    cell_ids = np.full(points.shape[0], -1, dtype=np.int32)
    weights = np.zeros((points.shape[0], cells.shape[1]))
    for point_id, point in enumerate(points):
        distances = np.linalg.norm(nodes - point, axis=1)
        nearest = int(np.argmin(distances))
        adjacent = np.flatnonzero(np.any(cells == nearest, axis=1)) if distances[nearest] <= tol else []
        if len(adjacent):
            cell_ids[point_id] = adjacent[0]
            weights[point_id, int(np.flatnonzero(cells[adjacent[0]] == nearest)[0])] = 1.0
            continue
        hit = inside(point)
        if hit is not None:
            cell_ids[point_id], weights[point_id] = hit
    if np.any(cell_ids < 0):
        raise ValueError("some points could not be located in the mesh")
    return _int(cell_ids), _float(weights)


def locate_points_in_triangles(nodes, cells, points, tol: float = 1e-4) -> tuple[Tensor, Tensor]:
    """Containing triangle ids and barycentric weights of points."""

    nodes, cells = np.asarray(nodes, dtype=float), np.asarray(cells, dtype=np.int32)
    origins = nodes[cells[:, 0]]
    transforms = np.stack((nodes[cells[:, 1]] - origins, nodes[cells[:, 2]] - origins), axis=-1)

    def inside(point):
        local = np.linalg.solve(transforms, (point - origins)[..., None])[..., 0]
        barycentric = np.column_stack((1.0 - local.sum(axis=1), local))
        hits = np.flatnonzero(np.all((barycentric >= -tol) & (barycentric <= 1.0 + tol), axis=1))
        return (int(hits[0]), barycentric[hits[0]]) if hits.size else None

    return _first_hit(np.asarray(points, dtype=float), nodes, cells, tol, inside)


def _bilinear_weights(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    values = 0.25 * np.asarray([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    gradients = 0.25 * np.stack(
        (np.asarray([-(1 - eta), 1 - eta, 1 + eta, -(1 + eta)]), np.asarray([-(1 - xi), -(1 + xi), 1 + xi, 1 - xi])), axis=1
    )
    return values, gradients


def _quad_weights(quad: np.ndarray, point: np.ndarray, tol: float) -> np.ndarray | None:
    """Q1 weights of a point on an edge (linear) or inside a convex quadrilateral (Newton)."""

    for start in range(4):
        stop = (start + 1) % 4
        edge = quad[stop] - quad[start]
        length_sq = float(edge @ edge)
        if length_sq <= tol * tol:
            continue
        t = float((point - quad[start]) @ edge / length_sq)
        if -tol <= t <= 1.0 + tol:
            t = min(max(t, 0.0), 1.0)
            if float(np.linalg.norm(point - (quad[start] + t * edge))) <= tol:
                weights = np.zeros(4)
                weights[start], weights[stop] = 1.0 - t, t
                return weights

    xi = eta = 0.0
    for _ in range(32):
        values, gradients = _bilinear_weights(xi, eta)
        residual = values @ quad - point
        if float(np.linalg.norm(residual)) <= tol and -1.0 - tol <= xi <= 1.0 + tol and -1.0 - tol <= eta <= 1.0 + tol:
            break
        try:
            step = np.linalg.solve(gradients.T @ quad, residual)
        except np.linalg.LinAlgError:
            return None
        xi, eta = xi - float(step[0]), eta - float(step[1])
        if float(np.linalg.norm(step)) <= 1e-12:
            break
    if not (-1.0 - tol <= xi <= 1.0 + tol and -1.0 - tol <= eta <= 1.0 + tol):
        return None
    values, _ = _bilinear_weights(min(max(xi, -1.0), 1.0), min(max(eta, -1.0), 1.0))
    return None if np.any(values < -tol) or np.any(values > 1.0 + tol) else values


def locate_points_in_quadrilaterals(nodes, cells, points, tol: float = 1e-4) -> tuple[Tensor, Tensor]:
    """Containing quadrilateral ids and bilinear Q1 weights of points."""

    nodes, cells = np.asarray(nodes, dtype=float), np.asarray(cells, dtype=np.int32)

    def inside(point):
        for cell_id, cell in enumerate(cells):
            weights = _quad_weights(nodes[cell], point, tol)
            if weights is not None:
                return cell_id, weights
        return None

    return _first_hit(np.asarray(points, dtype=float), nodes, cells, tol, inside)


def refine_triangle_mesh(nodes, cells) -> tuple[Tensor, Tensor, Tensor, dict[tuple[int, int], int]]:
    """Split each triangle into four through its edge midpoints (H2 refinement)."""

    points, midpoint, index = edge_midpoint_builder(nodes)
    refined, parents = [], []
    for parent, (n0, n1, n2) in enumerate(np.asarray(cells, dtype=np.int32).tolist()):
        n01, n12, n20 = midpoint(n0, n1), midpoint(n1, n2), midpoint(n2, n0)
        refined += [[n0, n01, n20], [n1, n12, n01], [n2, n20, n12], [n01, n12, n20]]
        parents += [parent] * 4
    return _float(points), _int(refined), _int(parents), index


def _orient_ccw(nodes, triangle: list[int]) -> list[int]:
    """Return the triangle with positive signed area."""

    p0, p1, p2 = (np.asarray(nodes[node], dtype=float) for node in triangle)
    e1, e2 = p1 - p0, p2 - p0
    return triangle if e1[0] * e2[1] - e1[1] * e2[0] > 0.0 else [triangle[0], triangle[2], triangle[1]]


def insert_surface_points_into_triangle_mesh(
    nodes, cells, boundary_edges, boundary_edge_cells, surface_node_ids, points, tol: float = 1e-4
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Insert top-surface points as vertices by fanning the boundary cells they split.

    Returns refined nodes, cells, parent cell ids, the node id of each point, and the refined surface path.
    """

    nodes, cells = np.asarray(nodes, dtype=float), np.asarray(cells, dtype=np.int32)
    surface = np.asarray(surface_node_ids, dtype=np.int32)
    points = np.asarray(points, dtype=float)
    unchanged = (_float(nodes), _int(cells), _int(np.arange(cells.shape[0])))
    if points.shape[0] == 0 or surface.shape[0] < 2:
        return (*unchanged, _int(np.empty(0)), _int(surface))

    owner = {tuple(sorted(edge)): cell for edge, cell in zip(np.asarray(boundary_edges).tolist(), np.asarray(boundary_edge_cells).tolist(), strict=True)}
    segments = list(zip(surface[:-1].tolist(), surface[1:].tolist(), strict=True))
    point_nodes = np.full(points.shape[0], -1, dtype=np.int32)
    on_segment: dict[int, list[tuple[float, int, np.ndarray]]] = {}
    for point_id, point in enumerate(points):
        distances = np.linalg.norm(nodes - point, axis=1)
        if distances.min() <= tol:
            point_nodes[point_id] = int(np.argmin(distances))
            continue
        best = (-1, 0.0, float("inf"))
        for segment_id, (a, b) in enumerate(segments):
            edge = nodes[b] - nodes[a]
            length_sq = float(edge @ edge)
            if length_sq <= tol:
                continue
            t = float((point - nodes[a]) @ edge / length_sq)
            distance = float(np.linalg.norm(point - (nodes[a] + t * edge)))
            if -tol <= t <= 1.0 + tol and distance <= tol and distance < best[2]:
                best = (segment_id, min(max(t, 0.0), 1.0), distance)
        segment_id, t, _ = best
        if segment_id < 0:
            raise ValueError("surface point could not be matched to a top boundary segment")
        if t <= tol or t >= 1.0 - tol:
            point_nodes[point_id] = segments[segment_id][0 if t <= tol else 1]
        else:
            on_segment.setdefault(segment_id, []).append((t, point_id, point))
    if not on_segment:
        return (*unchanged, _int(point_nodes), _int(surface))

    new_nodes = list(nodes)
    splits: dict[int, tuple[int, int, list[int]]] = {}
    for segment_id, items in on_segment.items():
        a, b = segments[segment_id]
        cell = owner[tuple(sorted((a, b)))]
        if cell in splits:
            raise ValueError("surface cell has multiple top boundary segments; unsupported topology")
        inserted, previous = [], None
        for t, point_id, point in sorted(items, key=lambda item: item[0]):
            if previous is not None and abs(t - previous[0]) <= tol:
                point_nodes[point_id] = previous[1]
                continue
            new_nodes.append(point)
            point_nodes[point_id] = len(new_nodes) - 1
            inserted.append(len(new_nodes) - 1)
            previous = (t, len(new_nodes) - 1)
        splits[cell] = (a, b, inserted)

    refined, parents = [], []
    for cell_id, cell in enumerate(cells.tolist()):
        if cell_id not in splits:
            refined.append(cell)
            parents.append(cell_id)
            continue
        a, b, inserted = splits[cell_id]
        interior = [node for node in cell if node not in (a, b)]
        if len(interior) != 1:
            raise ValueError("surface boundary cell must contain exactly one interior vertex")
        chain = [a, *inserted, b]
        for start, stop in zip(chain[:-1], chain[1:], strict=True):
            refined.append(_orient_ccw(new_nodes, [start, stop, interior[0]]))
            parents.append(cell_id)

    refined_surface = [int(surface[0])]
    for segment_id, (_, b) in enumerate(segments):
        refined_surface += [int(point_nodes[point_id]) for _, point_id, _ in sorted(on_segment.get(segment_id, []), key=lambda item: item[0])]
        refined_surface.append(b)
    return _float(np.asarray(new_nodes)), _int(refined), _int(parents), _int(point_nodes), _int(refined_surface)


def expand_columnar_triangle_mesh(nodes, cells, tol: float = 1e-6) -> tuple[Tensor, Tensor, Tensor] | None:
    """Complete terrain-following columnar meshes whose quads are split into single triangles.

    Every cell spans two adjacent node columns; the missing half-triangle of each strip
    quad is added (creating a bottom node where a column is one node short). Returns
    ``None`` when the mesh is not such a structure.
    """

    nodes, cells = np.asarray(nodes, dtype=float), np.asarray(cells, dtype=np.int32)
    if nodes.shape[0] == 0 or cells.shape[0] == 0:
        return None
    columns: list[list[int]] = []
    for node in np.argsort(nodes[:, 0], kind="mergesort").tolist():
        if columns and abs(nodes[node, 0] - nodes[columns[-1][0], 0]) <= tol:
            columns[-1].append(node)
        else:
            columns.append([node])
    sizes = [len(column) for column in columns]
    if len(columns) < 10 or min(sizes) < 2 or max(sizes) - min(sizes) > 1:
        return None

    column_of = np.full(nodes.shape[0], -1, dtype=np.int32)
    row_of = np.full(nodes.shape[0], -1, dtype=np.int32)
    for column_id, column in enumerate(columns):
        column.sort(key=lambda node: (-nodes[node, 1], nodes[node, 0]))
        for row, node in enumerate(column):
            column_of[node], row_of[node] = column_id, row

    expanded_nodes = nodes.tolist()
    expanded, parents = [], []
    for parent, cell in enumerate(cells):
        column_ids, rows = column_of[cell], row_of[cell]
        unique, counts = np.unique(column_ids, return_counts=True)
        if unique.shape[0] != 2 or unique[1] != unique[0] + 1 or sorted(counts.tolist()) != [1, 2]:
            return None
        pair_column, lone_column = int(unique[np.argmax(counts)]), int(unique[np.argmin(counts)])
        pair = column_ids == pair_column
        order = np.argsort(rows[pair])
        pair_nodes, pair_rows = cell[pair][order], rows[pair][order]
        if pair_rows[1] != pair_rows[0] + 1:
            return None
        lone_row = int(rows[~pair][0])
        if lone_row not in pair_rows:
            return None
        missing_row = int(pair_rows[1] if lone_row == pair_rows[0] else pair_rows[0])

        lone = columns[lone_column]
        if missing_row >= len(lone):
            if missing_row != len(lone) or len(lone) + 1 != max(sizes):
                return None
            depth = nodes[int(pair_nodes[1]), 1] - nodes[columns[pair_column][0], 1]
            expanded_nodes.append([float(nodes[lone[0], 0]), float(nodes[lone[0], 1] + depth)])
            lone.append(len(expanded_nodes) - 1)
        missing = int(lone[missing_row])
        if missing in cell.tolist():
            return None
        expanded += [cell.tolist(), _orient_ccw(expanded_nodes, [missing, int(pair_nodes[0]), int(pair_nodes[1])])]
        parents += [parent, parent]
    return _float(expanded_nodes), _int(expanded), _int(parents)


def build_quadratic_triangle_mesh(nodes, cells, boundary_edges) -> tuple[Tensor, Tensor, Tensor]:
    """Shared-edge P2 topology: cells ``(v0, v1, v2, m01, m12, m20)``, boundary edges ``(a, b, m)``."""

    points, midpoint, _ = edge_midpoint_builder(nodes)
    quadratic = [[n0, n1, n2, midpoint(n0, n1), midpoint(n1, n2), midpoint(n2, n0)] for n0, n1, n2 in np.asarray(cells).tolist()]
    boundary = [[a, b, midpoint(a, b)] for a, b in np.asarray(boundary_edges).tolist()]
    return _float(points), _int(quadratic), _int(boundary)


@dataclass(frozen=True)
class Mesh:
    """2D triangle or quadrilateral mesh with precomputed boundary and surface data."""

    nodes: Tensor
    cells: Tensor
    boundary_edges: Tensor
    boundary_edge_cells: Tensor
    boundary_edge_centers: Tensor
    boundary_edge_lengths: Tensor
    boundary_edge_normals: Tensor
    surface_node_ids: Tensor
    surface_edge_mask: Tensor
    surface_nodes: Tensor
    surface_reference_level: Tensor
    flat_surface: Tensor
    cell_areas: Tensor

    @classmethod
    def from_arrays(cls, nodes, cells, *, surface_node_ids=None) -> Mesh:
        """Build a mesh from coordinates and triangle/quadrilateral connectivity.

        Unused nodes are dropped. The top surface is inferred unless ``surface_node_ids``
        gives it as an ordered node path.
        """

        nodes = np.asarray(_float(nodes))
        cells = np.asarray(cells, dtype=np.int32)
        surface = None if surface_node_ids is None else np.asarray(surface_node_ids, dtype=np.int32)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError("nodes must have shape (num_nodes, 2)")
        if cells.ndim != 2 or cells.shape[1] not in (3, 4):
            raise ValueError("cells must have shape (num_cells, 3) or (num_cells, 4)")
        if np.any(cells < 0):
            raise ValueError("cells contain negative node indices")
        if cells.size and np.any(cells >= nodes.shape[0]):
            raise ValueError("cells reference nodes outside the mesh")
        used = np.unique(cells)
        if used.size != nodes.shape[0]:
            remap = np.full(nodes.shape[0], -1, dtype=np.int32)
            remap[used] = np.arange(used.size, dtype=np.int32)
            nodes, cells = nodes[used], remap[cells]
            surface = None if surface is None else remap[surface]

        areas = cell_areas_2d(nodes, cells)
        if bool(torch.any(areas <= 0.0)):
            raise ValueError("cells must define non-degenerate 2D polygons")
        edges, edge_cells = _boundary_topology(cells)
        centers, lengths, normals = _boundary_geometry(nodes, cells, edges, edge_cells)
        nodes64 = nodes.astype(float)
        surface_ids, mask, surface_nodes, level, flat = _surface_metadata(
            nodes64, edges, _infer_surface_path(nodes64, edges) if surface is None else surface
        )
        return cls(
            nodes=_float(nodes),
            cells=_int(cells),
            boundary_edges=_int(edges),
            boundary_edge_cells=_int(edge_cells),
            boundary_edge_centers=_float(centers),
            boundary_edge_lengths=_float(lengths),
            boundary_edge_normals=_float(normals),
            surface_node_ids=_int(surface_ids),
            surface_edge_mask=torch.as_tensor(mask),
            surface_nodes=_float(surface_nodes),
            surface_reference_level=torch.tensor(level, dtype=FLOAT_DTYPE),
            flat_surface=torch.tensor(flat),
            cell_areas=areas,
        )

    @classmethod
    def from_meshio(cls, mesh: meshio.Mesh) -> Mesh:
        """Create a mesh from a meshio mesh (quadrilaterals preferred over triangles)."""

        for kind in ("quad", "quadrilateral", "triangle"):
            if kind in mesh.cells_dict:
                return cls.from_arrays(mesh.points[:, :2], mesh.cells_dict[kind])
        raise ValueError("meshio mesh does not contain triangle or quadrilateral cells")

    @classmethod
    def from_file(cls, path: str | Path) -> Mesh:
        """Load a 2D triangle or quadrilateral mesh via meshio."""

        return cls.from_meshio(meshio.read(path))

    @property
    def node_count(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def cell_count(self) -> int:
        return int(self.cells.shape[0])

    @property
    def cell_node_count(self) -> int:
        return int(self.cells.shape[1])

    @property
    def is_triangle_mesh(self) -> bool:
        return self.cell_node_count == 3

    @property
    def is_quadrilateral_mesh(self) -> bool:
        return self.cell_node_count == 4

    @property
    def is_flat_surface(self) -> bool:
        """Whether the top boundary is flat within tolerance."""

        return bool(self.flat_surface)

    def locate_points(self, points, tol: float = 1e-4) -> tuple[Tensor, Tensor]:
        """Containing cell ids and interpolation weights of points."""

        locate = locate_points_in_triangles if self.is_triangle_mesh else locate_points_in_quadrilaterals
        return locate(self.nodes, self.cells, points, tol=tol)

    def _require_triangles(self, operation: str) -> None:
        if not self.is_triangle_mesh:
            raise ValueError(f"{operation} is currently implemented for triangle meshes only")

    def refine_uniform(self) -> tuple[Mesh, Tensor]:
        """Uniformly refine all triangles; returns the refined mesh and parent cell ids."""

        self._require_triangles("uniform refinement")
        nodes, cells, parents, midpoints = refine_triangle_mesh(self.nodes, self.cells)
        surface = self.surface_node_ids.tolist()
        refined_surface = [surface[0]]
        for a, b in zip(surface[:-1], surface[1:], strict=True):
            refined_surface += [midpoints[(min(a, b), max(a, b))], b]
        return Mesh.from_arrays(nodes, cells, surface_node_ids=refined_surface), parents

    def insert_surface_points(self, points, tol: float = 1e-4) -> tuple[Mesh, Tensor, Tensor]:
        """Insert top-surface points as mesh vertices; returns the mesh, parent cell ids, and point node ids."""

        self._require_triangles("surface point insertion")
        nodes, cells, parents, point_nodes, surface = insert_surface_points_into_triangle_mesh(
            self.nodes, self.cells, self.boundary_edges, self.boundary_edge_cells, self.surface_node_ids, points, tol=tol
        )
        return Mesh.from_arrays(nodes, cells, surface_node_ids=surface), parents, point_nodes

    def expand_columnar_cells(self, tol: float = 1e-6) -> tuple[Mesh, Tensor] | None:
        """Complete single-triangle terrain strips into triangulated quads (see :func:`expand_columnar_triangle_mesh`)."""

        expanded = expand_columnar_triangle_mesh(self.nodes, self.cells, tol=tol) if self.is_triangle_mesh else None
        if expanded is None:
            return None
        nodes, cells, parents = expanded
        return Mesh.from_arrays(nodes, cells, surface_node_ids=self.surface_node_ids), parents

    def build_quadratic_topology(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return shared-edge P2 nodes, cell connectivity, and boundary-edge connectivity."""

        self._require_triangles("quadratic topology")
        return build_quadratic_triangle_mesh(self.nodes, self.cells, self.boundary_edges)
