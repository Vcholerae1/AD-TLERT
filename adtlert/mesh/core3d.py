"""Three-dimensional tetrahedral mesh primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import meshio
import numpy as np

from adtlert.utils.dtypes import FLOAT_DTYPE, INT_DTYPE, NP_FLOAT_DTYPE
from adtlert.utils.torch_runtime import Array, torch_np


_TETRA_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


def tetrahedron_volumes(nodes: Array, cells: Array) -> Array:
    """Return absolute volumes for four-node tetrahedra."""

    nodes_np = np.asarray(nodes, dtype=NP_FLOAT_DTYPE)
    cells_np = np.asarray(cells, dtype=np.int32)
    cell_nodes = nodes_np[cells_np]
    jacobians = np.stack(
        (
            cell_nodes[:, 1] - cell_nodes[:, 0],
            cell_nodes[:, 2] - cell_nodes[:, 0],
            cell_nodes[:, 3] - cell_nodes[:, 0],
        ),
        axis=-1,
    )
    return torch_np.asarray(np.abs(np.linalg.det(jacobians)) / 6.0, dtype=FLOAT_DTYPE)


def _boundary_topology(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    face_owners: dict[tuple[int, int, int], list[int]] = {}
    for cell_id, cell in enumerate(cells):
        for local_face in _TETRA_FACES:
            face = tuple(sorted(int(cell[index]) for index in local_face))
            face_owners.setdefault(face, []).append(cell_id)

    boundary = [(face, owners[0]) for face, owners in face_owners.items() if len(owners) == 1]
    boundary.sort(key=lambda item: item[0])
    return (
        np.asarray([item[0] for item in boundary], dtype=np.int32),
        np.asarray([item[1] for item in boundary], dtype=np.int32),
    )


def _boundary_geometry(
    nodes: np.ndarray,
    cells: np.ndarray,
    faces: np.ndarray,
    face_cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    face_nodes = nodes[faces]
    centers = np.mean(face_nodes, axis=1)
    cross = np.cross(face_nodes[:, 1] - face_nodes[:, 0], face_nodes[:, 2] - face_nodes[:, 0])
    cross_norm = np.linalg.norm(cross, axis=1)
    areas = 0.5 * cross_norm
    normals = cross / cross_norm[:, None]
    cell_centers = np.mean(nodes[cells[face_cells]], axis=1)
    orientation = np.sum(normals * (centers - cell_centers), axis=1)
    normals *= np.where(orientation >= 0.0, 1.0, -1.0)[:, None]
    return centers, areas, normals


def locate_points_in_tetrahedra(
    nodes: Array,
    cells: Array,
    points: Array,
    tol: float = 1.0e-8,
) -> tuple[Array, Array]:
    """Locate points and return tetrahedron ids and barycentric weights."""

    nodes_np = np.asarray(nodes, dtype=float)
    cells_np = np.asarray(cells, dtype=np.int32)
    points_np = np.asarray(points, dtype=float)
    cell_nodes = nodes_np[cells_np]
    origins = cell_nodes[:, 0]
    jacobians = np.stack(
        (
            cell_nodes[:, 1] - origins,
            cell_nodes[:, 2] - origins,
            cell_nodes[:, 3] - origins,
        ),
        axis=-1,
    )
    inverse_jacobians = np.linalg.inv(jacobians)
    lower = np.min(cell_nodes, axis=1) - tol
    upper = np.max(cell_nodes, axis=1) + tol

    cell_ids = np.full(points_np.shape[0], -1, dtype=np.int32)
    weights = np.zeros((points_np.shape[0], 4), dtype=NP_FLOAT_DTYPE)
    for point_id, point in enumerate(points_np):
        nearest = int(np.argmin(np.linalg.norm(nodes_np - point, axis=1)))
        if np.linalg.norm(nodes_np[nearest] - point) <= tol:
            candidates = np.flatnonzero(np.any(cells_np == nearest, axis=1))
        else:
            candidates = np.flatnonzero(np.all((point >= lower) & (point <= upper), axis=1))
        for cell_id in candidates:
            local = inverse_jacobians[cell_id] @ (point - origins[cell_id])
            barycentric = np.asarray((1.0 - np.sum(local), *local), dtype=NP_FLOAT_DTYPE)
            if np.all(barycentric >= -tol) and np.all(barycentric <= 1.0 + tol):
                barycentric[np.abs(barycentric) <= tol] = 0.0
                barycentric /= np.sum(barycentric)
                cell_ids[point_id] = int(cell_id)
                weights[point_id] = barycentric
                break

    if np.any(cell_ids < 0):
        missing = np.flatnonzero(cell_ids < 0).tolist()
        raise ValueError(f"points lie outside the tetrahedral mesh: indices={missing}")
    return torch_np.asarray(cell_ids, dtype=INT_DTYPE), torch_np.asarray(weights, dtype=FLOAT_DTYPE)


@dataclass(frozen=True)
class Mesh3D:
    """Four-node tetrahedral mesh with boundary-face metadata."""

    nodes: Array
    cells: Array
    boundary_faces: Array
    boundary_face_cells: Array
    boundary_face_centers: Array
    boundary_face_areas: Array
    boundary_face_normals: Array
    surface_face_mask: Array
    surface_reference_level: Array
    flat_surface: Array
    cell_volumes: Array

    @classmethod
    def from_arrays(
        cls,
        nodes: Array,
        cells: Array,
        *,
        surface_face_mask: Array | None = None,
    ) -> "Mesh3D":
        node_array = np.asarray(nodes, dtype=NP_FLOAT_DTYPE)
        cell_array = np.asarray(cells, dtype=np.int32)
        if node_array.ndim != 2 or node_array.shape[1] != 3:
            raise ValueError("3D nodes must have shape (num_nodes, 3)")
        if cell_array.ndim != 2 or cell_array.shape[1] != 4:
            raise ValueError("3D cells must have shape (num_cells, 4)")
        if cell_array.size == 0:
            raise ValueError("tetrahedral mesh must contain at least one cell")
        if np.any(cell_array < 0) or np.any(cell_array >= node_array.shape[0]):
            raise ValueError("cells reference nodes outside the mesh")

        used = np.unique(cell_array.reshape(-1))
        if used.size != node_array.shape[0]:
            remap = np.full(node_array.shape[0], -1, dtype=np.int32)
            remap[used] = np.arange(used.size, dtype=np.int32)
            node_array = node_array[used]
            cell_array = remap[cell_array]

        volumes = np.asarray(tetrahedron_volumes(node_array, cell_array), dtype=float)
        if np.any(volumes <= 0.0):
            raise ValueError("cells must define non-degenerate tetrahedra")

        faces, face_cells = _boundary_topology(cell_array)
        centers, areas, normals = _boundary_geometry(node_array, cell_array, faces, face_cells)
        z_max = float(np.max(node_array[:, 2]))
        z_span = max(float(np.ptp(node_array[:, 2])), 1.0)
        surface_tol = max(1.0e-8, z_span * 1.0e-8)
        if surface_face_mask is None:
            # For a height-field terrain the ground surface is exactly the set
            # of exterior faces whose outward normal has a positive z component.
            surface_mask = normals[:, 2] > 1.0e-8
        else:
            surface_mask = np.asarray(surface_face_mask, dtype=bool).reshape(-1)
            if surface_mask.shape != (faces.shape[0],):
                raise ValueError(
                    "surface_face_mask must have one entry per boundary face "
                    f"({surface_mask.shape} != ({faces.shape[0]},))"
                )
        if not np.any(surface_mask):
            raise ValueError("could not identify any upward-facing terrain surface faces")
        surface_nodes = np.unique(faces[surface_mask].reshape(-1)) if np.any(surface_mask) else np.empty(0, dtype=int)
        flat_surface = bool(surface_nodes.size and np.ptp(node_array[surface_nodes, 2]) <= surface_tol)

        return cls(
            nodes=torch_np.asarray(node_array, dtype=FLOAT_DTYPE),
            cells=torch_np.asarray(cell_array, dtype=INT_DTYPE),
            boundary_faces=torch_np.asarray(faces, dtype=INT_DTYPE),
            boundary_face_cells=torch_np.asarray(face_cells, dtype=INT_DTYPE),
            boundary_face_centers=torch_np.asarray(centers, dtype=FLOAT_DTYPE),
            boundary_face_areas=torch_np.asarray(areas, dtype=FLOAT_DTYPE),
            boundary_face_normals=torch_np.asarray(normals, dtype=FLOAT_DTYPE),
            surface_face_mask=torch_np.asarray(surface_mask, dtype=bool),
            surface_reference_level=torch_np.asarray(z_max, dtype=FLOAT_DTYPE),
            flat_surface=torch_np.asarray(flat_surface, dtype=bool),
            cell_volumes=torch_np.asarray(volumes, dtype=FLOAT_DTYPE),
        )

    @classmethod
    def from_meshio(cls, mesh: meshio.Mesh) -> "Mesh3D":
        try:
            cells = mesh.cells_dict["tetra"]
        except KeyError as exc:
            raise ValueError("meshio mesh does not contain four-node tetrahedra") from exc
        return cls.from_arrays(mesh.points[:, :3], cells)

    @classmethod
    def from_file(cls, path: str | Path) -> "Mesh3D":
        return cls.from_meshio(meshio.read(path))

    @property
    def node_count(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def cell_count(self) -> int:
        return int(self.cells.shape[0])

    @property
    def dimension(self) -> int:
        return 3

    @property
    def is_flat_surface(self) -> bool:
        return bool(self.flat_surface)

    def locate_points(self, points: Array, tol: float = 1.0e-8) -> tuple[Array, Array]:
        return locate_points_in_tetrahedra(self.nodes, self.cells, points, tol=tol)

    def build_quadratic_topology(self) -> tuple[Array, Array, Array]:
        """Return shared-edge P2 nodes, cell DOFs, and boundary-face DOFs."""

        nodes = np.asarray(self.nodes, dtype=NP_FLOAT_DTYPE)
        cells = np.asarray(self.cells, dtype=np.int32)
        faces = np.asarray(self.boundary_faces, dtype=np.int32)
        quadratic_nodes = nodes.tolist()
        edge_midpoints: dict[tuple[int, int], int] = {}

        def midpoint(node_a: int, node_b: int) -> int:
            edge = tuple(sorted((int(node_a), int(node_b))))
            if edge not in edge_midpoints:
                edge_midpoints[edge] = len(quadratic_nodes)
                quadratic_nodes.append((0.5 * (nodes[edge[0]] + nodes[edge[1]])).tolist())
            return edge_midpoints[edge]

        quadratic_cells = []
        for n0, n1, n2, n3 in cells:
            quadratic_cells.append(
                (
                    int(n0), int(n1), int(n2), int(n3),
                    midpoint(n0, n1), midpoint(n1, n2), midpoint(n2, n0),
                    midpoint(n0, n3), midpoint(n1, n3), midpoint(n2, n3),
                )
            )

        quadratic_faces = []
        for n0, n1, n2 in faces:
            quadratic_faces.append(
                (int(n0), int(n1), int(n2), midpoint(n0, n1), midpoint(n1, n2), midpoint(n2, n0))
            )
        return (
            torch_np.asarray(quadratic_nodes, dtype=FLOAT_DTYPE),
            torch_np.asarray(quadratic_cells, dtype=INT_DTYPE),
            torch_np.asarray(quadratic_faces, dtype=INT_DTYPE),
        )
