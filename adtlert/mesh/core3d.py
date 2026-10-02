"""Three-dimensional tetrahedral mesh primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import meshio
import numpy as np
import torch

from adtlert.mesh.core import _float, _int, edge_midpoint_builder
from adtlert.utils.dtypes import FLOAT_DTYPE, NP_FLOAT_DTYPE

Tensor = torch.Tensor
_TETRA_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


def _edge_matrices(cell_nodes: np.ndarray) -> np.ndarray:
    return np.stack(
        [cell_nodes[:, index] - cell_nodes[:, 0] for index in (1, 2, 3)], axis=-1
    )


def tetrahedron_volumes(nodes, cells) -> Tensor:
    """Return absolute volumes of four-node tetrahedra."""

    cell_nodes = np.asarray(nodes, dtype=NP_FLOAT_DTYPE)[
        np.asarray(cells, dtype=np.int32)
    ]
    return _float(np.abs(np.linalg.det(_edge_matrices(cell_nodes))) / 6.0)


def _boundary_topology(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted boundary faces and their owning cells."""

    faces = np.sort(cells[:, _TETRA_FACES], axis=-1).reshape(-1, 3)
    unique, first, counts = np.unique(
        faces, axis=0, return_index=True, return_counts=True
    )
    boundary = counts == 1
    return unique[boundary].astype(np.int32), (first[boundary] // 4).astype(np.int32)


def _boundary_geometry(
    nodes: np.ndarray, cells: np.ndarray, faces: np.ndarray, face_cells: np.ndarray
):
    """Centers, areas, and outward unit normals of boundary faces."""

    face_nodes = nodes[faces]
    centers = np.mean(face_nodes, axis=1)
    cross = np.cross(
        face_nodes[:, 1] - face_nodes[:, 0], face_nodes[:, 2] - face_nodes[:, 0]
    )
    norm = np.linalg.norm(cross, axis=1)
    normals = cross / norm[:, None]
    outward = (
        np.sum(normals * (centers - np.mean(nodes[cells[face_cells]], axis=1)), axis=1)
        >= 0.0
    )
    return centers, 0.5 * norm, normals * np.where(outward, 1.0, -1.0)[:, None]


def locate_points_in_tetrahedra(
    nodes, cells, points, tol: float = 1.0e-8
) -> tuple[Tensor, Tensor]:
    """Containing tetrahedron ids and barycentric weights of points."""

    nodes, cells = np.asarray(nodes, dtype=float), np.asarray(cells, dtype=np.int32)
    points = np.asarray(points, dtype=float)
    cell_nodes = nodes[cells]
    origins = cell_nodes[:, 0]
    inverse = np.linalg.inv(_edge_matrices(cell_nodes))
    lower, upper = cell_nodes.min(axis=1) - tol, cell_nodes.max(axis=1) + tol

    cell_ids = np.full(points.shape[0], -1, dtype=np.int32)
    weights = np.zeros((points.shape[0], 4), dtype=NP_FLOAT_DTYPE)
    for point_id, point in enumerate(points):
        nearest = int(np.argmin(np.linalg.norm(nodes - point, axis=1)))
        if np.linalg.norm(nodes[nearest] - point) <= tol:
            candidates = np.flatnonzero(np.any(cells == nearest, axis=1))
        else:
            candidates = np.flatnonzero(
                np.all((point >= lower) & (point <= upper), axis=1)
            )
        for cell_id in candidates:
            local = inverse[cell_id] @ (point - origins[cell_id])
            barycentric = np.asarray(
                (1.0 - np.sum(local), *local), dtype=NP_FLOAT_DTYPE
            )
            if np.all(barycentric >= -tol) and np.all(barycentric <= 1.0 + tol):
                barycentric[np.abs(barycentric) <= tol] = 0.0
                cell_ids[point_id], weights[point_id] = (
                    cell_id,
                    barycentric / np.sum(barycentric),
                )
                break
    if np.any(cell_ids < 0):
        raise ValueError(
            f"points lie outside the tetrahedral mesh: indices={np.flatnonzero(cell_ids < 0).tolist()}"
        )
    return _int(cell_ids), _float(weights)


@dataclass(frozen=True)
class Mesh3D:
    """Four-node tetrahedral mesh with boundary-face metadata."""

    nodes: Tensor
    cells: Tensor
    boundary_faces: Tensor
    boundary_face_cells: Tensor
    boundary_face_centers: Tensor
    boundary_face_areas: Tensor
    boundary_face_normals: Tensor
    surface_face_mask: Tensor
    surface_reference_level: Tensor
    flat_surface: Tensor
    cell_volumes: Tensor

    @classmethod
    def from_arrays(cls, nodes, cells, *, surface_face_mask=None) -> Mesh3D:
        """Build a tetrahedral mesh; the ground surface defaults to upward-facing exterior faces."""

        nodes = np.asarray(nodes, dtype=NP_FLOAT_DTYPE)
        cells = np.asarray(cells, dtype=np.int32)
        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError("3D nodes must have shape (num_nodes, 3)")
        if cells.ndim != 2 or cells.shape[1] != 4:
            raise ValueError("3D cells must have shape (num_cells, 4)")
        if cells.size == 0:
            raise ValueError("tetrahedral mesh must contain at least one cell")
        if np.any(cells < 0) or np.any(cells >= nodes.shape[0]):
            raise ValueError("cells reference nodes outside the mesh")
        used = np.unique(cells)
        if used.size != nodes.shape[0]:
            remap = np.full(nodes.shape[0], -1, dtype=np.int32)
            remap[used] = np.arange(used.size, dtype=np.int32)
            nodes, cells = nodes[used], remap[cells]

        volumes = np.asarray(tetrahedron_volumes(nodes, cells), dtype=float)
        if np.any(volumes <= 0.0):
            raise ValueError("cells must define non-degenerate tetrahedra")
        faces, face_cells = _boundary_topology(cells)
        centers, areas, normals = _boundary_geometry(nodes, cells, faces, face_cells)
        if (
            surface_face_mask is None
        ):  # height-field terrain: exterior faces whose outward normal points up
            surface_mask = normals[:, 2] > 1.0e-8
        else:
            surface_mask = np.asarray(surface_face_mask, dtype=bool).reshape(-1)
            if surface_mask.shape != (faces.shape[0],):
                raise ValueError(
                    f"surface_face_mask must have one entry per boundary face ({surface_mask.shape} != ({faces.shape[0]},))"
                )
        if not np.any(surface_mask):
            raise ValueError(
                "could not identify any upward-facing terrain surface faces"
            )
        surface_nodes = np.unique(faces[surface_mask])
        surface_tol = max(1.0e-8, max(float(np.ptp(nodes[:, 2])), 1.0) * 1.0e-8)
        return cls(
            nodes=_float(nodes),
            cells=_int(cells),
            boundary_faces=_int(faces),
            boundary_face_cells=_int(face_cells),
            boundary_face_centers=_float(centers),
            boundary_face_areas=_float(areas),
            boundary_face_normals=_float(normals),
            surface_face_mask=torch.as_tensor(surface_mask),
            surface_reference_level=torch.tensor(
                float(np.max(nodes[:, 2])), dtype=FLOAT_DTYPE
            ),
            flat_surface=torch.tensor(
                bool(np.ptp(nodes[surface_nodes, 2]) <= surface_tol)
            ),
            cell_volumes=_float(volumes),
        )

    @classmethod
    def from_meshio(cls, mesh: meshio.Mesh) -> Mesh3D:
        if "tetra" not in mesh.cells_dict:
            raise ValueError("meshio mesh does not contain four-node tetrahedra")
        return cls.from_arrays(mesh.points[:, :3], mesh.cells_dict["tetra"])

    @classmethod
    def from_file(cls, path: str | Path) -> Mesh3D:
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

    def locate_points(self, points, tol: float = 1.0e-8) -> tuple[Tensor, Tensor]:
        return locate_points_in_tetrahedra(self.nodes, self.cells, points, tol=tol)

    def build_quadratic_topology(self) -> tuple[Tensor, Tensor, Tensor]:
        """Shared-edge P2 nodes, cell DOFs (vertices, then edges 01, 12, 20, 03, 13, 23), and boundary-face DOFs."""

        points, midpoint, _ = edge_midpoint_builder(self.nodes)
        cells = [
            [
                n0,
                n1,
                n2,
                n3,
                midpoint(n0, n1),
                midpoint(n1, n2),
                midpoint(n2, n0),
                midpoint(n0, n3),
                midpoint(n1, n3),
                midpoint(n2, n3),
            ]
            for n0, n1, n2, n3 in self.cells.tolist()
        ]
        faces = [
            [n0, n1, n2, midpoint(n0, n1), midpoint(n1, n2), midpoint(n2, n0)]
            for n0, n1, n2 in self.boundary_faces.tolist()
        ]
        return _float(points), _int(cells), _int(faces)
