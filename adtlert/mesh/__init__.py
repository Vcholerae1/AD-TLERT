"""Mesh primitives and meshio interoperability."""

from adtlert.mesh.core import (
    Mesh,
    build_quadratic_triangle_mesh,
    cell_areas_2d,
    extract_boundary_edges,
    locate_points_in_quadrilaterals,
    locate_points_in_triangles,
    refine_triangle_mesh,
    triangle_areas,
)
from adtlert.mesh.core3d import Mesh3D, locate_points_in_tetrahedra, tetrahedron_volumes

__all__ = [
    "Mesh",
    "Mesh3D",
    "build_quadratic_triangle_mesh",
    "cell_areas_2d",
    "extract_boundary_edges",
    "locate_points_in_quadrilaterals",
    "locate_points_in_tetrahedra",
    "locate_points_in_triangles",
    "refine_triangle_mesh",
    "tetrahedron_volumes",
    "triangle_areas",
]
