"""Finite-element bases, quadrature, and local element matrices."""

from adtlert.fem.boundary import assemble_local_boundary_mass, assemble_local_boundary_mass_p2, robin_boundary_coefficients
from adtlert.fem.tetrahedron import (
    TetrahedronP1Data,
    TetrahedronP2Data,
    build_tetrahedron_p1_data,
    build_tetrahedron_p2_data,
    p2_tetrahedron_shape_values,
    p2_triangle_mass_template,
)
from adtlert.fem.triangle import (
    P1ElementData,
    P2ElementData,
    TriangleQuadrature,
    assemble_local_mass,
    assemble_local_mass_p2,
    assemble_local_stiffness,
    assemble_local_stiffness_p2,
    build_p1_element_data,
    build_p2_element_data,
    p1_shape_functions,
    p2_shape_functions,
    reference_shape_gradients,
    reference_shape_gradients_p2,
    triangle_quadrature,
)

__all__ = [
    "P1ElementData",
    "P2ElementData",
    "TetrahedronP1Data",
    "TetrahedronP2Data",
    "TriangleQuadrature",
    "assemble_local_boundary_mass",
    "assemble_local_boundary_mass_p2",
    "assemble_local_mass",
    "assemble_local_mass_p2",
    "assemble_local_stiffness",
    "assemble_local_stiffness_p2",
    "build_p1_element_data",
    "build_p2_element_data",
    "build_tetrahedron_p1_data",
    "build_tetrahedron_p2_data",
    "p1_shape_functions",
    "p2_shape_functions",
    "p2_tetrahedron_shape_values",
    "p2_triangle_mass_template",
    "reference_shape_gradients",
    "reference_shape_gradients_p2",
    "robin_boundary_coefficients",
    "triangle_quadrature",
]
