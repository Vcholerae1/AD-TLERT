"""Finite-element basis and quadrature utilities."""

from adtlert.fem.assembly import (
    COORouting,
    assemble_global_bcoo,
    assemble_helmholtz_operator,
    assemble_local_mass,
    assemble_local_mass_p2,
    assemble_local_stiffness,
    assemble_local_stiffness_p2,
    build_coo_routing,
    build_coo_routing_from_connectivity,
)
from adtlert.fem.boundary import (
    BoundaryRouting,
    assemble_boundary_bcoo,
    assemble_local_boundary_mass,
    assemble_local_boundary_mass_p2,
    build_boundary_routing,
    build_boundary_routing_from_connectivity,
    robin_boundary_coefficients,
)
from adtlert.fem.p1 import (
    P1ElementData,
    build_p1_element_data,
    p1_shape_functions,
    reference_shape_gradients,
    triangle_quadrature,
)
from adtlert.fem.p2 import P2ElementData, build_p2_element_data, p2_shape_functions, reference_shape_gradients_p2
from adtlert.fem.tet_p1 import (
    TetrahedronP1Data,
    TetrahedronP2Data,
    build_tetrahedron_p1_data,
    build_tetrahedron_p2_data,
    p2_tetrahedron_shape_values,
    p2_triangle_mass_template,
)

__all__ = [
    "BoundaryRouting",
    "COORouting",
    "assemble_boundary_bcoo",
    "P1ElementData",
    "assemble_global_bcoo",
    "assemble_helmholtz_operator",
    "assemble_local_boundary_mass",
    "assemble_local_boundary_mass_p2",
    "assemble_local_mass",
    "assemble_local_mass_p2",
    "assemble_local_stiffness",
    "assemble_local_stiffness_p2",
    "build_boundary_routing",
    "build_boundary_routing_from_connectivity",
    "build_p1_element_data",
    "build_p2_element_data",
    "build_coo_routing",
    "build_coo_routing_from_connectivity",
    "p1_shape_functions",
    "p2_shape_functions",
    "reference_shape_gradients",
    "reference_shape_gradients_p2",
    "robin_boundary_coefficients",
    "triangle_quadrature",
    "P2ElementData",
    "TetrahedronP1Data",
    "TetrahedronP2Data",
    "build_tetrahedron_p1_data",
    "build_tetrahedron_p2_data",
    "p2_tetrahedron_shape_values",
    "p2_triangle_mass_template",
]
