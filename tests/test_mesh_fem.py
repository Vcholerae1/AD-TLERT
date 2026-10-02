import numpy as np
import pytest
import torch

from adtlert.fem import (
    assemble_local_boundary_mass,
    assemble_local_mass,
    assemble_local_stiffness,
    build_p1_element_data,
    build_p2_element_data,
    build_tetrahedron_p1_data,
    robin_boundary_coefficients,
)
from adtlert.fem.triangle import assemble_local_mass_p2, assemble_local_stiffness_p2
from adtlert.mesh import Mesh, Mesh3D


def grid_triangles(nx=6, ny=4, slope=0.0):
    xs, ys = np.meshgrid(
        np.arange(nx + 1, dtype=float), -np.arange(ny + 1, dtype=float)
    )
    nodes = np.column_stack((xs.ravel(), ys.ravel() + slope * xs.ravel()))
    index = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
    cells = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = (
                index[j, i],
                index[j, i + 1],
                index[j + 1, i + 1],
                index[j + 1, i],
            )
            cells += [[a, d, b], [b, d, c]]
    return nodes, np.asarray(cells)


def test_triangle_mesh_topology():
    nodes, cells = grid_triangles()
    mesh = Mesh.from_arrays(nodes, cells)
    assert mesh.is_triangle_mesh and mesh.cell_count == 48
    assert mesh.boundary_edges.shape[0] == 2 * (6 + 4)
    assert float(mesh.cell_areas.sum()) == pytest.approx(24.0)
    assert mesh.is_flat_surface
    # the inferred surface is the top row
    assert np.allclose(np.asarray(mesh.surface_nodes)[:, 1], 0.0)
    # outward normals point away from the owning cell
    centers = np.asarray(mesh.nodes)[np.asarray(mesh.cells)].mean(axis=1)[
        np.asarray(mesh.boundary_edge_cells)
    ]
    outward = np.sum(
        np.asarray(mesh.boundary_edge_normals)
        * (np.asarray(mesh.boundary_edge_centers) - centers),
        axis=1,
    )
    assert np.all(outward > 0)


def test_sloped_surface_is_not_flat():
    assert not Mesh.from_arrays(*grid_triangles(slope=0.2)).is_flat_surface


def test_locate_points_reproduces_coordinates():
    nodes, cells = grid_triangles()
    mesh = Mesh.from_arrays(nodes, cells)
    points = np.array([[0.3, -0.2], [2.5, -1.7], [5.9, -3.9], [3.0, -2.0]])
    cell_ids, weights = mesh.locate_points(points)
    corners = np.asarray(mesh.nodes)[np.asarray(mesh.cells)[np.asarray(cell_ids)]]
    assert np.allclose(np.einsum("pk,pkd->pd", np.asarray(weights), corners), points)
    assert np.allclose(np.asarray(weights).sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        mesh.locate_points(np.array([[50.0, 50.0]]))


def test_uniform_refinement_preserves_area_and_parents():
    mesh = Mesh.from_arrays(*grid_triangles())
    refined, parents = mesh.refine_uniform()
    assert refined.cell_count == 4 * mesh.cell_count
    assert float(refined.cell_areas.sum()) == pytest.approx(
        float(mesh.cell_areas.sum())
    )
    parent_area = np.bincount(
        np.asarray(parents),
        weights=np.asarray(refined.cell_areas),
        minlength=mesh.cell_count,
    )
    assert np.allclose(parent_area, np.asarray(mesh.cell_areas))


def test_quadratic_topology_counts():
    mesh = Mesh.from_arrays(*grid_triangles())
    nodes, cells, boundary = mesh.build_quadratic_topology()
    edge_count = (3 * mesh.cell_count + mesh.boundary_edges.shape[0]) // 2
    assert nodes.shape[0] == mesh.node_count + edge_count
    assert cells.shape == (mesh.cell_count, 6) and boundary.shape == (
        mesh.boundary_edges.shape[0],
        3,
    )


def test_quadrilateral_mesh_points_and_areas():
    xs, ys = np.meshgrid(np.arange(5.0), -np.arange(4.0))
    nodes = np.column_stack((xs.ravel(), ys.ravel()))
    index = np.arange(20).reshape(4, 5)
    cells = [
        [index[j, i], index[j, i + 1], index[j + 1, i + 1], index[j + 1, i]]
        for j in range(3)
        for i in range(4)
    ]
    mesh = Mesh.from_arrays(nodes, cells)
    assert mesh.is_quadrilateral_mesh and float(mesh.cell_areas.sum()) == pytest.approx(
        12.0
    )
    cell_ids, weights = mesh.locate_points(np.array([[1.25, -0.5]]))
    assert np.allclose(np.asarray(weights).sum(), 1.0)


@pytest.mark.parametrize("quadratic", [False, True])
def test_local_templates_have_zero_row_sums_and_total_mass(quadratic):
    mesh = Mesh.from_arrays(*grid_triangles())
    if quadratic:
        data = build_p2_element_data(mesh)
        stiffness, mass = (
            assemble_local_stiffness_p2(data, 1.0),
            assemble_local_mass_p2(data, 1.0),
        )
    else:
        data = build_p1_element_data(mesh)
        stiffness, mass = (
            assemble_local_stiffness(data, 1.0),
            assemble_local_mass(data, 1.0),
        )
    assert torch.allclose(
        stiffness.sum(dim=2), torch.zeros(()), atol=1e-5
    )  # constants have no gradient
    assert float(mass.sum()) == pytest.approx(float(mesh.cell_areas.sum()), rel=1e-5)
    assert torch.allclose(stiffness, stiffness.transpose(1, 2), atol=1e-6)


def test_boundary_templates_integrate_edge_length():
    mesh = Mesh.from_arrays(*grid_triangles())
    local = assemble_local_boundary_mass(mesh, 1.0)
    assert float(local.sum()) == pytest.approx(
        float(mesh.boundary_edge_lengths.sum()), rel=1e-6
    )


def test_robin_coefficients_vanish_on_the_surface():
    mesh = Mesh.from_arrays(*grid_triangles())
    coefficients = robin_boundary_coefficients(mesh, 1.0, np.array([3.0, 0.0]), 0.2)
    assert np.all(np.asarray(coefficients)[np.asarray(mesh.surface_edge_mask)] == 0.0)
    assert np.any(np.asarray(coefficients) != 0.0)


def test_tetrahedron_p1_data_unit_cube():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    mesh = Mesh3D.from_arrays(nodes, [[0, 1, 2, 3]])
    data = build_tetrahedron_p1_data(mesh)
    assert float(data.cell_volumes[0]) == pytest.approx(1.0 / 6.0)
    assert torch.allclose(
        data.stiffness_templates.sum(dim=2), torch.zeros(()), atol=1e-6
    )
