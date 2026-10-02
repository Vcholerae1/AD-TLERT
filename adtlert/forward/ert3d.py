"""Three-dimensional DC ERT forward operator on tetrahedral P1/P2 meshes (cuDSS sparse solves)."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import scipy.sparse as sp
import torch

from adtlert.fem.tetrahedron import (
    build_tetrahedron_p1_data,
    build_tetrahedron_p2_data,
    p2_tetrahedron_shape_values,
    p2_triangle_mass_template,
)
from adtlert.mesh import Mesh3D
from adtlert.survey import Survey
from adtlert.utils.dtypes import FLOAT_DTYPE


_CUDSS_LOGGER = logging.getLogger("adtlert.cudss")
_CUDSS_LOGGER.setLevel(logging.ERROR)


@dataclass(frozen=True)
class ForwardResponse3D:
    """Result of a true 3D DC ERT solve."""

    apparent_resistivity: torch.Tensor
    resistance: torch.Tensor
    electrode_potentials: torch.Tensor
    node_potentials: torch.Tensor


def _interpolation_matrix(
    mesh: Mesh3D,
    points: np.ndarray,
    *,
    element_order: int,
    cell_connectivity: np.ndarray,
    dof_count: int,
) -> sp.csr_matrix:
    cell_ids, weights = mesh.locate_points(points)
    cell_ids_np = np.asarray(cell_ids, dtype=np.int32)
    weights_np = np.asarray(weights, dtype=float)
    if element_order == 2:
        weights_np = p2_tetrahedron_shape_values(weights_np)
    cell_nodes = cell_connectivity[cell_ids_np]
    width = int(cell_nodes.shape[1])
    rows = np.repeat(np.arange(points.shape[0], dtype=np.int32), width)
    return sp.coo_matrix(
        (weights_np.reshape(-1), (rows, cell_nodes.reshape(-1))),
        shape=(points.shape[0], dof_count),
    ).tocsr()


def _local_face_indices(cell: np.ndarray, face: np.ndarray) -> np.ndarray:
    lookup = {int(node_id): local_id for local_id, node_id in enumerate(cell)}
    return np.asarray([lookup[int(node_id)] for node_id in face], dtype=np.int32)


@dataclass
class ERTForward3D:
    """Tetrahedral P1 forward operator following pyGIMLi's 3D DC/SR formulation.

    The singularity-removal path constructs a discrete point-source right-hand
    side by applying the unit-conductivity operator to the analytic half-space
    primary potential. This is the same primary/secondary identity used by
    pyGIMLi's ``DCSRMultiElectrodeModelling`` for its three-dimensional case.
    """

    mesh: Mesh3D
    survey: Survey
    boundary_mode: str = "mixed"
    singularity_removal: bool = True
    element_order: int = 1
    geometric_factor_mode: str = "auto"

    @classmethod
    def from_mesh_survey(
        cls,
        mesh: Mesh3D,
        survey: Survey,
        *,
        boundary_mode: str = "mixed",
        singularity_removal: bool = True,
        element_order: int = 1,
        geometric_factor_mode: str = "auto",
        **_: object,
    ) -> "ERTForward3D":
        return cls(
            mesh=mesh,
            survey=survey,
            boundary_mode=boundary_mode,
            singularity_removal=singularity_removal,
            element_order=element_order,
            geometric_factor_mode=geometric_factor_mode,
        )

    def __post_init__(self) -> None:
        def choice(name: str, options: tuple[str, ...]) -> str:
            value = str(getattr(self, name)).lower()
            if value not in options:
                raise ValueError(f"{name} must be one of: {sorted(options)}")
            return value

        self.boundary_mode = choice("boundary_mode", ("mixed", "dirichlet"))
        self.geometric_factor_mode = choice(
            "geometric_factor_mode", ("auto", "analytic", "numerical")
        )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "ADTLERT requires an NVIDIA GPU with CUDA (cuDSS sparse solves)"
            )
        if self.element_order not in (1, 2):
            raise ValueError("element_order must be 1 or 2")
        if self.survey.dimension != 3:
            raise ValueError(
                "ERTForward3D requires three-coordinate electrode positions"
            )
        self._cudss_state: dict[str, object] = {}
        self._cached_geometric_factors: torch.Tensor | None = None
        self._reset_solution_cache()
        electrodes = np.asarray(self.survey.electrode_positions, dtype=float)
        quads = np.asarray(self.survey.measurements, dtype=np.int32)
        self._current_electrode_ids = np.unique(quads[:, :2]).astype(np.int32)
        self._receiver_electrode_ids = np.unique(quads[:, 2:]).astype(np.int32)
        self._rhs_capacity = max(
            len(self._current_electrode_ids), len(self._receiver_electrode_ids)
        )
        self._current_electrode_map = np.full(
            self.survey.electrode_count, -1, dtype=np.int32
        )
        self._receiver_electrode_map = np.full(
            self.survey.electrode_count, -1, dtype=np.int32
        )
        self._current_electrode_map[self._current_electrode_ids] = np.arange(
            len(self._current_electrode_ids)
        )
        self._receiver_electrode_map[self._receiver_electrode_ids] = np.arange(
            len(self._receiver_electrode_ids)
        )
        if self.element_order == 2:
            element_data = build_tetrahedron_p2_data(self.mesh)
            self._dof_nodes = np.asarray(element_data.dof_nodes, dtype=float)
            self._cell_connectivity = np.asarray(
                element_data.cell_connectivity, dtype=np.int32
            )
            self._boundary_connectivity = np.asarray(
                element_data.boundary_connectivity, dtype=np.int32
            )
        else:
            element_data = build_tetrahedron_p1_data(self.mesh)
            self._dof_nodes = np.asarray(self.mesh.nodes, dtype=float)
            self._cell_connectivity = np.asarray(self.mesh.cells, dtype=np.int32)
            self._boundary_connectivity = np.asarray(
                self.mesh.boundary_faces, dtype=np.int32
            )
        self._electrode_matrix = _interpolation_matrix(
            self.mesh,
            electrodes,
            element_order=self.element_order,
            cell_connectivity=self._cell_connectivity,
            dof_count=self._dof_nodes.shape[0],
        )
        self._volume_templates = np.asarray(
            element_data.stiffness_templates, dtype=np.float64
        )
        self._cell_templates = self._volume_templates.copy()
        if self.boundary_mode == "mixed":
            self._add_mixed_boundary_templates()
        self._build_assembly_routing()

        self._unit_operator = self._assemble_operator(
            np.ones(self.mesh.cell_count, dtype=float)
        )
        if self.boundary_mode == "dirichlet":
            self._unit_operator = self._apply_dirichlet_matrix(self._unit_operator)
        self._source_rhs = self._build_source_rhs()

    @property
    def cell_count(self) -> int:
        return self.mesh.cell_count

    def _add_mixed_boundary_templates(self) -> None:
        """Add first-order far-field Robin terms to per-cell templates."""

        face_cells = np.asarray(self.mesh.boundary_face_cells, dtype=np.int32)
        centers = np.asarray(self.mesh.boundary_face_centers, dtype=float)
        areas = np.asarray(self.mesh.boundary_face_areas, dtype=float)
        normals = np.asarray(self.mesh.boundary_face_normals, dtype=float)
        surface_mask = np.asarray(self.mesh.surface_face_mask, dtype=bool)
        source_center = np.mean(
            np.asarray(self.survey.electrode_positions, dtype=float), axis=0
        )

        triangle_mass = (
            np.asarray(((2.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 1.0, 2.0))) / 12.0
        )
        for face_id in np.flatnonzero(~surface_mask):
            radial = centers[face_id] - source_center
            radius_sq = float(np.dot(radial, radial))
            if radius_sq <= 0.0:
                continue
            alpha = max(float(np.dot(radial, normals[face_id])) / radius_sq, 0.0)
            if alpha == 0.0:
                continue
            cell_id = int(face_cells[face_id])
            local = _local_face_indices(
                self._cell_connectivity[cell_id], self._boundary_connectivity[face_id]
            )
            face_mass = (
                p2_triangle_mass_template(areas[face_id])
                if self.element_order == 2
                else areas[face_id] * triangle_mass
            )
            self._cell_templates[cell_id][np.ix_(local, local)] += alpha * face_mass

    def _assemble_operator(self, conductivity: np.ndarray) -> sp.csr_matrix:
        values = np.asarray(conductivity, dtype=float).reshape(-1)
        if values.shape != (self.mesh.cell_count,):
            raise ValueError(f"conductivity must have shape ({self.mesh.cell_count},)")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("conductivity must contain positive finite values")
        contributions = (values[:, None] * self._flat_templates).reshape(-1)
        data = np.bincount(
            self._assembly_inverse,
            weights=contributions,
            minlength=self._csr_indices.size,
        )
        return sp.csr_matrix(
            (data, self._csr_indices, self._csr_indptr),
            shape=(self._dof_nodes.shape[0], self._dof_nodes.shape[0]),
        )

    def _build_assembly_routing(self) -> None:
        """Precompute local-entry to canonical CSR routing once per geometry."""

        cells = self._cell_connectivity
        width = int(cells.shape[1])
        rows = np.repeat(cells, width, axis=1).reshape(-1)
        cols = np.tile(cells, (1, width)).reshape(-1)
        dof_count = int(self._dof_nodes.shape[0])
        keys = rows.astype(np.int64) * dof_count + cols.astype(np.int64)
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        unique_rows = (unique_keys // dof_count).astype(np.int32)
        self._flat_templates = self._cell_templates.reshape(self.mesh.cell_count, -1)
        self._assembly_inverse = inverse.astype(np.int32)
        self._csr_indices = (unique_keys % dof_count).astype(np.int32)
        self._csr_indptr = np.concatenate(
            (
                (0,),
                np.cumsum(
                    np.bincount(unique_rows, minlength=dof_count), dtype=np.int64
                ),
            )
        ).astype(np.int32)

    def _dirichlet_node_ids(self) -> np.ndarray:
        faces = self._boundary_connectivity
        far_faces = faces[~np.asarray(self.mesh.surface_face_mask, dtype=bool)]
        return np.unique(far_faces.reshape(-1))

    def _apply_dirichlet_matrix(self, matrix: sp.csr_matrix) -> sp.csr_matrix:
        fixed = self._dirichlet_node_ids()
        result = matrix.tolil(copy=True)
        result[fixed, :] = 0.0
        result[:, fixed] = 0.0
        result[fixed, fixed] = 1.0
        return result.tocsr()

    def _primary_potential(self, source: np.ndarray) -> np.ndarray:
        nodes = self._dof_nodes
        surface_z = float(self.mesh.surface_reference_level)
        mirror = source.copy()
        mirror[2] = 2.0 * surface_z - source[2]
        direct_distance = np.linalg.norm(nodes - source, axis=1)
        mirror_distance = np.linalg.norm(nodes - mirror, axis=1)

        positive = direct_distance[direct_distance > 1.0e-12]
        # pyGIMLi's 3D nodal electrode singular value uses half the shortest
        # adjacent-node distance (its 2.5D Bessel path uses one sixth).
        fallback_radius = max(float(np.min(positive)) / 2.0, 1.0e-12)
        direct_distance = np.where(
            direct_distance > 1.0e-12, direct_distance, fallback_radius
        )
        mirror_distance = np.where(
            mirror_distance > 1.0e-12, mirror_distance, fallback_radius
        )
        return (1.0 / direct_distance + 1.0 / mirror_distance) / (4.0 * np.pi)

    def _build_source_rhs(self) -> np.ndarray:
        if self.singularity_removal and self.mesh.is_flat_surface:
            primary = np.stack(
                [
                    self._primary_potential(source)
                    for source in np.asarray(
                        self.survey.electrode_positions, dtype=float
                    )[self._current_electrode_ids]
                ],
                axis=0,
            )
            rhs = (self._unit_operator @ primary.T).T
        else:
            rhs = self._electrode_matrix[self._current_electrode_ids].toarray()
        if self.boundary_mode == "dirichlet":
            rhs[:, self._dirichlet_node_ids()] = 0.0
        padded = np.zeros((self._rhs_capacity, rhs.shape[1]), dtype=np.float64)
        padded[: rhs.shape[0]] = rhs
        return padded

    def _operator(self, conductivity: np.ndarray) -> sp.csr_matrix:
        """Assembled operator for ``conductivity``; a new model invalidates the cached fields."""

        values = np.asarray(conductivity, dtype=np.float64).reshape(-1)
        if self._cached_conductivity is None or not np.array_equal(
            values, self._cached_conductivity
        ):
            operator = self._assemble_operator(values)
            self._reset_solution_cache()
            self._cached_conductivity = values.copy()
            self._cached_operator = (
                self._apply_dirichlet_matrix(operator)
                if self.boundary_mode == "dirichlet"
                else operator
            )
        return self._cached_operator

    def _solve_cudss_rhs(
        self,
        operator: sp.csr_matrix,
        rhs: np.ndarray,
        *,
        factorize: bool,
        rhs_key: str,
    ) -> np.ndarray:
        """Solve multiple RHS columns with NVIDIA cuDSS through nvmath-python."""

        try:
            import cupy as cp
            import cupyx.scipy.sparse as cupy_sparse
            from nvmath.sparse.advanced import (
                DirectSolver,
                DirectSolverMatrixType,
                DirectSolverOptions,
            )
        except ImportError as exc:
            raise ImportError(
                "ERTForward3D requires CuPy and nvmath-python for the cuDSS backend"
            ) from exc

        canonical = operator
        matrix_data = self._cudss_state.get("matrix_data")
        if matrix_data is None:
            indptr = cp.asarray(canonical.indptr, dtype=cp.int32)
            indices = cp.asarray(canonical.indices, dtype=cp.int32)
            matrix_data = cp.asarray(canonical.data)
            matrix_gpu = cupy_sparse.csr_matrix(
                (matrix_data, indices, indptr), shape=canonical.shape
            )
            self._cudss_state.update(
                {
                    "matrix_data": matrix_data,
                    "matrix_gpu": matrix_gpu,
                    "indptr": indptr,
                    "indices": indices,
                }
            )
        else:
            if int(matrix_data.size) != int(canonical.data.size):
                raise RuntimeError("cuDSS operator sparsity changed after planning")
            matrix_data[...] = cp.asarray(canonical.data)
            matrix_gpu = self._cudss_state["matrix_gpu"]

        gpu_rhs_key = f"rhs_{rhs_key}"
        rhs_gpu = self._cudss_state.get(gpu_rhs_key)
        if rhs_gpu is None:
            rhs_gpu = cp.asfortranarray(
                cp.asarray(np.asarray(rhs, dtype=canonical.dtype).T)
            )
            self._cudss_state[gpu_rhs_key] = rhs_gpu
        solver = self._cudss_state.get("solver")
        if solver is None:
            options = DirectSolverOptions(
                sparse_system_type=DirectSolverMatrixType.SPD,
                logger=_CUDSS_LOGGER,
                blocking=True,
            )
            solver = DirectSolver(matrix_gpu, rhs_gpu, options=options)
            solver.plan()
            self._cudss_state["solver"] = solver
            factorize = True
        else:
            solver.reset_operands(b=rhs_gpu)
        if factorize:
            solver.factorize()
        solution = solver.solve()
        return np.asarray(cp.asnumpy(solution.T), dtype=np.float64)

    def _solve_node_fields(self, conductivity: torch.Tensor | float) -> np.ndarray:
        values = np.asarray(conductivity, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(self.mesh.cell_count, float(values), dtype=np.float64)
        values = values.reshape(-1)
        operator = self._operator(values)
        if self._cached_node_fields is None:
            self._cached_node_fields = self._solve_cudss_rhs(
                operator, self._source_rhs, factorize=True, rhs_key="sources"
            )
        return self._cached_node_fields[: len(self._current_electrode_ids)]

    def _measurement_values(
        self, node_fields: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        electrode_potentials = np.asarray(
            node_fields @ self._electrode_matrix.T, dtype=np.float64
        )
        quads = np.asarray(self.survey.measurements, dtype=np.int32)
        a, b, m, n = quads.T
        a = self._current_electrode_map[a]
        b = self._current_electrode_map[b]
        resistance = (
            electrode_potentials[a, m]
            - electrode_potentials[b, m]
            - electrode_potentials[a, n]
            + electrode_potentials[b, n]
        )
        return electrode_potentials, resistance

    def _expanded_fields(
        self, node_fields: np.ndarray, electrode_potentials: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        full_node_fields = np.full(
            (self.survey.electrode_count, node_fields.shape[1]),
            np.nan,
            dtype=np.float64,
        )
        full_electrode_potentials = np.full(
            (self.survey.electrode_count, self.survey.electrode_count),
            np.nan,
            dtype=np.float64,
        )
        full_node_fields[self._current_electrode_ids] = node_fields
        full_electrode_potentials[self._current_electrode_ids] = electrode_potentials
        return full_node_fields, full_electrode_potentials

    def _geometric_factors(self) -> torch.Tensor:
        use_numerical = self.geometric_factor_mode == "numerical" or (
            self.geometric_factor_mode == "auto" and not self.mesh.is_flat_surface
        )
        if not use_numerical:
            return self.survey.geometric_factors()
        if self._cached_geometric_factors is None:
            unit_fields = self._solve_node_fields(
                np.ones(self.mesh.cell_count, dtype=float)
            )
            _, unit_resistance = self._measurement_values(unit_fields)
            if np.any(np.abs(unit_resistance) <= np.finfo(float).tiny):
                raise ValueError(
                    "numerical geometric-factor solve produced zero resistance"
                )
            self._cached_geometric_factors = torch.as_tensor(
                1.0 / unit_resistance, dtype=FLOAT_DTYPE
            )
        return self._cached_geometric_factors

    def _response_from_fields(
        self,
        node_fields: np.ndarray,
        currents: torch.Tensor | float,
        *,
        include_fields: bool,
    ) -> ForwardResponse3D:
        electrode_potentials, resistance = self._measurement_values(node_fields)
        if include_fields:
            response_node_fields, response_electrode_potentials = self._expanded_fields(
                node_fields,
                electrode_potentials,
            )
        else:
            response_node_fields = np.empty((0, 0), dtype=np.float64)
            response_electrode_potentials = np.empty((0, 0), dtype=np.float64)
        current_array = np.asarray(currents, dtype=float)
        geometric_factors = np.abs(np.asarray(self._geometric_factors(), dtype=float))
        apparent = geometric_factors * resistance / current_array
        return ForwardResponse3D(
            apparent_resistivity=torch.as_tensor(apparent, dtype=FLOAT_DTYPE),
            resistance=torch.as_tensor(resistance, dtype=FLOAT_DTYPE),
            electrode_potentials=torch.as_tensor(
                response_electrode_potentials, dtype=FLOAT_DTYPE
            ),
            node_potentials=torch.as_tensor(response_node_fields, dtype=FLOAT_DTYPE),
        )

    def solve(
        self, conductivity: torch.Tensor | float, currents: torch.Tensor | float = 1.0
    ) -> ForwardResponse3D:
        return self._response_from_fields(
            self._solve_node_fields(conductivity), currents, include_fields=True
        )

    def resistance(self, conductivity: torch.Tensor | float) -> torch.Tensor:
        _, resistance = self._measurement_values(self._solve_node_fields(conductivity))
        return torch.as_tensor(resistance, dtype=FLOAT_DTYPE)

    def apparent_resistivity_values(
        self, conductivity: torch.Tensor | float, currents: torch.Tensor | float = 1.0
    ) -> torch.Tensor:
        resistance = np.asarray(self.resistance(conductivity), dtype=float)
        factors = np.abs(np.asarray(self._geometric_factors(), dtype=float))
        return torch.as_tensor(
            factors * resistance / np.asarray(currents, dtype=float), dtype=FLOAT_DTYPE
        )

    def apparent_resistivity_series(
        self, conductivities: torch.Tensor, currents: torch.Tensor | float = 1.0
    ) -> torch.Tensor:
        values = np.asarray(conductivities, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.mesh.cell_count:
            raise ValueError(
                f"conductivities must have shape (n_steps, {self.mesh.cell_count})"
            )
        currents_np = np.asarray(currents, dtype=float)
        rows = []
        for index, model in enumerate(values):
            step_currents = currents_np[index] if currents_np.ndim == 2 else currents_np
            rows.append(self.apparent_resistivity_values(model, currents=step_currents))
        return torch.stack(rows)

    def jacobian(
        self,
        conductivity: torch.Tensor | float,
        *,
        batch_size: int | None = None,
        cell_batch_size: int | None = None,
        include_robin_boundary_derivative: bool = False,
        normal_sensitivity: bool = True,
    ) -> torch.Tensor:
        del normal_sensitivity
        fields = self._solve_node_fields(conductivity)
        if self._cached_receiver_fields is None:
            point_source_rhs = (
                not self.singularity_removal or not self.mesh.is_flat_surface
            )
            if point_source_rhs and np.array_equal(
                self._current_electrode_ids, self._receiver_electrode_ids
            ):
                self._cached_receiver_fields = fields
            else:
                receiver_values = self._electrode_matrix[
                    self._receiver_electrode_ids
                ].toarray()
                receiver_rhs = np.zeros(
                    (self._rhs_capacity, receiver_values.shape[1]), dtype=np.float64
                )
                receiver_rhs[: receiver_values.shape[0]] = receiver_values
                if self.boundary_mode == "dirichlet":
                    receiver_rhs[:, self._dirichlet_node_ids()] = 0.0
                receiver_fields = self._solve_cudss_rhs(
                    self._cached_operator,
                    receiver_rhs,
                    factorize=False,
                    rhs_key="receivers",
                )
                self._cached_receiver_fields = receiver_fields[
                    : len(self._receiver_electrode_ids)
                ]
        receiver_basis_fields = self._cached_receiver_fields
        quads = np.asarray(self.survey.measurements, dtype=np.int32)
        cells = self._cell_connectivity
        templates = (
            self._cell_templates
            if include_robin_boundary_derivative
            else self._volume_templates
        )
        if batch_size is None:
            batch_size = min(max(self.survey.measurement_count, 1), 64)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        result = np.empty(
            (self.survey.measurement_count, self.mesh.cell_count), dtype=np.float64
        )
        width = int(cells.shape[1])
        for start in range(0, self.survey.measurement_count, batch_size):
            chunk = quads[start : start + batch_size]
            a, b, m, n = chunk.T
            current_fields = (
                fields[self._current_electrode_map[a]]
                - fields[self._current_electrode_map[b]]
            )
            receiver_fields = (
                receiver_basis_fields[self._receiver_electrode_map[m]]
                - receiver_basis_fields[self._receiver_electrode_map[n]]
            )
            chunk_size = int(chunk.shape[0])
            resolved_cell_batch_size = cell_batch_size
            if resolved_cell_batch_size is None:
                # Each local field block contains batch*cells*local_dofs
                # float64 values. Cap each of the two gathered blocks near 32 MB.
                resolved_cell_batch_size = max(
                    1, min(self.mesh.cell_count, 4_000_000 // (chunk_size * width))
                )
            if resolved_cell_batch_size < 1:
                raise ValueError("cell_batch_size must be >= 1")
            for cell_start in range(0, self.mesh.cell_count, resolved_cell_batch_size):
                cell_stop = min(
                    cell_start + resolved_cell_batch_size, self.mesh.cell_count
                )
                cell_nodes = cells[cell_start:cell_stop]
                current_local = current_fields[:, cell_nodes]
                receiver_local = receiver_fields[:, cell_nodes]
                result[start : start + chunk_size, cell_start:cell_stop] = -np.einsum(
                    "bci,cij,bcj->bc",
                    current_local,
                    templates[cell_start:cell_stop],
                    receiver_local,
                )
        return torch.as_tensor(result, dtype=FLOAT_DTYPE)

    def solve_with_jacobian(
        self,
        conductivity: torch.Tensor | float,
        currents: torch.Tensor | float = 1.0,
        *,
        batch_size: int | None = None,
        cell_batch_size: int | None = None,
        include_robin_boundary_derivative: bool = False,
        normal_sensitivity: bool = True,
        include_fields: bool = True,
        **_: object,
    ) -> tuple[ForwardResponse3D, torch.Tensor]:
        fields = self._solve_node_fields(conductivity)
        response = self._response_from_fields(
            fields, currents, include_fields=include_fields
        )
        jacobian = self.jacobian(
            conductivity,
            batch_size=batch_size,
            cell_batch_size=cell_batch_size,
            include_robin_boundary_derivative=include_robin_boundary_derivative,
            normal_sensitivity=normal_sensitivity,
        )
        return response, jacobian

    def prepare(
        self,
        conductivity: torch.Tensor | None = None,
        *,
        include_solver_state: bool = True,
    ) -> None:
        if conductivity is not None and include_solver_state:
            self._operator(np.asarray(conductivity, dtype=float))

    def _reset_solution_cache(self) -> None:
        self._cached_conductivity = self._cached_operator = None
        self._cached_node_fields = self._cached_receiver_fields = None

    def close(self) -> None:
        solver = self._cudss_state.get("solver")
        if solver is not None:
            try:
                solver.free()
            except Exception:  # best-effort release during teardown
                pass
        self._cudss_state.clear()
        self._reset_solution_cache()
