"""Three-dimensional DC ERT forward operator on tetrahedral P1/P2 meshes (cuDSS sparse solves)."""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch

from adtlert.fem.tetrahedron import (
    build_tetrahedron_p1_data,
    build_tetrahedron_p2_data,
    p2_tetrahedron_shape_values,
    p2_triangle_mass_template,
)
from adtlert.forward import kernels
from adtlert.forward.cudss import CUDA as _CUDA
from adtlert.forward.cudss import BatchedSolver
from adtlert.forward.discretization import SparsePattern
from adtlert.forward.field_cache import (
    FieldCache,
    factorization_name,
    newest_batches_first,
    step_batches,
)
from adtlert.forward.inputs import (
    cell_vector,
    check_finite,
    conductivity_model,
    conductivity_series,
    cotangent_series,
    measurement_vector,
    option,
    parameter_targets,
)
from adtlert.forward.kernels import CellAggregation, GroupSum
from adtlert.mesh import Mesh3D
from adtlert.survey import Survey
from adtlert.utils.dtypes import FLOAT_DTYPE

Tensor = torch.Tensor

PYGIMLI_ELECTRODE_RADIUS_FRACTION = 0.5
"""pyGIMLi evaluates the primary potential at an electrode node at this fraction of the
shortest distance to another node (one sixth in its 2.5D Bessel path)."""


@dataclass
class _Fields3D:
    """Assembled operator values, current-electrode fields and, once solved, receiver fields."""

    values: Tensor
    current: Tensor
    receiver: Tensor | None = None


@dataclass(frozen=True)
class ForwardResponse3D:
    """Result of a true 3D DC ERT solve; the potentials are ``None`` unless requested."""

    apparent_resistivity: Tensor
    resistance: Tensor
    electrode_potentials: Tensor | None = None
    node_potentials: Tensor | None = None


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
    """Tetrahedral P1/P2 forward operator following pyGIMLi's 3D DC/SR formulation.

    The singularity-removal path constructs a discrete point-source right-hand
    side by applying the unit-conductivity operator to the analytic half-space
    primary potential, the primary/secondary identity of pyGIMLi's
    ``DCSRMultiElectrodeModelling``. ``series_batch_steps`` models are factorized
    in one cuDSS batch; 3D factors are large, so the default is one.
    """

    mesh: Mesh3D
    survey: Survey
    boundary_mode: str = "mixed"
    singularity_removal: bool = True
    element_order: int = 1
    geometric_factor_mode: str = "auto"
    normal_field_cache_max_entries: int = 8
    series_batch_steps: int = 1

    @classmethod
    def from_mesh_survey(
        cls, mesh: Mesh3D, survey: Survey, **options: Any
    ) -> ERTForward3D:
        return cls(mesh=mesh, survey=survey, **options)

    def __post_init__(self) -> None:
        self.boundary_mode = option(
            self.boundary_mode, ("mixed", "dirichlet"), "boundary_mode"
        )
        self.geometric_factor_mode = option(
            self.geometric_factor_mode,
            ("auto", "analytic", "numerical"),
            "geometric_factor_mode",
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
        if int(self.series_batch_steps) < 1:
            raise ValueError("series_batch_steps must be >= 1")
        self.series_batch_steps = int(self.series_batch_steps)
        self._field_cache: FieldCache[_Fields3D] = FieldCache(
            self.normal_field_cache_max_entries
        )
        self.normal_field_cache_max_entries = self._field_cache.max_entries
        self._solver = BatchedSolver()
        self._cache: dict = {}
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

        self._unit_operator = self._build_unit_operator()
        self._source_rhs = self._build_source_rhs()
        self._receiver_rhs = self._build_receiver_rhs()
        self._shared_source_receiver_rhs = bool(
            (not self.singularity_removal or not self.mesh.is_flat_surface)
            and np.array_equal(
                self._current_electrode_ids, self._receiver_electrode_ids
            )
        )

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

    def _cached(self, key, compute):
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    def _on(self, name: str, tensor_fn, device: torch.device) -> Tensor:
        return self._cached((name, str(device)), lambda: tensor_fn().to(device))

    def _mapped_abmn(
        self, device: torch.device | str = "cpu"
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        def build():
            quads = np.asarray(self.survey.measurements, dtype=np.int32)
            a, b, m, n = quads.T
            mapped = np.stack(
                (
                    self._current_electrode_map[a],
                    self._current_electrode_map[b],
                    self._receiver_electrode_map[m],
                    self._receiver_electrode_map[n],
                ),
                axis=1,
            )
            return torch.as_tensor(mapped, dtype=torch.long)

        quads = self._on("mapped_abmn", build, torch.device(device))
        return quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]

    def _dirichlet_node_ids(self) -> np.ndarray:
        faces = self._boundary_connectivity
        far_faces = faces[~np.asarray(self.mesh.surface_face_mask, dtype=bool)]
        return np.unique(far_faces.reshape(-1))

    def _build_assembly_routing(self) -> None:
        """Precompute local-entry to canonical CSR routing once per geometry."""

        cells = self._cell_connectivity
        width = int(cells.shape[1])
        rows = np.repeat(cells, width, axis=1).reshape(-1)
        cols = np.tile(cells, (1, width)).reshape(-1)
        dof_count = int(self._dof_nodes.shape[0])
        keys = rows.astype(np.int64) * dof_count + cols.astype(np.int64)
        if self.boundary_mode == "dirichlet":
            fixed = self._dirichlet_node_ids()
            fixed_mask = np.zeros(dof_count, dtype=bool)
            fixed_mask[fixed] = True
            volume_pair = ~fixed_mask[rows] & ~fixed_mask[cols]
            active_pair = volume_pair | (rows == cols)
            unique_keys = np.unique(keys[active_pair])
            slots = np.searchsorted(unique_keys, keys)
            assembly_groups = np.where(volume_pair, slots, -1).astype(np.int64)
            sddmm_entries = np.where(volume_pair, slots, 0).astype(np.int64)
            fixed_diag_slots = np.searchsorted(
                unique_keys,
                fixed.astype(np.int64) * dof_count + fixed.astype(np.int64),
            ).astype(np.int64)
            pair_mask = volume_pair.reshape(self.mesh.cell_count, -1)
            self._flat_templates = np.where(
                pair_mask, self._cell_templates.reshape(self.mesh.cell_count, -1), 0.0
            )
            self._flat_volume_templates = np.where(
                pair_mask, self._volume_templates.reshape(self.mesh.cell_count, -1), 0.0
            )
        else:
            unique_keys, inverse = np.unique(keys, return_inverse=True)
            assembly_groups = inverse.astype(np.int64)
            sddmm_entries = inverse.astype(np.int64)
            fixed_diag_slots = np.empty(0, dtype=np.int64)
            self._flat_templates = self._cell_templates.reshape(
                self.mesh.cell_count, -1
            )
            self._flat_volume_templates = self._volume_templates.reshape(
                self.mesh.cell_count, -1
            )

        unique_rows = (unique_keys // dof_count).astype(np.int32)
        self._csr_indices = (unique_keys % dof_count).astype(np.int32)
        self._csr_indptr = np.concatenate(
            (
                (0,),
                np.cumsum(
                    np.bincount(unique_rows, minlength=dof_count), dtype=np.int64
                ),
            )
        ).astype(np.int32)
        self._assembly_groups = assembly_groups
        self._sddmm_entries = sddmm_entries.reshape(self.mesh.cell_count, width * width)
        self._fixed_diag_slots = fixed_diag_slots
        self._pattern = SparsePattern(
            shape=(dof_count, dof_count),
            volume_inverse=torch.as_tensor(assembly_groups, dtype=torch.long),
            boundary_inverse=torch.empty(0, dtype=torch.long),
            indptr=self._csr_indptr,
            indices=self._csr_indices,
        )

    def _build_unit_operator(self) -> sp.csr_matrix:
        valid = self._assembly_groups >= 0
        data = np.bincount(
            self._assembly_groups[valid],
            weights=self._flat_templates.reshape(-1)[valid],
            minlength=self._pattern.nnz,
        )
        if self._fixed_diag_slots.size:
            data[self._fixed_diag_slots] = 1.0
        return sp.csr_matrix(
            (data, self._pattern.indices, self._pattern.indptr),
            shape=self._pattern.shape,
        )

    def _primary_potential(self, source: np.ndarray) -> np.ndarray:
        nodes = self._dof_nodes
        surface_z = float(self.mesh.surface_reference_level)
        mirror = source.copy()
        mirror[2] = 2.0 * surface_z - source[2]
        direct_distance = np.linalg.norm(nodes - source, axis=1)
        mirror_distance = np.linalg.norm(nodes - mirror, axis=1)

        positive = direct_distance[direct_distance > 1.0e-12]
        fallback_radius = max(
            float(np.min(positive)) * PYGIMLI_ELECTRODE_RADIUS_FRACTION, 1.0e-12
        )
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

    def _build_receiver_rhs(self) -> np.ndarray:
        rhs = self._electrode_matrix[self._receiver_electrode_ids].toarray()
        if self.boundary_mode == "dirichlet":
            rhs[:, self._dirichlet_node_ids()] = 0.0
        padded = np.zeros((self._rhs_capacity, rhs.shape[1]), dtype=np.float64)
        padded[: rhs.shape[0]] = rhs
        return padded

    def _sigma(self, conductivity) -> Tensor:
        return conductivity_model(conductivity, self.mesh.cell_count, torch.float64)

    def _series(self, conductivities) -> Tensor:
        return conductivity_series(conductivities, self.mesh.cell_count, torch.float64)

    def _templates(self, *, robin: bool) -> Tensor:
        """Flattened per-cell operator templates on the GPU, with or without Robin terms."""

        if robin:
            return self._on(
                "templates",
                lambda: torch.as_tensor(self._flat_templates, dtype=torch.float64),
                _CUDA,
            )
        return self._on(
            "volume_templates",
            lambda: torch.as_tensor(self._flat_volume_templates, dtype=torch.float64),
            _CUDA,
        )

    def _assemble_homogeneous(self, sigmas: Tensor, *, robin: bool = True) -> Tensor:
        """Assemble ``sum_c sigma_{b,c} T_c`` on the GPU without fixed Dirichlet diagonals."""

        sigmas = sigmas.to(_CUDA, torch.float64)
        templates = self._templates(robin=robin)
        assemble = self._cached(
            "assembly",
            lambda: GroupSum(self._assembly_groups, self._pattern.nnz, _CUDA),
        )
        return assemble(
            (sigmas[:, :, None] * templates[None]).reshape(sigmas.shape[0], -1)
        )

    def _assemble_batch(self, sigmas: Tensor) -> Tensor:
        """Assemble system operators on the GPU, including unit Dirichlet diagonals when active."""

        values = self._assemble_homogeneous(sigmas, robin=True)
        if self._fixed_diag_slots.size:
            slots = self._on(
                "fixed_diag_slots",
                lambda: torch.as_tensor(self._fixed_diag_slots, dtype=torch.long),
                _CUDA,
            )
            values = values.clone()
            values[:, slots] = 1.0
        return values

    def _solve(
        self,
        values: Tensor,
        rhs: Tensor,
        *,
        refactorize: bool = True,
        factorization=None,
    ) -> Tensor:
        """Batched sparse solve ``A_b X_b = rhs_b`` on the GPU (plans are reused; see :class:`BatchedSolver`)."""

        return self._solver.solve(
            "3d",
            self._pattern,
            values.to(_CUDA, torch.float64),
            rhs.to(_CUDA, torch.float64),
            spd=True,
            refactorize=refactorize,
            factorization=factorization,
        )

    def _source_rhs_tensor(self) -> Tensor:
        return self._on(
            "source_rhs",
            lambda: torch.as_tensor(self._source_rhs, dtype=torch.float64),
            _CUDA,
        )

    def _receiver_rhs_tensor(self) -> Tensor:
        return self._on(
            "receiver_rhs",
            lambda: torch.as_tensor(self._receiver_rhs, dtype=torch.float64),
            _CUDA,
        )

    def _receiver_electrodes(self) -> Tensor:
        return self._receiver_rhs_tensor()[: len(self._receiver_electrode_ids)]

    def _pad_rhs(self, rhs: Tensor) -> Tensor:
        """Pad ``(B, n, dofs)`` right-hand sides to the plan's ``rhs_capacity`` rows."""

        missing = self._rhs_capacity - rhs.shape[1]
        if missing == 0:
            return rhs
        return torch.cat((rhs, rhs.new_zeros((rhs.shape[0], missing, rhs.shape[2]))), 1)

    def _solve_fields(self, sigmas: Tensor, factorization) -> list[_Fields3D]:
        values = self._assemble_batch(sigmas.to(_CUDA, torch.float64))
        rhs = self._source_rhs_tensor().expand(len(sigmas), -1, -1)
        solved = self._solve(values, rhs, factorization=factorization)
        fields = []
        for value, rows in zip(values, solved, strict=True):
            current = rows[: len(self._current_electrode_ids)]
            check_finite(current, "node fields")
            receiver = current if self._shared_source_receiver_rhs else None
            fields.append(_Fields3D(value, current, receiver))
        return fields

    def _fields_series(
        self, models: Tensor
    ) -> tuple[list[_Fields3D], list[bytes | None]]:
        return self._field_cache.solve_series(
            models, self.series_batch_steps, self._solve_fields
        )

    def _fields(self, conductivity) -> _Fields3D:
        return self._fields_series(self._sigma(conductivity)[None])[0][0]

    def _receivers(
        self, fields: list[_Fields3D], keys: list[bytes | None]
    ) -> list[Tensor]:
        """Receiver fields of every model, solving the missing ones with its operator."""

        receivers = {
            step: receiver
            for step, item in enumerate(fields)
            if (receiver := item.receiver) is not None
        }
        missing = [step for step in range(len(fields)) if step not in receivers]
        n_recv = len(self._receiver_electrode_ids)
        for batch in step_batches(missing, self.series_batch_steps)[::-1]:
            solved = self._solve(
                torch.stack([fields[step].values for step in batch]),
                self._receiver_rhs_tensor().expand(len(batch), -1, -1),
                factorization=factorization_name([keys[step] for step in batch]),
            )
            for step, rows in zip(batch, solved, strict=True):
                receiver = rows[:n_recv]
                check_finite(receiver, "receiver fields")
                receivers[step] = fields[step].receiver = receiver
        return [receivers[step] for step in range(len(fields))]

    def normal_field_cache_info(self) -> dict[str, int]:
        """Return LRU field-cache counters for profiling and regression tests."""

        return self._field_cache.info()

    def _measurement_resistances(self, current_fields: Tensor) -> Tensor:
        potentials = current_fields @ self._receiver_electrodes().T
        a, b, m, n = self._mapped_abmn(_CUDA)
        return (
            potentials[..., a, m]
            - potentials[..., b, m]
            - potentials[..., a, n]
            + potentials[..., b, n]
        )

    def _geometric_factors(self) -> Tensor:
        use_numerical = self.geometric_factor_mode == "numerical" or (
            self.geometric_factor_mode == "auto" and not self.mesh.is_flat_surface
        )
        if not use_numerical:
            return self.survey.geometric_factors()
        return self._cached("geometric_factors", self._numerical_geometric_factors)

    def _numerical_geometric_factors(self) -> Tensor:
        unit_values = self._on(
            "unit_values",
            lambda: torch.as_tensor(self._unit_operator.data, dtype=torch.float64)[
                None
            ],
            _CUDA,
        )
        solved = self._solve(
            unit_values, self._source_rhs_tensor()[None], factorization=None
        )
        unit_resistance = self._measurement_resistances(
            solved[0, : len(self._current_electrode_ids)]
        )
        if bool(
            torch.any(torch.abs(unit_resistance) <= torch.finfo(torch.float64).tiny)
        ):
            raise ValueError(
                "numerical geometric-factor solve produced zero resistance"
            )
        return (1.0 / unit_resistance).to(device="cpu", dtype=FLOAT_DTYPE)

    def _per_electrode(self, current_rows: np.ndarray) -> Tensor:
        """Rows of the current electrodes placed in an all-electrode array (NaN elsewhere)."""

        full = np.full(
            (self.survey.electrode_count, current_rows.shape[1]),
            np.nan,
            dtype=np.float64,
        )
        full[self._current_electrode_ids] = current_rows
        return torch.as_tensor(full, dtype=FLOAT_DTYPE)

    def _response(
        self, current: Tensor, currents, *, include_fields: bool
    ) -> ForwardResponse3D:
        resistance = self._measurement_resistances(current).to(
            device="cpu", dtype=FLOAT_DTYPE
        )
        apparent = (
            torch.abs(self._geometric_factors())
            * resistance
            / torch.as_tensor(currents, dtype=FLOAT_DTYPE)
        )
        if not include_fields:
            return ForwardResponse3D(apparent, resistance)
        nodes = current.detach().cpu().numpy()
        return ForwardResponse3D(
            apparent,
            resistance,
            electrode_potentials=self._per_electrode(
                np.asarray(nodes @ self._electrode_matrix.T)
            ),
            node_potentials=self._per_electrode(nodes),
        )

    def solve(
        self, conductivity: torch.Tensor | float, currents: torch.Tensor | float = 1.0
    ) -> ForwardResponse3D:
        return self._response(
            self._fields(conductivity).current, currents, include_fields=True
        )

    def resistance(self, conductivity: torch.Tensor | float) -> torch.Tensor:
        return self.resistance_series(self._sigma(conductivity)[None])[0]

    def resistance_series(self, conductivities) -> torch.Tensor:
        """Resistances of a ``(n_steps, n_cells)`` model series, ``(n_steps, n_data)``."""

        fields, _ = self._fields_series(self._series(conductivities))
        current = torch.stack([f.current for f in fields])
        resistance = self._measurement_resistances(current).to(
            device="cpu", dtype=FLOAT_DTYPE
        )
        check_finite(resistance, "resistances")
        return resistance

    def apparent_resistivity_values(
        self, conductivity: torch.Tensor | float, currents: torch.Tensor | float = 1.0
    ) -> torch.Tensor:
        factors = torch.abs(self._geometric_factors())
        current = torch.as_tensor(currents, dtype=FLOAT_DTYPE)
        return factors * self.resistance(conductivity) / current

    def apparent_resistivity_series(
        self, conductivities: torch.Tensor, currents: torch.Tensor | float = 1.0
    ) -> torch.Tensor:
        factors = torch.abs(self._geometric_factors())
        resistance = self.resistance_series(conductivities)
        current = torch.as_tensor(currents, dtype=FLOAT_DTYPE)
        if (
            current.ndim == 1
            and current.shape[0] == resistance.shape[0]
            and current.shape[0] != resistance.shape[1]
        ):
            current = current[:, None]
        return factors * resistance / current

    def _columns(
        self, cell_parameter_ids=None, parameter_count=None
    ) -> CellAggregation:
        """Reduction of per-cell values into forward cells or inversion parameters."""

        targets, count = parameter_targets(
            cell_parameter_ids, parameter_count, self.mesh.cell_count
        )
        digest = hashlib.blake2b(targets.tobytes(), digest_size=16).digest()
        return self._cached(
            ("columns", digest, count), lambda: CellAggregation(targets, count, _CUDA)
        )

    def _pair_cotangent(self, cotangent: Tensor) -> Tensor:
        """Transpose of the quadrupole difference map onto ``(..., n_curr, n_recv)``."""

        n_curr = len(self._current_electrode_ids)
        n_recv = len(self._receiver_electrode_ids)

        def build():
            a, b, m, n = self._mapped_abmn()
            groups = torch.cat(
                (
                    a * n_recv + m,
                    a * n_recv + n,
                    b * n_recv + m,
                    b * n_recv + n,
                )
            )
            return GroupSum(groups.numpy(), n_curr * n_recv, _CUDA)

        scatter = self._cached("pair_scatter", build)
        g = cotangent.to(_CUDA, torch.float64)
        signed = torch.cat((g, -g, -g, g), dim=-1)
        return scatter(signed).reshape(*g.shape[:-1], n_curr, n_recv)

    def _adjoint_rhs(self, pairs: Tensor) -> Tensor:
        """Adjoint right-hand sides of ``(B, n_curr, n_recv)`` pair cotangents."""

        return self._pad_rhs(pairs @ self._receiver_electrodes())

    def _cell_gradient(
        self,
        phi: Tensor,
        lam: Tensor,
        *,
        robin: bool = True,
        columns: CellAggregation | None = None,
    ) -> Tensor:
        """``-sum_s lam_s^T (dA/dsigma_c) phi_s`` per output column (SDDMM on the CSR pattern)."""

        q0 = self._pattern.sample(
            lam.to(_CUDA, torch.float64).T, phi.to(_CUDA, torch.float64)
        )
        entries = self._on(
            "sddmm_entries",
            lambda: torch.as_tensor(self._sddmm_entries, dtype=torch.long),
            _CUDA,
        )
        gradient = -(self._templates(robin=robin) * q0[entries]).sum(dim=1)
        if columns is not None:
            gradient = columns(gradient)
        return gradient.to(device="cpu", dtype=FLOAT_DTYPE)

    def vjp(self, conductivity, cotangent) -> Tensor:
        """Apply the transposed exact resistance Jacobian to a measurement cotangent."""

        cotangent = measurement_vector(
            cotangent, self.survey.measurement_count, FLOAT_DTYPE
        )
        return self.vjp_series(self._sigma(conductivity)[None], cotangent[None])[0]

    def vjp_series(self, conductivities, cotangents) -> Tensor:
        """Apply :meth:`vjp` to every step of a ``(n_steps, n_cells)`` model series.

        Adjoint solves are batched like the forward fields. The operators are symmetric, so a
        batch whose forward factorization the solver still holds (right after
        :meth:`resistance_series` of the same models) is solved without refactorizing.
        """

        models = self._series(conductivities)
        cotangents = cotangent_series(
            cotangents, models.shape[0], self.survey.measurement_count
        )
        fields, keys = self._fields_series(models)
        n_curr = len(self._current_electrode_ids)
        gradients: dict[int, Tensor] = {}
        for batch in newest_batches_first(len(fields), self.series_batch_steps):
            lam = self._solve(
                torch.stack([fields[step].values for step in batch]),
                self._adjoint_rhs(self._pair_cotangent(cotangents[batch])),
                factorization=factorization_name([keys[step] for step in batch]),
            )
            check_finite(lam, "adjoint fields")
            for step, rows in zip(batch, lam, strict=True):
                gradients[step] = self._cell_gradient(
                    fields[step].current, rows[:n_curr]
                )
        gradient = torch.stack([gradients[step] for step in range(len(fields))])
        check_finite(gradient, "conductivity gradient")
        return gradient

    def jvp(self, conductivity, delta_conductivity) -> Tensor:
        """Apply the exact resistance Jacobian to a conductivity perturbation."""

        sigma = self._sigma(conductivity)
        delta_sigma = cell_vector(
            delta_conductivity, self.mesh.cell_count, torch.float64
        )
        fields = self._fields(sigma)
        tangent = self._assemble_homogeneous(delta_sigma[None], robin=True)
        delta = self._solve(
            fields.values[None],
            self._pad_rhs(-self._pattern.matvec(tangent, fields.current[None])),
            factorization=factorization_name([self._field_cache.key(sigma)]),
        )[0, : len(self._current_electrode_ids)]
        check_finite(delta, "tangent fields")
        result = self._measurement_resistances(delta).to(
            device="cpu", dtype=FLOAT_DTYPE
        )
        check_finite(result, "resistance jvp")
        return result

    def normal_vjp(
        self, conductivity, cotangent, *, cell_parameter_ids=None, parameter_count=None
    ) -> Tensor:
        """Transpose of the normal-quadrupole sensitivity (see :meth:`normal_jvp`).

        ``cell_parameter_ids`` fuses the aggregation from forward cells into inversion
        parameters; cells with id ``-1`` are dropped.
        """

        cotangent = measurement_vector(
            cotangent, self.survey.measurement_count, FLOAT_DTYPE
        )
        return self.normal_vjp_series(
            self._sigma(conductivity)[None],
            cotangent[None],
            cell_parameter_ids=cell_parameter_ids,
            parameter_count=parameter_count,
        )[0]

    def normal_vjp_series(
        self,
        conductivities,
        cotangents,
        *,
        cell_parameter_ids=None,
        parameter_count=None,
    ) -> Tensor:
        """Apply :meth:`normal_vjp` to every step of a model series."""

        models = self._series(conductivities)
        cotangents = cotangent_series(
            cotangents, models.shape[0], self.survey.measurement_count
        )
        columns = self._columns(cell_parameter_ids, parameter_count)
        fields, keys = self._fields_series(models)
        receivers = self._receivers(fields, keys)
        pairs = self._pair_cotangent(cotangents)
        gradient = torch.stack(
            [
                self._cell_gradient(
                    item.current, pair @ receiver, robin=False, columns=columns
                )
                for item, receiver, pair in zip(fields, receivers, pairs, strict=True)
            ]
        )
        check_finite(gradient, "normal sensitivity vjp")
        return gradient

    def normal_jvp(self, conductivity, delta_conductivity) -> Tensor:
        """Matrix-free normal-quadrupole sensitivity without the Robin boundary derivative.

        This matches ``jacobian(..., include_robin_boundary_derivative=False)``, the
        inversion sensitivity convention, rather than differentiating :meth:`resistance`.
        """

        sigma = self._sigma(conductivity)
        delta_sigma = cell_vector(
            delta_conductivity, self.mesh.cell_count, torch.float64
        )
        fields, keys = self._fields_series(sigma[None])
        (receiver,) = self._receivers(fields, keys)
        tangent = self._assemble_homogeneous(delta_sigma[None], robin=False)
        gram = -self._pattern.matvec(tangent, fields[0].current[None])[0] @ receiver.T
        a, b, m, n = self._mapped_abmn(_CUDA)
        result = (gram[a, m] - gram[b, m] - gram[a, n] + gram[b, n]).to(
            device="cpu", dtype=FLOAT_DTYPE
        )
        check_finite(result, "normal sensitivity jvp")
        return result

    def jacobian(
        self,
        conductivity,
        *,
        batch_size: int | None = None,
        include_robin_boundary_derivative: bool = False,
        normal_sensitivity: bool = True,
        jacobian_cell_parameter_ids=None,
        jacobian_parameter_count: int | None = None,
    ) -> Tensor:
        """Explicit resistance Jacobian ``-(u_M - u_N)_c^T T_c (u_A - u_B)_c``.

        Rows are computed ``batch_size`` measurements at a time by the fused sensitivity
        kernel. 3D responses are not reciprocal-averaged, so ``normal_sensitivity`` has no
        effect; ``include_robin_boundary_derivative`` adds the far-field Robin terms.
        """

        del normal_sensitivity
        count = self.survey.measurement_count
        batch_size = min(max(count, 1), 64) if batch_size is None else int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        columns = self._columns(jacobian_cell_parameter_ids, jacobian_parameter_count)
        fields, keys = self._fields_series(self._sigma(conductivity)[None])
        (receiver,) = self._receivers(fields, keys)
        phi = torch.cat((fields[0].current, receiver))[None]
        n_curr = len(self._current_electrode_ids)
        local = self._cell_connectivity.shape[1]
        templates = self._templates(robin=include_robin_boundary_derivative)
        templates = columns.select(templates.reshape(1, -1, local, local), 1)
        cell_dofs = columns.select(
            self._on(
                "cell_dofs",
                lambda: torch.as_tensor(self._cell_connectivity, dtype=torch.long),
                _CUDA,
            ),
            0,
        )
        a, b, m, n = self._mapped_abmn(_CUDA)
        rows = [
            columns.columns(
                kernels.normal_sensitivity(
                    phi,
                    a[chunk],
                    b[chunk],
                    m[chunk] + n_curr,
                    n[chunk] + n_curr,
                    cell_dofs,
                    templates,
                )
            )
            for chunk in (
                slice(start, start + batch_size)
                for start in range(0, count, batch_size)
            )
        ]
        result = torch.cat(rows).to(device="cpu", dtype=FLOAT_DTYPE)
        check_finite(result, "jacobian")
        return result

    def jacobian_columnwise(self, conductivity) -> Tensor:
        """Materialize the exact resistance Jacobian with one JVP per cell (for testing)."""

        basis = torch.eye(self.mesh.cell_count, dtype=FLOAT_DTYPE)
        return torch.stack([self.jvp(conductivity, column) for column in basis], dim=1)

    def solve_with_jacobian(
        self,
        conductivity,
        currents=1.0,
        *,
        batch_size: int | None = None,
        include_robin_boundary_derivative: bool = False,
        normal_sensitivity: bool = True,
        include_fields: bool = False,
        jacobian_cell_parameter_ids=None,
        jacobian_parameter_count: int | None = None,
    ) -> tuple[ForwardResponse3D, Tensor]:
        """The forward response (potentials only with ``include_fields``) and :meth:`jacobian`."""

        sigma = self._sigma(conductivity)
        response = self._response(
            self._fields(sigma).current, currents, include_fields=include_fields
        )
        jacobian = self.jacobian(
            sigma,
            batch_size=batch_size,
            include_robin_boundary_derivative=include_robin_boundary_derivative,
            normal_sensitivity=normal_sensitivity,
            jacobian_cell_parameter_ids=jacobian_cell_parameter_ids,
            jacobian_parameter_count=jacobian_parameter_count,
        )
        return response, jacobian

    def prepare(
        self,
        conductivity: torch.Tensor | float | None = None,
        *,
        include_solver_state: bool = True,
    ) -> None:
        self._geometric_factors()
        if include_solver_state:
            self._fields(
                torch.ones(self.mesh.cell_count, dtype=FLOAT_DTYPE)
                if conductivity is None
                else conductivity
            )

    def close(self) -> None:
        self._field_cache.clear()
        self._solver.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
