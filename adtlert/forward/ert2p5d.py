"""2.5D multi-wavenumber ERT forward operator.

Every cosine-transform wavenumber is solved with the secondary-field formulation

    A(sigma) u_s = A(1) u_p - A(sigma) (rho_src u_p),    u = u_s + rho_src u_p,

where ``u_p`` is the unit-resistivity primary field of each electrode. Flat surfaces use
the analytic half-space primary on the input mesh. Terrain solves on an auxiliary H2/P1
discretization whose primary field is computed numerically on an H2/P2 discretization.
Both cases share one code path parameterized by :class:`Discretization`.
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.special import k0 as besselk0

from adtlert.forward.cudss import CUDA as _CUDA
from adtlert.forward.cudss import BatchedSolver
from adtlert.forward.discretization import (
    Discretization,
    build_discretization,
    exact_node_indices,
)
from adtlert.forward.integration import (
    build_inverse_cosine_weights,
    survey_wavenumber_bounds,
)
from adtlert.forward.kernels import GroupSum, normal_sensitivity
from adtlert.mesh import Mesh
from adtlert.survey import Survey
from adtlert.utils.dtypes import FLOAT_DTYPE, NP_FLOAT_DTYPE

Tensor = torch.Tensor

_TERRAIN_CACHE_VERSION = "terrain_auxiliary_v2"


@dataclass(frozen=True)
class ForwardResponse:
    """Result of a 2.5D ERT forward solve."""

    apparent_resistivity: Tensor
    resistance: Tensor
    electrode_potentials: Tensor
    integrated_potentials: Tensor
    wavenumbers: Tensor
    weights: Tensor


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _choice(value: str, choices: tuple[str, ...], name: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in choices:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(choices))}")
    return normalized


def _cell_vector(values, count: int) -> Tensor:
    array = torch.as_tensor(values, dtype=FLOAT_DTYPE)
    if array.ndim == 0:
        return array.expand(count)
    if array.shape != (count,):
        raise ValueError(f"conductivity must be scalar or shape ({count},)")
    return array


def _measurement_vector(values, count: int, dtype) -> Tensor:
    array = torch.as_tensor(values, dtype=FLOAT_DTYPE)
    if array.ndim == 0:
        array = array.expand(count)
    elif array.shape != (count,):
        raise ValueError(f"cotangent must be scalar or shape ({count},)")
    return array.to(dtype)


def _check_finite(values: Tensor, context: str) -> None:
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError(f"{context} produced non-finite values")


def _halfspace_primary(
    nodes: np.ndarray, source: np.ndarray, wavenumber: float, surface: float
) -> np.ndarray:
    """Analytic unit-resistivity half-space primary field (image source when buried)."""

    distance = np.linalg.norm(nodes - source, axis=1)
    values = np.zeros(nodes.shape[0])
    if abs(source[1] - surface) <= 1e-8:
        valid = distance > 1e-12
        values[valid] = besselk0(distance[valid] * wavenumber) / np.pi
        return values
    mirrored = np.linalg.norm(nodes - (source[0], 2.0 * surface - source[1]), axis=1)
    valid = (distance > 1e-12) & (mirrored > 1e-12)
    values[valid] = (
        besselk0(distance[valid] * wavenumber) + besselk0(mirrored[valid] * wavenumber)
    ) / (2.0 * np.pi)
    return values


@dataclass
class _Fields:
    """Assembled operator values and total fields for one conductivity model."""

    values: Tensor
    total: Tensor
    _device_copies: dict = field(default_factory=dict)

    def on(self, device: torch.device) -> Tensor:
        if device.type == "cpu":
            return self.total
        if str(device) not in self._device_copies:
            self._device_copies[str(device)] = self.total.to(device)
        return self._device_copies[str(device)]


@dataclass(frozen=True, eq=False)
class ERTForward2p5D:
    """2.5D ERT forward operator with analytic (flat) or numerical (terrain) primary fields."""

    mesh: Mesh
    survey: Survey
    wavenumbers: Tensor
    weights: Tensor
    discretization: Discretization
    primary_potential_discretization: Discretization | None
    geometric_discretization: Discretization | None
    source_cell_ids: Tensor
    source_node_ids: Tensor
    numerical_h2_refined: bool
    numerical_p2_refined: bool
    topographic_geometric_factor_mode: str
    terrain_cache_dir: Path | None
    normal_field_cache_max_entries: int = 8
    _cache: dict = field(default_factory=dict, init=False, repr=False)
    _solver: BatchedSolver = field(
        default_factory=BatchedSolver, init=False, repr=False
    )
    _field_cache: OrderedDict = field(
        default_factory=OrderedDict, init=False, repr=False
    )

    @classmethod
    def from_mesh_survey(
        cls,
        mesh: Mesh,
        survey: Survey,
        quadrature_order: int = 2,
        numerical_h2_refined: bool = True,
        numerical_p2_refined: bool = True,
        topographic_geometric_factor_mode: str = "analytic",
        terrain_cache_dir: str | Path | None = None,
        normal_field_cache_max_entries: int = 8,
    ) -> ERTForward2p5D:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "ADTLERT requires an NVIDIA GPU with CUDA (cuDSS sparse solves)"
            )
        gf_mode = _choice(
            topographic_geometric_factor_mode,
            ("analytic", "numerical"),
            "topographic_geometric_factor_mode",
        )
        if int(normal_field_cache_max_entries) < 0:
            raise ValueError("normal_field_cache_max_entries must be non-negative")
        quadrature = build_inverse_cosine_weights(*survey_wavenumber_bounds(survey))

        def build(name, **options):
            return build_discretization(
                name,
                mesh,
                survey,
                quadrature.wavenumbers,
                quadrature_order=quadrature_order,
                **options,
            )

        terrain = not mesh.is_flat_surface
        potential = geometric = None
        if terrain:
            # Terrain solves run in float64; historically this also switches Torch's default dtype.
            torch.set_default_dtype(torch.float64)
            discretization = build(
                "primary", auxiliary=True, refine=numerical_h2_refined
            )
            potential = build(
                "primary_potential",
                auxiliary=True,
                refine=numerical_h2_refined,
                quadratic=True,
            )
            if gf_mode == "numerical":
                geometric = build(
                    "geometric",
                    auxiliary=True,
                    refine=numerical_h2_refined,
                    quadratic=numerical_p2_refined,
                )
        else:
            discretization = build("native", auxiliary=False)

        positions = np.asarray(survey.electrode_positions, dtype=float)
        nodes = np.asarray(mesh.nodes, dtype=float)
        distances = np.linalg.norm(nodes[None, :, :] - positions[:, None, :], axis=2)
        nearest = np.argmin(distances, axis=1)
        source_node_ids = np.where(
            distances[np.arange(len(nearest)), nearest] <= 1e-2, nearest, -1
        )

        return cls(
            mesh=mesh,
            survey=survey,
            wavenumbers=quadrature.wavenumbers,
            weights=quadrature.weights,
            discretization=discretization,
            primary_potential_discretization=potential,
            geometric_discretization=geometric,
            source_cell_ids=mesh.locate_points(survey.electrode_positions)[0].long(),
            source_node_ids=torch.as_tensor(source_node_ids, dtype=torch.long),
            numerical_h2_refined=numerical_h2_refined,
            numerical_p2_refined=numerical_p2_refined,
            topographic_geometric_factor_mode=gf_mode,
            terrain_cache_dir=_terrain_cache_dir(terrain_cache_dir),
            normal_field_cache_max_entries=int(normal_field_cache_max_entries),
        )

    # -- configuration ------------------------------------------------------

    @property
    def use_numerical_primary(self) -> bool:
        return self.primary_potential_discretization is not None

    @property
    def use_numerical_geometric_factors(self) -> bool:
        return self.geometric_discretization is not None

    @property
    def dtype(self) -> torch.dtype:
        """Field precision: float64 on terrain, ``FLOAT_DTYPE`` otherwise."""

        return torch.float64 if self.use_numerical_primary else FLOAT_DTYPE

    @property
    def _device(self) -> torch.device:
        """Device of the dense adjoint contractions (fields are cached on the host)."""

        return _CUDA

    def _cached(self, key, compute):
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    def _on(self, name: str, tensor_fn, device: torch.device) -> Tensor:
        return self._cached((name, str(device)), lambda: tensor_fn().to(device))

    def _abmn(
        self, device: torch.device | str = "cpu"
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        quads = self._on(
            "abmn", lambda: self.survey.measurements.long(), torch.device(device)
        )
        return quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]

    # -- assembly -----------------------------------------------------------

    def _volume_templates(
        self, d: Discretization, device: torch.device | str = "cpu"
    ) -> Tensor:
        """``K + k_w^2 M`` per wavenumber, shape ``(W, cells, k, k)``."""

        def build():
            wavenumber_sq = torch.square(self.wavenumbers.to(self.dtype))
            return d.stiffness.to(self.dtype) + wavenumber_sq[
                :, None, None, None
            ] * d.mass.to(self.dtype)

        return self._on(f"{d.name}.volume_templates", build, torch.device(device))

    def _boundary_templates(
        self, d: Discretization, device: torch.device | str = "cpu"
    ) -> Tensor:
        def build():
            return d.boundary_geometries.to(self.dtype)[
                :, :, None, None
            ] * d.boundary_mass.to(self.dtype)

        return self._on(f"{d.name}.boundary_templates", build, torch.device(device))

    def _assemble(self, d: Discretization, conductivity: Tensor) -> Tensor:
        """CSR values of ``A_w(conductivity)`` for every wavenumber, shape ``(W, nnz)``."""

        sigma = conductivity.to(self.dtype)
        volume = sigma[d.parent_cell_ids][None, :, None, None] * self._volume_templates(
            d
        )
        boundary = (
            sigma[d.parent_cell_ids[d.boundary_cells]][None, :]
            * d.boundary_geometries.to(self.dtype)
        )[:, :, None, None] * d.boundary_mass.to(self.dtype)
        count = self.wavenumbers.shape[0]
        values = torch.zeros((count, d.pattern.nnz), dtype=self.dtype)
        values.index_add_(1, d.pattern.volume_inverse, volume.reshape(count, -1))
        return values.index_add_(
            1, d.pattern.boundary_inverse, boundary.reshape(count, -1)
        )

    def _sigma(self, conductivity) -> Tensor:
        return _cell_vector(conductivity, self.mesh.cell_count)

    # -- linear solves ------------------------------------------------------

    def _solve(
        self,
        d: Discretization,
        values: Tensor,
        rhs: Tensor,
        *,
        refactorize: bool = True,
    ) -> Tensor:
        """Batched sparse solve ``A_b X_b = rhs_b`` (plans are reused; see :class:`BatchedSolver`)."""

        return self._solver.solve(
            d.name,
            d.pattern,
            values.to(self.dtype),
            rhs.to(self.dtype),
            spd=d.spd,
            refactorize=refactorize,
        )

    # -- primary fields and terrain caches ----------------------------------

    def _disk_path(self, name: str, d: Discretization) -> Path | None:
        if self.terrain_cache_dir is None:
            return None
        digest = hashlib.sha256()
        for label, value in (
            ("version", _TERRAIN_CACHE_VERSION),
            ("name", name),
            ("dtype", str(self.dtype)),
            (
                "options",
                (
                    self.numerical_h2_refined,
                    self.numerical_p2_refined,
                    self.topographic_geometric_factor_mode,
                ),
            ),
        ):
            digest.update(f"{label}={value!r};".encode())
        for array in (
            self.mesh.nodes, self.mesh.cells, self.mesh.surface_node_ids, self.survey.electrode_positions,
            self.survey.measurements, self.wavenumbers, self.weights, d.parent_cell_ids, d.dof_nodes, d.cell_dofs,
            d.boundary_dofs, d.electrode_matrix, d.pattern.indptr, d.pattern.indices, d.stiffness, d.mass,
            d.boundary_mass, d.boundary_geometries,
        ):  # fmt: skip
            array = np.ascontiguousarray(np.asarray(array))
            digest.update(f"{array.dtype}{array.shape}".encode())
            digest.update(array.tobytes())
        return self.terrain_cache_dir / f"{digest.hexdigest()}.npz"

    def _disk_cached(self, name: str, d: Discretization, compute) -> Tensor:
        """Memory- and (optionally) disk-cached unit potential stack of shape ``(W, E, dofs)``."""

        def load_or_compute():
            path = self._disk_path(name, d)
            shape = (
                self.wavenumbers.shape[0],
                self.survey.electrode_count,
                d.dof_count,
            )
            if path is not None and path.exists():
                try:
                    with np.load(path, allow_pickle=False) as payload:
                        cached = payload["value"]
                    if cached.shape == shape:
                        return torch.as_tensor(cached, dtype=self.dtype)
                except (
                    Exception
                ):  # corrupt or partial cache files are simply recomputed
                    pass
            value = compute()
            if path is not None:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
                    np.savez(tmp, value=value.numpy())
                    tmp.replace(path)
                except OSError:
                    pass
            return value

        return self._cached(name, load_or_compute)

    def _unit_potentials(self, d: Discretization) -> Tensor:
        """Unit-conductivity potentials of every electrode on ``d``."""

        def compute():
            values = self._assemble(
                d, torch.ones(self.mesh.cell_count, dtype=FLOAT_DTYPE)
            )
            rhs = d.electrode_matrix.to(self.dtype).expand(
                self.wavenumbers.shape[0], -1, -1
            )
            return self._solve(d, values, rhs)

        return self._disk_cached(f"{d.name}_sub_potentials", d, compute)

    def _unit_primary(self) -> Tensor:
        """Unit-resistivity primary field ``u_p`` on the field discretization, ``(W, E, dofs)``."""

        if self.use_numerical_primary:
            potential = self.primary_potential_discretization

            def project():
                selection = exact_node_indices(
                    potential.dof_nodes, self.discretization.dof_nodes
                )
                return self._unit_potentials(potential)[..., torch.as_tensor(selection)]

            return self._disk_cached(
                "auxiliary_sub_potentials", self.discretization, project
            )
        return self._cached("analytic_primary", self._analytic_primary)

    def _analytic_primary(self) -> Tensor:
        nodes = np.asarray(self.mesh.nodes, dtype=float)
        cells = np.asarray(self.mesh.cells)
        surface = float(np.max(nodes[:, 1]))
        node_ids = self.source_node_ids.tolist()
        radii = {}
        for node in set(node_ids) - {-1}:
            neighbors = np.setdiff1d(
                np.unique(cells[np.any(cells == node, axis=1)]), [node]
            )
            radii[node] = (
                np.min(np.linalg.norm(nodes[neighbors] - nodes[node], axis=1))
                if neighbors.size
                else None
            )
        positions = np.asarray(self.survey.electrode_positions, dtype=float)
        primary = np.zeros((self.wavenumbers.shape[0], len(positions), nodes.shape[0]))
        for w, wavenumber in enumerate(self.wavenumbers.double().tolist()):
            for s, (source, node) in enumerate(zip(positions, node_ids, strict=True)):
                primary[w, s] = _halfspace_primary(nodes, source, wavenumber, surface)
                if (
                    node >= 0
                ):  # pyGIMLi's singular value: K0 at one sixth of the nearest-neighbor distance
                    radius = radii[node]
                    primary[w, s, node] = (
                        0.0
                        if radius is None
                        else besselk0(radius / 6.0 * wavenumber) / np.pi
                    )
        return torch.as_tensor(primary, dtype=FLOAT_DTYPE)

    def _reference_rhs(self) -> Tensor:
        """``A(1) u_p``, the conductivity-independent part of the secondary-field RHS."""

        def compute():
            d = self.discretization
            ones = torch.ones(self.mesh.cell_count, dtype=FLOAT_DTYPE)
            return d.pattern.matvec(self._assemble(d, ones), self._unit_primary())

        return self._cached("reference_rhs", compute)

    def _source_resistivities(self, sigma: Tensor) -> Tensor:
        """Resistivity at each electrode: geometric mean over cells sharing its node, else its cell."""

        def node_cell_weights():
            cells = np.asarray(self.mesh.cells)
            weights = np.zeros(
                (self.survey.electrode_count, self.mesh.cell_count),
                dtype=NP_FLOAT_DTYPE,
            )
            counts = np.ones(self.survey.electrode_count, dtype=NP_FLOAT_DTYPE)
            for source, node in enumerate(self.source_node_ids.tolist()):
                if node >= 0:
                    mask = np.any(cells == node, axis=1)
                    weights[source, mask], counts[source] = 1.0, np.count_nonzero(mask)
            return torch.as_tensor(weights), torch.as_tensor(counts)

        weights, counts = self._cached("source_cell_weights", node_cell_weights)
        node_rho = torch.exp(
            torch.einsum("ec,c->e", weights, -torch.log(sigma)) / counts
        )
        return torch.where(
            self.source_node_ids >= 0, node_rho, 1.0 / sigma[self.source_cell_ids]
        )

    # -- field solves -------------------------------------------------------

    def _field_cache_key(self, sigma: Tensor) -> bytes | None:
        if self.normal_field_cache_max_entries < 1:
            return None
        host = np.ascontiguousarray(sigma.detach().cpu().numpy())
        return (
            hashlib.blake2b(host.view(np.uint8), digest_size=16).digest()
            + host.dtype.str.encode()
        )

    def _fields(self, conductivity) -> _Fields:
        """Total fields for a conductivity model, served from an LRU cache when possible."""

        sigma = self._sigma(conductivity)
        key = self._field_cache_key(sigma)
        if key is not None:
            counter = "hits" if key in self._field_cache else "misses"
            self._cache[counter] = self._cache.get(counter, 0) + 1
            if key in self._field_cache:
                self._field_cache.move_to_end(key)
                return self._field_cache[key]

        d = self.discretization
        values = self._assemble(d, sigma)
        primary = (
            self._unit_primary() * self._source_resistivities(sigma)[None, :, None]
        )
        rhs = self._reference_rhs() - d.pattern.matvec(values, primary)
        fields = _Fields(values, self._solve(d, values, rhs) + primary)
        _check_finite(fields.total, "total fields")
        if key is not None:
            self._field_cache[key] = fields
            while len(self._field_cache) > self.normal_field_cache_max_entries:
                self._field_cache.popitem(last=False)
        return fields

    def normal_field_cache_info(self) -> dict[str, int]:
        """Return LRU field-cache counters for profiling and regression tests."""

        return {
            "entries": len(self._field_cache),
            "max_entries": self.normal_field_cache_max_entries,
            "hits": self._cache.get("hits", 0),
            "misses": self._cache.get("misses", 0),
        }

    # -- measurement maps ---------------------------------------------------

    def _integrate(self, fields: Tensor) -> Tensor:
        return torch.tensordot(self.weights.to(fields.dtype), fields, dims=([0], [0]))

    def _receivers(self) -> tuple[Tensor, Tensor]:
        """Receiver rows ``E[M] - E[N]`` (normal) and ``E[A] - E[B]`` (reciprocal)."""

        def build():
            a, b, m, n = self._abmn()
            electrodes = self.discretization.electrode_matrix
            return (electrodes[m] - electrodes[n]).to(self.dtype), (
                electrodes[a] - electrodes[b]
            ).to(self.dtype)

        return self._cached("receivers", build)

    def _normal_reciprocal(self, integrated: Tensor) -> tuple[Tensor, Tensor]:
        """Normal (AB source, MN receiver) and reciprocal (MN source, AB receiver) resistances."""

        a, b, m, n = self._abmn()
        rows = torch.arange(self.survey.measurement_count)
        normal_receiver, current_receiver = self._receivers()
        normal = integrated @ normal_receiver.T
        reciprocal = integrated @ current_receiver.T
        return normal[a, rows] - normal[b, rows], reciprocal[m, rows] - reciprocal[
            n, rows
        ]

    def _adjoint_rhs(self, cotangent: Tensor, *, reciprocal: bool = False) -> Tensor:
        """Transpose of the (reciprocal) measurement map for ``(Q, D)`` cotangents → ``(W, Q*E, dofs)``."""

        a, b, m, n = self._abmn()
        normal_receiver, current_receiver = self._receivers()
        positive, negative, receiver = (
            (m, n, current_receiver) if reciprocal else (a, b, normal_receiver)
        )
        weighted = cotangent.to(self.dtype)[:, :, None] * receiver[None]
        sources = torch.zeros(
            (cotangent.shape[0], self.survey.electrode_count, receiver.shape[1]),
            dtype=self.dtype,
        )
        sources.index_add_(1, positive, weighted).index_add_(1, negative, -weighted)
        rhs = self.weights.to(self.dtype)[:, None, None, None] * sources[None]
        return rhs.reshape(self.wavenumbers.shape[0], -1, receiver.shape[1])

    @staticmethod
    def _combine(normal: Tensor, reciprocal: Tensor) -> Tensor:
        return torch.sqrt(torch.abs(normal * reciprocal))

    @staticmethod
    def _combine_scale(normal: Tensor, reciprocal: Tensor) -> tuple[Tensor, Tensor]:
        """``sign(n r)`` and the floored combined resistance used by the reciprocal chain rule."""

        combined = torch.sqrt(torch.abs(normal * reciprocal))
        return torch.sign(normal * reciprocal), torch.clamp_min(combined, 1e-30)

    def _geometric_factors(self) -> Tensor:
        if not self.use_numerical_geometric_factors:
            return self.survey.geometric_factors()
        return self._cached("geometric_factors", self._numerical_geometric_factors)

    def _numerical_geometric_factors(self) -> Tensor:
        d = self.geometric_discretization
        integrated = self._integrate(self._unit_potentials(d))
        potentials = integrated @ d.electrode_matrix.to(integrated.dtype).T
        a, b, m, n = self._abmn()
        rows = torch.arange(self.survey.measurement_count)
        sources = potentials[a] - potentials[b]
        return 1.0 / (sources[rows, m] - sources[rows, n])

    def _response(
        self, integrated: Tensor, normal: Tensor, reciprocal: Tensor, currents
    ) -> ForwardResponse:
        resistance = self._combine(normal, reciprocal)
        current = torch.as_tensor(currents, dtype=FLOAT_DTYPE)
        electrode_matrix = self.discretization.electrode_matrix.to(integrated.dtype)
        return ForwardResponse(
            apparent_resistivity=torch.abs(self._geometric_factors())
            * resistance
            / current,
            resistance=resistance,
            electrode_potentials=integrated @ electrode_matrix.T,
            integrated_potentials=integrated,
            wavenumbers=self.wavenumbers,
            weights=self.weights,
        )

    def _measure(self, conductivity) -> tuple[_Fields, Tensor, Tensor, Tensor]:
        fields = self._fields(conductivity)
        integrated = self._integrate(fields.total)
        _check_finite(integrated, "integrated potentials")
        return fields, integrated, *self._normal_reciprocal(integrated)

    # -- public forward API -------------------------------------------------

    def solve(self, conductivity, currents=1.0) -> ForwardResponse:
        """Solve the multi-wavenumber 2.5D forward problem for a conductivity model."""

        _, integrated, normal, reciprocal = self._measure(conductivity)
        return self._response(integrated, normal, reciprocal, currents)

    def resistance(self, conductivity) -> Tensor:
        """Return the reciprocal-averaged resistance of each measurement."""

        _, _, normal, reciprocal = self._measure(conductivity)
        return self._combine(normal, reciprocal)

    def apparent_resistivity_values(self, conductivity, currents=1.0) -> Tensor:
        """Return apparent resistivity values without allocating a ForwardResponse."""

        current = torch.as_tensor(currents, dtype=FLOAT_DTYPE)
        return (
            torch.abs(self._geometric_factors())
            * self.resistance(conductivity)
            / current
        )

    def apparent_resistivity_series(self, conductivities, currents=1.0) -> Tensor:
        """Apparent resistivity for a ``(n_steps, n_cells)`` model series (currents may be per step)."""

        models = self._series(conductivities)
        current = torch.as_tensor(currents, dtype=FLOAT_DTYPE)
        return torch.stack(
            [
                self.apparent_resistivity_values(
                    model, current[step] if current.ndim == 2 else current
                )
                for step, model in enumerate(models)
            ]
        )

    def _series(self, conductivities) -> Tensor:
        models = torch.as_tensor(conductivities, dtype=FLOAT_DTYPE)
        if models.ndim != 2 or models.shape[1] != self.mesh.cell_count:
            raise ValueError(
                f"conductivities must have shape (n_steps, {self.mesh.cell_count})"
            )
        return models

    def prepare(self, conductivity=None, *, include_solver_state: bool = True) -> None:
        """Populate geometry, primary-field, and (optionally) solver/field caches before timed solves."""

        self._unit_primary()
        self._reference_rhs()
        self._receivers()
        self._geometric_factors()
        if include_solver_state:
            self._fields(
                torch.ones(self.mesh.cell_count)
                if conductivity is None
                else conductivity
            )

    # -- derivatives --------------------------------------------------------

    def _target_cells(
        self, cell_parameter_ids, parameter_count
    ) -> tuple[np.ndarray, int]:
        """Map discretization cells to output columns (forward cells or inversion parameters)."""

        parents = self.discretization.parent_cell_ids.numpy()
        if cell_parameter_ids is None:
            if parameter_count is not None:
                raise ValueError("parameter_count requires cell_parameter_ids")
            return parents, self.mesh.cell_count
        ids = np.asarray(
            cell_parameter_ids.cpu()
            if isinstance(cell_parameter_ids, Tensor)
            else cell_parameter_ids
        )
        ids = ids.astype(np.int64).reshape(-1)
        if ids.shape != (self.mesh.cell_count,):
            raise ValueError(
                f"cell_parameter_ids must have shape ({self.mesh.cell_count},)"
            )
        if np.any(ids < -1) or not np.any(ids >= 0):
            raise ValueError(
                "cell_parameter_ids may only contain -1 or non-negative ids, with at least one active"
            )
        count = int(ids.max()) + 1 if parameter_count is None else int(parameter_count)
        if count <= int(ids.max()):
            raise ValueError(
                "parameter_count is smaller than the largest cell parameter id"
            )
        return ids[parents], count

    def _aggregation(
        self, targets: np.ndarray, count: int
    ) -> tuple[Tensor | None, GroupSum]:
        """Active discretization cells (``None`` when all are) and their reduction into ``count`` columns."""

        digest = hashlib.blake2b(targets.tobytes(), digest_size=16).digest()

        def build():
            active = np.flatnonzero(targets >= 0)
            cells = (
                None
                if active.size == targets.size
                else torch.as_tensor(active, device=_CUDA)
            )
            return cells, GroupSum(targets[active], count, _CUDA)

        return self._cached(("aggregation", digest, count), build)

    def _cell_gradient(
        self, phi: Tensor, lam: Tensor, *, robin: bool = True, targets=None, count=None
    ) -> Tensor:
        """``-sum_{w,s} lam_ws^T dA_w/dsigma phi_ws`` per output column; ``lam`` may lead with a batch axis.

        The volume term is the transpose of assembly: ``sum_ws lam phi^T`` is sampled on the
        sparsity pattern (SDDMM) and contracted with the stiffness/mass templates, so no
        ``W x S x C x k`` gathered blocks are formed. Reductions are deterministic.
        """

        d = self.discretization
        if targets is None:
            targets, count = d.parent_cell_ids.numpy(), self.mesh.cell_count
        active, reduce = self._aggregation(targets, count)
        phi, lam = phi.to(_CUDA), lam.to(_CUDA)
        if lam.ndim == 4:
            return torch.stack(
                [
                    self._cell_gradient(
                        phi, block, robin=robin, targets=targets, count=count
                    )
                    for block in lam.unbind(1)
                ]
            )

        W, S, N = phi.shape
        k2 = torch.square(self.wavenumbers.to(_CUDA, phi.dtype))[:, None, None]
        right = phi.reshape(W * S, N)
        q0 = d.pattern.sample(lam.reshape(W * S, N).T, right)
        q2 = d.pattern.sample((k2 * lam).reshape(W * S, N).T, right)
        entries = self._on(
            f"{d.name}.entries",
            lambda: d.pattern.volume_inverse.reshape(d.cell_dofs.shape[0], -1),
            _CUDA,
        )
        stiffness, mass = (
            self._on(
                f"{d.name}.{name}",
                lambda t=t: t.reshape(entries.shape).to(self.dtype),
                _CUDA,
            )
            for name, t in (("stiffness", d.stiffness), ("mass", d.mass))
        )
        gradient = -(stiffness * q0[entries] + mass * q2[entries]).sum(dim=1)
        if robin:
            boundary_dofs = self._on(
                f"{d.name}.boundary_dofs", lambda: d.boundary_dofs, _CUDA
            )
            boundary = -torch.einsum(
                "wsbi,wbij,wsbj->b",
                lam[..., boundary_dofs],
                self._boundary_templates(d, _CUDA),
                phi[..., boundary_dofs],
            )
            to_cells = self._cached(
                (d.name, "boundary_to_cells"),
                lambda: GroupSum(d.boundary_cells.numpy(), d.cell_dofs.shape[0], _CUDA),
            )
            gradient = gradient + to_cells(boundary)
        return reduce(gradient if active is None else gradient[active]).cpu()

    def vjp(self, conductivity, cotangent) -> Tensor:
        """Apply the transposed resistance Jacobian to a measurement cotangent."""

        fields, _, normal, reciprocal = self._measure(conductivity)
        sign, scale = self._combine_scale(normal, reciprocal)
        common = (
            0.5
            * _measurement_vector(cotangent, self.survey.measurement_count, self.dtype)
            * sign
            / scale
        )
        rhs = self._adjoint_rhs((common * reciprocal)[None]) + self._adjoint_rhs(
            (common * normal)[None], reciprocal=True
        )
        lam = self._solve(self.discretization, fields.values, rhs)
        _check_finite(lam, "adjoint fields")
        gradient = self._cell_gradient(fields.on(self._device), lam)
        _check_finite(gradient, "conductivity gradient")
        return gradient

    def jvp(self, conductivity, delta_conductivity) -> Tensor:
        """Apply the resistance Jacobian to a conductivity perturbation."""

        d = self.discretization
        fields, _, normal, reciprocal = self._measure(conductivity)
        tangent_rhs = -d.pattern.matvec(
            self._assemble(d, self._sigma(delta_conductivity)), fields.total
        )
        delta = self._solve(d, fields.values, tangent_rhs)
        _check_finite(delta, "tangent fields")
        delta_normal, delta_reciprocal = self._normal_reciprocal(self._integrate(delta))
        sign, scale = self._combine_scale(normal, reciprocal)
        return (
            0.5 * sign * (delta_normal * reciprocal + normal * delta_reciprocal) / scale
        )

    def normal_vjp(
        self, conductivity, cotangent, *, cell_parameter_ids=None, parameter_count=None
    ) -> Tensor:
        """Transpose of the normal-quadrupole sensitivity (see :meth:`normal_jvp`).

        ``cell_parameter_ids`` fuses the aggregation from forward cells into inversion
        parameters; cells with id ``-1`` are ignored.
        """

        targets, count = self._target_cells(cell_parameter_ids, parameter_count)
        device = self._device
        phi = self._fields(conductivity).on(device)
        weights = _measurement_vector(
            cotangent, self.survey.measurement_count, phi.dtype
        )
        a, b, m, n = self._abmn()
        sources = self.survey.electrode_count
        pairs = torch.zeros(sources * sources, dtype=phi.dtype).index_add_(
            0, torch.cat((m * sources + a, m * sources + b, n * sources + a, n * sources + b)),
            torch.cat((weights, -weights, -weights, weights)),
        ).reshape(sources, sources).to(device)  # fmt: skip
        current = torch.einsum("ef,wfn->wen", pairs, phi)
        receiver = self.weights.to(device, phi.dtype)[:, None, None] * phi
        gradient = self._cell_gradient(
            current, receiver, robin=False, targets=targets, count=count
        )
        _check_finite(gradient, "normal sensitivity vjp")
        return gradient

    def normal_jvp(self, conductivity, delta_conductivity) -> Tensor:
        """Matrix-free normal-quadrupole sensitivity without the Robin boundary derivative.

        This matches ``jacobian(..., normal_sensitivity=True, include_robin_boundary_derivative=False)``,
        the inversion sensitivity convention, rather than differentiating :meth:`resistance`.
        """

        d, device = self.discretization, self._device
        phi = self._fields(conductivity).on(device)
        direction = self._sigma(delta_conductivity).to(device, phi.dtype)[
            d.parent_cell_ids.to(device)
        ]
        volume = direction[None, :, None, None] * self._volume_templates(d, device)
        count = self.wavenumbers.shape[0]
        assemble = self._cached(
            (d.name, "volume_assembly"),
            lambda: GroupSum(d.pattern.volume_inverse.numpy(), d.pattern.nnz, _CUDA),
        )
        tangent = assemble(volume.reshape(count, -1))
        gram = torch.einsum(
            "w,wen,wfn->ef",
            self.weights.to(device, phi.dtype),
            phi,
            -d.pattern.matvec(tangent, phi),
        )
        a, b, m, n = self._abmn(device)
        result = (gram[m, a] - gram[m, b] - gram[n, a] + gram[n, b]).cpu()
        _check_finite(result, "normal sensitivity jvp")
        return result

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
        cotangents = torch.as_tensor(cotangents, dtype=FLOAT_DTYPE)
        if tuple(cotangents.shape) != (models.shape[0], self.survey.measurement_count):
            raise ValueError(
                f"cotangents must have shape {(models.shape[0], self.survey.measurement_count)}"
            )
        return torch.stack(
            [
                self.normal_vjp(
                    model,
                    cotangent,
                    cell_parameter_ids=cell_parameter_ids,
                    parameter_count=parameter_count,
                )
                for model, cotangent in zip(models, cotangents, strict=True)
            ]
        )

    # -- explicit Jacobians -------------------------------------------------

    def _batch_size(self, batch_size: int | None, direct: bool) -> int:
        if batch_size is None:
            if not direct:
                return 8
            return (
                min(self.survey.measurement_count, 64)
                if self.mesh.cell_count >= 4096
                else self.survey.measurement_count
            )
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        return int(batch_size)

    def _normal_jacobian(
        self, fields: _Fields, batch_size: int, targets: np.ndarray, count: int
    ) -> Tensor:
        """Direct normal sensitivity ``-sum_w w_w (u_M - u_N)^T dA_w (u_A - u_B)`` (fused Triton kernel)."""

        d = self.discretization
        active, reduce = self._aggregation(targets, count)
        templates = self.weights.to(_CUDA, self.dtype)[
            :, None, None, None
        ] * self._volume_templates(d, _CUDA)
        cell_dofs = self._on(f"{d.name}.cell_dofs", lambda: d.cell_dofs, _CUDA)
        if active is not None:
            templates, cell_dofs = templates[:, active], cell_dofs[active]
        phi = fields.on(_CUDA)
        a, b, m, n = self._abmn(_CUDA)
        rows = [
            reduce(
                normal_sensitivity(
                    phi, a[chunk], b[chunk], m[chunk], n[chunk], cell_dofs, templates
                )
            )
            for chunk in (
                slice(start, start + batch_size)
                for start in range(0, self.survey.measurement_count, batch_size)
            )
        ]
        return torch.cat(rows).cpu()

    def _adjoint_jacobian(
        self,
        fields,
        normal,
        reciprocal,
        batch_size: int,
        robin: bool,
        normal_sensitivity: bool,
    ) -> Tensor:
        """Jacobian rows from batched adjoint solves (identity cotangents, last block zero-padded)."""

        count = self.survey.measurement_count
        sign, scale = self._combine_scale(normal, reciprocal)
        eye = torch.eye(count, dtype=normal.dtype)
        phi = fields.on(self._device)
        rows = []
        for start in range(0, count, batch_size):
            cotangent = torch.zeros((batch_size, count), dtype=normal.dtype)
            cotangent[: min(batch_size, count - start)] = eye[
                start : start + batch_size
            ]
            if normal_sensitivity:
                rhs = self._adjoint_rhs(cotangent)
            else:
                common = 0.5 * cotangent * sign[None, :] / scale[None, :]
                rhs = self._adjoint_rhs(
                    common * reciprocal[None, :]
                ) + self._adjoint_rhs(common * normal[None, :], reciprocal=True)
            lam = self._solve(
                self.discretization, fields.values, rhs, refactorize=start == 0
            )
            lam = lam.reshape(
                self.wavenumbers.shape[0], batch_size, self.survey.electrode_count, -1
            )
            rows.append(self._cell_gradient(phi, lam, robin=robin)[: count - start])
        return torch.cat(rows)

    def solve_with_jacobian(
        self,
        conductivity,
        currents=1.0,
        *,
        batch_size: int | None = None,
        include_robin_boundary_derivative: bool = False,
        normal_sensitivity: bool = True,
        jacobian_cell_parameter_ids=None,
        jacobian_parameter_count: int | None = None,
    ) -> tuple[ForwardResponse, Tensor]:
        """Solve the forward problem and materialize ``d resistance / d conductivity``.

        The default normal-quadrupole sensitivity omits the Robin boundary derivative. Pass
        ``include_robin_boundary_derivative=True, normal_sensitivity=False`` for the exact
        derivative of the reciprocal-averaged response. ``jacobian_cell_parameter_ids``
        accumulates the direct normal sensitivity into parameter columns.
        """

        fields, integrated, normal, reciprocal = self._measure(conductivity)
        direct = normal_sensitivity and not include_robin_boundary_derivative
        batch_size = self._batch_size(batch_size, direct)
        if direct:
            targets, count = self._target_cells(
                jacobian_cell_parameter_ids, jacobian_parameter_count
            )
            jacobian = self._normal_jacobian(fields, batch_size, targets, count)
        else:
            jacobian = self._adjoint_jacobian(
                fields,
                normal,
                reciprocal,
                batch_size,
                include_robin_boundary_derivative,
                normal_sensitivity,
            )
        return self._response(integrated, normal, reciprocal, currents), jacobian

    def jacobian(
        self,
        conductivity,
        *,
        batch_size: int | None = None,
        include_robin_boundary_derivative: bool = False,
        normal_sensitivity: bool = True,
    ) -> Tensor:
        """Materialize the resistance Jacobian (see :meth:`solve_with_jacobian`)."""

        return self.solve_with_jacobian(
            conductivity,
            batch_size=batch_size,
            include_robin_boundary_derivative=include_robin_boundary_derivative,
            normal_sensitivity=normal_sensitivity,
        )[1]

    def jacobian_columnwise(self, conductivity) -> Tensor:
        """Materialize the exact resistance Jacobian with one JVP per cell (for testing)."""

        basis = torch.eye(self.mesh.cell_count, dtype=FLOAT_DTYPE)
        return torch.stack([self.jvp(conductivity, column) for column in basis], dim=1)

    # -- resources ----------------------------------------------------------

    def close(self) -> None:
        """Drop cached fields and release cuDSS solver resources."""

        self._field_cache.clear()
        self._solver.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _terrain_cache_dir(cache_dir: str | Path | None) -> Path | None:
    cache_dir = (
        cache_dir
        if cache_dir is not None
        else os.environ.get("ADTLERT_TERRAIN_CACHE_DIR", "").strip()
    )
    return Path(cache_dir).expanduser() if cache_dir else None
