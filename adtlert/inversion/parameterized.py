"""Forward wrapper whose inversion parameters live on a coarser mesh than the solve.

Internal module: names with a leading underscore are shared inside the ``adtlert.inversion`` package.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree

from adtlert.forward import ERTForward2p5D
from adtlert.forward.modeling import log_response_and_jacobian
from adtlert.inversion.config import ArrayLike
from adtlert.inversion.forward_calls import (
    _forward_and_jacobian_log,
    _forward_log_response,
)
from adtlert.inversion.inputs import _as_log_model
from adtlert.inversion.regularization import (
    cell_edges,
)
from adtlert.mesh import Mesh
from adtlert.utils.dtypes import FLOAT_DTYPE


class ParameterizedERTForward2p5D:
    """ERT forward wrapper with a full solve mesh and a smaller parameter mesh.

    ``background_mode`` sets forward cells outside the parameter domain from:
    ``"pygimli_prolongation"`` (pyGIMLi's neighbor-weighted resistivity prolongation),
    ``"nearest"`` (nearest parameter center), or ``"fixed_mean"`` (mean log-resistivity).
    """

    def __init__(
        self,
        forward: ERTForward2p5D,
        parameter_cell_ids: ArrayLike,
        *,
        regularization_mesh: Mesh | None = None,
        forward_cell_parameter_ids: ArrayLike | None = None,
        background_mode: str = "pygimli_prolongation",
    ) -> None:
        self.forward_operator = forward
        self.survey = forward.survey
        self.mesh = forward.mesh
        self.regularization_mesh = regularization_mesh
        cell_count = int(forward.mesh.cell_count)
        self.parameter_cell_ids = np.asarray(parameter_cell_ids, dtype=np.int32).ravel()
        if self.parameter_cell_ids.size == 0:
            raise ValueError("parameter_cell_ids must not be empty")
        if np.any(self.parameter_cell_ids < 0) or np.any(
            self.parameter_cell_ids >= cell_count
        ):
            raise ValueError(
                "parameter_cell_ids reference cells outside the forward mesh"
            )
        if np.unique(self.parameter_cell_ids).size != self.parameter_cell_ids.size:
            raise ValueError("parameter_cell_ids must be unique")
        if background_mode not in ("pygimli_prolongation", "nearest", "fixed_mean"):
            raise ValueError(
                "background_mode must be 'pygimli_prolongation', 'nearest', or 'fixed_mean'"
            )
        self.background_mode = background_mode

        if forward_cell_parameter_ids is None:
            ids = np.full(cell_count, -1, dtype=np.int32)
            ids[self.parameter_cell_ids] = np.arange(
                self.parameter_cell_ids.size, dtype=np.int32
            )
        else:
            ids = np.asarray(forward_cell_parameter_ids, dtype=np.int32).ravel()
            if ids.shape != (cell_count,):
                raise ValueError(
                    f"forward_cell_parameter_ids must have one entry per forward mesh cell ({ids.shape} != ({cell_count},))"
                )
            if np.any(ids < -1):
                raise ValueError(
                    "forward_cell_parameter_ids may only contain -1 or non-negative parameter ids"
                )
            active_ids = np.unique(ids[ids >= 0])
            if active_ids.size == 0:
                raise ValueError(
                    "forward_cell_parameter_ids must contain at least one active parameter cell"
                )
            if not np.array_equal(active_ids, np.arange(active_ids[-1] + 1)):
                raise ValueError(
                    "forward_cell_parameter_ids must use contiguous ids starting at 0"
                )
        self.forward_cell_parameter_ids = ids
        self._n_parameters = int(ids.max()) + 1
        if self.parameter_cell_ids.size != self._n_parameters:
            raise ValueError(
                "parameter_cell_ids must contain one representative forward cell per inversion parameter "
                f"({self.parameter_cell_ids.size} != {self._n_parameters})"
            )
        if not np.array_equal(
            ids[self.parameter_cell_ids], np.arange(self._n_parameters)
        ):
            raise ValueError(
                "parameter_cell_ids must be ordered representatives of forward_cell_parameter_ids"
            )
        if (
            regularization_mesh is not None
            and int(regularization_mesh.cell_count) != self._n_parameters
        ):
            raise ValueError(
                "regularization mesh cell count must match inversion parameter count "
                f"({regularization_mesh.cell_count} != {self._n_parameters})"
            )

        cells = np.arange(cell_count, dtype=np.int32)
        self._active_forward_mask = ids >= 0
        self._active_forward_cell_ids = cells[self._active_forward_mask]
        self._active_parameter_ids = ids[self._active_forward_cell_ids]
        self.background_cell_ids = cells[~self._active_forward_mask]
        self._background_parameter_ids = (
            self._nearest_parameters()
            if background_mode == "nearest"
            else np.empty(0, dtype=np.int32)
        )
        self._resistivity_prolongation_matrix = (
            self._prolongation() if background_mode == "pygimli_prolongation" else None
        )
        self._jacobian_projection = self._projection()
        self._torch_maps: dict[str, torch.Tensor] = {}

    @classmethod
    def from_mesh_survey(
        cls,
        mesh: Mesh,
        survey,
        parameter_cell_ids: ArrayLike,
        *,
        regularization_mesh: Mesh | None = None,
        background_mode: str = "pygimli_prolongation",
        **forward_kwargs: Any,
    ) -> ParameterizedERTForward2p5D:
        forward_cell_parameter_ids = forward_kwargs.pop(
            "forward_cell_parameter_ids", None
        )
        return cls(
            ERTForward2p5D.from_mesh_survey(mesh, survey, **forward_kwargs),
            parameter_cell_ids,
            regularization_mesh=regularization_mesh,
            forward_cell_parameter_ids=forward_cell_parameter_ids,
            background_mode=background_mode,
        )

    @property
    def cell_count(self) -> int:
        return self._n_parameters

    def close(self) -> None:
        self.forward_operator.close()

    def _nearest_parameters(self) -> np.ndarray:
        if self.background_cell_ids.size == 0:
            return np.empty(0, dtype=np.int32)
        centers = np.mean(
            np.asarray(self.mesh.nodes, dtype=float)[np.asarray(self.mesh.cells)],
            axis=1,
        )
        sums = np.zeros((self.cell_count, centers.shape[1]))
        counts = np.zeros(self.cell_count)
        np.add.at(
            sums, self._active_parameter_ids, centers[self._active_forward_cell_ids]
        )
        np.add.at(counts, self._active_parameter_ids, 1.0)
        if np.any(counts <= 0.0):
            raise ValueError(
                "each inversion parameter must own at least one forward mesh cell"
            )
        _, nearest = cKDTree(sums / counts[:, None]).query(
            centers[self.background_cell_ids]
        )
        return np.asarray(nearest, dtype=np.int32)

    def _projection(self) -> sp.csr_matrix:
        """``d log(rho_forward) / d log(rho_parameter)`` used to project Jacobian columns."""

        rows, cols = [self._active_forward_cell_ids], [self._active_parameter_ids]
        data = [np.ones(self._active_forward_cell_ids.size)]
        background = self.background_cell_ids
        if background.size and self.background_mode == "nearest":
            rows.append(background)
            cols.append(self._background_parameter_ids)
            data.append(np.ones(background.size))
        elif background.size and self.background_mode == "fixed_mean":
            rows.append(np.repeat(background, self.cell_count))
            cols.append(
                np.tile(np.arange(self.cell_count, dtype=np.int32), background.size)
            )
            data.append(
                np.full(background.size * self.cell_count, 1.0 / self.cell_count)
            )
        shape = (int(self.mesh.cell_count), self.cell_count)
        return sp.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=shape,
        ).tocsr()

    def _prolongation(self) -> sp.csr_matrix:
        """Replicate pyGIMLi's marker prolongation: background rows are neighbor-weighted averages."""

        cell_count = int(self.mesh.cell_count)
        matrix = np.zeros((cell_count, self.cell_count))
        matrix[self._active_forward_cell_ids, self._active_parameter_ids] = 1.0
        if self.background_cell_ids.size == 0:
            return sp.csr_matrix(matrix)

        nodes = np.asarray(self.mesh.nodes, dtype=float)
        edge_cells: dict[tuple[int, ...], list[int]] = {}
        for cell_id, cell in enumerate(np.asarray(self.mesh.cells, dtype=np.int32)):
            for edge in cell_edges(cell):
                edge_cells.setdefault(tuple(sorted(edge)), []).append(cell_id)
        neighbors: list[list[tuple[int, float]]] = [[] for _ in range(cell_count)]
        for edge, owners in edge_cells.items():
            tangent = nodes[edge[1]] - nodes[edge[0]]
            length = float(np.linalg.norm(tangent))
            if len(owners) != 2 or length <= 0.0:
                continue
            weight = abs(float(tangent[1])) / length + 1.0e-6
            neighbors[owners[0]].append((owners[1], weight))
            neighbors[owners[1]].append((owners[0], weight))

        known = self._active_forward_mask.copy()
        unknown = set(self.background_cell_ids.tolist())
        while unknown:
            assignments = []
            for cell_id in sorted(unknown):
                row, total = np.zeros(self.cell_count), 0.0
                for neighbor, weight in neighbors[cell_id]:
                    if known[neighbor]:
                        row += matrix[neighbor] * weight
                        total += weight
                if total > 1.0e-8:
                    assignments.append((cell_id, row / total))
            if not assignments:
                raise ValueError(
                    "could not prolongate background forward cells from active parameter cells"
                )
            for cell_id, row in assignments:
                matrix[cell_id], known[cell_id] = row, True
                unknown.remove(cell_id)
        return sp.csr_matrix(matrix)

    def _full_log_model(self, log_resistivity: ArrayLike) -> np.ndarray:
        return self._full_log_model_and_projection(log_resistivity)[0]

    def _full_log_model_and_projection(
        self, log_resistivity: ArrayLike
    ) -> tuple[np.ndarray, sp.csr_matrix]:
        parameter_log = np.asarray(log_resistivity, dtype=float).ravel()
        if parameter_log.shape != (self.cell_count,):
            raise ValueError(f"log_resistivity must have shape ({self.cell_count},)")
        if self.background_mode == "pygimli_prolongation":
            full = np.asarray(
                self._resistivity_prolongation_matrix @ np.exp(parameter_log),
                dtype=float,
            ).ravel()
            if np.any(full <= 0.0) or not np.all(np.isfinite(full)):
                raise ValueError(
                    "prolongated forward resistivity contains non-positive or non-finite values"
                )
            return np.log(full), self._jacobian_projection
        full_log = np.empty(int(self.mesh.cell_count))
        full_log[self._active_forward_cell_ids] = parameter_log[
            self._active_parameter_ids
        ]
        if self.background_cell_ids.size:
            full_log[self.background_cell_ids] = (
                parameter_log[self._background_parameter_ids]
                if self.background_mode == "nearest"
                else float(np.mean(parameter_log))
            )
        return full_log, self._jacobian_projection

    def log_model_to_full(self, log_parameters: torch.Tensor) -> torch.Tensor:
        """Differentiable ``(..., n_parameters) -> (..., n_forward_cells)`` log-resistivity map.

        The torch counterpart of :meth:`_full_log_model` (float64, on the input's device),
        used to backpropagate through the background extension.
        """

        exponential = self.background_mode == "pygimli_prolongation"
        device = log_parameters.device
        if str(device) not in self._torch_maps:
            matrix = (
                self._resistivity_prolongation_matrix
                if exponential
                else self._jacobian_projection
            ).tocoo()
            self._torch_maps[str(device)] = torch.sparse_coo_tensor(
                np.vstack((matrix.row, matrix.col)),
                matrix.data,
                matrix.shape,
                dtype=torch.float64,
                device=device,
            ).coalesce()
        flat = log_parameters.to(torch.float64).reshape(-1, log_parameters.shape[-1]).T
        full = torch.sparse.mm(
            self._torch_maps[str(device)], torch.exp(flat) if exponential else flat
        ).T
        if exponential:
            full = torch.log(full)
        return full.reshape(*log_parameters.shape[:-1], full.shape[-1])

    def forward_and_jacobian(
        self,
        resistivity_model: ArrayLike,
        log_transform: bool = True,
        *,
        include_robin_boundary_derivative: bool | None = None,
        normal_sensitivity: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        log_model = _as_log_model(
            resistivity_model, self.cell_count, log_transform, "resistivity_model"
        )
        full_log, projection = self._full_log_model_and_projection(log_model)
        robin = bool(include_robin_boundary_derivative)
        normal = True if normal_sensitivity is None else bool(normal_sensitivity)
        if self.background_mode == "pygimli_prolongation" and normal and not robin:
            # Fused aggregation of forward-cell sensitivities into parameter columns.
            return log_response_and_jacobian(
                self.forward_operator,
                torch.as_tensor(np.exp(-full_log), dtype=FLOAT_DTYPE),
                chain=np.exp(-log_model),
                jacobian_cell_parameter_ids=self.forward_cell_parameter_ids,
                jacobian_parameter_count=self.cell_count,
            )
        predicted, full_jacobian = _forward_and_jacobian_log(
            self.forward_operator, full_log, robin=robin, normal=normal
        )
        return predicted, np.asarray((projection.T @ full_jacobian.T).T, dtype=float)

    def response(self, resistivity_model: ArrayLike) -> np.ndarray:
        return self.forward(resistivity_model, log_transform=False)

    def forward(
        self, resistivity_model: ArrayLike, log_transform: bool = True
    ) -> np.ndarray:
        log_model = _as_log_model(
            resistivity_model, self.cell_count, log_transform, "resistivity_model"
        )
        response = _forward_log_response(
            self.forward_operator, self._full_log_model(log_model)
        )
        return response if log_transform else np.exp(response)
