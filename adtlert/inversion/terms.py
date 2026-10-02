"""Model-space regularization terms of the inversion objective.

Internal module: names with a leading underscore are shared inside the ``adtlert.inversion`` package.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from adtlert.inversion.config import _regularization_domain
from adtlert.inversion.inputs import _sensor_system
from adtlert.inversion.objective import _model_difference_operator, _reference_roughness
from adtlert.inversion.regularization import (
    build_spatial_regularization,
    build_temporal_regularization,
)
from adtlert.inversion.state import _diagonal, _regularization_value_and_derivative


class RegularizationTerms:
    """Model-space terms of the objective as scaled ``(A, b)`` least-squares blocks.

    Combines spatial smoothness (or damping), temporal smoothness, joint-frame coupling, and
    soft sensor constraints, optionally in the physical (e.g. water-content) domain.
    :meth:`blocks` linearizes every active term at the current states.
    """

    def __init__(
        self, forward, config, n_cells: int, n_times: int, *, single: bool
    ) -> None:
        self.forward, self.config, self.n_cells, self.n_times, self.single = (
            forward,
            config,
            n_cells,
            n_times,
            single,
        )
        self.joint_frame = (
            not single and config.temporal_regularization_mode == "joint_frame"
        )
        self.temporal_weight = 0.0 if single else config.temporal_regularization
        joint_frame, temporal_weight = self.joint_frame, self.temporal_weight
        spatial = build_spatial_regularization(config.spatial_regularization)
        temporal = build_temporal_regularization(config.temporal_regularization_type)
        if (
            joint_frame
            and temporal_weight > 0.0
            and temporal.name not in ("first_order_l2", "second_order_l2")
        ):
            raise ValueError(
                "robust temporal regularization is currently supported for temporal_regularization_mode='separate' only"
            )
        if (
            joint_frame
            and config.regularization > 0.0
            and spatial.name not in ("identity", "first_order")
        ):
            raise ValueError(
                "robust spatial regularization is currently supported for temporal_regularization_mode='separate' only"
            )

        roughness = spatial.matrix(forward, n_cells, z_weight=config.z_weight)
        roughness_all = sp.block_diag([roughness] * n_times, format="csr")
        sensor = None if single else _sensor_system(config, n_cells, n_times)
        physical = _regularization_domain(config) == "physical" and (
            config.regularization > 0.0
            or (not joint_frame and temporal_weight > 0.0)
            or sensor is not None
        )
        self.spatial, self.temporal, self.roughness = spatial, temporal, roughness
        self.roughness_all, self.sensor, self.physical = roughness_all, sensor, physical

    def blocks(
        self, states: np.ndarray, reference: np.ndarray | None
    ) -> list[tuple[sp.spmatrix, np.ndarray]]:
        """Scaled ``(A, b)`` blocks of every model-space term, linearized at ``states``."""

        if self.physical:
            domain, derivative = _regularization_value_and_derivative(
                states, self.config
            )
            projection = _diagonal(derivative)
            reference_domain = (
                None
                if reference is None
                else _regularization_value_and_derivative(reference, self.config)[0]
            )
        else:
            domain, derivative, projection, reference_domain = (
                states,
                np.ones_like(states),
                None,
                reference,
            )
        domain_vec = np.asarray(domain, dtype=float).reshape(-1, order="F")
        reference_vec = (
            None
            if reference_domain is None
            else np.asarray(reference_domain, dtype=float).reshape(-1, order="F")
        )

        def project(matrix, columns=None):
            if projection is None:
                return matrix
            chain = projection if columns is None else _diagonal(derivative[:, columns])
            return (matrix @ chain).tocsr()

        def linear_block(operator, scale, *, identity=False):
            current = operator @ domain_vec
            target = _reference_roughness(
                self.config.regularization_mode,
                current,
                operator,
                reference_vec,
                identity=identity,
            )
            return project(scale * operator), scale * (target - current)

        blocks = []
        scale = float(np.sqrt(self.config.regularization))
        if self.config.regularization > 0.0 and self.joint_frame:
            frame = [self.roughness_all]
            if self.temporal_weight > 0.0:
                frame.append(
                    self.temporal.matrix(
                        self.n_cells, self.n_times, scale=float(self.temporal_weight)
                    )
                )
            blocks.append(linear_block(sp.vstack(frame, format="csr"), scale))
        elif (
            self.config.regularization > 0.0
            and self.spatial.name == "model_difference_smoothness"
            and not self.single
        ):
            blocks.append(
                linear_block(
                    _model_difference_operator(self.roughness, self.n_times), scale
                )
            )
        elif self.config.regularization > 0.0 and self.spatial.name in (
            "identity",
            "first_order",
        ):
            blocks.append(
                linear_block(
                    self.roughness_all, scale, identity=self.spatial.name == "identity"
                )
            )
        elif (
            self.config.regularization > 0.0
        ):  # IRLS-reweighted robust self.spatial terms, one block per timestep
            per_time = []
            for t in range(self.n_times):
                domain_t = np.asarray(domain[:, t], dtype=float)
                current = self.roughness @ domain_t
                target = _reference_roughness(
                    self.config.regularization_mode, current, self.roughness,
                    None if reference_domain is None else np.asarray(reference_domain[:, t], dtype=float),
                )  # fmt: skip
                matrix, rhs = self.spatial.linearized_system(
                    self.forward,
                    domain_t,
                    self.n_cells,
                    reference_roughness=target,
                    scale=scale,
                    z_weight=self.config.z_weight,
                )
                per_time.append((project(matrix, t), rhs))
            blocks.append(
                (
                    sp.block_diag([m for m, _ in per_time], format="csr"),
                    np.concatenate([r for _, r in per_time]),
                )
            )

        if not self.joint_frame and self.temporal_weight > 0.0:
            options = {
                "threshold": self.config.active_time_threshold,
                "minimum_weight": self.config.active_time_minimum_weight,
            }
            matrix, rhs = self.temporal.linearized_system(
                domain_vec, self.n_cells, self.n_times, scale=float(np.sqrt(self.temporal_weight)),
                **(options if self.temporal.name == "active_time_constraint" else {}),
            )  # fmt: skip
            blocks.append((project(matrix), rhs))

        if self.sensor is not None:
            matrix, target = self.sensor
            sensor_scale = float(np.sqrt(self.config.sensor_constraint))
            blocks.append(
                (
                    sensor_scale * project(matrix),
                    sensor_scale * (target - matrix @ domain_vec),
                )
            )
        return blocks
