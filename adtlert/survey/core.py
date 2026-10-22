"""Survey primitives for ABMN-style ERT measurements."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from adtlert.utils.dtypes import FLOAT_DTYPE, INT_DTYPE

REMOTE_ELECTRODE = -1
"""ABMN index of an electrode at infinity (pole configurations); its potential is zero."""


def with_remote_slot(potentials: torch.Tensor) -> torch.Tensor:
    """Append the zero row and column that :meth:`Survey.measurement_slots` uses for a remote
    electrode to ``(..., sources, receivers)`` electrode potentials."""

    return torch.nn.functional.pad(potentials, (0, 1, 0, 1))


@dataclass(frozen=True)
class Survey:
    """Electrode geometry and ABMN measurement indexing.

    ``REMOTE_ELECTRODE`` (-1) in a measurement places that electrode at infinity, as in
    pole-dipole (B), dipole-pole (N) and pole-pole (B and N) arrays.
    """

    electrode_positions: torch.Tensor
    measurements: torch.Tensor

    @classmethod
    def from_arrays(cls, electrode_positions, measurements) -> Survey:
        """Build a survey from electrode coordinates and ABMN indices."""

        positions = torch.as_tensor(electrode_positions, dtype=FLOAT_DTYPE)
        quads = torch.as_tensor(measurements, dtype=INT_DTYPE)
        if positions.ndim != 2 or positions.shape[1] not in (2, 3):
            raise ValueError(
                "electrode_positions must have shape (num_electrodes, 2 or 3)"
            )
        if quads.ndim != 2 or quads.shape[1] != 4:
            raise ValueError("measurements must have shape (num_measurements, 4)")
        if bool(torch.any(quads < REMOTE_ELECTRODE)):
            raise ValueError(
                "measurements contain negative electrode indices other than "
                f"{REMOTE_ELECTRODE} (remote)"
            )
        if quads.numel() and bool(torch.any(quads >= positions.shape[0])):
            raise ValueError("measurements reference electrodes outside the survey")
        remote = quads == REMOTE_ELECTRODE
        if bool(torch.any(remote[:, :2].all(dim=1) | remote[:, 2:].all(dim=1))):
            raise ValueError(
                "each current (AB) and potential (MN) pair needs an electrode in the survey"
            )
        distinct = torch.where(remote, -1 - torch.arange(4, dtype=quads.dtype), quads)
        if bool(torch.any(torch.diff(torch.sort(distinct, dim=1).values, dim=1) == 0)):
            raise ValueError("each ABMN measurement must use four distinct electrodes")
        return cls(electrode_positions=positions, measurements=quads)

    @property
    def electrode_count(self) -> int:
        return int(self.electrode_positions.shape[0])

    @property
    def measurement_count(self) -> int:
        return int(self.measurements.shape[0])

    @property
    def dimension(self) -> int:
        """Spatial dimension of the electrode coordinates."""

        return int(self.electrode_positions.shape[1])

    def measurement_slots(self) -> torch.Tensor:
        """ABMN indices with a remote electrode mapped to ``electrode_count``.

        Per-electrode arrays padded with one zero entry then give the remote electrode its
        zero potential without special cases.
        """

        quads = self.measurements.long()
        return torch.where(quads == REMOTE_ELECTRODE, self.electrode_count, quads)

    def geometric_factors(self) -> torch.Tensor:
        """Analytic half-space geometric factors ``2 pi / (1/AM - 1/AN - 1/BM + 1/BN)``.

        Terms with a remote electrode vanish.
        """

        quads = self.measurements.long()
        points = self.electrode_positions[quads.clamp_min(0)]
        real = quads != REMOTE_ELECTRODE

        def inverse_distance(p: int, q: int) -> torch.Tensor:
            distance = torch.linalg.norm(points[:, p] - points[:, q], dim=-1)
            return torch.where(real[:, p] & real[:, q], 1.0 / distance, 0.0)

        return (
            2.0
            * math.pi
            / (
                inverse_distance(0, 2)
                - inverse_distance(0, 3)
                - inverse_distance(1, 2)
                + inverse_distance(1, 3)
            )
        )

    def apparent_resistivity(self, voltages, currents=1.0) -> torch.Tensor:
        """Convert measured voltages to apparent resistivity."""

        voltages = torch.as_tensor(voltages, dtype=FLOAT_DTYPE)
        return (
            self.geometric_factors()
            * voltages
            / torch.as_tensor(currents, dtype=FLOAT_DTYPE)
        )
