"""Survey primitives for ABMN-style ERT measurements."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from adtlert.utils.dtypes import FLOAT_DTYPE, INT_DTYPE


@dataclass(frozen=True)
class Survey:
    """Electrode geometry and ABMN measurement indexing."""

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
        if bool(torch.any(quads < 0)):
            raise ValueError("measurements contain negative electrode indices")
        if quads.numel() and bool(torch.any(quads >= positions.shape[0])):
            raise ValueError("measurements reference electrodes outside the survey")
        if bool(torch.any(torch.diff(torch.sort(quads, dim=1).values, dim=1) == 0)):
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

    def geometric_factors(self) -> torch.Tensor:
        """Analytic half-space geometric factors ``2 pi / (1/AM - 1/AN - 1/BM + 1/BN)``."""

        a, b, m, n = self.electrode_positions[self.measurements.long()].unbind(dim=1)
        distance = lambda p, q: torch.linalg.norm(p - q, dim=-1)  # noqa: E731
        return (
            2.0
            * math.pi
            / (
                1.0 / distance(a, m)
                - 1.0 / distance(a, n)
                - 1.0 / distance(b, m)
                + 1.0 / distance(b, n)
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
