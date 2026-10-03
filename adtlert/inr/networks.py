"""Compact deterministic coordinate networks for fast ERT inversion."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

FREQUENCY_SPACINGS = ("log_uniform", "octave")


def fourier_frequency_bank(levels: int, spacing: str = "log_uniform") -> torch.Tensor:
    """Angular frequencies for coordinates normalized to ``[-1, 1]``.

    ``"octave"`` (``pi * 2**k``) gives every level above the first a common
    period of 1, half the normalized domain. Poorly constrained cells then
    receive shifted replicas of the resolved structure. ``"log_uniform"``
    keeps the same highest frequency, starts at ``pi / 2`` (period 4, twice
    the domain), and uses an irrational level ratio, so no subset of levels
    shares an in-domain period.
    """

    if levels < 0:
        raise ValueError("Fourier levels must be non-negative")
    if spacing not in FREQUENCY_SPACINGS:
        raise ValueError(f"frequency_spacing must be one of {FREQUENCY_SPACINGS}")
    if levels == 0:
        return torch.empty(0, dtype=torch.float32)
    if spacing == "octave":
        multipliers = 2.0 ** np.arange(levels, dtype=float)
    else:
        multipliers = np.geomspace(0.5, 2.0 ** (levels - 1), levels)
    return math.pi * torch.as_tensor(multipliers, dtype=torch.float32)


class MultiscaleFourierEncoder(nn.Module):
    """Deterministic Fourier encoding with separate spatial/time bandwidths.

    A fixed frequency bank removes the expensive architecture sweep used by
    random Fourier features. Training code precomputes this module's output
    once because inversion coordinates are fixed. Frequencies are persistent
    buffers, so loading a checkpoint restores the bank it was trained with.
    """

    def __init__(
        self,
        input_dimensions: int,
        spatial_levels: int = 4,
        temporal_levels: int = 3,
        frequency_spacing: str = "log_uniform",
    ) -> None:
        super().__init__()
        if input_dimensions not in (1, 2, 3):
            raise ValueError(
                "input_dimensions must be 1 (time), 2 (space), or 3 (space-time)"
            )
        if spatial_levels < 0 or temporal_levels < 0:
            raise ValueError("Fourier levels must be non-negative")
        self.input_dimensions = int(input_dimensions)
        self.spatial_levels = int(spatial_levels if input_dimensions >= 2 else 0)
        self.temporal_levels = int(temporal_levels if input_dimensions in (1, 3) else 0)
        self.frequency_spacing = str(frequency_spacing)
        for name, levels in (
            ("spatial", self.spatial_levels),
            ("temporal", self.temporal_levels),
        ):
            frequencies = fourier_frequency_bank(levels, self.frequency_spacing)
            self.register_buffer(f"{name}_frequencies", frequencies, persistent=True)
            self.register_buffer(
                f"{name}_level_weights",
                torch.ones(levels, dtype=torch.float32),
                persistent=True,
            )

    @property
    def output_dimensions(self) -> int:
        spatial_features = (
            2 * 2 * self.spatial_levels if self.input_dimensions >= 2 else 0
        )
        temporal_features = (
            2 * self.temporal_levels if self.input_dimensions in (1, 3) else 0
        )
        return self.input_dimensions + spatial_features + temporal_features

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.shape[-1] != self.input_dimensions:
            raise ValueError(
                f"coordinates must end with {self.input_dimensions} dimensions"
            )
        features = [coordinates]
        if self.input_dimensions >= 2 and self.spatial_levels:
            spatial_phase = coordinates[..., :2, None] * self.spatial_frequencies
            features.extend(
                (
                    torch.sin(spatial_phase).flatten(-2),
                    torch.cos(spatial_phase).flatten(-2),
                )
            )
        if self.input_dimensions in (1, 3) and self.temporal_levels:
            temporal_index = 0 if self.input_dimensions == 1 else 2
            temporal_phase = (
                coordinates[..., temporal_index : temporal_index + 1]
                * self.temporal_frequencies
            )
            features.extend((torch.sin(temporal_phase), torch.cos(temporal_phase)))
        return torch.cat(features, dim=-1)

    def feature_mask(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Return feature weights in the exact order produced by ``forward``."""

        parts = [torch.ones(self.input_dimensions, dtype=dtype, device=device)]
        if self.input_dimensions >= 2:
            spatial = self.spatial_level_weights.to(device=device, dtype=dtype)
            for _ in range(4):
                parts.append(spatial)
        if self.input_dimensions in (1, 3):
            temporal = self.temporal_level_weights.to(device=device, dtype=dtype)
            parts.extend((temporal, temporal))
        return torch.cat(parts)

    def weigh(self, features: torch.Tensor) -> torch.Tensor:
        """Scale encoded ``features`` by the current progressive level weights."""

        return features * self.feature_mask(
            dtype=features.dtype, device=features.device
        )

    @staticmethod
    def _progressive_weights(
        levels: int, start_levels: int, progress: float
    ) -> torch.Tensor:
        if levels == 0:
            return torch.empty(0, dtype=torch.float32)
        start = min(max(int(start_levels), 0), levels)
        if start == levels:
            return torch.ones(levels, dtype=torch.float32)
        position = float(np.clip(progress, 0.0, 1.0)) * (levels - start)
        indices = torch.arange(levels, dtype=torch.float32)
        weights = torch.clamp(position - (indices - start), 0.0, 1.0)
        weights[:start] = 1.0
        return 0.5 - 0.5 * torch.cos(math.pi * weights)

    @torch.no_grad()
    def set_progressive_levels(
        self,
        progress: float,
        *,
        spatial_start_levels: int = 2,
        temporal_start_levels: int = 1,
    ) -> None:
        """Smoothly unlock high-frequency features from coarse to fine."""

        self.spatial_level_weights.copy_(
            self._progressive_weights(
                self.spatial_levels, spatial_start_levels, progress
            )
        )
        self.temporal_level_weights.copy_(
            self._progressive_weights(
                self.temporal_levels, temporal_start_levels, progress
            )
        )


def _mlp(
    input_width: int,
    hidden_width: int,
    hidden_layers: int,
    *,
    zero_output: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(int(hidden_layers)):
        layers.extend((nn.Linear(int(input_width), int(hidden_width)), nn.SiLU()))
        input_width = hidden_width
    output = nn.Linear(int(input_width), 1)
    if zero_output:
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
    return nn.Sequential(*layers, output)


class _BoundedLogResistivity(nn.Module):
    """Maps a residual to ``log rho``: ``log rho0 + r``, or a shifted sigmoid inside ``resistivity_bounds``."""

    def _init_output(
        self, initial_resistivity: float, resistivity_bounds: tuple[float, float] | None
    ) -> None:
        if not np.isfinite(initial_resistivity) or initial_resistivity <= 0.0:
            raise ValueError("initial_resistivity must be positive and finite")
        self.initial_resistivity = float(initial_resistivity)
        self.resistivity_bounds = (
            None
            if resistivity_bounds is None
            else tuple(float(v) for v in resistivity_bounds)
        )
        self.bounded_output = resistivity_bounds is not None

        def buffer(name: str, value: float) -> None:
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32))

        buffer("initial_log_resistivity", math.log(initial_resistivity))
        if resistivity_bounds is None:
            log_lower = log_upper = float("nan")
            initial_logit = 0.0
        else:
            lower, upper = self.resistivity_bounds
            if not 0.0 < lower < initial_resistivity < upper:
                raise ValueError(
                    "resistivity_bounds must strictly contain initial_resistivity"
                )
            log_lower, log_upper = math.log(lower), math.log(upper)
            fraction = (math.log(initial_resistivity) - log_lower) / (
                log_upper - log_lower
            )
            initial_logit = math.log(fraction / (1.0 - fraction))
        buffer("log_lower_bound", log_lower)
        buffer("log_upper_bound", log_upper)
        buffer("initial_logit", initial_logit)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.forward_encoded(self.encode(coordinates))

    def _encoders(self) -> tuple[MultiscaleFourierEncoder, ...]:
        raise NotImplementedError

    def set_progressive_levels(
        self,
        progress: float,
        *,
        spatial_start_levels: int = 2,
        temporal_start_levels: int = 1,
    ) -> None:
        """Unlock the Fourier levels of every encoder (levels an encoder lacks are ignored)."""

        for encoder in self._encoders():
            encoder.set_progressive_levels(
                progress,
                spatial_start_levels=spatial_start_levels,
                temporal_start_levels=temporal_start_levels,
            )

    @classmethod
    def from_configuration(cls, configuration: dict[str, object]):
        """Rebuild a network from :meth:`configuration`; ``load_state_dict`` then restores it."""

        return cls(**configuration)

    def _output(self, residual: torch.Tensor) -> torch.Tensor:
        if not self.bounded_output:
            return self.initial_log_resistivity + residual
        fraction = torch.sigmoid(self.initial_logit + residual)
        return (
            self.log_lower_bound
            + (self.log_upper_bound - self.log_lower_bound) * fraction
        )


class MultiscaleINR(_BoundedLogResistivity):
    """Small bounded residual MLP mapping coordinates to log resistivity.

    The final layer is zero-initialized, so the initial field is exactly the
    supplied homogeneous resistivity. Physical bounds use a shifted sigmoid
    rather than clipping, retaining useful gradients at every optimization
    step.
    """

    def __init__(
        self,
        *,
        input_dimensions: int = 2,
        spatial_levels: int = 4,
        temporal_levels: int = 3,
        hidden_width: int = 64,
        hidden_layers: int = 2,
        initial_resistivity: float = 100.0,
        resistivity_bounds: tuple[float, float] | None = (10.0, 20000.0),
        frequency_spacing: str = "log_uniform",
    ) -> None:
        super().__init__()
        if hidden_width < 1 or hidden_layers < 1:
            raise ValueError("hidden_width and hidden_layers must be positive")
        self.hidden_width = int(hidden_width)
        self.hidden_layers = int(hidden_layers)
        self.encoder = MultiscaleFourierEncoder(
            input_dimensions=input_dimensions,
            spatial_levels=spatial_levels,
            temporal_levels=temporal_levels,
            frequency_spacing=frequency_spacing,
        )
        self.mlp = _mlp(
            self.encoder.output_dimensions,
            hidden_width,
            hidden_layers,
            zero_output=True,
        )
        self._init_output(initial_resistivity, resistivity_bounds)

    @property
    def input_dimensions(self) -> int:
        return self.encoder.input_dimensions

    def encode(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.encoder(coordinates)

    def _encoders(self) -> tuple[MultiscaleFourierEncoder, ...]:
        return (self.encoder,)

    def configuration(self) -> dict[str, object]:
        """Return the constructor settings required for a compatible checkpoint."""

        return {
            "input_dimensions": self.input_dimensions,
            "spatial_levels": self.encoder.spatial_levels,
            "temporal_levels": self.encoder.temporal_levels,
            "hidden_width": self.hidden_width,
            "hidden_layers": self.hidden_layers,
            "initial_resistivity": self.initial_resistivity,
            "resistivity_bounds": self.resistivity_bounds,
            "frequency_spacing": self.encoder.frequency_spacing,
        }

    def forward_encoded(self, encoded_coordinates: torch.Tensor) -> torch.Tensor:
        return self._output(
            self.mlp(self.encoder.weigh(encoded_coordinates)).squeeze(-1)
        )


class JointSpatioTemporalINR(MultiscaleINR):
    """A single Fourier-feature MLP representing ``log rho(x, z, t)``."""

    def __init__(self, **kwargs: object) -> None:
        input_dimensions = int(kwargs.pop("input_dimensions", 3))
        if input_dimensions != 3:
            raise ValueError("JointSpatioTemporalINR requires input_dimensions=3")
        super().__init__(input_dimensions=3, **kwargs)


class DualNetworkINR(_BoundedLogResistivity):
    """Rank-one space/time INR with independent Fourier-feature networks.

    The model is ``log rho(x, z, t) = bounded(log rho0 + S(x, z) * A(t))``.
    Only the temporal output layer is zero-initialized. The initial model is
    therefore exactly homogeneous while the output layer retains a useful
    gradient through the already-varying spatial mask.
    """

    def __init__(
        self,
        *,
        spatial_levels: int = 5,
        temporal_levels: int = 4,
        spatial_hidden_width: int = 64,
        temporal_hidden_width: int = 32,
        spatial_hidden_layers: int = 2,
        temporal_hidden_layers: int = 2,
        initial_resistivity: float = 100.0,
        resistivity_bounds: tuple[float, float] | None = (10.0, 1000.0),
        frequency_spacing: str = "log_uniform",
    ) -> None:
        super().__init__()
        if (
            min(
                spatial_hidden_width,
                temporal_hidden_width,
                spatial_hidden_layers,
                temporal_hidden_layers,
            )
            < 1
        ):
            raise ValueError("hidden widths and layer counts must be positive")
        self.spatial_hidden_width = int(spatial_hidden_width)
        self.temporal_hidden_width = int(temporal_hidden_width)
        self.spatial_hidden_layers = int(spatial_hidden_layers)
        self.temporal_hidden_layers = int(temporal_hidden_layers)
        self.spatial_encoder = MultiscaleFourierEncoder(
            2,
            spatial_levels=spatial_levels,
            temporal_levels=0,
            frequency_spacing=frequency_spacing,
        )
        self.temporal_encoder = MultiscaleFourierEncoder(
            1,
            spatial_levels=0,
            temporal_levels=temporal_levels,
            frequency_spacing=frequency_spacing,
        )
        self.spatial_mlp = _mlp(
            self.spatial_encoder.output_dimensions,
            spatial_hidden_width,
            spatial_hidden_layers,
        )
        self.temporal_mlp = _mlp(
            self.temporal_encoder.output_dimensions,
            temporal_hidden_width,
            temporal_hidden_layers,
            zero_output=True,
        )
        self._init_output(initial_resistivity, resistivity_bounds)

    @property
    def input_dimensions(self) -> int:
        return 3

    def encode(self, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
            raise ValueError(
                "DualNetworkINR coordinates must have shape (time, cell, 3)"
            )
        spatial_coordinates = coordinates[0, :, :2]
        temporal_coordinates = coordinates[:, 0, 2:3]
        return self.spatial_encoder(spatial_coordinates), self.temporal_encoder(
            temporal_coordinates
        )

    def _encoders(self) -> tuple[MultiscaleFourierEncoder, ...]:
        return self.spatial_encoder, self.temporal_encoder

    def configuration(self) -> dict[str, object]:
        return {
            "spatial_levels": self.spatial_encoder.spatial_levels,
            "temporal_levels": self.temporal_encoder.temporal_levels,
            "spatial_hidden_width": self.spatial_hidden_width,
            "temporal_hidden_width": self.temporal_hidden_width,
            "spatial_hidden_layers": self.spatial_hidden_layers,
            "temporal_hidden_layers": self.temporal_hidden_layers,
            "initial_resistivity": self.initial_resistivity,
            "resistivity_bounds": self.resistivity_bounds,
            "frequency_spacing": self.spatial_encoder.frequency_spacing,
        }

    def forward_encoded(
        self, encoded_coordinates: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        spatial_encoded, temporal_encoded = encoded_coordinates
        spatial_features = self.spatial_encoder.weigh(spatial_encoded)
        temporal_features = self.temporal_encoder.weigh(temporal_encoded)
        spatial_mask = torch.sigmoid(self.spatial_mlp(spatial_features).squeeze(-1))
        temporal_amplitude = self.temporal_mlp(temporal_features).squeeze(-1)
        return self._output(temporal_amplitude[:, None] * spatial_mask[None, :])


__all__ = [
    "FREQUENCY_SPACINGS",
    "DualNetworkINR",
    "JointSpatioTemporalINR",
    "MultiscaleFourierEncoder",
    "MultiscaleINR",
    "fourier_frequency_bank",
]
