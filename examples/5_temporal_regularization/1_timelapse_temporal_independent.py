from __future__ import annotations

from _temporal_regularization_runner import run_temporal_regularization_case


if __name__ == "__main__":
    raise SystemExit(
        run_temporal_regularization_case(
            temporal_regularization_type="temporal_smoothness",
            temporal_regularization=0.0,
            output_name="independent",
        )
    )
