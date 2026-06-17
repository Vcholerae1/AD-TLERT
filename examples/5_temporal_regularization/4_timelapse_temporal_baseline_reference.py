from __future__ import annotations

from _temporal_regularization_runner import run_temporal_regularization_case


if __name__ == "__main__":
    raise SystemExit(
        run_temporal_regularization_case(
            temporal_regularization_type="baseline_reference",
            temporal_regularization=10.0,
            output_name="baseline_reference",
        )
    )
