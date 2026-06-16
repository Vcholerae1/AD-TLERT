from __future__ import annotations

from _spatial_regularization_runner import run_spatial_regularization_case


if __name__ == "__main__":
    raise SystemExit(
        run_spatial_regularization_case(
            spatial_regularization="model_difference_smoothness",
            output_name="model_difference_smoothness",
        )
    )
