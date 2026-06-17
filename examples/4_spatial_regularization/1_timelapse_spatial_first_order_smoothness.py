from __future__ import annotations

from _spatial_regularization_runner import run_spatial_regularization_case


if __name__ == "__main__":
    raise SystemExit(
        run_spatial_regularization_case(
            spatial_regularization="first_order_smoothness",
            output_name="first_order_smoothness",
        )
    )
