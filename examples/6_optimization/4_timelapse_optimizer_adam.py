from __future__ import annotations

from _optimization_runner import run_optimizer_case


if __name__ == "__main__":
    raise SystemExit(
        run_optimizer_case(
            optimizer="adam",
            output_name="adam",
        )
    )
