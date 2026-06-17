from __future__ import annotations

from _optimization_runner import run_optimizer_case


if __name__ == "__main__":
    raise SystemExit(
        run_optimizer_case(
            optimizer="gauss_newton_cgls",
            output_name="gauss_newton_cgls",
        )
    )
