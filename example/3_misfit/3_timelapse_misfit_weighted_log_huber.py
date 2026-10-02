from __future__ import annotations

from _misfit_runner import run_misfit_case

if __name__ == "__main__":
    raise SystemExit(
        run_misfit_case(
            misfit="weighted_log_huber",
            output_name="weighted_log_huber",
        )
    )
