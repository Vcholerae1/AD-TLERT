from __future__ import annotations

from _misfit_runner import run_misfit_case

if __name__ == "__main__":
    raise SystemExit(
        run_misfit_case(
            misfit="log_data_difference_l2",
            output_name="log_data_difference_l2",
        )
    )
