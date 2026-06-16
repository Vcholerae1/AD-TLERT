from __future__ import annotations

import json

from _optimization_runner import collect_all_metrics, write_aggregate_table


TARGET_OPTIMIZERS = (
    "gauss_newton_cgls",
    "lbfgs_b",
    "nonlinear_cg",
    "adam",
)


def main() -> int:
    rows = collect_all_metrics()
    if not rows:
        print("No optimization benchmark files found under result/6_optimization")
        return 1

    rows = [row for row in rows if str(row.get("optimizer", "")) in TARGET_OPTIMIZERS]
    if not rows:
        print(
            "No matching benchmark files found for target optimizers: "
            + ", ".join(TARGET_OPTIMIZERS)
        )
        return 1

    json_path, csv_path = write_aggregate_table(rows=rows)
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
