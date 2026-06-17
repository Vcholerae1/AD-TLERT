# Time-Lapse Optimization Algorithm Comparison

This folder benchmarks different nonlinear optimizers on the same time-lapse ERT inversion setup.

## Compared Optimizers

- `gauss_newton_cgls`
- `lbfgs_b`
- `nonlinear_cg`
- `adam`

Each case calls `examples/2_timelapsedERT_inversion_deepert.py` with fixed controls:

- inversion mode: `windowed`
- window: `size=3`, `step=1`
- data misfit: `weighted_log_l2`
- spatial regularization: `first_order_smoothness`
- temporal regularization: `temporal_smoothness` with `alpha=10`

## Metrics Recorded

For each optimizer, the runner stores:

- peak memory (`max_rss_kb`, from `/usr/bin/time -v`)
- total iterations (sum of window iterations)
- runtime (`runtime_wall_sec`)
- final objective function value (`final_objective_value`, data chi2)

Per-case benchmark file:

- `result/6_optimization/<optimizer>/optimization_benchmark.json`

Aggregate summary files:

- `result/6_optimization/optimization_benchmark_summary.json`
- `result/6_optimization/optimization_benchmark_summary.csv`

## Run

Run one optimizer:

```bash
uv run python examples/6_optimization/2_timelapse_optimizer_lbfgs_b.py
```

Run all optimizers and build aggregate summary:

```bash
uv run python examples/6_optimization/0_run_all_optimizers.py
```

Pass extra inversion args (forwarded to every optimizer case), for example:

```bash
uv run python examples/6_optimization/0_run_all_optimizers.py --max-iterations 10
```
