# Time-Lapse Temporal Regularization Comparison

This folder provides parallel inversion scripts for the same time-lapse ERT setup,
changing only the temporal regularization term.

## Cases

- `1_timelapse_temporal_independent.py`
- `2_timelapse_temporal_smoothness.py`
- `3_timelapse_temporal_active_time_constraint.py`
- `4_timelapse_temporal_baseline_reference.py`

All cases call `examples/2_timelapsedERT_inversion_deepert.py` with identical
windowed settings and save to:

- `result/5_temporal_regularization/independent`
- `result/5_temporal_regularization/temporal_smoothness`
- `result/5_temporal_regularization/active_time_constraint`
- `result/5_temporal_regularization/baseline_reference`

Fixed controls in this comparison:

- data misfit: `weighted_log_l2`
- spatial regularization: `first_order_smoothness`
- optimizer: `gauss_newton_cgls`

Temporal controls in each case:

- independent: `--temporal-regularization 0`
- temporal smoothness: `--temporal-regularization-type temporal_smoothness --temporal-regularization 10`
- active time constraint: `--temporal-regularization-type active_time_constraint --temporal-regularization 10`
- baseline reference: `--temporal-regularization-type baseline_reference --temporal-regularization 10`

## Run

Run one case:

```bash
uv run python examples/5_temporal_regularization/2_timelapse_temporal_smoothness.py
```

Run all four:

```bash
uv run python examples/5_temporal_regularization/0_run_all_temporal_regularizations.py
```

Pass extra inversion args to override defaults, for example:

```bash
uv run python examples/5_temporal_regularization/0_run_all_temporal_regularizations.py --max-iterations 10 --no-plot
```
