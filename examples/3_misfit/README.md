# Time-Lapse Misfit Comparison

This folder provides parallel inversion scripts for the same time-lapse ERT setup,
changing only the data misfit term.

## Cases

- `1_timelapse_misfit_weighted_log_l2.py`
- `2_timelapse_misfit_weighted_log_l1.py`
- `3_timelapse_misfit_weighted_log_huber.py`
- `4_timelapse_misfit_log_data_difference_l2.py`

All cases call `examples/2_timelapsedERT_inversion_deepert.py` with identical
windowed settings and save to:

- `result/3_misfit/weighted_log_l2`
- `result/3_misfit/weighted_log_l1`
- `result/3_misfit/weighted_log_huber`
- `result/3_misfit/log_data_difference_l2`

## Run

Run one case:

```bash
uv run python examples/3_misfit/1_timelapse_misfit_weighted_log_l2.py
```

Run all four:

```bash
uv run python examples/3_misfit/0_run_all_misfits.py
```

Pass extra inversion args to override defaults, for example:

```bash
uv run python examples/3_misfit/0_run_all_misfits.py --max-iterations 10 --no-plot
```
