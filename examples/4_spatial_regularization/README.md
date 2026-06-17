# Time-Lapse Spatial Regularization Comparison

This folder provides parallel inversion scripts for the same time-lapse ERT setup,
changing only the spatial regularization term.

## Cases

- `1_timelapse_spatial_first_order_smoothness.py`
- `2_timelapse_spatial_model_difference_smoothness.py`
- `3_timelapse_spatial_structural_prior.py`
- `4_timelapse_spatial_huber.py`

All cases call `examples/2_timelapsedERT_inversion_deepert.py` with identical
windowed settings and save to:

- `result/4_spatial_regularization/first_order_smoothness`
- `result/4_spatial_regularization/model_difference_smoothness`
- `result/4_spatial_regularization/structural_prior`
- `result/4_spatial_regularization/spatial_huber`

The `structural_prior` case uses the default structural-unit file resolved by
`examples/2_timelapsedERT_inversion_deepert.py`:
`parflow_models/petrophysical_models_2d/class2d_y2.npy`.

Fixed controls in this comparison:

- data misfit: `weighted_log_l2`
- temporal regularization: `temporal_smoothness`
- optimizer: `gauss_newton_cgls`

## Run

Run one case:

```bash
uv run python examples/4_spatial_regularization/1_timelapse_spatial_first_order_smoothness.py
```

Run all four:

```bash
uv run python examples/4_spatial_regularization/0_run_all_spatial_regularizations.py
```

Pass extra inversion args to override defaults, for example:

```bash
uv run python examples/4_spatial_regularization/0_run_all_spatial_regularizations.py --max-iterations 10 --no-plot
```
