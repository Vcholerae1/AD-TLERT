# 7) Physical-Parameter Comparison (Water Content)

This folder compares two ways to obtain water content (`theta`) from time-lapse ERT inversion:

1. Invert **resistivity** first, then convert `rho -> saturation -> theta` using petrophysical relation.
2. Invert **water content** (`theta`) directly with Deepert's differentiable petrophysical transform.

## Scripts

- `1_timelapse_resistivity_then_convert.py`
  - Runs `examples/2_timelapsedERT_inversion_deepert.py` with resistivity parameterization.
  - Post-processes `final_models.npy` into:
    - `final_saturation_from_resistivity_models.npy`
    - `final_water_content_from_resistivity_models.npy`

- `2_timelapse_ad_theta.py`
  - Runs `examples/2_timelapsedERT_inversion_deepert.py` with:
    - `--petrophysical-transform water_content`
  - Produces `final_water_content_models.npy` directly.

- `0_run_all_physical_parameter.py`
  - Runs both cases in sequence.

- `3_compare_physical_parameter_results.ipynb`
  - Plot layout:
    - 3 rows x 4 cols maps (`Day 1 / 190 / 294 / 365`):
      - True water content
      - Water content from resistivity-conversion
      - Water content from AD theta inversion
    - 1 row x 1 col yearly regolith series comparison.

## Default output

- `result/7_physical_parameter/resistivity_then_convert`
- `result/7_physical_parameter/ad_theta`

## Quick start

Run both cases:

```bash
uv run python examples/7_physical_parameter/0_run_all_physical_parameter.py
```

Then open and run:

- `examples/7_physical_parameter/3_compare_physical_parameter_results.ipynb`
