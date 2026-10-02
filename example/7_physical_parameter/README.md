# 7) Physical-Parameter Comparison (Water Content)

This folder compares two ways to obtain water content (`theta`) from time-lapse ERT inversion:

1. Invert **resistivity** first, then convert `rho -> saturation -> theta` using petrophysical relation.
2. Invert **saturation** directly with ADTLERT's differentiable petrophysical transform, then output `theta`.

## Scripts

- `1_timelapse_resistivity_then_convert.py`
  - Runs `examples/2_timelapsedERT_inversion_adtlert.py` with resistivity parameterization.
  - Post-processes `final_models.npy` into:
    - `final_saturation_from_resistivity_models.npy`
    - `final_water_content_from_resistivity_models.npy`

- `2_timelapse_ad_saturation.py`
  - Runs `examples/2_timelapsedERT_inversion_adtlert.py` with:
    - `--petrophysical-transform saturation`
  - Produces `final_water_content_models.npy` (or reconstructs it from saturation if needed).

- `0_run_all_physical_parameter.py`
  - Runs both cases in sequence.

- `3_compare_physical_parameter_results.ipynb`
  - Plot layout:
    - 3 rows x 4 cols maps (`Day 1 / 190 / 294 / 365`):
      - True water content
      - Water content from resistivity-conversion
      - Water content from AD saturation inversion
    - 1 row x 1 col yearly regolith series comparison.

## Default output

- `result/7_physical_parameter/resistivity_then_convert`
- `result/7_physical_parameter/ad_saturation`

## Quick start

Run both cases:

```bash
uv run python examples/7_physical_parameter/0_run_all_physical_parameter.py
```

Then open and run:

- `examples/7_physical_parameter/3_compare_physical_parameter_results.ipynb`
