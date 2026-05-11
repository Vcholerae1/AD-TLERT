# deepert

Differentiable 2.5D ERT forward and time-lapse inversion tooling in JAX.

This repository is intentionally small. It keeps:

- `deepert/`: the core numerical implementation.
- `1_forward_t04368(1).ipynb`: single T04368 forward example.
- `2_single_inversion_t04368.ipynb`: single T04368 inversion example.
- `1_forward_1year.ipynb`: 365-day forward example.
- `2_timelapsedERT_inversion.ipynb`: 365-day time-lapse inversion example.
- `2d_resistivity_model/` and `models_1year_1day/`: input data used by the examples.

## Install

```bash
uv sync
```

The terrain examples read ParFlow PFB files, so install the example extra when
running those paths:

```bash
uv sync --extra examples
```

## Core Forward API

```python
import jax.numpy as jnp

from deepert.forward import ERTForward2p5D

forward = ERTForward2p5D.from_mesh_survey(mesh, survey)
conductivity = jnp.asarray(1.0 / resistivity, dtype=jnp.float32)
response = forward.solve(conductivity)

rhoa = response.apparent_resistivity
jacobian = forward.jacobian(conductivity)
```

`ERTForwardModeling` provides a small wrapper for mesh/data-like objects:

```python
from deepert.forward import ERTForwardModeling

modeling = ERTForwardModeling(mesh=mesh, data=scheme)
log_rhoa, jac = modeling.forward_and_jacobian(log_resistivity, log_transform=True)
```

## Terrain Workflow

```python
import numpy as np

from deepert.workflows import (
    build_terrain_forward_case,
    discover_resistivity_slices,
    parse_pftcl,
    read_slope_x,
    run_terrain_forward,
    run_terrain_forward_series,
    save_terrain_forward_dat,
    save_terrain_forward_npz,
)

grid = parse_pftcl("models_1year_1day/sc2d_6.out.pftcl")
slope_x = read_slope_x("models_1year_1day/sc2d_6.out.slope_x.pfb", y_index=2)
rho_2d = np.load("2d_resistivity_model/resistivity2d_y2_t04368.npy")

case = build_terrain_forward_case(rho_2d, grid, slope_x, y_index=2)
rhoa = run_terrain_forward(case)
save_terrain_forward_dat("synthetic_ert_terrain_vardz_t04368.dat", case, rhoa)
save_terrain_forward_npz("synthetic_ert_terrain_vardz_t04368.npz", case, rhoa)

pairs = discover_resistivity_slices("2d_resistivity_model", y_index=2)
manifest, failures = run_terrain_forward_series(
    pairs,
    grid,
    slope_x,
    "result/timelapsedERT_forward",
    y_index=2,
)
```

## Inversion API

```python
import numpy as np

from deepert.inversion import ERTInversion, InversionConfig

config = InversionConfig(
    max_iterations=8,
    data_std=0.05,
    regularization=1.0e-2,
    spatial_regularization="first_order",
    model_bounds=(10.0, 20000.0),
)

initial_model = np.full(forward.mesh.cell_count, np.median(observed_rhoa))
result = ERTInversion(
    forward=forward,
    observed_data=observed_rhoa,
    config=config,
).setup().run(initial_model)

final_model = result.final_model
predicted_rhoa = result.predicted_data
coverage = result.coverage
chi2_history = result.iteration_chi2
```

Windowed time-lapse inversion:

```python
from deepert.inversion import WindowedTimeLapseERTInversion

config = InversionConfig(
    max_iterations=6,
    data_std=np.log1p(relative_error_by_time),
    regularization=50.0,
    temporal_regularization=10.0,
    temporal_regularization_mode="separate",
    spatial_regularization="first_order",
    model_bounds=(0.001, 1.0e4),
    max_log_step=None,
    line_search=True,
)

result = WindowedTimeLapseERTInversion(
    forward=forward,
    observed_data=observed_rhoa_by_time,
    config=config,
    window_size=3,
    window_step=1,
).setup().run(initial_model)

final_models = result.final_models
```

## Validation

```bash
uv run python -m compileall deepert
uv build
```
