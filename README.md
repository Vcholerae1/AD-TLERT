# deepert

Differentiable 2.5D ERT forward and time-lapse inversion tooling on Torch.

The repository contains:

- `deepert/`: the core numerical implementation.
- `example/single_time/`: Deepert/pyGIMLi forward examples and
  Deepert/pyGIMLi/ResIPy single-time inversion examples.
- `example/window_363/`: Deepert/pyGIMLi/ResIPy inversion examples for 365
  time steps using 3-step sliding windows (363 windows).
- `parflow_models/` and `resistivity_models_2d/`: input data used by the
  examples.

See [`example/README.md`](example/README.md) for commands and backend
requirements.

## Install

```bash
uv sync
```

The terrain examples read ParFlow PFB files and build Triangle inversion
meshes, so install the example extra when running those paths:

```bash
uv sync --extra examples
```

## Core Forward API

```python
import torch

from deepert.forward import ERTForward2p5D

forward = ERTForward2p5D.from_mesh_survey(mesh, survey)
conductivity = torch.as_tensor(1.0 / resistivity, dtype=torch.float32)
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

grid = parse_pftcl("parflow_models/sc2d_6.out.pftcl")
slope_x = read_slope_x("parflow_models/sc2d_6.out.slope_x.pfb", y_index=2)
rho_2d = np.load("resistivity_models_2d/resistivity2d_y2_t04536.npy")

case = build_terrain_forward_case(rho_2d, grid, slope_x, y_index=2)
rhoa = run_terrain_forward(case)
save_terrain_forward_dat("synthetic_ert_terrain_vardz_t04536.dat", case, rhoa)
save_terrain_forward_npz("synthetic_ert_terrain_vardz_t04536.npz", case, rhoa)

pairs = discover_resistivity_slices("resistivity_models_2d", y_index=2)
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
uv run pytest
uv run python -m compileall deepert
uv build
```
