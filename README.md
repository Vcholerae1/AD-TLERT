# AD-TLERT

Differentiable 2.5D ERT and tetrahedral 3D DC forward tooling on Torch.

The repository contains:

- `adtlert/`: the core numerical implementation.
- `example/single_time/`: ADTLERT/pyGIMLi forward examples and
  ADTLERT/pyGIMLi/ResIPy single-time inversion examples.
- `example/window_363/`: ADTLERT/pyGIMLi/ResIPy inversion examples for 365
  time steps using 3-step sliding windows (363 windows).
- `parflow_models/` and `resistivity_models_2d/`: input data used by the
  examples.


  

See [`example/README.md`](example/README.md) for commands and dependency
requirements.

## Install

ADTLERT is a CUDA library: every sparse solve runs on NVIDIA cuDSS and the
adjoint contractions run on the GPU. It requires Linux, an NVIDIA GPU, and a
driver that supports CUDA 12. CuPy, cuDSS, and nvmath-python are installed as
regular dependencies:

```bash
python -m pip install adtlert
```

Only one CuPy CUDA variant may be installed in an environment.

For development from a checkout, use `uv`. The terrain examples read ParFlow
PFB files and build Triangle inversion meshes:

```bash
uv sync --extra examples
```

## Core Forward API

```python
import torch

from adtlert.forward import ERTForward2p5D

forward = ERTForward2p5D.from_mesh_survey(mesh, survey)
conductivity = torch.as_tensor(1.0 / resistivity, dtype=torch.float32)
response = forward.solve(conductivity)

rhoa = response.apparent_resistivity
jacobian = forward.jacobian(conductivity)
```

For matrix-free differentiation, use the PyTorch autograd bridge. Its reverse
pass calls `forward.vjp(...)`, while PyTorch forward AD calls
`forward.jvp(...)`; no explicit Jacobian is materialized:

```python
from adtlert.forward import apparent_resistivity_autograd

conductivity = conductivity.requires_grad_()
rhoa = apparent_resistivity_autograd(conductivity, forward, currents=1.0)
loss = data_misfit(rhoa, observed_rhoa)
loss.backward()
conductivity_gradient = conductivity.grad
```

True 3D DC forward modelling uses a tetrahedral mesh and three-coordinate
electrodes. It follows pyGIMLi's 3D singularity-removal formulation, with an
analytic half-space primary field and a mixed far-field boundary:

```python
from adtlert.forward import ERTForward3D
from adtlert.mesh import Mesh3D
from adtlert.survey import Survey

mesh3d = Mesh3D.from_file("mesh3d.vtu")
survey3d = Survey.from_arrays(electrode_xyz, abmn)
forward3d = ERTForward3D.from_mesh_survey(mesh3d, survey3d)
response3d, jacobian3d = forward3d.solve_with_jacobian(conductivity3d)
```

For topography, use quadratic tetrahedra and numerical geometric factors. The
surface is inferred from upward-facing exterior triangles; pass
`surface_face_mask=` to `Mesh3D.from_arrays` when boundary classification must
be explicit:

```python
forward3d = ERTForward3D.from_mesh_survey(
    mesh3d,
    survey3d,
    element_order=2,
    geometric_factor_mode="auto",  # numerical for terrain, analytic when flat
)
```

For large explicit sensitivities, `jacobian(..., batch_size=32,
cell_batch_size=20000)` bounds temporary field memory. Assembly routing,
active AB/MN electrode sets, numerical geometric factors, sparse plans, and
fixed GPU right-hand sides are cached across inversion iterations.

The initial 3D implementation supports flat-surface half-space models,
four-node tetrahedral geometry, P1 or ten-node P2 basis functions, NVIDIA cuDSS
sparse factorization, analytic or terrain-aware numerical
geometric factors, exact adjoint cell sensitivities, first-order face-neighbor
regularization, and the existing single-time inversion API. Large-scale
matrix-free time-lapse inversion remains on the 3D roadmap.

`ERTForwardModeling` provides a small wrapper for mesh/data-like objects:

```python
from adtlert.forward import ERTForwardModeling

modeling = ERTForwardModeling(mesh=mesh, data=scheme)
log_rhoa, jac = modeling.forward_and_jacobian(log_resistivity, log_transform=True)
```

## Terrain Workflow

```python
import numpy as np

from adtlert.workflows import (
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

from adtlert.inversion import ERTInversion, InversionConfig

config = InversionConfig(
    max_iterations=8,
    data_std=0.05,
    regularization=1.0e-2,
    spatial_regularization="first_order",
    model_bounds=(10.0, 20000.0),
)

initial_model = np.full(forward.mesh.cell_count, np.median(observed_rhoa))
result = (
    ERTInversion(
        forward=forward,
        observed_data=observed_rhoa,
        config=config,
    )
    .setup()
    .run(initial_model)
)

final_model = result.final_model
predicted_rhoa = result.predicted_data
coverage = result.coverage
chi2_history = result.iteration_chi2
```

Windowed time-lapse inversion:

```python
from adtlert.inversion import WindowedTimeLapseERTInversion

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

result = (
    WindowedTimeLapseERTInversion(
        forward=forward,
        observed_data=observed_rhoa_by_time,
        config=config,
        window_size=3,
        window_step=1,
    )
    .setup()
    .run(initial_model)
)

final_models = result.final_models
```

## Validation

```bash
uv run pytest
uv run python -m compileall adtlert
uv build
```
