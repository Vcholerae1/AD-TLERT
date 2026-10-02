# Architecture

ADTLERT computes 2.5D (and 3D) DC resistivity responses on the GPU and inverts them. Everything
differentiable runs through PyTorch tensors; sparse solves use NVIDIA cuDSS.

## Package map

| Package | Responsibility |
|---|---|
| `adtlert.mesh` | Triangle/quadrilateral (`Mesh`) and tetrahedral (`Mesh3D`) meshes: boundary topology, surface detection, point location, refinement. |
| `adtlert.fem` | Local element matrices (P1/P2 triangles, P1/P2 tetrahedra) and Robin boundary coefficients. |
| `adtlert.survey` | Electrodes, ABMN quadruples, analytic geometric factors. |
| `adtlert.forward` | The forward operators. `discretization` builds the element space and sparse pattern; `cudss` plans and reuses batched cuDSS solvers; `kernels` holds the Triton sensitivity kernel, the SDDMM wrapper and deterministic reductions; `ert2p5d`/`ert3d` are the operators; `autograd` exposes the exact VJP/JVP to PyTorch; `modeling` is the facade used by inversions. |
| `adtlert.inversion` | Log-space inversion. `config` (settings, results, validation), `inputs`, `state` (petrophysical maps), `terms` (`RegularizationTerms`), `objective` (misfits, coverage, line search), `matrix_free` (gradients), `optimizers`, `parameterized` (parameter mesh coarser than the solve mesh) and `core` (the engine and public entry points). |
| `adtlert.inr` | Coordinate networks (`networks`), the autograd physics bridge (`physics`) and the training loop (`train`). |
| `adtlert.workflows` | ParFlow slices -> terrain-following cases, `.dat` I/O, series forward runs, triangle inversion meshes. |

Modules in `adtlert.inversion` other than the public classes in `__init__` are internal: names
with a leading underscore are shared inside the package and are not API.

## The 2.5D solve

Each cosine-transform wavenumber solves the secondary-field problem

    A(sigma) u_s = A(1) u_p - A(sigma) (rho_src u_p),     u = u_s + rho_src u_p

with `u_p` the unit-resistivity primary field of every electrode. Flat surfaces use the analytic
half-space primary on the input mesh. Terrain solves on a refined auxiliary discretization whose
primary is computed numerically on a quadratic one. Both are the same code path, parameterized
by a `Discretization`. Total fields are cached by conductivity (an LRU of
`normal_field_cache_max_entries`), so a VJP right after a forward does not re-solve; the INR
trainers check that the cache holds at least one batch of timesteps.

## Derivatives

- `jacobian`/`solve_with_jacobian`: explicit sensitivities. The default "normal" sensitivity omits
  the Robin boundary derivative; `include_robin_boundary_derivative=True, normal_sensitivity=False`
  is the exact derivative of the reciprocity-averaged response.
- `vjp`/`jvp`: exact, matrix-free. `normal_vjp`/`normal_jvp`: the normal-sensitivity convention.
- The volume term of every adjoint gradient is the transpose of assembly: `sum lam phi^T` sampled on
  the sparsity pattern, contracted with the stiffness and mass templates.

## Conventions

- Precision: `FLOAT_DTYPE` is fixed at import (float32, or float64 with `ADTLERT_ENABLE_FLOAT64=1`).
  Terrain solves are always float64, which also switches Torch's default dtype for the process.
- Determinism: our own GPU reductions are bitwise repeatable (`GroupSum`, a fixed Triton summation
  order). cuDSS is not by default; see the changelog.
- Cell models are ordered like the mesh cells; time-lapse models are `(n_cells, n_times)`, data
  are `(n_times, n_measurements)`.

## Tests

`uv run pytest` runs everything the machine supports; `-m "not gpu"` runs the CPU-only subset
(this is what CI runs). The GPU suite checks adjoint consistency (dot tests), the exact Jacobian
against finite differences and the resistivity-scaling identity, the Triton kernel against an
einsum reference, deterministic reductions, the inversion loops and INR training.
