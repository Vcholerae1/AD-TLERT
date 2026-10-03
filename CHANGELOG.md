# Changelog

## 0.2.1 - 2026-10-04

### Added

- `ERTForward2p5D.resistance_series` and `vjp_series`; `apparent_resistivity_autograd` accepts a
  `(n_steps, n_cells)` series. Uncached models are solved together, up to `series_batch_steps`
  (new constructor option, default 8) models per cuDSS batch of `steps * W` systems.
- `BatchedSolver.solve(..., factorization=...)` names the factorized matrices. The adjoint (and
  tangent) solve of a batch reuses the forward factorization when the solver still holds it;
  any other solve in between invalidates it, so a stale factorization is never used.

### Changed

- `ERTForward2p5D` keeps its fields on the GPU: assembly, the secondary-field right-hand side,
  wavenumber integration, the measurement maps and the adjoint right-hand side run there, and
  only per-datum or per-cell results return to the host. The field cache now holds GPU tensors
  (one `(W, E, dofs)` field and the operator values per entry). Public return types and devices
  are unchanged.
- Transfer resistances are read from the `(E, E)` electrode potentials instead of contracting the
  fields with `(D, dofs)` receiver rows (about `2D / E` times fewer FP64 flops); the adjoint
  scatters cotangents onto electrode pairs with `GroupSum`.
- `matrix_free_log_rhoa_series` (INR physics) solves the timesteps of a call as one series.
- Terrain INR step (1250 cells, 5151 DOFs, 17 wavenumbers, 48 electrodes, 3 timesteps forward +
  backward, RTX 4070): 400 ms -> 93 ms. Host-device copies 76 ms -> 0.1 ms; six numeric
  factorizations (27 ms) -> one batched factorization (5 ms).
- GPU assembly, the operator product and the measurement-map adjoint use fixed-order reductions
  (`GroupSum`, padded rows), so they are bitwise repeatable. Results agree with 0.2.0 to within
  cuDSS run-to-run noise.

## 0.2.0 - 2026-10-03

ADTLERT is now a CUDA library. 0.1.x had a CPU/SciPy build; the two are different products.

### Breaking

- Requires Linux and an NVIDIA GPU (CUDA 12). cuDSS and nvmath-python are regular dependencies;
  the `cuda12` extra and CuPy are gone (the fused sensitivity kernel is written in Triton).
- nvmath-python >= 1.0 (cuDSS 0.8) is required; it solves terrain systems an order of magnitude faster than 0.8.
- Removed the SciPy solver backend and the `linear_solver_backend` argument of `ERTForward2p5D`,
  `ERTForward3D`, `ERTForwardModeling`, `MappedERTForwardModeling` and the terrain workflows.
- Removed `adtlert.utils.torch_runtime` (the JAX-port shim that patched `torch.Tensor`) and the
  unused sparse-assembly API: `assemble_global_bcoo`, `assemble_helmholtz_operator`,
  `build_coo_routing*`, `build_boundary_routing*`, `assemble_boundary_bcoo`, `BCOO`, `CSR`.
- `INRConfig.require_cuda` is gone (CUDA is always required); `prepare_cuda_forward` no longer takes it.
- `INRConfig` and `INRResult` are grouped (see "Migrating INR code" below); the old flat keyword
  arguments and result attributes are gone, with no compatibility aliases.
- Examples switch Torch to float64 with `torch.set_default_dtype` after importing adtlert
  (`FLOAT_DTYPE` is fixed at import; set `ADTLERT_ENABLE_FLOAT64=1` for float64 throughout).

### Migrating INR code

`INRConfig` keeps `max_iterations`, `target_chi2`, `device`, `log_every`, `prepare_solver`,
`snapshot_interval`, `progress_callback` and `extra_penalty`; every other option moved into a group.

| Before (`INRConfig(...)`) | After |
|---|---|
| `optimizer`, `learning_rate`, `weight_decay`, `gradient_clip_norm`, `scheduler`, `learning_rate_milestones`, `plateau_patience`, `plateau_factor`, `minimum_learning_rate` | `optimization=Optimization(...)`, same names |
| `progressive_encoding`, `progressive_full_iteration`, `progressive_spatial_start_levels`, `progressive_temporal_start_levels` | `progressive=Progressive(enabled, full_iteration, spatial_start_levels, temporal_start_levels)` |
| `spatial_regularization`, `temporal_regularization`, `regularization_huber_delta` | `regularization=Regularization(spatial, temporal, huber_delta)` |
| `time_window_size`, `time_window_step`, `full_evaluation_interval`, `alternate_window_direction` | `windows=Windows(size, step, full_evaluation_interval, alternate_direction)` |

| Before (`INRResult`) | After |
|---|---|
| `chi2_history`, `rms_history`, `objective_history` | `history.chi2`, `history.rms`, `history.objective` |
| `spatial_penalty_history`, `temporal_penalty_history`, `extra_penalty_history` | `history.spatial_penalty`, `history.temporal_penalty`, `history.extra_penalty` |
| `full_chi2_iterations`, `full_chi2_history` | `history.full_iterations`, `history.full_chi2` |
| `elapsed_seconds`, `forward_seconds`, `backward_seconds`, `optimizer_seconds` | `timing.elapsed`, `timing.forward`, `timing.backward`, `timing.optimizer` |
| `physics_forward_timesteps`, `physics_vjp_timesteps` | `timing.forward_timesteps`, `timing.vjp_timesteps` |
| `optimizer`, `scheduler`, `time_window_size`, `time_window_step` | `config.optimization.optimizer`, `config.optimization.scheduler`, `config.windows.size`, `config.windows.step` |

`log_resistivity`, `resistivity`, `predicted_log_data`, `predicted_data`, `iterations`,
`best_iteration`, `best_chi2`, `stop_reason`, `device` and `gpu_report` are unchanged. Training
numerics are bitwise identical to the flat API. Networks can be rebuilt with
`Network.from_configuration(network.configuration())` followed by `load_state_dict`.

### Added

- `adtlert.inr`: implicit neural representations (`MultiscaleINR`, `JointSpatioTemporalINR`,
  `DualNetworkINR`) trained through the matrix-free physics, with windowed time-lapse training.
- `ParameterizedERTForward2p5D.log_model_to_full`: differentiable background extension.
- `parameter_max_cell_area` for `build_source_position_triangle_inversion_case`.
- A pytest suite (`tests/`; GPU tests are skipped without a GPU).

### Changed

- One 2.5D code path for flat and terrain surfaces; one inversion engine for single-time,
  time-lapse and windowed inversion; modules split by responsibility (see `ARCHITECTURE.md`).
- Sensitivities: one autotuned Triton kernel for any element type (terrain quadrilaterals:
  244 ms -> 6 ms); adjoint gradients use the transpose of assembly applied to an SDDMM instead
  of gathering `W x S x C x k` field blocks.
- Reductions with repeated indices use `GroupSum`, a fixed reduction tree that is bitwise
  repeatable (CUDA `index_add_` and cuSPARSE are not).

### Fixed

- Terrain `ERTForward2p5D.vjp` crashed with a float32/float64 mismatch; the terrain adjoint now
  runs in float64 throughout (dot-test error 1.8e-4 -> 1e-10).
- Terrain with `topographic_geometric_factor_mode="numerical"` crashed with the same mismatch.
- `close()` left a stale cuDSS solver behind on the flat path, which corrupted later solves;
  `ERTForward3D.close()` called a method that does not exist and never freed its solver.
- Matrix-free single-time coverage mixed in the regularization gradient through an aliased array.
- A top-level `example.py` installed by `nvidia-ml-py` shadowed `from example...` imports.

### Known limitations

- cuDSS is not bitwise deterministic by default (float32 fields vary by ~1e-6 between runs, which
  Gauss-Newton amplifies). Its `DETERMINISTIC_MODE` makes runs identical at ~45 % higher cost on
  terrain problems; nvmath-python does not expose it yet.
- The Gauss-Newton linear solve, regularization assembly and petrophysical maps still run on the CPU.
