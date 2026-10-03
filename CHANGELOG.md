# Changelog

## Unreleased (planned 0.2.0)

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
- Examples switch Torch to float64 with `torch.set_default_dtype` after importing adtlert
  (`FLOAT_DTYPE` is fixed at import; set `ADTLERT_ENABLE_FLOAT64=1` for float64 throughout).

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
