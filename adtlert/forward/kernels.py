"""GPU building blocks for sensitivities: a fused CUDA kernel and deterministic reductions.

Scatter-adds with repeated indices (``index_add_`` on CUDA) use atomics whose order is
not fixed, so their float sums vary between runs. :class:`GroupSum` replaces them with a
fixed CSR reduction, and the sensitivity kernel writes every output entry exactly once.

The kernel is CUDA C++ compiled at first use by NVRTC through ``cuda.core`` (both ship as
wheels for Linux and Windows), so no compiler toolchain or Triton is needed.
"""

from __future__ import annotations

import contextlib
import warnings

import numpy as np
import torch

Tensor = torch.Tensor


def _csr(indptr, indices, values, size) -> Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Sparse CSR tensor support is in beta"
        )
        return torch.sparse_csr_tensor(indptr, indices, values, size=size)


class GroupSum:
    """Deterministic ``out[g] = sum_{i: groups[i] = g} x[i]`` along the last axis.

    Entries with a negative group are dropped. The sum is a fixed tree: each group's entries
    are laid out in padded blocks, reduced with a plain ``sum`` over a fixed axis, and the
    block sums are reduced the same way until one value per group remains. Only gathers and
    fixed-shape reductions are involved, so results are bitwise repeatable for any group
    size (cuSPARSE and atomics are not).
    """

    def __init__(
        self, groups: np.ndarray, count: int, device: torch.device, block: int = 64
    ):
        groups = np.asarray(groups, dtype=np.int64)
        self.count = int(count)
        entries = np.flatnonzero(groups >= 0)
        owner = groups[entries]
        zero_slot = groups.size
        self._levels: list[Tensor] = []
        while True:
            order = np.argsort(owner, kind="stable")
            entries, owner = entries[order], owner[order]
            sizes = np.bincount(owner, minlength=self.count)
            width = int(
                min(block, 1 << max(int(sizes.max(initial=1)) - 1, 0).bit_length())
            )
            blocks = -(-sizes // width)
            block_owner = np.repeat(np.arange(self.count), blocks)
            starts = np.cumsum(sizes) - sizes
            within = (
                np.arange(block_owner.size) - (np.cumsum(blocks) - blocks)[block_owner]
            )
            position = (
                starts[block_owner][:, None]
                + within[:, None] * width
                + np.arange(width)[None, :]
            )
            valid = position < (starts + sizes)[block_owner][:, None]
            index = np.where(
                valid,
                entries[np.minimum(position, max(entries.size - 1, 0))],
                zero_slot,
            )
            self._levels.append(torch.as_tensor(index, device=device))
            if blocks.max(initial=0) <= 1:
                break
            zero_slot = block_owner.size
            entries, owner = np.arange(zero_slot), block_owner
        self._target = torch.as_tensor(block_owner, device=device)

    def __call__(self, x: Tensor) -> Tensor:
        for level in self._levels:
            padded = torch.cat((x, x.new_zeros(*x.shape[:-1], 1)), dim=-1)
            x = padded[..., level].sum(dim=-1)
        out = x.new_zeros(*x.shape[:-1], self.count)
        return out.index_copy_(x.ndim - 1, self._target, x)


class CellAggregation:
    """Sums per-cell values into output columns; cells whose target is ``-1`` are dropped."""

    def __init__(self, targets: np.ndarray, count: int, device: torch.device):
        targets = np.asarray(targets, dtype=np.int64)
        active = np.flatnonzero(targets >= 0)
        self.active = (
            None
            if active.size == targets.size
            else torch.as_tensor(active, device=device)
        )
        identity = count == targets.size and np.array_equal(targets, np.arange(count))
        self._sum = None if identity else GroupSum(targets[active], count, device)

    def select(self, values: Tensor, dim: int) -> Tensor:
        """The active cells of ``values`` along ``dim``."""

        return values if self.active is None else values.index_select(dim, self.active)

    def columns(self, active_values: Tensor) -> Tensor:
        """Sum values over the active cells (last axis) into the output columns."""

        return active_values if self._sum is None else self._sum(active_values)

    def __call__(self, values: Tensor) -> Tensor:
        return self.columns(self.select(values, -1))


def sampled_products(
    indptr: Tensor, indices: Tensor, shape, left: Tensor, right: Tensor
) -> Tensor:
    """Values of ``left @ right`` on a CSR sparsity pattern (SDDMM), in pattern order."""

    pattern = _csr(
        indptr,
        indices,
        torch.zeros(indices.numel(), dtype=left.dtype, device=left.device),
        shape,
    )
    return torch.sparse.sampled_addmm(pattern, left, right, beta=0.0).values()


_BLOCK_CELLS, _BLOCK_DATA = 32, 4

_SENSITIVITY_SOURCE = r"""
// One thread owns one (measurement, cell) entry and accumulates it in the fixed order
// (w, i, j), so results are bitwise repeatable. Blocks cover 32 cells x 4 measurements;
// consecutive blocks walk the measurements of a cell block first, which keeps that block's
// field entries in cache. Row index S (one past the last field row) is a remote electrode,
// whose potential is zero.
template <typename T>
__device__ __forceinline__ T potential(const T* field, int row, int S, long long N, int node)
{
    return row < S ? __ldg(field + row * N + node) : T(0);
}

template <typename T, int K>
__global__ void __launch_bounds__(BLOCK_CELLS * BLOCK_DATA) normal_sensitivity(
    const T* __restrict__ phi, const int* __restrict__ a, const int* __restrict__ b,
    const int* __restrict__ m, const int* __restrict__ n, const int* __restrict__ cells,
    const T* __restrict__ templates, T* __restrict__ out,
    int W, int S, long long N, int D, int C)
{
    const int data_blocks = (D + BLOCK_DATA - 1) / BLOCK_DATA;
    const int c = (blockIdx.x / data_blocks) * BLOCK_CELLS + threadIdx.x;
    const int d = (blockIdx.x % data_blocks) * BLOCK_DATA + threadIdx.y;
    if (c >= C || d >= D) return;
    int nodes[K];
#pragma unroll
    for (int k = 0; k < K; ++k) nodes[k] = __ldg(cells + (long long)c * K + k);
    const int ia = __ldg(a + d), ib = __ldg(b + d), im = __ldg(m + d), in = __ldg(n + d);
    T acc = 0;
    for (int w = 0; w < W; ++w) {
        const T* field = phi + (long long)w * S * N;
        const T* weights = templates + ((long long)w * C + c) * (K * K);
        T current[K];
#pragma unroll
        for (int j = 0; j < K; ++j)
            current[j] = potential(field, ia, S, N, nodes[j]) - potential(field, ib, S, N, nodes[j]);
#pragma unroll
        for (int i = 0; i < K; ++i) {
            const T receiver =
                potential(field, im, S, N, nodes[i]) - potential(field, in, S, N, nodes[i]);
#pragma unroll
            for (int j = 0; j < K; ++j)
                acc += receiver * (__ldg(weights + i * K + j) * current[j]);
        }
    }
    out[(long long)d * C + c] = -acc;
}
"""

_CUDA_TYPES = {torch.float32: "float", torch.float64: "double"}
_compiled: dict = {}


@contextlib.contextmanager
def _cuda_core_device(device: torch.device):
    """Make ``device`` current for ``cuda.core``, restoring PyTorch's current device afterwards."""

    from cuda.core import Device

    with torch.cuda.device(device):
        cuda_device = Device(torch.cuda.current_device())
        cuda_device.set_current()
        yield cuda_device


def _sensitivity_kernel(index: int, dtype: torch.dtype, local_dofs: int):
    """Compile (once per device, dtype and local DOF count) and load the kernel.

    Requires ``index`` to be current (see :func:`_cuda_core_device`).
    """

    from cuda.core import Program, ProgramOptions

    key = (index, dtype, local_dofs)
    if key not in _compiled:
        major, minor = torch.cuda.get_device_capability(index)
        name = f"normal_sensitivity<{_CUDA_TYPES[dtype]}, {local_dofs}>"
        options = ProgramOptions(
            arch=f"sm_{major}{minor}",
            std="c++17",
            define_macro=[
                ("BLOCK_CELLS", str(_BLOCK_CELLS)),
                ("BLOCK_DATA", str(_BLOCK_DATA)),
            ],
        )
        program = Program(_SENSITIVITY_SOURCE, "c++", options)
        module = program.compile("cubin", name_expressions=(name,))
        _compiled[key] = module.get_kernel(name)
    return _compiled[key]


def normal_sensitivity(
    phi: Tensor,
    a: Tensor,
    b: Tensor,
    m: Tensor,
    n: Tensor,
    cell_dofs: Tensor,
    templates: Tensor,
) -> Tensor:
    """``out[d, c] = -sum_w (u_M - u_N)_c^T T_wc (u_A - u_B)_c`` for every measurement and cell.

    ``phi`` is ``(W, E, N)``, ``cell_dofs`` ``(C, K)``, ``templates`` ``(W, C, K, K)``; the
    gather and contraction are fused, so no ``W x D x C x K`` field blocks are formed. An
    electrode index ``E`` is a remote electrode with zero potential.
    """

    from cuda.core import LaunchConfig, launch

    if phi.dtype not in _CUDA_TYPES:
        raise TypeError("normal_sensitivity supports float32 and float64 fields")
    phi = phi.contiguous()
    templates = templates.to(phi.dtype).contiguous()
    cells = cell_dofs.to(device=phi.device, dtype=torch.int32).contiguous()
    a, b, m, n = (
        index.to(device=phi.device, dtype=torch.int32).contiguous()
        for index in (a, b, m, n)
    )
    (W, E, N), (C, K), D = phi.shape, cells.shape, a.shape[0]
    out = torch.empty((D, C), dtype=phi.dtype, device=phi.device)
    if C == 0 or D == 0:
        return out
    blocks = -(-C // _BLOCK_CELLS) * -(-D // _BLOCK_DATA)
    with _cuda_core_device(phi.device) as cuda_device:
        kernel = _sensitivity_kernel(cuda_device.device_id, phi.dtype, K)
        stream = cuda_device.create_stream(torch.cuda.current_stream(phi.device))
        launch(
            stream,
            LaunchConfig(grid=blocks, block=(_BLOCK_CELLS, _BLOCK_DATA)),
            kernel,
            *(x.data_ptr() for x in (phi, a, b, m, n, cells, templates, out)),
            np.int32(W),
            np.int32(E),
            np.int64(N),
            np.int32(D),
            np.int32(C),
        )
    return out
