"""GPU building blocks for sensitivities: a fused Triton kernel and deterministic reductions.

Scatter-adds with repeated indices (``index_add_`` on CUDA) use atomics whose order is
not fixed, so their float sums vary between runs. :class:`GroupSum` replaces them with a
fixed CSR reduction, and the Triton kernel writes every output entry exactly once.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import triton
import triton.language as tl

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
        entries = np.flatnonzero(
            groups >= 0
        )  # indices into the input of the current level
        owner = groups[entries]
        n_inputs = groups.size
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
                valid, entries[np.minimum(position, max(entries.size - 1, 0))], n_inputs
            )
            self._levels.append(
                torch.as_tensor(index, device=device)
            )  # n_inputs is the zero pad
            if blocks.max(initial=0) <= 1:
                break
            n_inputs = block_owner.size
            entries, owner = np.arange(n_inputs), block_owner
        self._target = torch.as_tensor(block_owner, device=device)

    def __call__(self, x: Tensor) -> Tensor:
        for level in self._levels:
            padded = torch.cat((x, x.new_zeros(*x.shape[:-1], 1)), dim=-1)
            x = padded[..., level].sum(dim=-1)
        out = x.new_zeros(*x.shape[:-1], self.count)
        return out.index_copy_(x.ndim - 1, self._target, x)


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


_CONFIGS = [
    triton.Config({"BLOCK_D": block_d, "BLOCK_C": block_c}, num_warps=warps)
    for block_d, block_c, warps in (
        (8, 32, 4),
        (16, 32, 4),
        (16, 64, 4),
        (32, 32, 4),
        (8, 64, 2),
        (32, 64, 8),
    )
]


@triton.autotune(configs=_CONFIGS, key=["D", "C", "W", "K"])
@triton.jit
def _normal_sensitivity_kernel(
    phi, a, b, m, n, cells, templates, out, W, S, N, D, C,
    K: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_C: tl.constexpr,
):  # fmt: skip
    # Purely elementwise with a fixed summation order (w, i, j): every launch configuration
    # produces bitwise identical results, so autotuning cannot change the numbers.
    rows = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    row_ok, col_ok = rows < D, cols < C
    tile_ok = row_ok[:, None] & col_ok[None, :]
    source_a = tl.load(a + rows, mask=row_ok, other=0).to(tl.int64)[:, None] * N
    source_b = tl.load(b + rows, mask=row_ok, other=0).to(tl.int64)[:, None] * N
    receiver_m = tl.load(m + rows, mask=row_ok, other=0).to(tl.int64)[:, None] * N
    receiver_n = tl.load(n + rows, mask=row_ok, other=0).to(tl.int64)[:, None] * N
    acc = tl.zeros((BLOCK_D, BLOCK_C), dtype=out.dtype.element_ty)
    for w in range(W):
        field = phi + w.to(tl.int64) * S * N
        block = templates + (w * C + cols).to(tl.int64) * (K * K)
        for i in tl.static_range(K):
            node_i = tl.load(cells + cols * K + i, mask=col_ok, other=0).to(tl.int64)[
                None, :
            ]
            receiver = tl.load(field + receiver_m + node_i, mask=tile_ok, other=0.0)
            receiver -= tl.load(field + receiver_n + node_i, mask=tile_ok, other=0.0)
            for j in tl.static_range(K):
                node_j = tl.load(cells + cols * K + j, mask=col_ok, other=0).to(
                    tl.int64
                )[None, :]
                current = tl.load(field + source_a + node_j, mask=tile_ok, other=0.0)
                current -= tl.load(field + source_b + node_j, mask=tile_ok, other=0.0)
                weight = tl.load(block + i * K + j, mask=col_ok, other=0.0)[None, :]
                acc += receiver * (weight * current)
    tl.store(out + rows[:, None].to(tl.int64) * C + cols[None, :], -acc, mask=tile_ok)


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
    gather and contraction are fused, so no ``W x D x C x K`` field blocks are formed.
    """

    phi, templates = phi.contiguous(), templates.contiguous()
    cells = cell_dofs.to(torch.int32).contiguous()
    a, b, m, n = (index.to(torch.int32).contiguous() for index in (a, b, m, n))
    (W, E, N), (C, K), D = phi.shape, cells.shape, a.shape[0]
    out = torch.empty((D, C), dtype=phi.dtype, device=phi.device)

    def grid(meta):
        return triton.cdiv(D, meta["BLOCK_D"]), triton.cdiv(C, meta["BLOCK_C"])

    _normal_sensitivity_kernel[grid](
        phi,
        a,
        b,
        m,
        n,
        cells,
        templates,
        out,
        W,
        E,
        N,
        D,
        C,
        K=K,
    )
    return out
