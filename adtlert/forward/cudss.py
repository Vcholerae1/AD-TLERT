"""NVIDIA cuDSS direct solves of batched sparse systems, with plan reuse.

``nvmath-python`` wraps cuDSS; the symbolic analysis (reordering) is the expensive part and
depends only on the sparsity pattern, so one solver is planned per pattern and right-hand-side
shape, and later solves refresh the matrix values and right-hand sides in place.
"""

from __future__ import annotations

import contextlib
import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

CUDA = torch.device("cuda")
LOGGER = logging.getLogger("adtlert.cudss")
LOGGER.setLevel(logging.ERROR)


def create_solver(matrices, rhs, *, spd: bool):
    """Plan a batched ``DirectSolver`` (SPD, or symmetric with nested-dissection reordering)."""

    try:
        from nvmath.sparse import advanced
    except ImportError as exc:
        raise ImportError("ADTLERT requires nvmath-python (cuDSS)") from exc
    matrix_type = (
        advanced.DirectSolverMatrixType.SPD
        if spd
        else advanced.DirectSolverMatrixType.SYMMETRIC
    )
    options = advanced.DirectSolverOptions(
        sparse_system_type=matrix_type, logger=LOGGER, blocking=True
    )
    solver = advanced.DirectSolver(matrices, rhs, options=options)
    if not spd:
        if hasattr(advanced, "DirectSolverAlgType"):  # nvmath-python < 1.0
            solver.plan_config.algorithm = advanced.DirectSolverAlgType.ALG_1
        else:
            solver.plan_config.reordering_algorithm = (
                advanced.DirectSolverReorderingAlg.NESTED_DISSECTION
            )
    solver.plan()
    return solver


@dataclass(frozen=True)
class CsrStructure:
    """Sparsity structure of a CSR matrix (anything with these attributes works as a pattern)."""

    indptr: np.ndarray
    indices: np.ndarray
    shape: tuple[int, int]


class BatchedSolver:
    """Solves ``A_b X_b = rhs_b`` for a family of matrices sharing one CSR pattern."""

    def __init__(self) -> None:
        self._plans: dict[Any, dict[str, Any]] = {}

    def solve(
        self,
        key,
        pattern,
        values: torch.Tensor,
        rhs: torch.Tensor,
        *,
        spd: bool,
        refactorize: bool = True,
    ) -> torch.Tensor:
        """Solve for ``rhs`` of shape ``(batch, n_rhs, n)`` given ``values`` of shape ``(batch, nnz)``.

        ``pattern`` provides ``indptr``, ``indices`` and ``shape``. ``key`` names the family of
        systems; plans are additionally keyed by the right-hand-side shape. The numeric
        factorization is redone unless ``refactorize=False`` (same matrices, new right-hand sides).
        """

        origin = rhs.device
        values, rhs = values.to(CUDA), rhs.to(CUDA, values.dtype)
        plan = self._plans.get((key, tuple(rhs.shape)))
        if plan is None:
            indptr, indices = (
                torch.as_tensor(x, device=CUDA)
                for x in (pattern.indptr, pattern.indices)
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Sparse CSR tensor support is in beta"
                )
                matrices = [
                    torch.sparse_csr_tensor(
                        indptr, indices, value.clone(), size=pattern.shape
                    )
                    for value in values
                ]
            buffer = rhs.clone()
            plan = {"matrices": matrices, "rhs": buffer, "view": buffer.transpose(1, 2)}
            plan["solver"] = create_solver(matrices, plan["view"], spd=spd)
            self._plans[(key, tuple(rhs.shape))] = plan
            refactorize = True
        else:
            for matrix, value in zip(plan["matrices"], values, strict=True):
                matrix.values().copy_(value)
            plan["rhs"].copy_(rhs)
            plan["solver"].reset_operands(b=plan["view"])
        if refactorize:
            plan["solver"].factorize()
        return plan["solver"].solve().transpose(1, 2).to(origin, copy=True).contiguous()

    def close(self) -> None:
        """Release the solvers (best effort)."""

        for plan in self._plans.values():
            with contextlib.suppress(Exception):  # teardown must not raise
                plan["solver"].free()
        self._plans.clear()
