"""Optimizer registry and model-update rules for ADTLERT inversions."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import cg, lsqr


@dataclass(frozen=True)
class LinearizedOptimizer:
    """Metadata for a linearized inversion update backend."""

    name: str
    description: str


@dataclass(frozen=True)
class OptimizationAlgorithm:
    """Metadata for an outer nonlinear inversion optimizer."""

    name: str
    description: str
    uses_linearized_solver: bool = False
    gradient_based: bool = False


_LINEARIZED_OPTIMIZERS = {
    optimizer.name: optimizer
    for optimizer in (
        LinearizedOptimizer("lsqr", "SciPy LSQR on the assembled linearized system."),
        LinearizedOptimizer("gpu_cgls", "GPU CGLS on the assembled linearized system."),
        LinearizedOptimizer(
            "normal_cg", "SciPy conjugate-gradient solve on normal equations."
        ),
        LinearizedOptimizer(
            "pyhydro_cgls", "PyHydroGeophysX-style CGLS on normal equations."
        ),
    )
}

_OPTIMIZATION_ALGORITHMS = {
    algorithm.name: algorithm
    for algorithm in (
        OptimizationAlgorithm(
            "gauss_newton_cgls",
            "Damped Gauss-Newton update solved with the configured CGLS/LSQR backend.",
            uses_linearized_solver=True,
        ),
        OptimizationAlgorithm(
            "levenberg_marquardt",
            "Levenberg-Marquardt damped Gauss-Newton update.",
            uses_linearized_solver=True,
        ),
        OptimizationAlgorithm(
            "lbfgs",
            "Limited-memory BFGS using ADTLERT matrix-free normal VJP gradients and line search.",
            gradient_based=True,
        ),
        OptimizationAlgorithm(
            "lbfgs_b",
            "Projected limited-memory BFGS using matrix-free normal VJP gradients.",
            gradient_based=True,
        ),
        OptimizationAlgorithm(
            "nonlinear_cg",
            "Nonlinear conjugate-gradient descent using matrix-free normal VJP gradients.",
            gradient_based=True,
        ),
        OptimizationAlgorithm(
            "adam",
            "Adam first-order optimizer using ADTLERT matrix-free normal VJP gradients.",
            gradient_based=True,
        ),
    )
}


def _lookup(registry: dict, name, kind: type, label: str):
    if isinstance(name, kind):
        return name
    try:
        return registry[str(name).strip().lower().replace("-", "_")]
    except KeyError as exc:
        raise ValueError(
            f"unknown {label}={name!r}; available choices: {', '.join(sorted(registry))}"
        ) from exc


def available_linearized_optimizers() -> tuple[str, ...]:
    """Return canonical linearized solver names."""

    return tuple(sorted(_LINEARIZED_OPTIMIZERS))


def available_optimization_algorithms() -> tuple[str, ...]:
    """Return canonical outer optimization algorithm names."""

    return tuple(sorted(_OPTIMIZATION_ALGORITHMS))


def build_linearized_optimizer(name: str | LinearizedOptimizer) -> LinearizedOptimizer:
    """Resolve a linearized optimizer from a registered name."""

    return _lookup(
        _LINEARIZED_OPTIMIZERS, name, LinearizedOptimizer, "linearized_solver"
    )


def build_optimization_algorithm(
    name: str | OptimizationAlgorithm,
) -> OptimizationAlgorithm:
    """Resolve an outer optimization algorithm from a registered name."""

    return _lookup(_OPTIMIZATION_ALGORITHMS, name, OptimizationAlgorithm, "optimizer")


# ---------------------------------------------------------------------------
# Update rules. ``config`` is an ``InversionConfig``.
# ---------------------------------------------------------------------------


def limit_step(delta: np.ndarray, max_step: float | None) -> np.ndarray:
    """Scale ``delta`` so its largest entry does not exceed ``max_step``."""

    largest = float(np.max(np.abs(delta))) if delta.size else 0.0
    if max_step is None or largest <= max_step:
        return delta
    return delta * (max_step / largest)


def linearized_gradient(matrix: sp.spmatrix, rhs: np.ndarray) -> np.ndarray:
    """Gradient of ``||A dm - b||^2 / 2`` at ``dm = 0``."""

    return -np.asarray(
        matrix.T @ np.asarray(rhs, dtype=float).reshape(-1), dtype=float
    ).reshape(-1)


def _descent(direction: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    """Fall back to steepest descent if a quasi-Newton/CG update loses descent."""

    if not np.all(np.isfinite(direction)) or float(np.dot(direction, gradient)) >= 0.0:
        return -gradient
    return direction


def _lbfgs_direction(
    gradient: np.ndarray, history: list[tuple[np.ndarray, np.ndarray, float]]
) -> np.ndarray:
    """L-BFGS two-loop recursion over stored ``(s, y, 1 / y.s)`` pairs."""

    if not history:
        return -gradient
    q = gradient.copy()
    alphas = []
    for s_vec, y_vec, rho in reversed(history):
        alphas.append(float(rho * np.dot(s_vec, q)))
        q -= alphas[-1] * y_vec
    s_last, y_last, _ = history[-1]
    yy = float(np.dot(y_last, y_last))
    r = (float(np.dot(s_last, y_last) / yy) if yy > 0.0 else 1.0) * q
    for (s_vec, y_vec, rho), alpha in zip(history, reversed(alphas)):
        r += s_vec * (alpha - float(rho * np.dot(y_vec, r)))
    return -r


def first_order_step(
    current: np.ndarray, gradient: np.ndarray, state: dict[str, Any], config
) -> np.ndarray:
    """Model increment of a first-order/quasi-Newton optimizer from the current gradient."""

    algorithm = build_optimization_algorithm(config.optimization_algorithm).name
    current = np.asarray(current, dtype=float).reshape(-1)
    grad = np.asarray(gradient, dtype=float).reshape(-1)
    if current.shape != grad.shape:
        raise ValueError("current_state and gradient shape mismatch")
    if not np.all(np.isfinite(grad)):
        raise ValueError("optimizer gradient contains non-finite values")
    if not np.any(grad):
        return np.zeros_like(grad)
    max_step = (
        config.max_log_step
        if config.max_log_step is not None
        else config.optimizer_max_step
    )

    if algorithm == "nonlinear_cg":  # Polak-Ribiere+
        previous_grad, previous_dir = (
            state.get("nonlinear_cg_gradient"),
            state.get("nonlinear_cg_direction"),
        )
        direction = -grad
        if previous_grad is not None and previous_dir is not None:
            beta = max(
                0.0,
                float(
                    np.dot(grad, grad - previous_grad)
                    / max(
                        float(np.dot(previous_grad, previous_grad)), np.finfo(float).eps
                    )
                ),
            )
            direction = -grad + beta * previous_dir
        direction = _descent(direction, grad)
        state["nonlinear_cg_gradient"], state["nonlinear_cg_direction"] = (
            grad.copy(),
            direction.copy(),
        )
        return limit_step(direction, max_step)

    if algorithm in ("lbfgs", "lbfgs_b"):
        history = state.setdefault("lbfgs_history", [])
        if state.get("lbfgs_state") is not None:
            s_vec, y_vec = (
                current - state["lbfgs_state"],
                grad - state["lbfgs_gradient"],
            )
            ys = float(np.dot(y_vec, s_vec))
            if (
                ys > 1.0e-12
                and np.all(np.isfinite(s_vec))
                and np.all(np.isfinite(y_vec))
            ):
                history.append((s_vec.copy(), y_vec.copy(), 1.0 / ys))
                del history[: -int(config.lbfgs_history)]
        direction = _descent(_lbfgs_direction(grad, history), grad)
        state["lbfgs_state"], state["lbfgs_gradient"] = current.copy(), grad.copy()
        return limit_step(direction, max_step)

    if algorithm == "adam":
        beta1, beta2 = float(config.adam_beta1), float(config.adam_beta2)
        step = int(state.get("adam_step", 0)) + 1
        m = beta1 * state.get("adam_m", np.zeros_like(grad)) + (1.0 - beta1) * grad
        v = beta2 * state.get("adam_v", np.zeros_like(grad)) + (1.0 - beta2) * (
            grad * grad
        )
        state.update(adam_step=step, adam_m=m, adam_v=v)
        direction = -(m / (1.0 - beta1**step)) / (
            np.sqrt(v / (1.0 - beta2**step)) + float(config.adam_epsilon)
        )
        return limit_step(direction, max_step)

    raise ValueError(
        f"optimizer={config.optimization_algorithm!r} is not a first-order optimizer"
    )


def linearized_step(
    matrix: sp.spmatrix,
    rhs: np.ndarray,
    current: np.ndarray,
    state: dict[str, Any],
    config,
) -> np.ndarray:
    """Model increment for the stacked linearized system ``A dm ~= b``."""

    algorithm = build_optimization_algorithm(config.optimization_algorithm)
    if not algorithm.uses_linearized_solver:
        return first_order_step(
            current, linearized_gradient(matrix, rhs), state, config
        )
    matrix, rhs = matrix.tocsr(), np.asarray(rhs, dtype=float).reshape(-1)
    if algorithm.name == "levenberg_marquardt" and config.lm_damping > 0.0:
        n_parameters = int(np.asarray(current).size)
        matrix = sp.vstack(
            (matrix, np.sqrt(config.lm_damping) * sp.eye(n_parameters, format="csr")),
            format="csr",
        )
        rhs = np.concatenate((rhs, np.zeros(n_parameters)))
    return limit_step(solve_linearized(matrix, rhs, config), config.max_log_step)


def solve_linearized(matrix: sp.spmatrix, rhs: np.ndarray, config) -> np.ndarray:
    """Least-squares solution of ``A x ~= b`` with the configured backend."""

    solver = config.linearized_solver
    if solver == "gpu_cgls":
        solution = _gpu_cgls(
            matrix,
            rhs,
            max_iterations=config.cgls_max_iterations,
            tolerance=config.cgls_tolerance,
        )
    elif solver == "pyhydro_cgls":
        normal_rhs = np.asarray(matrix.T @ rhs, dtype=float).reshape(-1, 1)
        solution = _pyhydro_cgls(
            (matrix.T @ matrix).tocsr(),
            normal_rhs,
            config.cgls_max_iterations,
            config.cgls_tolerance,
        )
        solution = solution.ravel()
    elif solver == "normal_cg":
        normal_rhs = np.asarray(matrix.T @ rhs, dtype=float).ravel()
        solution, info = cg(
            (matrix.T @ matrix).tocsr(),
            normal_rhs,
            rtol=config.cgls_tolerance,
            atol=0.0,
            maxiter=config.cgls_max_iterations,
        )
        if info < 0:
            raise ValueError(f"normal_cg failed with illegal input/info={info}")
    else:
        solution = lsqr(
            matrix,
            rhs,
            atol=config.lsqr_atol,
            btol=config.lsqr_btol,
            iter_lim=config.lsqr_iter_limit,
        )[0]
    if not np.all(np.isfinite(solution)):
        raise ValueError("linearized inversion update contains non-finite values")
    return np.asarray(solution, dtype=float)


def _gpu_cgls(
    matrix: sp.spmatrix, rhs: np.ndarray, *, max_iterations: int, tolerance: float
) -> np.ndarray:
    """CGLS for ``min ||A x - b||`` on the GPU; ``A`` and ``A^T`` are both stored as CSR."""

    cpu = matrix.tocsr()
    dtype = (
        torch.float64
        if cpu.dtype == np.float64 or np.asarray(rhs).dtype == np.float64
        else torch.float32
    )

    def csr(m):
        tensors = (torch.as_tensor(x, device="cuda") for x in (m.indptr, m.indices))
        return torch.sparse_csr_tensor(
            *tensors, torch.as_tensor(m.data, dtype=dtype, device="cuda"), size=m.shape
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Sparse CSR tensor support is in beta"
        )
        system, transpose = csr(cpu), csr(cpu.T.tocsr())
    r = torch.as_tensor(np.asarray(rhs).ravel(), dtype=dtype, device="cuda")
    x = torch.zeros(system.shape[1], dtype=dtype, device="cuda")
    s = transpose @ r
    p = s.clone()
    gamma = torch.dot(s, s)
    gamma0 = float(gamma)
    if gamma0 <= 0.0 or float(torch.dot(r, r)) <= 0.0:
        return x.cpu().numpy()
    for _ in range(int(max_iterations)):
        q = system @ p
        denominator = torch.dot(q, q)
        if float(denominator) <= 0.0:
            break
        alpha = gamma / denominator
        x = x + alpha * p
        r = r - alpha * q
        s = transpose @ r
        gamma_new = torch.dot(s, s)
        if float(gamma_new) <= 0.0 or float(gamma_new) / gamma0 < float(tolerance):
            break
        p = s + (gamma_new / gamma) * p
        gamma = gamma_new
    return x.cpu().numpy().astype(float)


def _pyhydro_cgls(
    matrix, rhs: np.ndarray, max_iterations: int, tolerance: float
) -> np.ndarray:
    """PyHydroGeophysX's CGLS routine (stops on the relative residual ``||r||^2 / ||b||^2``)."""

    b = np.asarray(rhs, dtype=float).reshape(-1, 1)
    x = np.zeros((matrix.shape[1], 1))
    r = b.copy()
    s = np.asarray(matrix.T @ r, dtype=float).reshape(-1, 1)
    p = s.copy()
    gamma = float((s.T @ s).item())
    rr0 = float((r.T @ r).item())
    if rr0 <= 0.0 or gamma <= 0.0:
        return x
    for _ in range(int(max_iterations)):
        q = np.asarray(matrix @ p, dtype=float).reshape(-1, 1)
        denominator = float((q.T @ q).item())
        if denominator <= 0.0:
            break
        alpha = gamma / denominator
        x += alpha * p
        r -= alpha * q
        s = np.asarray(matrix.T @ r, dtype=float).reshape(-1, 1)
        gamma_new = float((s.T @ s).item())
        if gamma <= 0.0:
            break
        p = s + float(gamma_new / gamma) * p
        gamma = gamma_new
        if float((r.T @ r).item()) / rr0 < float(tolerance):
            break
    return x
