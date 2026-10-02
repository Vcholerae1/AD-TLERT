"""The fused Triton sensitivity kernel and the deterministic reductions."""

import numpy as np
import pytest
import torch
import triton

from adtlert.forward import ERTForward2p5D, kernels
from tests.conftest import conductivity

pytestmark = pytest.mark.gpu

CUDA = torch.device("cuda")


def random_problem(k, dtype, W=5, S=9, N=60, C=40, D=17, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    draw = lambda *shape: torch.randn(*shape, generator=gen, dtype=dtype).to(CUDA)  # noqa: E731
    phi, templates = draw(W, S, N), draw(W, C, k, k)
    cells = torch.randint(0, N, (C, k), generator=gen).to(CUDA)
    a, b, m, n = (torch.randint(0, S, (D,), generator=gen).to(CUDA) for _ in range(4))
    return phi, a, b, m, n, cells, templates


def reference(phi, a, b, m, n, cells, templates):
    current = (phi[:, a] - phi[:, b])[..., cells]
    receiver = (phi[:, m] - phi[:, n])[..., cells]
    return -torch.einsum("wdci,wcij,wdcj->dc", receiver, templates, current)


@pytest.mark.parametrize("k", [3, 4, 6])
@pytest.mark.parametrize("dtype, tol", [(torch.float64, 1e-12), (torch.float32, 1e-4)])
def test_triton_sensitivity_matches_the_einsum_reference(k, dtype, tol):
    problem = random_problem(k, dtype)
    expected, actual = reference(*problem), kernels.normal_sensitivity(*problem)
    assert actual.shape == expected.shape
    assert float((actual - expected).abs().max() / expected.abs().max()) < tol


def test_every_autotune_configuration_gives_identical_results():
    phi, a, b, m, n, cells, templates = random_problem(4, torch.float64)
    outputs = []
    for config in kernels._CONFIGS:
        out = torch.empty((a.shape[0], cells.shape[0]), dtype=phi.dtype, device=CUDA)
        block_d, block_c = config.kwargs["BLOCK_D"], config.kwargs["BLOCK_C"]
        grid = (triton.cdiv(a.shape[0], block_d), triton.cdiv(cells.shape[0], block_c))
        kernels._normal_sensitivity_kernel.fn[grid](
            phi, a.int(), b.int(), m.int(), n.int(), cells.int(), templates, out, *phi.shape, a.shape[0], cells.shape[0],
            K=4, BLOCK_D=block_d, BLOCK_C=block_c, num_warps=config.num_warps,
        )  # fmt: skip
        outputs.append(out)
    assert all(torch.equal(outputs[0], other) for other in outputs[1:])


def test_sensitivity_kernel_handles_ragged_tiles():
    problem = random_problem(
        3, torch.float64, C=37, D=5
    )  # neither dimension fills a block
    assert torch.allclose(
        kernels.normal_sensitivity(*problem), reference(*problem), rtol=1e-10
    )


def test_group_sum_matches_index_add_and_is_repeatable():
    groups = np.array([2, 0, 2, -1, 1, 0, 2, 2])
    x = torch.randn(3, 8, dtype=torch.float64, device=CUDA)
    expected = torch.zeros(3, 4, dtype=torch.float64, device=CUDA)
    expected.index_add_(
        1, torch.as_tensor(groups[groups >= 0], device=CUDA), x[:, groups >= 0]
    )
    summed = kernels.GroupSum(groups, 4, CUDA)
    assert torch.allclose(summed(x), expected)
    assert all(torch.equal(summed(x), summed(x)) for _ in range(5))
    assert torch.equal(
        summed(x)[:, 3], torch.zeros(3, dtype=torch.float64, device=CUDA)
    )  # empty group
    assert summed(x.float()).dtype == torch.float32  # one matrix per dtype


@pytest.mark.parametrize(
    "size, groups, block",
    [(200_000, 5, 64), (2_000, 5, 64), (200_000, 5_000, 64), (1_000, 3, 4)],
)
def test_group_sum_is_bitwise_repeatable_for_any_group_size(size, groups, block):
    """cuSPARSE reductions of long rows are not repeatable; the tree reduction must be."""

    labels = np.random.default_rng(0).integers(-1, groups, size)
    x = torch.randn(size, dtype=torch.float32, device=CUDA)
    summed = kernels.GroupSum(labels, groups, CUDA, block=block)
    first = summed(x)
    assert all(torch.equal(first, summed(x)) for _ in range(20))
    exact = torch.stack(
        [
            x[torch.as_tensor(labels == g, device=CUDA)].double().sum()
            for g in range(groups)
        ]
    )
    assert float((first.double() - exact).abs().max()) < 1e-3 * (
        1 + float(exact.abs().max())
    )


def test_group_sum_edge_cases():
    assert kernels.GroupSum(np.array([-1, -1]), 3, CUDA)(
        torch.ones(2, device=CUDA)
    ).tolist() == [0.0, 0.0, 0.0]
    summed = kernels.GroupSum(np.array([2, 2, 0]), 4, CUDA)
    assert summed(torch.tensor([1.0, 2.0, 4.0], device=CUDA)).tolist() == [
        4.0,
        0.0,
        3.0,
        0.0,
    ]
    batch = torch.arange(6.0, device=CUDA).reshape(2, 3)
    assert summed(batch).tolist() == [[2.0, 0.0, 1.0, 0.0], [5.0, 0.0, 7.0, 0.0]]


def test_sampled_products_match_dense_products_on_the_pattern():
    dense = torch.randn(6, 4, dtype=torch.float64, device=CUDA) @ torch.randn(
        4, 6, dtype=torch.float64, device=CUDA
    )
    left, right = (
        torch.randn(6, 5, dtype=torch.float64, device=CUDA),
        torch.randn(5, 6, dtype=torch.float64, device=CUDA),
    )
    pattern = (torch.rand(6, 6, device=CUDA) > 0.5) | torch.eye(
        6, dtype=torch.bool, device=CUDA
    )
    rows, cols = pattern.nonzero(as_tuple=True)
    indptr = torch.cat((rows.new_zeros(1), torch.bincount(rows, minlength=6).cumsum(0)))
    values = kernels.sampled_products(indptr, cols, (6, 6), left, right)
    assert torch.allclose(values, (left @ right)[rows, cols])
    assert dense.shape == (6, 6)


@pytest.mark.parametrize("slope", [0.0, 0.03])
def test_cell_gradient_matches_the_gathered_einsum(slope):
    """The SDDMM adjoint gradient equals the direct gather/contract formula it replaces."""

    from tests.conftest import quad_case

    case = quad_case(slope)
    forward = ERTForward2p5D.from_mesh_survey(case.mesh, case.survey)
    d = forward.discretization
    phi = forward._fields(conductivity(case)).on(CUDA)
    lam = torch.randn_like(phi)
    templates = forward._volume_templates(d, CUDA)
    cells = d.cell_dofs.to(CUDA)
    per_cell = -torch.einsum(
        "wsci,wcij,wscj->c", lam[..., cells], templates, phi[..., cells]
    )
    expected = torch.zeros(
        case.mesh.cell_count, dtype=phi.dtype, device=CUDA
    ).index_add_(0, d.parent_cell_ids.to(CUDA), per_cell)
    actual = forward._cell_gradient(phi, lam, robin=False).to(CUDA)
    assert float((actual - expected).abs().max() / expected.abs().max()) < 1e-8
    batched = forward._cell_gradient(
        phi, torch.stack([lam, 2 * lam], dim=1), robin=False
    )
    assert (
        batched.shape[0] == 2
    )  # leading batch axis of the adjoint fields becomes the leading axis of the result
    assert torch.allclose(
        batched[1].to(CUDA), 2 * actual, rtol=1e-9
    ) and torch.allclose(batched[0].to(CUDA), actual, rtol=1e-9)


def test_robin_term_of_the_gradient_matches_direct_contraction(flat_case):
    forward = ERTForward2p5D.from_mesh_survey(flat_case.mesh, flat_case.survey)
    d = forward.discretization
    phi = forward._fields(conductivity(flat_case)).on(CUDA)
    lam = torch.randn_like(phi)
    boundary = d.boundary_dofs.to(CUDA)
    per_edge = -torch.einsum(
        "wsbi,wbij,wsbj->b",
        lam[..., boundary],
        forward._boundary_templates(d, CUDA),
        phi[..., boundary],
    )
    with_robin = forward._cell_gradient(phi, lam, robin=True)
    without = forward._cell_gradient(phi, lam, robin=False)
    expected = torch.zeros(
        flat_case.mesh.cell_count, dtype=phi.dtype, device=CUDA
    ).index_add_(0, d.boundary_cells.to(CUDA), per_edge)
    assert (
        float(
            ((with_robin - without).to(CUDA) - expected).abs().max()
            / expected.abs().max()
        )
        < 1e-9
    )
