"""Tests for the eigenglacier projection, including the MPS chunking workaround."""

import pytest
import torch

from pism_emulator.emulators.nnemulator import (
    MPS_MAX_INNER_DIM,
    DNNEmulator,
    project_basis,
)


def _inputs(n_nodes: int, q: int = 7, batch: int = 5, dtype=torch.float64):
    g = torch.Generator().manual_seed(0)
    C = torch.randn(batch, q, generator=g, dtype=dtype)
    V = torch.randn(n_nodes, q, generator=g, dtype=dtype)
    return C, V


def test_cpu_uses_plain_product() -> None:
    """Off MPS the projection is the plain matrix product."""
    C, V = _inputs(1000)
    assert torch.equal(project_basis(C, V), C @ V.T)


@pytest.mark.parametrize("chunk", [1, 64, 333, 1000, 5000])
def test_chunked_matches_plain_with_gradients(chunk: int) -> None:
    """
    Chunked projection equals the plain product, values and gradients.

    Parameters
    ----------
    chunk : int
        Rows of the basis per block.
    """
    C, V = _inputs(1000)
    G = torch.randn(5, 1000, dtype=torch.float64)

    C1 = C.clone().requires_grad_()
    out1 = C1 @ V.T
    out1.backward(G)

    C2 = C.clone().requires_grad_()
    out2 = project_basis(C2, V, chunk=chunk)
    out2.backward(G)

    torch.testing.assert_close(out2, out1)
    torch.testing.assert_close(C2.grad, C1.grad)


def test_blocked_hessian_and_vector_input() -> None:
    """Second derivatives and 1-D coefficients match the plain product."""
    _, V = _inputs(1000)
    w = torch.rand(1000, dtype=torch.float64)

    def loss(plain: bool):
        def f(c):
            out = c @ V.T if plain else project_basis(c, V, chunk=128)
            return (w * torch.tanh(out) ** 2).sum()

        return f

    c0 = torch.randn(7, dtype=torch.float64)
    H_plain = torch.autograd.functional.hessian(loss(True), c0)
    H_block = torch.autograd.functional.hessian(loss(False), c0)
    torch.testing.assert_close(H_block, H_plain)


def test_chunk_cache_follows_in_place_updates() -> None:
    """The cached chunks are rebuilt after an in-place change of the basis."""
    C, V = _inputs(100)
    first = project_basis(C, V, chunk=30)
    with torch.no_grad():
        V.mul_(2.0)
    torch.testing.assert_close(project_basis(C, V, chunk=30), 2.0 * first)


def test_emulator_after_load_state_dict_uses_new_basis() -> None:
    """A cached chunking must not survive loading different weights."""
    n, q, p = 50, 4, 3
    g = torch.Generator().manual_seed(1)
    V1, V2 = torch.randn(n, q, generator=g), torch.randn(n, q, generator=g)
    F_mean, area = torch.zeros(n), torch.ones(n) / n
    e1 = DNNEmulator(p, V1, F_mean, area, depth=1, dropout=0.0).eval()
    e2 = DNNEmulator(p, V2, F_mean, area, depth=1, dropout=0.0).eval()
    x = torch.randn(2, p, generator=g)
    project_basis(torch.ones(1, q), e1.V_hat, chunk=7)  # populate the cache
    e1.load_state_dict(e2.state_dict())
    coeffs = torch.ones(1, q)
    torch.testing.assert_close(project_basis(coeffs, e1.V_hat, chunk=7), coeffs @ V2.T)
    torch.testing.assert_close(e1(x), e2(x))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_mps_large_grid_gradient_is_accurate() -> None:
    """
    Regression test for the MPS matmul error above 2**15 inner dimension.

    With a plain ``coeffs @ V_hat.T`` the gradient with respect to ``coeffs`` was
    5-60 % wrong for this shape (torch 2.13, macOS 26, M2 Max).
    """
    n_nodes = 649_404
    assert n_nodes > MPS_MAX_INNER_DIM
    g = torch.Generator().manual_seed(0)
    C = torch.randn(256, 100, generator=g)
    V = torch.randn(n_nodes, 100, generator=g)
    G = torch.randn(256, n_nodes, generator=g)
    ref = G.double() @ V.double()

    Cm = C.to("mps").requires_grad_()
    out = project_basis(Cm, V.to("mps"))
    out.backward(G.to("mps"))
    err = (Cm.grad.cpu().double() - ref).norm() / ref.norm()
    assert err < 1e-5
