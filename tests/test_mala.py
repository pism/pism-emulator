"""
Tests for the MALA samplers in ``pism_emulator.mcmc.mala``.

The toy target (``mala_toy``) has two parameters, so its posterior moments are
computed exactly by quadrature and compared against long chains.
"""

# pylint: disable=protected-access
from pathlib import Path

import numpy as np
import pytest
import torch
from mala_toy import (
    SIGMA,
    X_MAX,
    X_MIN,
    Y_TARGET,
    LinearModel,
    exact_moments,
    make_sampler,
    run_chain,
)
from scipy import stats

GOLDEN = Path(__file__).parent / "test_data" / "mala_legacy_golden.pt"


def test_neg_log_prob_matches_scipy() -> None:
    """Student-t likelihood plus Beta prior equals the scipy densities."""
    s = make_sampler(alpha=0.7)
    x = torch.tensor([0.3, 0.8])
    y = LinearModel()(x).numpy()
    log_like = stats.t.logpdf(
        Y_TARGET.numpy(), df=4.0, loc=y, scale=SIGMA.numpy()
    ).sum()
    x_bar = ((x - X_MIN) / (X_MAX - X_MIN)).numpy()
    log_prior = stats.beta.logpdf(x_bar, 3.0, 3.0).sum()
    expected = -(0.7 * log_like + log_prior)
    np.testing.assert_allclose(s.neg_log_prob(x).item(), expected, rtol=1e-5)


def test_proposal_logpdf_matches_multivariate_normal() -> None:
    """``_proposal_logpdf`` is the density of N(mu, 2h H^-1)."""
    s = make_sampler()
    H = torch.tensor([[3.0, 0.4], [0.4, 1.5]])
    Hinv = torch.linalg.inv(H)
    h = 0.3
    y, mu = torch.tensor([0.2, -0.1]), torch.tensor([0.05, 0.1])
    log_det_Hinv = torch.logdet(Hinv)
    got = s._proposal_logpdf(y, mu, H, log_det_Hinv, h, s._two_pi)
    ref = torch.distributions.MultivariateNormal(mu, covariance_matrix=2 * h * Hinv)
    torch.testing.assert_close(got, ref.log_prob(y), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("use_eig", [False, True])
def test_local_geometry_is_consistent(use_eig: bool) -> None:
    """
    Check that Hpos is |H|, Hinv its inverse and log_det_Hinv its log-determinant.

    Parameters
    ----------
    use_eig : bool
        Use the eigendecomposition variant.
    """
    s = make_sampler()
    x = torch.tensor([0.4, 0.6], requires_grad=True)
    geom = s._local_geometry_eig if use_eig else s._local_geometry
    U, g, Hpos, Hinv, log_det_Hinv = geom(x)
    H = torch.autograd.functional.hessian(s.neg_log_prob, x.detach())
    lam, Q = torch.linalg.eigh(H)
    absH = Q @ torch.diag(lam.abs()) @ Q.T
    torch.testing.assert_close(U, s.neg_log_prob(x))
    torch.testing.assert_close(
        g, torch.autograd.grad(s.neg_log_prob(x), x)[0], rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(Hpos, absH, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(Hpos @ Hinv, torch.eye(2), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(log_det_Hinv, -torch.logdet(absH), rtol=1e-4, atol=1e-4)


def test_local_geometry_ill_conditioned() -> None:
    """The default metric stays exact for an ill-conditioned Hessian."""
    s = make_sampler()
    Q, _ = torch.linalg.qr(
        torch.randn(4, 4, generator=torch.Generator().manual_seed(0))
    )
    H0 = Q @ torch.diag(torch.tensor([1.0, 10.0, 1e3, 1e4])) @ Q.T
    s.neg_log_prob = lambda x: 0.5 * x @ H0 @ x  # type: ignore[method-assign,assignment]
    _, _, Hpos, Hinv, log_det_Hinv = s._local_geometry(
        torch.ones(4, requires_grad=True)
    )
    torch.testing.assert_close(Hpos @ Hinv, torch.eye(4), rtol=0, atol=5e-3)
    torch.testing.assert_close(log_det_Hinv, -torch.logdet(H0), rtol=1e-4, atol=1e-3)


def test_reparametrized_target_is_change_of_variables() -> None:
    """U_phi(phi) = U_X(X(phi)) - log|dX/dphi|, with the likelihood evaluated at X."""
    r, b = make_sampler(reparam=True), make_sampler()
    phi = torch.tensor([0.3, -0.4])
    expected = b.neg_log_prob(r.phi_to_X(phi)) - r.log_jacobian(phi)
    torch.testing.assert_close(r.neg_log_prob(phi), expected)


def test_reparametrization_round_trip_and_jacobian() -> None:
    """Check the phi <-> X round trip, and that log_jacobian is log|dX/dphi|."""
    s = make_sampler(reparam=True)
    phi = torch.tensor([-1.2, 0.7])
    torch.testing.assert_close(s.X_to_phi(s.phi_to_X(phi)), phi, rtol=1e-4, atol=1e-4)
    J = torch.autograd.functional.jacobian(s.phi_to_X, phi)
    torch.testing.assert_close(s.log_jacobian(phi), torch.logdet(J))


@pytest.mark.parametrize("reparam", [False, True])
def test_find_map_reaches_minimum(reparam: bool) -> None:
    """
    Check that LBFGS finds the mode (gradient ~0 there), without accumulated grads.

    Parameters
    ----------
    reparam : bool
        Use the reparametrized sampler.
    """
    s = make_sampler(reparam=reparam)
    x0 = torch.zeros(2) if reparam else torch.full((2,), 0.5)
    x_map = s.find_MAP(x0, max_iter=200).detach().requires_grad_(True)
    g = torch.autograd.grad(s.neg_log_prob(x_map), x_map)[0]
    assert g.abs().max() < 1e-3
    assert x_map.grad is None or x_map.grad.abs().max() < 1e-3


@pytest.mark.parametrize("legacy", [False, True])
def test_lp_is_log_posterior_of_each_draw(legacy: bool) -> None:
    """
    Check that ``lp`` is finite and equals -neg_log_prob of the kept draw.

    Parameters
    ----------
    legacy : bool
        Use the legacy proposal.
    """
    s = make_sampler(proposal="legacy" if legacy else "manifold")
    res = run_chain(s, seed=0, burn=20, samples=60)
    assert torch.isfinite(res["lp"]).all()
    assert res["accept"].any()
    with torch.no_grad():
        expected = torch.stack([-s.neg_log_prob(x) for x in res["samples"]])
    torch.testing.assert_close(res["lp"], expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "name, kwargs",
    [
        ("current", {}),
        ("eig", {"use_eig": True}),
    ],
)
def test_legacy_reproduces_old_sampler(name: str, kwargs: dict) -> None:
    """
    Check that ``proposal="legacy"`` reproduces the pre-fix chains draw for draw.

    The reparametrized sampler is not included: before the fix its likelihood
    applied phi -> X twice, so its old chains cannot (and should not) be
    reproduced.

    Parameters
    ----------
    name : str
        Key of the golden record.
    kwargs : dict
        Extra sampler options.
    """
    golden = torch.load(GOLDEN)[name]
    s = make_sampler(proposal="legacy", **kwargs)
    res = run_chain(s, seed=1234, burn=50, samples=150)
    torch.testing.assert_close(res["samples"], golden["samples"])
    torch.testing.assert_close(res["accept"], golden["accept"])
    torch.testing.assert_close(res["step_size"], golden["step_size"])


@pytest.mark.slow
@pytest.mark.parametrize(
    "reparam, proposal", [(False, "manifold"), (True, "manifold"), (False, "fixed")]
)
def test_posterior_moments_match_exact(reparam: bool, proposal: str) -> None:
    """
    Check that four chains reproduce the exact posterior mean and sd.

    Parameters
    ----------
    reparam : bool
        Use the reparametrized sampler.
    proposal : str
        Sampler proposal.
    """
    mean, var = exact_moments()
    draws = torch.cat(
        [
            run_chain(
                make_sampler(reparam=reparam, proposal=proposal),
                seed=c,
                burn=500,
                samples=3000,
            )["samples"]
            for c in range(4)
        ]
    )
    torch.testing.assert_close(draws.mean(0), mean, rtol=0, atol=0.01)
    torch.testing.assert_close(draws.std(0), var.sqrt(), rtol=0.08, atol=0)


def test_fixed_proposal_computes_one_hessian_per_chain() -> None:
    """Check that proposal="fixed" evaluates the Hessian once per chain."""
    s = make_sampler(proposal="fixed")
    for seed in (0, 1):
        s.hessian_counter = 0
        res = run_chain(s, seed=seed, burn=20, samples=80)
        assert s.hessian_counter == 1
        assert res["accept"].any()
        with torch.no_grad():
            expected = torch.stack([-s.neg_log_prob(x) for x in res["samples"]])
        torch.testing.assert_close(res["lp"], expected, rtol=1e-5, atol=1e-5)


def test_new_sampler_rejects_unsupported_options() -> None:
    """Delayed acceptance and stale metrics exist only for the legacy proposal."""
    with pytest.raises(ValueError):
        make_sampler(delayed_accept=True)
    with pytest.raises(ValueError):
        make_sampler(hess_refresh=5)
    make_sampler(delayed_accept=True, hess_refresh=5, proposal="legacy")
    with pytest.raises(ValueError):
        make_sampler(proposal="exact")
    with pytest.warns(DeprecationWarning):
        make_sampler(metric_mode="current")
