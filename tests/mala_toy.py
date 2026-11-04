"""
Small analytic target for the MALA sampler tests.

A linear "emulator" ``Y = A X`` with three observations of two parameters, a
Student-t likelihood and Beta(3, 3) priors on [0, 1]^2. The posterior is
non-Gaussian, so its Hessian changes with position, and two-dimensional, so its
moments can be computed exactly by quadrature on a grid.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from pism_emulator.mcmc.mala import MALASamplerModule, ReparametrizedMALASamplerModule

A = torch.tensor([[1.0, 0.5], [0.2, 1.5], [1.0, -1.0]])
Y_TARGET = torch.tensor([0.9, 1.1, 0.1])
SIGMA = torch.tensor([0.3, 0.3, 0.3])
X_MIN = torch.zeros(2)
X_MAX = torch.ones(2)


class LinearModel(nn.Module):
    """Linear stand-in for an emulator: ``forward(X, add_mean=True) = A X``."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("A", A.clone())

    def forward(self, X: Tensor, add_mean: bool = False) -> Tensor:
        """
        Evaluate the linear model.

        Parameters
        ----------
        X : torch.Tensor
            Parameter vector(s), shape ``(..., 2)``.
        add_mean : bool, optional
            Ignored; the model has zero mean.

        Returns
        -------
        torch.Tensor
            Predictions, shape ``(..., 3)``.
        """
        _ = add_mean
        return X @ self.A.T


def make_sampler(reparam: bool = False, **kwargs) -> MALASamplerModule:
    """
    Build a sampler for the toy target.

    Parameters
    ----------
    reparam : bool, optional
        Build a :class:`ReparametrizedMALASamplerModule` instead.
    **kwargs
        Passed to the sampler.

    Returns
    -------
    MALASamplerModule
        The sampler.
    """
    defaults = dict(
        alpha=1.0,
        nu=4.0,
        metric_mode="current",
        delayed_accept=False,
        h0=0.5,
        show_progress=False,
    )
    defaults.update(kwargs)
    cls = ReparametrizedMALASamplerModule if reparam else MALASamplerModule
    return cls(LinearModel(), X_MIN, X_MAX, Y_TARGET, SIGMA, **defaults)


def run_chain(
    sampler: MALASamplerModule,
    seed: int,
    burn: int,
    samples: int,
    x0: Tensor | None = None,
) -> dict:
    """
    Run one chain through ``predict_step`` without a Trainer.

    Parameters
    ----------
    sampler : MALASamplerModule
        The sampler.
    seed : int
        Seed for torch's global RNG.
    burn, samples : int
        Burn-in and kept draws.
    x0 : torch.Tensor, optional
        Starting point in the sampler's own coordinates; default the centre.

    Returns
    -------
    dict
        Output of ``predict_step``.
    """
    sampler.burn, sampler.samples = burn, samples
    sampler.hparams.burn, sampler.hparams.samples = burn, samples
    if x0 is None:
        x0 = torch.zeros(2) if isinstance(sampler, ReparametrizedMALASamplerModule) else torch.full((2,), 0.5)
    torch.manual_seed(seed)
    return sampler.predict_step((torch.tensor([0]), x0.unsqueeze(0)), 0)


def exact_moments(n: int = 801) -> tuple[Tensor, Tensor]:
    """
    Posterior mean and variance of the toy target by grid quadrature.

    Parameters
    ----------
    n : int, optional
        Grid points per dimension.

    Returns
    -------
    mean, var : torch.Tensor
        Shape ``(2,)`` each.
    """
    s = make_sampler()
    g = torch.linspace(0.0, 1.0, n + 2, dtype=torch.float64)[1:-1]
    X1, X2 = torch.meshgrid(g, g, indexing="ij")
    pts = torch.stack([X1.ravel(), X2.ravel()], dim=1).float()
    with torch.no_grad():
        U = torch.stack([s.neg_log_prob(p) for p in pts]).double()
    w = torch.exp(-(U - U.min()))
    w = w / w.sum()
    P = pts.double()
    mean = (w[:, None] * P).sum(0)
    var = (w[:, None] * (P - mean) ** 2).sum(0)
    return mean.float(), var.float()
