# Copyright (C) 2021-25 Andy Aschwanden, Douglas C Brinkerhoff
#
# This file is part of pism-emulator.
#
# PISM-EMULATOR is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# PISM-EMULATOR is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with PISM; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

# pylint: disable=not-callable,too-many-lines,too-many-instance-attributes,arguments-differ,too-many-statements,arguments-renamed
"""
MALA Sampler.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import math
import sys
import time
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pism_emulator.mcmc.writer import DiskPredictionWriter

tqdm.set_lock(tqdm.get_lock())
_TQDM_NONFATAL = (AttributeError, ValueError, OSError)


class ChainInitDataset(Dataset):
    """
    Dataset of initial states for running multiple MCMC chains.

    This dataset stores a 2-D tensor of initial parameter vectors and
    yields `(chain_id, init_vector)` pairs suitable for a DataLoader.

    Parameters
    ----------
    inits : torch.Tensor
        A tensor of shape ``(n_chains, dim)`` containing the initial
        parameter vectors for each chain. Must be 2-dimensional.

    Attributes
    ----------
    inits : torch.Tensor
        The stored initial states with shape ``(n_chains, dim)``.

    Examples
    --------
    >>> inits = torch.randn(4, 10)  # 4 chains, 10 parameters each
    >>> ds = ChainInitDataset(inits)
    >>> len(ds)
    4
    >>> chain_id, x0 = ds[0]
    >>> chain_id
    0
    >>> x0.shape
    torch.Size([10])
    """

    def __init__(self, inits: torch.Tensor) -> None:
        """
        Init.

        Parameters
        ----------
        inits : torch.Tensor
            Initial values.
        """
        if inits.ndim != 2:
            raise ValueError(
                f"`inits` must be 2-D (n_chains, dim); got shape {tuple(inits.shape)}"
            )
        self.inits: torch.Tensor = inits  # (n_chains, dim)

    def __len__(self) -> int:
        """
        Return the number of chains.

        Returns
        -------
        int
          The length.
        """
        return int(self.inits.shape[0])

    def __getitem__(self, i: int) -> tuple[int, torch.Tensor]:
        """
        Get the ``i``-th chain's initial state.

        Parameters
        ----------
        i : int
            Index of the chain to retrieve.

        Returns
        -------
        tuple of (int, torch.Tensor)
            A pair ``(chain_id, init_vector)`` where ``chain_id == i`` and
            ``init_vector`` has shape ``(dim,)``.
        """
        return i, self.inits[i]


class MALASamplerModule(pl.LightningModule):
    """
    Manifold MALA (mMALA) sampler implemented as a LightningModule.

    This module **does not** perform training; it runs independent MCMC chains
    using a manifold Metropolis-Adjusted Langevin Algorithm and returns
    posterior draws (optionally with per-step statistics).

    Parameters
    ----------
    model : pl.LightningModule
        Forward model used inside the likelihood. Must behave like a
        :class:`torch.nn.Module` and accept a 1D parameter tensor ``X`` with
        signature ``model(X, add_mean=True) -> prediction``.
    X_min : array-like of float or torch.Tensor, shape (D,)
        Element-wise lower bounds used to normalize parameters for the Beta prior.
    X_max : array-like of float or torch.Tensor, shape (D,)
        Element-wise upper bounds used to normalize parameters for the Beta prior.
    Y_target : array-like or torch.Tensor, shape (N,)
        Observed target vector on which the emulator is conditioned.
    sigma_hat : array-like or torch.Tensor, shape (N,)
        Per-node standard deviation used in the Student-t likelihood.
    alpha : float, default=0.01
        Weight on the (log) likelihood contribution in the posterior.
    alpha_b : float, default=3.0
        Beta prior shape parameter :math:`\\alpha` for normalized parameters.
    beta_b : float, default=3.0
        Beta prior shape parameter :math:`\\beta` for normalized parameters.
    nu : float, default=1.0
        Degrees of freedom of the Student-t likelihood.
    metric_mode : {"manifold", "current"}, default="manifold"
        Reverse proposal metric. If ``"manifold"``, compute geometry at the
        proposal :math:`x'`; if ``"current"``, reuse :math:`H(x)` for both
        directions.
    hess_refresh : int, default=1
        Recompute the local geometry (gradient & Hessian) every ``N`` steps (``N >= 1``).
    delayed_accept : bool, default=False
        Enable two-stage delayed acceptance to avoid computing geometry at
        obviously poor proposals.
    adapt_method : {"dual", "ema"}, default="ema"
        Step-size adaptation during burn-in (dual-averaging à la NUTS/Stan, or
        exponentially weighted moving average).
    h0 : float, default=0.1
        Initial step size.
    h_min : float, default=1e-3
        Minimum allowed step size during adaptation.
    h_max : float, default=1.0
        Maximum allowed step size during adaptation.
    use_eig : bool, default=False
        Use linalg.eigh to compute :math:`H(x)`.
    acc_target : float, default=0.25
        Target acceptance probability used by the adaptation logic.
    dual_t0 : float, default=10.0
        Dual-averaging stabilizer (see Hoffman & Gelman 2014).
    dual_kappa : float, default=0.75
        Dual-averaging shrinkage exponent.
    dual_gamma : float, default=0.05
        Dual-averaging learning-rate scale.
    k_adapt : float, default=0.01
        EMA adaptation gain (only if ``adapt_method="ema"``).
    beta : float, default=0.99
        EMA decay factor (only if ``adapt_method="ema"``).
    burn : int, default=500
        Number of burn-in iterations (discarded).
    samples : int, default=2000
        Number of post-burn samples stored per chain.
    show_progress : bool, default=True
        If True, render per-chain progress bars via :mod:`tqdm`.
    pbar_update_every : int, default=10
        Update rate (in steps) for progress-bar postfix text.
    q : int, default=100
        Reserved parameter (e.g., truncation rank for a low-rank geometry); not
        used in the default exact‐Hessian path.
    seed : int or None, default=None
        Base RNG seed; the effective per-chain seed is derived from this value.
    **kwargs : Any
        Ignored. Included for forward compatibility with Lightning APIs.

    Attributes
    ----------
    model : torch.nn.Module
        The wrapped forward model (set to ``eval()``; parameters frozen).
    X_min, X_max : torch.Tensor, shape (D,)
        Registered buffers holding bounds for prior normalization.
    Y_target : torch.Tensor, shape (N,)
        Registered buffer with the observation vector.
    sigma_hat : torch.Tensor, shape (N,)
        Registered buffer with per-node standard deviations.
    alpha, alpha_b, beta_b, nu : torch.Tensor
        Scalar buffers storing distribution hyperparameters.
    samples : int
        Number of post-burn samples per chain.
    burn : int
        Number of burn-in iterations per chain.
    show_progress : bool
        Whether this instance attempts to draw progress bars.

    Returns
    -------
    dict (per-chain, from ``predict_step``)
        A mapping with:
        - ``"chain"`` : int — chain id (rank),
        - ``"samples"`` : torch.Tensor, shape (S, D) — post-burn samples,
        - ``"lp"`` : torch.Tensor, shape (S,) — log posterior per step (optional),
        - ``"step_size"`` : torch.Tensor, shape (S,) — step size per step (optional),
        - ``"accept"`` : torch.Tensor, shape (S,) of bool — acceptance indicator.

        When running multi-process CPU sampling (``ddp_spawn``), predictions may
        be written to disk via a callback and re-assembled by the driver.

    See Also
    --------
    lightning.pytorch.Trainer.predict : Run inference with Lightning modules.

    Notes
    -----
    * Designed for ``Trainer.predict`` with either:
      - single-process (GPU/MPS/CPU), or
      - multi-process CPU via ``ddp_spawn`` (one chain per process).
    * The negative log posterior combines a Student-t likelihood (``nu``) and
      independent Beta priors on normalized parameters
      :math:`\\bar X = (X - X_{\\min}) / (X_{\\max} - X_{\\min})`.
    * Local geometry is obtained via exact Hessian by default; if you add a
      low-rank path, use ``q`` to control the truncation rank.

    Examples
    --------
    Create and run 4 CPU chains with DDP‐spawn:

    >>> sampler = MALASamplerModule(model, X_min, X_max, Y_obs, sigma_hat, samples=2000)
    >>> dl = DataLoader(ChainInitDataset(inits), batch_size=1, shuffle=False)
    >>> trainer = pl.Trainer(accelerator="cpu", devices=4, strategy="ddp_spawn")
    >>> preds = trainer.predict(sampler, dl, return_predictions=True)
    >>> # stack all chains' samples:
    >>> chains = torch.stack([p["samples"] for p in preds])  # (C, S, D)
    """

    def __init__(
        self,
        model: pl.LightningModule,
        X_min: Tensor | np.ndarray | list[float],
        X_max: Tensor | np.ndarray | list[float],
        Y_target: Tensor | np.ndarray | list[float],
        sigma_hat: Tensor | np.ndarray | list[float],
        *,
        alpha: float = 0.01,
        alpha_b: float = 3.0,
        beta_b: float = 3.0,
        nu: float = 1.0,
        metric_mode: Literal["manifold", "current"] = "manifold",
        hess_refresh: int = 1,
        delayed_accept: bool = False,
        adapt_method: Literal["dual", "ema"] = "ema",
        h0: float = 0.1,
        h_min: float = 1e-3,
        h_max: float = 1.0,
        use_eig: bool = False,
        acc_target: float = 0.25,
        dual_t0: float = 10.0,
        dual_kappa: float = 0.75,
        dual_gamma: float = 0.05,
        k_adapt: float = 0.01,
        beta: float = 0.99,
        burn: int = 500,
        samples: int = 2000,
        show_progress: bool = True,
        pbar_update_every: int = 10,
        q: int = 100,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])  # everything except the model

        _ = (
            adapt_method,
            h0,
            h_min,
            h_max,
            acc_target,
            dual_t0,
            dual_kappa,
            dual_gamma,
            k_adapt,
            beta,
        )

        if isinstance(model, torch.nn.Module):
            self.model: torch.nn.Module = model.eval()
        else:
            raise TypeError("`model` must be an nn.Module / LightningModule")

        # Buffers follow device automatically
        self.register_buffer(
            "X_min", torch.as_tensor(X_min, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "X_max", torch.as_tensor(X_max, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "Y_target", torch.as_tensor(Y_target, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "sigma_hat",
            torch.as_tensor(sigma_hat, dtype=torch.float32),
            persistent=False,
        )

        self.register_buffer(
            "alpha", torch.as_tensor(alpha, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "alpha_b", torch.as_tensor(alpha_b, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "beta_b", torch.as_tensor(beta_b, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "nu", torch.as_tensor(nu, dtype=torch.float32), persistent=False
        )

        self.register_buffer(
            "_two_pi",
            torch.tensor(2.0 * math.pi, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("q", torch.tensor(q, dtype=torch.int), persistent=False)

        # Numerics / state
        self._base_seed: int = 0 if seed is None else int(seed)
        self._eps_beta: float = 1e-7
        self._eps_eig: float = 1e-6
        self._pbar: Optional[tqdm] = None
        self._rank: int = 0
        self._step_count: int = 0
        self._total_steps: int = 0
        self.burn: int = int(burn)
        self.delayed_accept: bool = bool(delayed_accept)
        self.hessian_counter: int = 0
        self.hess_refresh: int = int(hess_refresh)
        self.metric_mode: Literal["manifold", "current"] = metric_mode
        self.pbar_update_every: int = int(pbar_update_every)
        self.samples: int = int(samples)
        self.show_progress: bool = show_progress
        self.use_eig: bool = use_eig

    def configure_optimizers(self) -> None:
        """
        Lightning hook. This module does not optimize/train any parameters.
        """
        return None

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """
        Forward model wrapper.

        Parameters
        ----------
        *args
            Positional arguments. If provided, ``args[0]`` must be ``X``.
        **kwargs
            Keyword arguments. Must contain ``X`` if ``*args`` is empty.

        Returns
        -------
        torch.Tensor
            Model prediction consistent with ``Y_target`` shape.
        """
        if args:
            X = args[0]
        elif "X" in kwargs:
            X = kwargs["X"]
        else:
            raise TypeError("forward() missing required argument: X")

        return self.model(X, add_mean=True)

    def neg_log_prob(self, X: Tensor) -> Tensor:
        """
        Negative log-posterior at ``X``.

        Combines a Student-t log-likelihood with element-wise Beta priors
        on parameters transformed to :math:`[0,1]` via
        ``(X - X_min) / (X_max - X_min)`` (clamped to avoid log(0)).

        Parameters
        ----------
        X : torch.Tensor
            Parameter vector (leaf tensor with ``requires_grad=True`` recommended).

        Returns
        -------
        torch.Tensor
            Scalar tensor with the negative log-posterior.
        """
        Y_pred = self.forward(X)
        r = Y_pred - self.Y_target
        t = r / self.sigma_hat
        nu = self.nu

        # Student-t log-likelihood (sum over observation dimension)
        log_like = (
            torch.special.gammaln((nu + 1) * 0.5)
            - torch.special.gammaln(nu * 0.5)
            - 0.5 * torch.log(torch.pi * nu)
            - torch.log(self.sigma_hat)
            - 0.5 * (nu + 1.0) * torch.log1p((t * t) / nu)
        ).sum()

        # Beta prior on normalized parameters
        X_bar = torch.clamp(
            (X - self.X_min) / (self.X_max - self.X_min),
            self._eps_beta,
            1 - self._eps_beta,
        )
        log_prior = (
            (self.alpha_b - 1.0) * torch.log(X_bar)
            + (self.beta_b - 1.0) * torch.log(1.0 - X_bar)
            + torch.lgamma(self.alpha_b + self.beta_b)
            - torch.lgamma(self.alpha_b)
            - torch.lgamma(self.beta_b)
        ).sum()

        return -(self.alpha * log_like + log_prior)

    @torch.enable_grad()
    def _local_geometry_eig(
        self, X: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute local geometry at ``X``: negative log-posterior, gradient and Hessian.

        Parameters
        ----------
        X : torch.Tensor
            Current state (requires grad).

        Returns
        -------
        tuple
            ``(log_pi, g, Hpos, Hinv, log_det_Hinv)``.
        """
        log_pi = self.neg_log_prob(X)
        self.hessian_counter += 1

        g = torch.autograd.grad(log_pi, X, retain_graph=True, create_graph=False)[0]
        H = torch.autograd.functional.hessian(
            self.neg_log_prob, X, vectorize=False, create_graph=False
        )
        H = 0.5 * (H + H.T)

        # Eigen decomposition (symmetric)
        lam, Q = torch.linalg.eigh(H)
        lam = lam.real
        Q = Q.real
        lam_p = torch.sqrt(lam * lam + self._eps_eig)
        Hpos = Q @ torch.diag(lam_p) @ Q.T
        Hinv = Q @ torch.diag(1.0 / lam_p) @ Q.T
        log_det_Hinv = torch.sum(torch.log(1.0 / lam_p))
        return log_pi, g, Hpos, Hinv, log_det_Hinv

    @torch.enable_grad()
    def _local_geometry(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute local geometry.

        Compute local geometry of the negative log-posterior at a point ``X``:
        value, gradient, and a positive-(semi)definite Hessian proxy together with
        its inverse and log-determinant factor.

        This MPS-safe implementation avoids eigen-decomposition by forming the
        symmetric Hessian ``H`` via autograd, then operating on the SPD matrix
        ``S = H @ H + eps * I``. The matrix square root and inverse square root of
        ``S`` are obtained through a Newton–Schulz iteration (matmul-only), and the
        log-determinant is computed from a Cholesky factorization with adaptive
        jitter.

        Parameters
        ----------
        X : torch.Tensor
            Current state (parameter vector) at which to evaluate the local geometry.
            Must require gradients (``requires_grad=True``). Shape ``(d,)`` or ``(d, 1)``
            and on the same device/dtype expected by the model.

        Returns
        -------
        log_pi : torch.Tensor
            Scalar negative log-posterior evaluated at ``X``; shape ``()``.
        g : torch.Tensor
            Gradient of ``log_pi`` w.r.t. ``X``; shape ``(d,)``.
        Hpos : torch.Tensor
            Positive-(semi)definite matrix approximating
            ``sqrt(H @ H + eps * I)``; shape ``(d, d)``.
        Hinv : torch.Tensor
            Inverse of ``Hpos``, i.e., ``(H @ H + eps * I)^{-1/2}``; shape ``(d, d)``.
        log_det_Hinv : torch.Tensor
            Scalar equal to ``-0.5 * log|H @ H + eps * I|``.

        Notes
        -----
        - ``H`` is constructed via ``torch.autograd.functional.hessian`` and
          symmetrized as ``0.5 * (H + H.T)``.
        - The SPD matrix ``S = H @ H + eps * I`` shares eigenvectors with ``H`` and
          has eigenvalues ``lambda(H)^2 + eps``; thus
          ``Hpos = S^{1/2}`` and ``Hinv = S^{-1/2}``.
        - The Newton–Schulz iteration uses only matrix multiplications and additions,
          making it compatible with MPS where ``torch.linalg.eigh`` may be unavailable.
        - The Cholesky step employs adaptive jitter to ensure positive-definiteness.
          If Cholesky still fails, a small CPU fallback using ``slogdet`` may be used
          solely for the log-determinant (dimension is typically small).
        """
        # 1) Negative log-posterior and gradient
        log_pi = self.neg_log_prob(X)
        self.hessian_counter += 1

        g = torch.autograd.grad(log_pi, X, retain_graph=True, create_graph=False)[0]

        # 2) Hessian (symmetrize for safety)
        H = torch.autograd.functional.hessian(
            self.neg_log_prob, X, vectorize=False, create_graph=False
        )
        H = 0.5 * (H + H.T)

        # 3) Build SPD matrix S = H^2 + eps*I (but we’ll add eps adaptively below)
        d = X.numel()
        I = torch.eye(d, device=H.device, dtype=H.dtype)
        S = H @ H
        S = 0.5 * (S + S.T)  # enforce symmetry to reduce numerical asymmetry

        def _mat_sqrt_inv_sqrt(
            A: torch.Tensor, iters: int = 6
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Compute square root and inverse square root.

            Compute the matrix square root and inverse square root of a symmetric
            positive-(semi)definite matrix using the Newton–Schulz iteration.

            The iteration is applied to the scaled matrix ``A / ||A||_F`` for
            numerical stability, then rescaled to obtain ``A^{1/2}`` and
            ``A^{-1/2}``. The method uses only matrix multiplications and additions,
            making it suitable for devices where eigen-decompositions are not
            available.

            Parameters
            ----------
            A : torch.Tensor
                Square input matrix of shape ``(d, d)`` on the current device and
                dtype. Should be symmetric and (near) positive-definite; in practice
                you may add a small jitter ``eps * I`` before calling this routine.
            iters : int, optional
                Number of Newton–Schulz iterations. Typical values in ``[5, 8]``
                yield ~1e-4 to ~1e-6 relative accuracy for well-conditioned inputs.
                Default is ``6``.

            Returns
            -------
            root : torch.Tensor
                Matrix square root ``A^{1/2}`` of shape ``(d, d)`` with the same
                device and dtype as ``A``.
            invrt : torch.Tensor
                Matrix inverse square root ``A^{-1/2}`` of shape ``(d, d)`` with the
                same device and dtype as ``A``.

            Notes
            -----
            The iteration updates are

            ``Y_{k+1} = Y_k * (3I - Z_k * Y_k) / 2``,
            ``Z_{k+1} = (3I - Z_k * Y_k) * Z_k / 2``,

            initialized with ``Y_0 = A_s`` and ``Z_0 = I``, where
            ``A_s = A / ||A||_F``. Upon convergence,
            ``root = Y * sqrt(||A||_F)`` and ``invrt = Z / sqrt(||A||_F)``.

            References
            ----------
            * Higham, N. J. and Lin, L. (2011). "A New Scaling and Squaring Algorithm
              for the Matrix Exponential." SIAM J. Matrix Anal. Appl.
            * Ionescu, C. et al. (2015). "Matrix Backpropagation for Deep Networks
              with Structured Layers." ICCV.
            """
            # Scale for better convergence (Frobenius norm is fine and device-friendly)
            scale = A.norm(p="fro").clamp_min(1e-12)
            As = A / scale

            Y = As.clone()  # approx sqrt(As)
            Z = torch.eye(
                A.size(-1), device=A.device, dtype=A.dtype
            )  # approx invsqrt(As)

            I = Z  # reuse allocated identity
            for _ in range(iters):
                T = 0.5 * (3.0 * I - Z @ Y)
                Y = Y @ T
                Z = T @ Z

            root = Y * torch.sqrt(scale)
            invrt = Z / torch.sqrt(scale)
            return root, invrt

        # 4) Adaptive jitter so Cholesky(S + eps I) succeeds for log-det
        eps = float(getattr(self, "_eps_eig", 1e-8))
        growth = 10.0
        max_tries = 6

        Hpos = Hinv = log_det_Hinv = None  # will be set below

        for _ in range(max_tries):
            S_eps = S + eps * I
            try:
                # Cholesky is fast/stable and implemented on MPS
                L = torch.linalg.cholesky(S_eps)
                # Now compute √ and √⁻¹ with NS iteration (matmuls only → MPS OK)
                Hpos, Hinv = _mat_sqrt_inv_sqrt(S_eps, iters=6)
                # log|Hinv| = -1/2 * log|S_eps|
                log_det_S = 2.0 * torch.log(torch.diagonal(L)).sum()
                log_det_Hinv = -0.5 * log_det_S
                break
            except RuntimeError:
                # Leading minor not PD or other numeric issue: increase jitter
                eps *= growth
        else:
            # Final fallback: get logdet robustly on CPU via slogdet (dimension is tiny)
            S_cpu = (S + eps * I).detach().cpu()
            sign, logabs = torch.linalg.slogdet(S_cpu)  # works for SPD/PSD
            if sign.item() <= 0:
                eps *= growth
                sign, logabs = torch.linalg.slogdet(
                    (S_cpu + eps * torch.eye(d)).contiguous()
                )
            log_det_Hinv = (-0.5 * logabs).to(S.device)
            # Still produce Hpos/Hinv on-device
            Hpos, Hinv = _mat_sqrt_inv_sqrt(S + eps * I, iters=7)

        return log_pi, g, Hpos, Hinv, log_det_Hinv

    @staticmethod
    def _proposal_logpdf(
        y: Tensor,
        mu: Tensor,
        H: Tensor,
        log_det_Hinv: Tensor,
        h: float,
        two_pi: Tensor,
    ) -> Tensor:
        r"""
        Log-density of Gaussian proposal :math:`\\mathcal{N}(\\mu,\, 2h\\,H^{-1})`.

        Parameters
        ----------
        y : torch.Tensor
            Evaluation point (shape ``(D,)``).
        mu : torch.Tensor
            Mean vector (shape ``(D,)``).
        H : torch.Tensor
            Positive-definite matrix proportional to inverse covariance (``(D, D)``).
        log_det_Hinv : torch.Tensor
            Scalar log-determinant of :math:`H^{-1}`.
        h : float
            MALA step size.
        two_pi : torch.Tensor
            Constant tensor with value :math:`2\\pi`.

        Returns
        -------
        torch.Tensor
            Scalar log-density value.
        """
        d = y.numel()
        delta = (y - mu).unsqueeze(-1)
        quad = (delta.transpose(0, 1) @ (H @ delta)).squeeze() / (2.0 * h)
        logdet_Sigma = d * np.log(2.0 * h) + log_det_Hinv
        return -0.5 * (d * torch.log(two_pi) + logdet_Sigma + quad)

    def _mala_step(
        self,
        X: Tensor,
        h: float,
        local: Optional[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = None,
    ) -> tuple[
        Tensor, Optional[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]], int, float
    ]:
        """
        Perform one mMALA/DA-mMALA step.

        Parameters
        ----------
        X : torch.Tensor
            Current state (leaf tensor with ``requires_grad=True``).
        h : float
            Current step size.
        local : tuple or None, optional
            Cached local geometry at ``X`` as returned by :meth:`_local_geometry`.

        Returns
        -------
        tuple
            ``(X_next, local_next, accepted, alpha)`` where:

            * ``X_next`` : next state (possibly unchanged on reject)
            * ``local_next`` : geometry tuple at the accepted state or ``None`` if invalidated
            * ``accepted`` : 1 if accepted else 0
            * ``alpha`` : acceptance probability used for the decision
        """
        # refresh metric at x if needed
        if (
            (local is None)
            or (self.hess_refresh == 1)
            or (self._step_count % self.hess_refresh == 0)
        ):
            if self.use_eig:
                local = self._local_geometry_eig(X)
            else:
                local = self._local_geometry(X)
        log_pi, _, H, Hinv, log_det_Hinv = local

        # Propose
        with torch.no_grad():
            L = torch.linalg.cholesky(2.0 * h * Hinv)
            eps = torch.randn_like(X)
            Xp = (X + L @ eps).detach()
        Xp.requires_grad_(True)

        # Target at proposal
        log_pi_p = self.neg_log_prob(Xp)
        # forward q(x'|x) using H(x)
        logq_f = self._proposal_logpdf(Xp, X, H, log_det_Hinv, h, self._two_pi)

        if not self.delayed_accept or self.metric_mode == "current":
            # reverse q(x|x') also using H(x)  (CURRENT metric variant)
            logq_r = self._proposal_logpdf(X, Xp, H, log_det_Hinv, h, self._two_pi)
            log_alpha = -log_pi_p + logq_r + log_pi - logq_f
            alpha = (
                torch.exp(torch.clamp(log_alpha, max=0.0))
                if torch.isfinite(log_alpha)
                else X.new_zeros(())
            )
            accept = torch.rand(()) <= alpha
            if accept:
                X = Xp.detach().requires_grad_(True)
                local = None  # recompute metric at new point next step
                s = 1
            else:
                s = 0
            self._step_count += 1
            return X, local, s, float(alpha.detach())

        # Delayed-acceptance (manifold)
        logq_r_cheap = self._proposal_logpdf(X, Xp, H, log_det_Hinv, h, self._two_pi)
        log_alpha_cheap = -log_pi_p + logq_r_cheap + log_pi - logq_f
        a1 = (
            torch.exp(torch.clamp(log_alpha_cheap, max=0.0))
            if torch.isfinite(log_alpha_cheap)
            else X.new_zeros(())
        )
        if torch.rand(()) > a1:
            self._step_count += 1
            return X, local, 0, float(a1.detach())

        # compute geometry at x' only if stage-1 accepted
        loc_p = self._local_geometry(Xp)
        _, _, H_p, _, log_det_Hinv_p = loc_p
        logq_r_true = self._proposal_logpdf(X, Xp, H_p, log_det_Hinv_p, h, self._two_pi)
        # correction gate
        log_alpha_corr = logq_r_true - logq_r_cheap
        a2 = (
            torch.exp(torch.clamp(log_alpha_corr, max=0.0))
            if torch.isfinite(log_alpha_corr)
            else X.new_zeros(())
        )
        if torch.rand(()) <= a2:
            X = Xp.detach().requires_grad_(True)
            local = loc_p
            s = 1
        else:
            s = 0

        self._step_count += 1
        return X, local, s, float((a1 * a2).detach())

    # ------------------------------------------------------------------ #
    # MAP search                                                         #
    # ------------------------------------------------------------------ #
    def find_MAP(
        self,
        X: Tensor,
        max_iter: int = 100,
        lr: float = 1.0,
    ) -> Tensor:
        """
        Find a MAP estimate via L-BFGS on the negative log-posterior.

        Parameters
        ----------
        X : torch.Tensor
            Initial point (will be detached and made a leaf with grad).
        max_iter : int, default=100
            Maximum number of L-BFGS iterations (passed to the optimizer).
        lr : float, default=1.0
            L-BFGS learning rate.

        Returns
        -------
        torch.Tensor
            The optimized MAP point (same device as module buffers).
        """
        X = X.detach().to(device=self.device, dtype=torch.float32).requires_grad_(True)

        def closure() -> Tensor:
            """
            Closure.

            Returns
            -------

            torch.Tensor
                Loss tensor.
            """
            self.zero_grad(set_to_none=True)
            loss = self.neg_log_prob(X)
            loss.backward()
            return loss

        opt = torch.optim.LBFGS(
            [X], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe"
        )
        _ = opt.step(closure)
        self.zero_grad(set_to_none=True)
        return X

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Any]:
        """
        One predict step = one chain.

        Parameters
        ----------
        batch : torch.Tensor
            Tuple-like batch where ``batch[0]`` is an integer chain id and
            ``batch[1]`` is the initial parameter vector for that chain.
        batch_idx : int
            Lightning-provided batch index (unused).
        dataloader_idx : int, optional
            Dataloader index provided by Lightning (unused).

        Returns
        -------
        dict
            Dictionary with fields:

            * ``chain`` : int, the chain id
            * ``samples`` : (S, D) tensor of post-burn samples on CPU
            * ``lp`` : (S,) tensor, log posterior per kept step (CPU)
            * ``step_size`` : (S,) tensor, step size per kept step (CPU)
            * ``accept`` : (S,) boolean tensor, acceptance indicator per kept step (CPU)
        """
        _ = batch_idx
        _ = dataloader_idx

        chain_id, _X0 = batch
        chain_id = (
            int(chain_id) if isinstance(chain_id, torch.Tensor) else int(chain_id)
        )
        if _X0.dim() > 1 and _X0.size(0) == 1:
            _X0 = _X0.squeeze(0)
        X = _X0.detach().requires_grad_(True)

        burn, samples = int(self.hparams.burn), int(self.hparams.samples)
        total = burn + samples

        # step-size & adaptation params
        h = float(self.hparams.h0)
        h_min, h_max = float(self.hparams.h_min), float(self.hparams.h_max)
        acc_target = float(self.hparams.acc_target)
        adapt_method = self.hparams.adapt_method
        k_adapt, beta = float(self.hparams.k_adapt), float(self.hparams.beta)
        dual_t0, dual_kappa, dual_gamma = (
            float(self.hparams.dual_t0),
            float(self.hparams.dual_kappa),
            float(self.hparams.dual_gamma),
        )

        kept: list[Tensor] = []

        # per-step histories (kept as tensors, sliced post-burn once)
        lp_hist: list[Tensor] = []
        h_hist: list[Tensor] = []
        acc_hist: list[Tensor] = []

        local = None
        acc_ema = acc_target
        if adapt_method == "dual":
            n = 0
            log_h = math.log(max(h, 1e-12))
            mu = math.log(10.0 * h)
            Hbar = 0.0
            log_hbar = log_h

        dev = X.device
        for t in range(total):
            X, local, s, _ = self._mala_step(X, h, local)

            # log posterior (as tensor; avoid .item() inside loop)
            if local is not None and len(local) >= 1:
                lp_t = (-local[0]).detach()
            else:
                lp_t = torch.tensor(float("nan"), device=dev)

            lp_hist.append(lp_t.to(torch.float32))
            h_hist.append(torch.tensor(h, device=dev, dtype=torch.float32))
            acc_hist.append(torch.tensor(s, device=dev, dtype=torch.int8))

            if t >= burn:
                kept.append(X.detach())

            # adapt during burn-in
            if t < burn:
                if adapt_method == "dual":
                    n += 1
                    alpha_n = float(s)
                    Hbar = (1.0 - 1.0 / (n + dual_t0)) * Hbar + (
                        1.0 / (n + dual_t0)
                    ) * (acc_target - alpha_n)
                    log_h = mu - (math.sqrt(n) / dual_gamma) * Hbar
                    log_hbar = (n ** (-dual_kappa)) * log_h + (
                        1.0 - n ** (-dual_kappa)
                    ) * log_hbar
                    h = float(math.exp(log_h))
                else:  # EMA
                    acc_ema = beta * acc_ema + (1.0 - beta) * float(s)
                    h = h * (1.0 + k_adapt * math.copysign(1.0, acc_ema - acc_target))
                h = min(max(h, h_min), h_max)
            elif t == burn and adapt_method == "dual":
                h = float(math.exp(log_hbar))
                h = min(max(h, h_min), h_max)

            # display-only fetch (safe occasional .item())
            logP_disp = (
                float(lp_t.detach().cpu().item())
                if torch.isfinite(lp_t)
                else float("nan")
            )
            self._update_bar(t, h, logP_disp)

            # re-leaf for next step
            X = X.detach().requires_grad_(True)

        samples_out = torch.stack(kept)  # (S, D)
        lp_arr = torch.stack(lp_hist)[burn:].to(torch.float32)  # (S,)
        h_arr = torch.stack(h_hist)[burn:].to(torch.float32)  # (S,)
        acc_arr = torch.stack(acc_hist)[burn:].to(torch.bool)  # (S,)

        return {
            "chain": chain_id,
            "samples": samples_out,
            "lp": lp_arr,
            "step_size": h_arr,
            "accept": acc_arr,
        }

    # ------------------------------------------------------------------ #
    # Progress bar helpers                                               #
    # ------------------------------------------------------------------ #
    def on_predict_start(self) -> None:
        """
        Lightning hook executed at the beginning of prediction.

        Initializes RNG seeds, prepares progress-bar layout, and computes the
        total number of steps as ``burn + samples``.
        """
        self._rank = int(getattr(self.trainer, "global_rank", 0))
        s = self._base_seed + 1000 * self._rank + 12345
        torch.manual_seed(s)
        np.random.seed(s % (2**32))
        self._total_steps = self.burn + self.samples

        bar_fmt = (
            f"chain {self._rank}: "
            "{percentage:>3.0f}%|{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )
        disable_bars = not sys.stdout.isatty() or not self.show_progress

        self._pbar = tqdm(
            total=self._total_steps,
            position=self._rank,
            leave=True,
            ncols=120,
            dynamic_ncols=False,
            ascii=True,
            bar_format=bar_fmt,
            mininterval=0.25,
            disable=disable_bars,
        )

    def on_predict_end(self) -> None:
        """
        Lightning hook executed at the end of prediction.

        Cleans up the progress bar (if enabled).
        """
        p = self._pbar
        self._pbar = None
        if p is None:
            return

        # Best-effort: cosmetic updates; ignore non-fatal tqdm/IO/state issues
        with contextlib.suppress(*_TQDM_NONFATAL):
            p.set_postfix_str("done", refresh=True)
            p.refresh()

        # Best-effort: disable + close; ignore non-fatal tqdm/IO/state issues
        with contextlib.suppress(*_TQDM_NONFATAL):
            p.disable = True
            p.close()

    def _update_bar(self, t: int, h: float, logp: float) -> None:
        """
        Update the chain's progress bar.

        Updates the progress bar.

        Parameters
        ----------
        t : int
            Current step (0-based).
        h : float
            Current step size (for display).
        logp : float
            Current (display) log posterior value.

        Notes
        -----
        The bar postfix is updated at most every ``pbar_update_every`` steps
        to reduce rendering overhead in multi-process runs.
        """
        if self._pbar is None:
            return
        sample_str = "—" if t < self.burn else f"{t - self.burn + 1}"
        if (t % self.pbar_update_every) == 0 or (t + 1) == self._total_steps:
            self._pbar.set_postfix_str(
                f"h={h:.3f}  logp={logp:.3f}  sample={sample_str}", refresh=True
            )
        self._pbar.update(1)


def make_trainer_for_chains(accelerator: str, n_chains: int) -> pl.Trainer:
    """
    Create a minimal Trainer configured for single- or multi-chain inference.

    CPU runs can use multiple processes (one per chain). GPU/MPS runs use a single
    device and run one chain per call to ``predict``.

    Parameters
    ----------
    accelerator : str
        One of ``{"cpu", "cuda", "gpu", "mps"}`` as accepted by Lightning.
    n_chains : int
        Number of chains to run. On CPU, this sets the number of processes.

    Returns
    -------
    pl.Trainer
        A Trainer instance with logging/checkpointing disabled and inference mode
        enabled as appropriate for prediction-only workloads.
    """
    if accelerator.lower() == "cpu" and n_chains > 1:
        devices = n_chains
        strategy = "ddp_spawn"  # safe on macOS/Windows; uses spawn
    else:
        devices = 1  # one chain per (single) GPU/MPS
        strategy = "auto"

    return pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=False,
        enable_checkpointing=False,
        inference_mode=False,  # we need autograd for MALA
        num_sanity_val_steps=0,
    )


def run_sampling(
    sampler: pl.LightningModule,
    inits: Tensor,
    accelerator: str = "cpu",
    tmp_dir: str | Path = "./_preds",
) -> dict[str, Tensor]:
    """
    Run MCMC chains.

    Parameters
    ----------
    sampler : pl.LightningModule
        The sampling module implementing ``predict_step`` that returns a dict with
        at least ``{"chain": int, "samples": Tensor}`` and optionally
        ``"lp"``, ``"step_size"``, and ``"accept"`` (all shaped ``(S,)``).
    inits : torch.Tensor
        Initial states per chain of shape ``(C, D)``.
    accelerator : str, default "cpu"
        Lightning accelerator string (``"cpu"``, ``"cuda"``/``"gpu"``, or ``"mps"``).
    tmp_dir : str or pathlib.Path, default "./_preds"
        Temporary directory for per-chain files in multi-process CPU mode.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with keys:
          - ``"samples"`` : ``(C, S, D)`` float32 tensor
          - ``"lp"`` : ``(C, S)`` float32 tensor (if provided)
          - ``"step_size"`` : ``(C, S)`` float32 tensor (if provided)
          - ``"accept"`` : ``(C, S)`` bool tensor (if provided)
    """
    dl = DataLoader(ChainInitDataset(inits), batch_size=1, shuffle=False, num_workers=0)

    n_chains = int(inits.shape[0])
    multi_cpu = (accelerator == "cpu") and (n_chains > 1)

    def _stack_single_outs(outs: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
        """
        Stack single-process prediction outputs into chain-major tensors.

        This helper filters out ``None`` entries, sorts records by the integer value
        of the ``"chain"`` field, and stacks common keys across chains. It always
        returns a ``"samples"`` tensor of shape ``(C, ...)`` where ``C`` is the
        number of chains. Optional keys among ``{"lp", "step_size", "accept"}``
        are included if present in **all** records and stacked to shape ``(C, ...)``
        as well.

        Parameters
        ----------
        outs : Sequence[dict[str, Tensor]]
            Iterable of per-process prediction dictionaries. Each dictionary must
            contain:
              - ``"chain"``: an integer (or string convertible to int) identifying the chain
              - ``"samples"``: a :class:`torch.Tensor` with arbitrary trailing shape
            May additionally contain keys ``"lp"``, ``"step_size"``, and ``"accept"``.
            ``None`` entries are ignored.

        Returns
        -------
        dict[str, Tensor]
            Dictionary with stacked tensors:
              - ``"samples"`` : ``(C, ...)``
              - ``"lp"``, ``"step_size"``, ``"accept"`` : ``(C, ...)`` if available for all chains.

        Notes
        -----
        Records are sorted by ``int(d["chain"])`` before stacking to ensure
        deterministic chain ordering.
        """
        outs = [o for o in outs if o is not None]
        outs = sorted(outs, key=lambda d: int(d["chain"]))
        stats: dict[str, Tensor] = {
            "samples": torch.stack([d["samples"] for d in outs])
        }
        for k in ("lp", "step_size", "accept"):
            if all((k in d) for d in outs):
                stats[k] = torch.stack([d[k] for d in outs])
        return stats

    def _stack_from_disk(
        pred_dir: str | Path, expected_chains: int | None = None
    ) -> dict[str, Tensor]:
        """
        Load and stack per-chain prediction tensors saved on disk.

        Finds files matching ``rank*_chain*.pt`` in ``pred_dir``, loads them with
        :func:`torch.load`, sorts by the ``"chain"`` field, and stacks common keys
        across chains. As with :func:`_stack_single_outs`, ``"samples"`` is required
        and optional keys among ``{"lp", "step_size", "accept"}`` are stacked only
        if present in **all** records.

        Parameters
        ----------
        pred_dir : str or pathlib.Path
            Directory containing files named like ``rank*_chain*.pt``. Each file
            must serialize a dict with at least ``"chain"`` and ``"samples"``.
        expected_chains : int, optional
            If provided, validate that the number of stacked chains equals this
            value. A mismatch raises a :class:`RuntimeError`.

        Returns
        -------
        dict[str, Tensor]
            Dictionary with stacked tensors:
              - ``"samples"`` : ``(C, ...)``
              - ``"lp"``, ``"step_size"``, ``"accept"`` : ``(C, ...)`` if available for all chains.

        Raises
        ------
        RuntimeError
            If no matching files are found in ``pred_dir``.
        RuntimeError
            If ``expected_chains`` is given and does not match the number of
            stacked chains.

        Notes
        -----
        Files are sorted by ``int(record["chain"])`` to ensure deterministic chain
        ordering independent of filename ordering.
        """
        pred_dir = Path(pred_dir)
        files = sorted(pred_dir.glob("rank*_chain*.pt"))
        if not files:
            raise RuntimeError(f"No prediction files found in {pred_dir}")
        recs = [torch.load(f) for f in files]
        recs.sort(key=lambda r: int(r["chain"]))

        stats: dict[str, Tensor] = {
            "samples": torch.stack([r["samples"] for r in recs])
        }
        for k in ("lp", "step_size", "accept"):
            if all((k in r) for r in recs):
                stats[k] = torch.stack([r[k] for r in recs])
        if expected_chains is not None and stats["samples"].shape[0] != expected_chains:
            raise RuntimeError(
                f"Expected {expected_chains} chains, got {stats['samples'].shape[0]}."
            )
        return stats

    if multi_cpu:
        wall_start = time.perf_counter()
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=n_chains,
            strategy="ddp_spawn",
            logger=False,
            enable_checkpointing=False,
            inference_mode=False,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            callbacks=[DiskPredictionWriter(tmp_dir, write_interval="batch")],
        )
        _ = trainer.predict(sampler, dl, return_predictions=False)
        wall_secs = time.perf_counter() - wall_start
        rank_zero_info(
            f"[predict/ddp_spawn] Elapsed: {wall_secs:.2f}s "
            f"({str(dt.timedelta(seconds=int(wall_secs)))})"
        )
        stats = _stack_from_disk(tmp_dir, expected_chains=n_chains)
    else:
        timer = Timer()
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=1,
            logger=False,
            enable_checkpointing=False,
            inference_mode=False,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
            callbacks=[timer],
        )
        outs = trainer.predict(sampler, dl, return_predictions=True)
        secs = timer.time_elapsed("predict") or 0.0
        rank_zero_info(
            f"[predict] Elapsed: {secs:.2f}s ({str(dt.timedelta(seconds=int(secs)))})"
        )
        stats = _stack_single_outs(outs)

    return stats


class ReparametrizedMALASamplerModule(
    MALASamplerModule
):  # pylint: disable=too-many-ancestors
    """
    Reparametrized MALA sampler using sigmoid transformation for bounded parameters.

    This sampler extends MALASamplerModule by transforming bounded parameters to an
    unbounded space where sampling is performed. The transformation ensures that
    samples remain within the specified bounds [X_min, X_max] by:

    1. Sampling in unbounded space φ ∈ (-∞, ∞)
    2. Transforming to bounded space: X = X_min + (X_max - X_min) * sigmoid(φ)
    3. Adding Jacobian correction to the log probability

    This approach avoids hard rejections at boundaries and can improve mixing for
    parameters that tend to explore near their bounds.

    Parameters
    ----------
    model : pl.LightningModule
        Forward model used inside the likelihood.
    X_min : array-like of float or torch.Tensor, shape (D,)
        Element-wise lower bounds for parameters.
    X_max : array-like of float or torch.Tensor, shape (D,)
        Element-wise upper bounds for parameters.
    Y_target : array-like or torch.Tensor, shape (N,)
        Observed target vector.
    sigma_hat : array-like or torch.Tensor, shape (N,)
        Per-node standard deviation used in the Student-t likelihood.
    **kwargs : Any
        Additional arguments passed to MALASamplerModule.

    Notes
    -----
    The transformation used is:
        X(φ) = X_min + (X_max - X_min) * sigmoid(φ)

    where sigmoid(φ) = 1 / (1 + exp(-φ)).

    The Jacobian correction added to the log probability is:
        log|dX/dφ| = Σ_i [log(X_max[i] - X_min[i]) + log(sigmoid(φ[i])) + log(1 - sigmoid(φ[i]))]

    Examples
    --------
    >>> sampler = ReparametrizedMALASamplerModule(
    ...     model, X_min, X_max, Y_obs, sigma_hat, samples=2000
    ... )
    >>> # Initial values in unbounded space (will be transformed to bounded space)
    >>> phi_init = torch.zeros(6)  # starts at midpoint of bounds
    >>> dl = DataLoader(ChainInitDataset(phi_init.unsqueeze(0)), batch_size=1)
    >>> trainer = pl.Trainer(accelerator="cuda", devices=1)
    >>> preds = trainer.predict(sampler, dl)
    """

    def __init__(
        self,
        model: pl.LightningModule,
        X_min: Tensor | np.ndarray | list[float],
        X_max: Tensor | np.ndarray | list[float],
        Y_target: Tensor | np.ndarray | list[float],
        sigma_hat: Tensor | np.ndarray | list[float],
        **kwargs: Any,
    ) -> None:
        """
        Initialize the reparametrized MALA sampler.

        Parameters
        ----------
        model : pl.LightningModule
            Forward model used inside the likelihood.
        X_min : Tensor | np.ndarray | list[float]
            Element-wise lower bounds for parameters.
        X_max : Tensor | np.ndarray | list[float]
            Element-wise upper bounds for parameters.
        Y_target : Tensor | np.ndarray | list[float]
            Observed target vector.
        sigma_hat : Tensor | np.ndarray | list[float]
            Per-node standard deviation used in the Student-t likelihood.
        **kwargs : Any
            Additional arguments passed to MALASamplerModule.
        """
        super().__init__(model, X_min, X_max, Y_target, sigma_hat, **kwargs)

        # Precompute range for efficiency
        self.register_buffer(
            "X_range",
            self.X_max - self.X_min,
            persistent=False,
        )

    def phi_to_X(self, phi: Tensor) -> Tensor:
        """
        Transform from unbounded space φ to bounded space X.

        Parameters
        ----------
        phi : torch.Tensor
            Parameter vector in unbounded space.

        Returns
        -------
        torch.Tensor
            Parameter vector in bounded space [X_min, X_max].
        """
        sigmoid_phi = torch.sigmoid(phi)
        return self.X_min + self.X_range * sigmoid_phi

    def X_to_phi(self, X: Tensor) -> Tensor:
        """
        Transform from bounded space X to unbounded space φ.

        Uses the logit function (inverse sigmoid).

        Parameters
        ----------
        X : torch.Tensor
            Parameter vector in bounded space [X_min, X_max].

        Returns
        -------
        torch.Tensor
            Parameter vector in unbounded space.
        """
        # Normalize to [0, 1] with small epsilon to avoid log(0)
        eps = 1e-7
        X_normalized = torch.clamp((X - self.X_min) / self.X_range, eps, 1 - eps)
        # logit(p) = log(p / (1 - p))
        return torch.log(X_normalized / (1 - X_normalized))

    def log_jacobian(self, phi: Tensor) -> Tensor:
        """
        Compute log-determinant of Jacobian for the transformation φ → X.

        For the sigmoid transformation X = a + (b - a) * sigmoid(φ), the Jacobian is:
            dX_i/dφ_i = (b_i - a_i) * sigmoid(φ_i) * (1 - sigmoid(φ_i))

        Since the transformation is element-wise, the Jacobian is diagonal and:
            log|det(J)| = Σ_i log|(b_i - a_i) * sigmoid(φ_i) * (1 - sigmoid(φ_i))|

        Parameters
        ----------
        phi : torch.Tensor
            Parameter vector in unbounded space.

        Returns
        -------
        torch.Tensor
            Scalar log-determinant of the Jacobian.
        """
        sigmoid_phi = torch.sigmoid(phi)
        # log|J| = Σ [log(range) + log(sigmoid) + log(1 - sigmoid)]
        log_jac = (
            torch.log(self.X_range)
            + torch.log(sigmoid_phi)
            + torch.log(1 - sigmoid_phi)
        ).sum()
        return log_jac

    def neg_log_prob(self, phi: Tensor) -> Tensor:  # pylint: disable=arguments-renamed
        """
        Negative log-posterior in the reparametrized space.

        Transforms φ to X, evaluates the original negative log-posterior,
        and subtracts the Jacobian correction.

        Parameters
        ----------
        phi : torch.Tensor
            Parameter vector in unbounded space (requires grad).

        Returns
        -------
        torch.Tensor
            Scalar negative log-posterior in the reparametrized space.
        """
        X = self.phi_to_X(phi)

        # Original negative log-posterior in bounded space
        neg_log_p_X = super().neg_log_prob(X)

        # Jacobian correction (subtract because we want to add log|J| to log-prob)
        log_jac = self.log_jacobian(phi)

        # In reparametrized space: -log p(φ) = -log p(X(φ)) - log|dX/dφ|
        return neg_log_p_X - log_jac

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """
        Forward model wrapper for reparametrized space.

        Accepts φ in unbounded space, transforms to X, and evaluates the model.

        Parameters
        ----------
        *args
            Positional arguments. If provided, args[0] must be φ.
        **kwargs
            Keyword arguments. Must contain φ if *args is empty.

        Returns
        -------
        torch.Tensor
            Model prediction.
        """
        if args:
            phi = args[0]
        elif "X" in kwargs:
            # For compatibility, also accept X keyword
            phi = kwargs["X"]
        elif "phi" in kwargs:
            phi = kwargs["phi"]
        else:
            raise TypeError("forward() missing required argument: phi (or X)")

        X = self.phi_to_X(phi)
        return self.model(X, add_mean=True)

    def find_MAP(
        self,
        phi: Tensor,
        max_iter: int = 100,
        lr: float = 1.0,
        use_adam: bool = False,
        n_restarts: int = 1,
    ) -> Tensor:
        """
        Find MAP estimate in the reparametrized space.

        Parameters
        ----------
        phi : torch.Tensor
            Initial point in unbounded space.
        max_iter : int, default=100
            Maximum optimizer iterations (per restart if using Adam).
        lr : float, default=1.0
            Learning rate (1.0 for L-BFGS, 0.01-0.1 recommended for Adam).
        use_adam : bool, default=False
            If True, use Adam optimizer with multiple restarts instead of L-BFGS.
            Adam is more robust to steep gradients near boundaries.
        n_restarts : int, default=1
            Number of random restarts when using Adam (ignored for L-BFGS).

        Returns
        -------
        torch.Tensor
            MAP point in unbounded space φ.

        Notes
        -----
        For problems with steep gradients near boundaries (common with sigmoid
        transformations), consider using Adam with multiple restarts:
        ``find_MAP(phi, use_adam=True, lr=0.05, n_restarts=5, max_iter=1000)``
        """
        if use_adam:
            return self._find_MAP_adam(phi, max_iter, lr, n_restarts)
        return self._find_MAP_lbfgs(phi, max_iter, lr)

    def _find_MAP_lbfgs(self, phi: Tensor, max_iter: int, lr: float) -> Tensor:
        """
        Find MAP using L-BFGS optimizer (original implementation).

        Parameters
        ----------
        phi : torch.Tensor
            Initial point in unbounded space.
        max_iter : int
            Maximum number of L-BFGS iterations.
        lr : float
            L-BFGS learning rate.

        Returns
        -------
        torch.Tensor
            MAP point in unbounded space φ.
        """
        phi = (
            phi.detach()
            .to(device=self.device, dtype=torch.float32)
            .requires_grad_(True)
        )

        def closure() -> Tensor:
            """
            Closure for L-BFGS optimizer.

            Returns
            -------
            torch.Tensor
                Loss value (negative log posterior).
            """
            self.zero_grad(set_to_none=True)
            loss = self.neg_log_prob(phi)
            if not torch.isfinite(loss):
                return torch.tensor(1e10, device=loss.device, dtype=loss.dtype)
            loss.backward()
            if phi.grad is not None:
                if torch.isnan(phi.grad).any() or torch.isinf(phi.grad).any():
                    phi.grad.zero_()
                    return torch.tensor(1e10, device=loss.device, dtype=loss.dtype)
                torch.nn.utils.clip_grad_norm_([phi], max_norm=100.0)
            return loss

        opt = torch.optim.LBFGS(
            [phi], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe"
        )
        try:
            _ = opt.step(closure)
            if torch.isnan(phi).any():
                raise ValueError("Optimization produced NaN")
        except (RuntimeError, ValueError) as e:
            print(f"First optimization attempt failed: {e}")
            print("Retrying with conservative settings...")
            phi = (
                phi.detach()
                .to(device=self.device, dtype=torch.float32)
                .requires_grad_(True)
            )
            opt = torch.optim.LBFGS(
                [phi],
                lr=0.01,
                max_iter=max_iter * 3,
                line_search_fn=None,
                tolerance_grad=1e-4,
                tolerance_change=1e-6,
            )
            _ = opt.step(closure)

        self.zero_grad(set_to_none=True)
        return phi

    def _find_MAP_adam(
        self, phi_init: Tensor, max_iter: int, lr: float, n_restarts: int
    ) -> Tensor:
        """
        Find MAP using Adam optimizer with multiple random restarts.

        More robust to steep gradients near boundaries than L-BFGS.

        Parameters
        ----------
        phi_init : torch.Tensor
            Initial point in unbounded space (used for first restart).
        max_iter : int
            Maximum number of Adam iterations per restart.
        lr : float
            Adam learning rate (typically 0.01-0.1).
        n_restarts : int
            Number of random restarts to try.

        Returns
        -------
        torch.Tensor
            MAP point in unbounded space φ (best across all restarts).
        """
        best_phi = None
        best_loss = float("inf")

        for restart in range(n_restarts):
            # Initialize: use provided init for first restart, random for others
            if restart == 0:
                phi = phi_init.detach().clone()
            else:
                # Random initialization: sample from standard normal
                phi = torch.randn_like(phi_init) * 0.5

            phi = phi.to(device=self.device, dtype=torch.float32).requires_grad_(True)

            # Adam optimizer
            opt = torch.optim.Adam([phi], lr=lr, betas=(0.9, 0.999))

            # Learning rate scheduler: reduce on plateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=50
            )

            prev_loss = float("inf")
            patience_counter = 0
            patience = 100  # Early stopping patience

            for _ in range(max_iter):
                opt.zero_grad()
                loss = self.neg_log_prob(phi)

                if not torch.isfinite(loss):
                    # Skip this restart if we hit numerical issues
                    break

                loss.backward()

                # Clip gradients
                if phi.grad is not None:
                    if torch.isnan(phi.grad).any() or torch.isinf(phi.grad).any():
                        break
                    torch.nn.utils.clip_grad_norm_([phi], max_norm=10.0)

                opt.step()
                scheduler.step(loss.detach())

                # Early stopping: check for convergence
                if abs(prev_loss - loss.item()) < 1e-6:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
                else:
                    patience_counter = 0

                prev_loss = loss.item()

            # Evaluate final loss
            with torch.no_grad():
                final_loss = self.neg_log_prob(phi).item()

            if torch.isfinite(torch.tensor(final_loss)) and final_loss < best_loss:
                best_loss = final_loss
                best_phi = phi.detach().clone()

        if best_phi is None:
            # All restarts failed, return original
            print("Warning: All Adam restarts failed, returning initial point")
            return phi_init.detach().to(device=self.device, dtype=torch.float32)

        self.zero_grad(set_to_none=True)
        return best_phi

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Any]:
        """
        Run one chain in the reparametrized space.

        The input initial state should be in unbounded space (φ). The output
        samples are transformed back to bounded space (X) for interpretability.

        Parameters
        ----------
        batch : Any
            Tuple-like batch where batch[0] is chain id and batch[1] is φ_init.
        batch_idx : int
            Lightning-provided batch index.
        dataloader_idx : int, optional
            Dataloader index.

        Returns
        -------
        dict
            Dictionary with:
            - "chain" : int
            - "samples" : (S, D) tensor in BOUNDED space X (transformed from φ)
            - "samples_phi" : (S, D) tensor in UNBOUNDED space φ
            - "lp" : (S,) log posterior
            - "step_size" : (S,) step sizes
            - "accept" : (S,) acceptance indicators
        """
        # Call parent's predict_step which runs MALA in φ space
        result = super().predict_step(batch, batch_idx, dataloader_idx)

        # result["samples"] are in φ space, transform to X space for output
        samples_phi = result["samples"]  # (S, D) in unbounded space

        # Transform each sample to bounded space
        samples_X = torch.stack(
            [self.phi_to_X(phi) for phi in samples_phi]
        )  # (S, D) in bounded space

        # Return both representations
        result["samples"] = samples_X  # Main output in bounded space
        result["samples_phi"] = samples_phi  # Also save unbounded space

        return result
