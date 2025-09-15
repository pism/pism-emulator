#!/bin/env python3
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

import os
from pathlib import Path
import time
from argparse import ArgumentParser
from os.path import join
from typing import Literal
import arviz as az
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from lightning import LightningModule
from scipy.stats import beta
from tqdm.auto import tqdm
import matplotlib.pylab as plt

from pism_emulator.datasets import PISMDatasetXRP as PISMDataset
from pism_emulator.nnemulator import NNEmulator


from typing import Callable, Sequence
import numpy as np
import torch
from torch import Tensor


class mMALASampler:
    """
    mMALA Sampler (manifold Metropolis-adjusted Langevin)

    Parameters
    ----------
    model : LightningModule
        Emulator that maps X -> log10(Y_pred) (i.e., returns log10; sampler uses 10**model(X)).
    X_min, X_max : float or Tensor
        Element-wise lower/upper bounds for X (for Beta prior support).
    Y_target : array-like
        Target observations in linear scale (same shape as emulator output after masking).
    sigma_hat : array-like
        Per-node observational std deviation (same shape as Y_target).
    alpha : float or Tensor, default 0.01
        Weight between likelihood and prior in V(X) (negative log-posterior).
    alpha_b, beta_b : float or Tensor, default 3.0
        Beta prior parameters on the normalized parameters X̄ = (X-X_min)/(X_max-X_min).
    nu : float or Tensor, default 1.0
        Degrees of freedom for the Student-t likelihood on residuals.
    emulator_dir : str, optional
        Unused here (kept for backward compatibility).
    device : {"cpu","cuda"}, default "cpu"
        Device to run computations on.

    Notes
    -----
    * The sampler constructs proposals with covariance Σ = 2 h H^{-1}(X), where
      H(X) is a positive-definite metric derived from the Hessian of V(X)
      with an eigenvalue stabilization.
    * Acceptance ratio uses q(x'|x) with H evaluated at x, and q(x|x') with H evaluated at x'.
      This is important for the *manifold* (state-dependent) metric.
    """

    def __init__(
        self,
        model,
        X_min: float | torch.Tensor,
        X_max: float | torch.Tensor,
        Y_target: np.ndarray | Tensor,
        sigma_hat: np.ndarray | Tensor,
        metric_mode: Literal["manifold", "current"] = "manifold",
        hess_refresh: int = 1,
        delayed_accept: bool = False,
        alpha: float | torch.Tensor = 0.01,
        alpha_b: float | torch.Tensor = 3.0,
        beta_b: float | torch.Tensor = 3.0,
        nu: float | torch.Tensor = 1.0,
        emulator_dir: str = "./emulator",
        device: str = "cpu",
    ):
        super().__init__()
        self.model = model.eval()

        to_t = lambda v: v if isinstance(v, torch.Tensor) else torch.tensor(v)
        self.device = device
        self.X_min = to_t(X_min).to(device=device, dtype=torch.float32)
        self.X_max = to_t(X_max).to(device=device, dtype=torch.float32)
        self.Y_target = to_t(Y_target).to(device=device, dtype=torch.float32)
        self.sigma_hat = to_t(sigma_hat).to(device=device, dtype=torch.float32)
        self.alpha = to_t(alpha).to(device=device, dtype=torch.float32)
        self.alpha_b = to_t(alpha_b).to(device=device, dtype=torch.float32)
        self.beta_b = to_t(beta_b).to(device=device, dtype=torch.float32)
        self.nu = to_t(nu).to(device=device, dtype=torch.float32)

        self.emulator_dir = emulator_dir
        self.hessian_counter = 0

        # Small constants for stability
        self._eps_beta: float = 1e-7  # clamp for Beta prior support (0,1)
        self._eps_eig: float = 1e-6  # eigenvalue stabilization in metric
        self._two_pi: Tensor = torch.tensor(
            2.0 * np.pi, device=self.device, dtype=torch.float32
        )
        self.metric_mode = metric_mode
        self.hess_refresh = int(hess_refresh)
        self.delayed_accept = bool(delayed_accept)
        self._step_count = 0

    # ---------------------------
    # Objective (negative log posterior)
    # ---------------------------
    def V(self, X: Tensor) -> Tensor:
        """
        Compute negative log posterior (up to constant):
        V(X) = -( alpha * log_likelihood + log_prior )

        Returns
        -------
        Tensor (scalar)
        """
        # Emulator outputs log10; recover linear prediction
        Y_pred = 10.0 ** self.model(X, add_mean=True)

        r = Y_pred - self.Y_target
        t = r / self.sigma_hat
        nu = self.nu

        # Student-t log-likelihood (per element), summed
        # log t_pdf = lgamma((ν+1)/2) - lgamma(ν/2) - 0.5*log(πν) - log(σ)
        #            - (ν+1)/2 * log(1 + (t^2)/ν)
        log_like = torch.sum(
            torch.lgamma((nu + 1.0) * 0.5)
            - torch.lgamma(nu * 0.5)
            - 0.5
            * torch.log(
                self._two_pi / 2.0 * nu
            )  # same as 0.5*log(pi*nu), but re-using _two_pi
            - torch.log(self.sigma_hat)
            - ((nu + 1.0) * 0.5) * torch.log1p((t * t) / nu)
        )

        # Beta prior on normalized parameters in (0,1)
        X_bar = (X - self.X_min) / (self.X_max - self.X_min)
        X_bar = torch.clamp(X_bar, self._eps_beta, 1.0 - self._eps_beta)

        log_prior = torch.sum(
            (self.alpha_b - 1.0) * torch.log(X_bar)
            + (self.beta_b - 1.0) * torch.log(1.0 - X_bar)
            + torch.lgamma(self.alpha_b + self.beta_b)
            - torch.lgamma(self.alpha_b)
            - torch.lgamma(self.beta_b)
        )

        return -(self.alpha * log_like + log_prior)

    # ---------------------------
    # Local metric (gradient, Hessian, stabilized inverse and log-det)
    # ---------------------------
    @torch.enable_grad()
    def _local_geometry(
        self, X: Tensor, eps_eig: float | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Return (log_pi, grad, H, Hinv, log_det_Hinv) at X with eigenvalue stabilization.
        """
        if eps_eig is None:
            eps_eig = self._eps_eig

        log_pi = self.V(X)
        self.hessian_counter += 1

        g = torch.autograd.grad(log_pi, X, retain_graph=True, create_graph=True)[0]
        H = torch.autograd.functional.hessian(self.V, X, create_graph=False)

        # Symmetrize (defensive) and stabilize eigenvalues: λ' = sqrt(λ^2 + eps)
        H = 0.5 * (H + H.T)
        lam, Q = torch.linalg.eig(H)
        lam = torch.real(lam)
        Q = torch.real(Q)
        lam_p = torch.sqrt(lam * lam + eps_eig)  # positive
        H_pos = Q @ torch.diag(lam_p) @ Q.T
        Hinv = Q @ torch.diag(1.0 / lam_p) @ Q.T
        log_det_Hinv = torch.sum(torch.log(1.0 / lam_p))

        return log_pi, g, H_pos, Hinv, log_det_Hinv

    # ---------------------------
    # Gaussian proposal log-pdf with Σ = 2 h H^{-1}
    # ---------------------------
    @staticmethod
    def _proposal_logpdf(
        y: Tensor, mu: Tensor, H: Tensor, log_det_Hinv: Tensor, h: float, two_pi: Tensor
    ) -> Tensor:
        """
        log N(y | mu, Σ), with Σ = 2 h H^{-1}. Uses H and log|H^{-1}|.
        """
        d = y.shape[0]
        delta = (y - mu).unsqueeze(-1)  # (d,1)
        # Quadratic term: δᵀ Σ^{-1} δ = (1/(2h)) δᵀ H δ
        quad = (delta.transpose(0, 1) @ (H @ delta)).squeeze() / (2.0 * h)
        logdet_Sigma = d * np.log(2.0 * h) + log_det_Hinv
        return -0.5 * (d * torch.log(two_pi) + logdet_Sigma + quad)

    # ---------------------------
    # Propose one MALA step
    # ---------------------------

    def MALA_step(self, X: torch.Tensor, h: float, local_data=None):
        """One MALA update; returns (X_new, new_local_data, accepted_flag)."""
        # (A) ensure local geometry at x (optionally throttled)
        need_refresh = (local_data is None) or (
            self.hess_refresh > 1 and (self._step_count % self.hess_refresh) == 0
        )
        if need_refresh:
            local_data = self._local_geometry(X)
        log_pi, g, H, Hinv, log_det_Hinv = local_data

        # (B) propose with Σ = 2 h H^{-1}(x)
        with torch.no_grad():
            L = torch.linalg.cholesky(2.0 * h * Hinv)
            X_prop = (X + L @ torch.randn_like(X)).detach()
        X_prop.requires_grad_(True)

        # Need exact target at proposal either way
        log_pi_prop = self.V(X_prop)

        # Common forward log q term (uses H at x)
        logq_fwd = self._proposal_logpdf(X_prop, X, H, log_det_Hinv, h, self._two_pi)

        # If DA disabled or metric_mode == "current", do the simple path
        if (not self.delayed_accept) or (self.metric_mode == "current"):
            # Use current metric for reverse as well
            logq_rev = self._proposal_logpdf(
                X, X_prop, H, log_det_Hinv, h, self._two_pi
            )
            log_alpha = -log_pi_prop + logq_rev + log_pi - logq_fwd
            alpha = (
                torch.exp(torch.clamp(log_alpha, max=0.0))
                if torch.isfinite(log_alpha)
                else torch.zeros((), device=self.device)
            )
            accept = torch.rand((), device=self.device) <= alpha

            if accept:
                X_new = X_prop.detach().requires_grad_(True)
                # refresh at next step (or immediate if you prefer)
                new_local = None
                s = 1
            else:
                X_new = X
                new_local = local_data
                s = 0

            self._step_count += 1
            return X_new, new_local, s

        # -------------------------------
        # (C) Delayed Acceptance (manifold)
        # Stage 1: cheap reverse uses H(x)
        logq_rev_cheap = self._proposal_logpdf(
            X, X_prop, H, log_det_Hinv, h, self._two_pi
        )
        log_alpha_cheap = -log_pi_prop + logq_rev_cheap + log_pi - logq_fwd

        # First gate
        if torch.isfinite(log_alpha_cheap):
            a1 = torch.exp(torch.clamp(log_alpha_cheap, max=0.0))
        else:
            a1 = torch.zeros((), device=self.device)
        u1 = torch.rand((), device=self.device)
        if u1 > a1:
            # Early reject w/o proposal geometry
            self._step_count += 1
            return X, local_data, 0

        # Stage 2: compute proposal geometry & exact reverse
        local_prop = self._local_geometry(X_prop)  # <- expensive, only if gate 1 passed
        _, _, H_prop, _, log_det_Hinv_prop = local_prop
        logq_rev_true = self._proposal_logpdf(
            X, X_prop, H_prop, log_det_Hinv_prop, h, self._two_pi
        )

        # Correction gate: exp(Δ_true - Δ_cheap)
        log_alpha_corr = logq_rev_true - logq_rev_cheap
        if torch.isfinite(log_alpha_corr):
            a2 = torch.exp(torch.clamp(log_alpha_corr, max=0.0))
        else:
            a2 = torch.zeros((), device=self.device)
        u2 = torch.rand((), device=self.device)
        if u2 <= a2:
            # Final accept
            X_new = X_prop.detach().requires_grad_(True)
            new_local = local_prop
            s = 1
        else:
            X_new = X
            new_local = local_data
            s = 0

        self._step_count += 1
        return X_new, new_local, s

    def torch_find_MAP(
        self,
        X: torch.tensor,
        X_keys,
        X_mean,
        X_std,
        n_iters: int = 25,
        verbose: bool = False,
        print_interval: int = 10,
    ):
        # L-BFGS
        def closure():
            opt.zero_grad()
            loss = self.V(X)
            loss.backward()
            return loss

        opt = torch.optim.LBFGS(
            [X], lr=0.1, max_iter=n_iters, line_search_fn="strong_wolfe"
        )

        for i in range(n_iters):
            log_pi = self.V(X)
            log_pi.backward()
            opt.step(closure)
            opt.zero_grad()

        print(f"\nFinal iter: {i:d}, log(P): {log_pi:.1f}\n")
        print(
            "".join(
                [
                    f"{key}: {(val * std + mean):.3f}\n"
                    for key, val, std, mean in zip(
                        X_keys,
                        X.data.cpu().numpy(),
                        X_std,
                        X_mean,
                    )
                ]
            )
        )
        return X

    # ---------------------------
    # Find MAP via (quasi-)Newton line search
    # ---------------------------
    def find_MAP(
        self,
        X: Tensor,
        X_keys: Sequence[str],
        X_mean: Tensor,
        X_std: Tensor,
        n_iters: int = 51,
        verbose: bool = False,
        print_interval: int = 10,
    ) -> Tensor:
        print("******** Finding MAP point ********")
        alphas = torch.logspace(-4, 0, 11, device=self.device)

        for i in range(n_iters):
            log_pi, g, _, Hinv, _ = self._local_geometry(X)
            p = Hinv @ (-g)
            # simple line search along p
            vals = []
            for a in alphas:
                X_try = (X + a * p).detach().requires_grad_(True)
                vals.append(self.V(X_try).detach().cpu().item())
            gamma = alphas[int(np.nanargmin(vals))]
            X = (X + gamma * p).detach().requires_grad_(True)

            if verbose and (i % print_interval == 0):
                print(f"iter: {i:4d}, -V (log post): {-log_pi.item():.3f}")
                print(
                    "".join(
                        f"{k}: {(x* s + m):.4f}\n"
                        for k, x, m, s in zip(
                            X_keys, X.detach().cpu().numpy(), X_mean, X_std
                        )
                    )
                )

        log_pi = self.V(X).detach()
        print(f"\nFinal iter: {i:d}, -V (log post): {-log_pi.item():.3f}\n")
        print(
            "".join(
                f"{k}: {(x* s + m):.4f}\n"
                for k, x, m, s in zip(X_keys, X.detach().cpu().numpy(), X_mean, X_std)
            )
        )
        return X

    # ---------------------------
    # Sampling
    # ---------------------------
    def sample(
        self,
        X: Tensor,
        chain: int = 0,
        burn: int = 1000,
        samples: int = 10001,
        h: float = 0.1,
        h_max: float = 1.0,
        acc_target: float = 0.25,
        k_adapt: float = 0.01,
        beta: float = 0.99,
        model_index: int = 0,
        save_interval: int | None = None,
        print_interval: int = 50,
        save_callback: Callable[[np.ndarray, int, int], None] | None = None,
    ) -> np.ndarray:
        """
        Run MALA and return posterior samples (array of shape [N, dim]).

        Parameters
        ----------
        X : Tensor
            Initial position (requires_grad=True is not required; will be set).
        chain : int
            Chain id (used only in the progress bar).
        burn : int
            Number of burn-in steps (discarded).
        samples : int
            Number of posterior draws to keep.
        h, h_max : float
            Step size and its upper bound.
        acc_target : float
            Target acceptance probability for adaptation.
        k_adapt : float
            Adaptation gain for step size updates.
        beta : float
            Exponential smoothing for acceptance rate display.
        model_index : int
            Unused here (kept for compatibility).
        save_interval : int or None
            If provided and `save_callback` is set, call it every `save_interval`
            kept samples with the array accumulated so far.
        print_interval : int
            Tqdm description update cadence.
        save_callback : callable or None
            Function called as `save_callback(X_posterior_so_far, chain, model_index)`.

        Returns
        -------
        np.ndarray
            Posterior draws, shape (samples, dim).
        """
        X = X.detach().to(device=self.device, dtype=torch.float32).requires_grad_(True)

        kept: list[Tensor] = []
        acc_ema = acc_target
        total_steps = burn + samples
        progress = tqdm(range(total_steps), position=chain, leave=True)

        local = None
        kept_count = 0

        for t in progress:
            X, local, s = self.MALA_step(X, h, local_data=local)

            # After burn-in, store
            if t >= burn:
                kept.append(X.detach())
                kept_count += 1

            # Robbins-Monro style step adaptation on smoothed acceptance
            acc_ema = beta * acc_ema + (1.0 - beta) * float(s)
            h = min(h * (1.0 + k_adapt * np.sign(acc_ema - acc_target)), h_max)

            # Progress bar (update sparsely)
            if (t % print_interval) == 0 or (t == total_steps - 1):
                log_p = local[0].item() if local is not None else float("nan")
                progress.set_description(
                    f"chain={chain}, t={t:6d} | acc={acc_ema:.2f} | h={h:.3f} | logP={-log_p:.2f}"
                )

            # Periodic save hook AFTER burn-in:
            if (
                save_interval is not None
                and save_callback is not None
                and t >= burn
                and ((kept_count % save_interval) == 0)
            ):
                X_post = torch.stack(kept).cpu().numpy().astype("float32", copy=False)
                save_callback(X_post, chain, model_index)

        X_posterior = torch.stack(kept).cpu().numpy().astype("float32", copy=False)
        return X_posterior


class MALASampler(object):
    """
    mMALA Sampler

    Author: Douglas C Brinkerhoff, University of Montana
    Creates a manifold Metropolis-adjusted Langevin algorithm (mMALA)

    Example::

        sampler = MALASampler(
            emulator, X_min, X_max, Y_target, sigma_hat
        )
        >>> X_map = sampler.find_MAP(X_0)
        >>> X_posterior = sampler.sample(
        >>>     X_map,
        >>>     samples=1000,
        >>>     burn=1000
        >>> )

    Args:
        model: LightningModule
        X_min (array or Tensor): minimum of distribution
        X_max (array or Tensor): maximum of distribution
        Y_target (array or Tensor): scale of the distribution
        sigma_hat (array or Tensor): covariance
        alpha (float): adjusts the weighting between the prior and the likelihood
        alpha_b (float or Tensor):  1st concentration parameter of the distribution
        (often referred to as alpha)
        beta_b (float or Tensor): 2nd concentration parameter of the distribution
        (often referred to as beta)
    """

    def __init__(
        self,
        model: LightningModule,
        X_min: float | torch.Tensor,
        X_max: float | torch.Tensor,
        Y_target: np.ndarray | torch.Tensor,
        sigma_hat: np.ndarray | torch.Tensor,
        alpha: float | torch.Tensor = 0.01,
        alpha_b: float | torch.Tensor = 3.0,
        beta_b: float | torch.Tensor = 3.0,
        nu: float | torch.Tensor = 1.0,
        emulator_dir="./emulator",
        device="cpu",
    ):
        super().__init__()
        self.model = model.eval()
        self.X_min = (
            torch.tensor(X_min, dtype=torch.float32, device=device)
            if not isinstance(X_min, torch.Tensor)
            else X_min.to(device)
        )
        self.X_max = (
            torch.tensor(X_max, dtype=torch.float32, device=device)
            if not isinstance(X_max, torch.Tensor)
            else X_max.to(device)
        )
        self.Y_target = (
            torch.tensor(Y_target, dtype=torch.float32, device=device)
            if not isinstance(Y_target, torch.Tensor)
            else Y_target.to(device)
        )
        self.sigma_hat = (
            torch.tensor(sigma_hat, dtype=torch.float32, device=device)
            if not isinstance(sigma_hat, torch.Tensor)
            else sigma_hat.to(device)
        )
        self.alpha = (
            torch.tensor(alpha, dtype=torch.float32, device=device)
            if not isinstance(alpha, torch.Tensor)
            else alpha.to(device)
        )
        self.alpha_b = (
            torch.tensor(alpha_b, dtype=torch.float32, device=device)
            if not isinstance(alpha_b, torch.Tensor)
            else alpha_b.to(device)
        )
        self.beta_b = (
            torch.tensor(beta_b, dtype=torch.float32, device=device).to(device)
            if not isinstance(beta_b, torch.Tensor)
            else beta_b.to(device)
        )
        self.nu = (
            torch.tensor(nu, dtype=torch.float32, device=device)
            if not isinstance(nu, torch.Tensor)
            else nu.to(device)
        )
        self.emulator_dir = emulator_dir
        self.hessian_counter = 0
        self.device = device

    def torch_find_MAP(
        self,
        X: torch.tensor,
        X_keys,
        X_mean,
        X_std,
        n_iters: int = 51,
        verbose: bool = False,
        print_interval: int = 10,
    ):
        # L-BFGS
        def closure():
            opt.zero_grad()
            loss = self.V(X)
            loss.backward()
            return loss

        opt = torch.optim.LBFGS([X], lr=0.1, max_iter=25, line_search_fn="strong_wolfe")

        for i in range(n_iters):
            log_pi = self.V(X)
            log_pi.backward()
            opt.step(closure)
            opt.zero_grad()

        print(f"\nFinal iter: {i:d}, log(P): {log_pi:.1f}\n")
        print(
            "".join(
                [
                    f"{key}: {(val * std + mean):.3f}\n"
                    for key, val, std, mean in zip(
                        X_keys,
                        X.data.cpu().numpy(),
                        X_std,
                        X_mean,
                    )
                ]
            )
        )
        return X

    def find_MAP(
        self,
        X: torch.tensor,
        X_keys,
        X_mean,
        X_std,
        n_iters: int = 51,
        verbose: bool = False,
        print_interval: int = 10,
    ):
        print("***********************************************")
        print("***********************************************")
        print("Finding MAP point")
        print("***********************************************")
        print("***********************************************")
        # Line search distances
        alphas = torch.logspace(-4, 0, 11)
        # Find MAP point
        for i in range(n_iters):
            log_pi, g, _, Hinv, log_det_Hinv = self.get_log_like_gradient_and_hessian(
                X, compute_hessian=True
            )
            # - f'(x) / f''(x)
            # g = f'(x), Hinv = 1 / f''(x)
            p = Hinv @ -g
            # Line search
            alpha_index = np.nanargmin(
                [
                    self.get_log_like_gradient_and_hessian(
                        X + alpha * p, compute_hessian=False
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    for alpha in alphas
                ]
            )
            gamma = alphas[alpha_index]
            mu = X + gamma * p
            X.data = mu.data
            if verbose & (i % print_interval == 0):
                print("===============================================")
                print(f"iter: {i:d}, log(P): {log_pi:.1f}\n")
                print(
                    "".join(
                        [
                            f"{key}: {(val * std + mean):.3f}\n"
                            for key, val, std, mean in zip(
                                X_keys,
                                X.data.cpu().numpy(),
                                X_std,
                                X_mean,
                            )
                        ]
                    )
                )
        print(f"\nFinal iter: {i:d}, log(P): {log_pi:.1f}\n")
        print(
            "".join(
                [
                    f"{key}: {(val * std + mean):.3f}\n"
                    for key, val, std, mean in zip(
                        X_keys,
                        X.data.cpu().numpy(),
                        X_std,
                        X_mean,
                    )
                ]
            )
        )
        return X

    def V(
        self,
        X,
    ):
        """
        The log likelihood and log prior could be written as
        log_likelihood = torch.distributions.StudentT(nu).log_prob(t).sum()
        log_prior = torch.distributions.Beta(alpha_b, beta_b).log_prob(X_bar).sum()
        but X_bar may contain negative values, for which the Beta distribution bails but torch.log
        returns NaN, which is then just ignored. We could do
        log_prior = torch.distributions.Beta(alpha_b, beta_b, validate_args=False).log_prob(X_bar).sum()

        """
        Y_pred = 10 ** self.model(X, add_mean=True)
        r = Y_pred - self.Y_target
        sigma_hat = self.sigma_hat
        t = r / sigma_hat
        nu = self.nu
        X_min = self.X_min
        X_max = self.X_max
        alpha_b = self.alpha_b
        beta_b = self.beta_b

        # Likelihood
        log_likelihood = torch.sum(
            torch.lgamma((nu + 1) / 2.0)
            - torch.lgamma(nu / 2.0)
            - torch.log(torch.sqrt(torch.pi * nu) * sigma_hat)
            - (nu + 1) / 2.0 * torch.log(1 + 1.0 / nu * t**2)
        )
        # Prior
        X_bar = (X - X_min) / (X_max - X_min)
        log_prior = torch.sum(
            (alpha_b - 1) * torch.log(X_bar)
            + (beta_b - 1) * torch.log(1 - X_bar)
            + torch.lgamma(alpha_b + beta_b)
            - torch.lgamma(alpha_b)
            - torch.lgamma(beta_b)
        )
        return -(self.alpha * log_likelihood + log_prior)

    def get_log_like_gradient_and_hessian(self, X, eps=1e-2, compute_hessian=False):
        log_pi = self.V(X)
        if compute_hessian:
            self.hessian_counter += 1
            g = torch.autograd.grad(log_pi, X, retain_graph=True, create_graph=True)[0]
            H = torch.autograd.functional.hessian(self.V, X, create_graph=True)
            lamda, Q = torch.linalg.eig(H)
            lamda, Q = torch.real(lamda), torch.real(Q)
            lamda_prime = torch.sqrt(lamda**2 + eps)
            lamda_prime_inv = 1.0 / lamda_prime
            H = Q @ torch.diag(lamda_prime) @ Q.T
            Hinv = Q @ torch.diag(lamda_prime_inv) @ Q.T
            log_det_Hinv = torch.sum(torch.log(lamda_prime_inv))
            return log_pi, g, H, Hinv, log_det_Hinv
        else:
            return log_pi

    def draw_sample(self, mu, cov, eps=1e-10):
        L = torch.linalg.cholesky(
            cov + eps * torch.eye(cov.shape[0], device=self.device)
        )
        return mu + L @ torch.randn(L.shape[0], device=self.device)

    def get_proposal_likelihood(self, Y, mu, inverse_cov, log_det_cov):
        # - 0.5 * log_det_Hinv - 0.5 * (Y - mu) @ H / (2*h) * (Y - mu)
        # Log-likelihood of a Multivariate Normal distribution
        k = Y.shape[0]
        return (
            -0.5 * log_det_cov
            - 0.5 * (Y - mu) @ inverse_cov @ (Y - mu)
            + k * torch.log(torch.tensor(2) * torch.pi)
        )

    def MALA_step(self, X, h, local_data=None):
        if local_data is not None:
            pass
        else:
            local_data = self.get_log_like_gradient_and_hessian(X, compute_hessian=True)

        log_pi, g, H, Hinv, log_det_Hinv = local_data
        X_ = self.draw_sample(X, 2 * h * Hinv).detach()
        X_.requires_grad = True

        log_pi_ = self.get_log_like_gradient_and_hessian(X_, compute_hessian=False)
        # logq = torch.distributions.MultivariateNormal(X_, precision_matrix=H / (2 * h)).log_prob(X).sum()
        # logq_ = torch.distributions.MultivariateNormal(X, precision_matrix=H / (2 * h)).log_prob(X_).sum()
        logq = self.get_proposal_likelihood(X_, X, H / (2 * h), log_det_Hinv)
        logq_ = self.get_proposal_likelihood(X, X_, H / (2 * h), log_det_Hinv)

        # alpha = min(1, P * Q_ / (P_ * Q))
        # s = self.MetropolisHastingsAcceptance(log_pi, log_pi_, logq, logq_)
        # if s == 1:
        #     local_data = self.get_log_like_gradient_and_hessian(X, compute_hessian=True)
        log_alpha = -log_pi_ + logq_ + log_pi - logq
        alpha = torch.exp(min(log_alpha, torch.tensor([0.0], device=self.device)))
        u = torch.rand(1, device=self.device)
        if u <= alpha and log_alpha != np.inf:
            X.data = X_.data
            local_data = self.get_log_like_gradient_and_hessian(X, compute_hessian=True)
            s = 1
        else:
            s = 0

        return X, local_data, s

    def sample(
        self,
        X,
        chain: int = 0,
        burn: int = 1000,
        samples: int = 10001,
        h: float = 0.1,
        h_max: float = 1.0,
        acc_target: float = 0.25,
        k: float = 0.01,
        beta: float = 0.99,
        model_index: int = 0,
        save_interval: int = 1000,
        print_interval: int = 50,
    ):

        local_data = None
        m_vars = []
        acc = acc_target
        progress = tqdm(range(samples + burn), position=chain, leave=True)
        for i in progress:
            X, local_data, s = self.MALA_step(X, h, local_data=local_data)
            if i >= burn:
                m_vars.append(X.detach())
            acc = beta * acc + (1 - beta) * s
            h = min(h * (1 + k * np.sign(acc - acc_target)), h_max)
            log_p = local_data[0].item()
            desc = f"chain: {chain}, sample: {(i):d}, accept rate: {acc:.2f}, step size: {h:.2f}, log(P): {log_p:.1f} "
            progress.set_description(desc=desc)

            if (i + burn % save_interval == 0) & (i >= burn):
                X_posterior = torch.stack(m_vars).cpu().numpy()
                df = pd.DataFrame(
                    data=X_posterior.astype("float32") * dataset.X_std.cpu().numpy()
                    + dataset.X_mean.cpu().numpy(),
                    columns=dataset.X_keys,
                )
                if out_format == "csv":
                    df.to_csv(
                        join(
                            posterior_dir,
                            f"X_posterior_model_{model_index}_chain_{chain}.csv.gz",
                        )
                    )
                elif out_format == "parquet":
                    df.to_parquet(
                        join(
                            posterior_dir,
                            f"X_posterior_model_{model_index}_chain_{chain}.parquet",
                        )
                    )
                else:
                    raise NotImplementedError(f"{out_format} not implemented")

        X_posterior = torch.stack(m_vars).cpu().numpy()
        return X_posterior


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", default=False, action="store_true")
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--emulator_dir", default="emulator_ensemble")
    parser.add_argument("--model_index", type=int, default=0)
    parser.add_argument("--num_iterations", type=int, default=100000)
    parser.add_argument("--out_format", choices=["csv", "parquet"], default="parquet")
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument(
        "--samples_file", default="../data/samples/velocity_calibration_samples_100.csv"
    )
    parser.add_argument(
        "--target_file",
        default="../data/observed_speeds/greenland_vel_mosaic250_v1_g9000m.nc",
    )
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("TRAINING_FILES", nargs="*", help="PISM netCDF files")

    parser = NNEmulator.add_model_specific_args(parser)
    args = parser.parse_args()
    hparams = vars(args)

    checkpoint = args.checkpoint
    device = args.device
    emulator_dir = args.emulator_dir
    alpha = args.alpha
    model_index = args.model_index
    chains = args.chains
    samples = args.samples
    burn = args.burn
    out_format = args.out_format
    samples_file = args.samples_file
    target_file = args.target_file
    thin = args.thin
    training_files = args.TRAINING_FILES

    posterior_dir = f"{emulator_dir}/posterior_samples/"
    if not os.path.isdir(posterior_dir):
        os.makedirs(posterior_dir)

    dataset = PISMDataset(
        training_files=training_files,
        samples_file=samples_file,
        target_file=target_file,
        thin=thin,
        target_corr_threshold=0,
        target_error_var="velsurf_mag_error",
        target_var="velsurf_mag",
    )

    X = dataset.X
    X_min = X.cpu().numpy().min(axis=0) - 1e-3
    X_max = X.cpu().numpy().max(axis=0) + 1e-3
    n_parameters = dataset.n_parameters

    torch.manual_seed(0)
    np.random.seed(0)
    emulator_file = join(emulator_dir, "emulator", f"emulator_{model_index}.h5")

    state_dict = torch.load(emulator_file, weights_only=True)
    e = NNEmulator(
        state_dict["l_1.weight"].shape[1],
        state_dict["V_hat"].shape[1],
        state_dict["V_hat"],
        state_dict["F_mean"],
        state_dict["area"],
        hparams,
    )
    e.load_state_dict(state_dict)
    e.to(device)

    Y_target = dataset.Y_target
    if dataset.target_has_error:
        sigma = dataset.Y_target_error
        sigma[sigma < 10] = 10
    else:
        sigma = 10

    rho = 1.0 / (1e4**2)
    point_area = (dataset.grid_resolution * thin) ** 2
    K = point_area * rho
    sigma_hat = np.sqrt(sigma**2 / K**2)

    # Eq 23 in SI
    # this is 2.0 in the paper
    alpha_b = 3.0
    beta_b = 3.0
    X_prior = (
        beta.rvs(alpha_b, beta_b, size=(samples, n_parameters)) * (X_max - X_min)
        + X_min
    )
    # Initial condition for MAP. Note that using 0 yields similar results
    X_0 = torch.tensor(
        X_prior.mean(axis=0), requires_grad=True, dtype=torch.float, device=device
    )

    start = time.process_time()
    sampler = mMALASampler(
        e,
        X_min,
        X_max,
        Y_target,
        sigma_hat,
        emulator_dir=emulator_dir,
        device=device,
        alpha=alpha,
        metric_mode="current",
        hess_refresh=2,
        delayed_accept=False,
    )
    a = time.time()
    X_map = sampler.torch_find_MAP(X_0, dataset.X_keys, dataset.X_mean, dataset.X_std)
    e = time.time()
    print(f"Finding MAP {(e-a):.0f}s")

    result = Parallel(n_jobs=chains)(
        delayed(sampler.sample)(
            X_map,
            samples=samples,
            model_index=int(model_index),
            burn=burn,
            chain=c,
            save_interval=1000,
            print_interval=100,
        )
        for c in range(chains)
    )
    print(sampler.hessian_counter)
    print(time.process_time() - start)

    # result: iterable of chains, each array (draw, dim)
    chains = [np.asarray(Xp, dtype=np.float32, order="C") for Xp in result]
    arr = np.stack(chains, axis=0)  # (chain, draw, dim)

    # Denormalize ONCE (no in-place *= / += in a loop)
    X_mean = np.asarray(dataset.X_mean.cpu().numpy(), dtype=np.float32)
    X_std = np.asarray(dataset.X_std.cpu().numpy(), dtype=np.float32)
    arr_denorm = arr * X_std[None, None, :] + X_mean[None, None, :]

    # Build one InferenceData with all chains
    posterior = {name: arr_denorm[:, :, i] for i, name in enumerate(dataset.X_keys)}
    idata = az.from_dict(posterior=posterior)  # infers chain/draw from (C, S)

    # Save to Zarr (overwrite)
    out_dir = Path(posterior_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_zarr = out_dir / f"X_posterior_model_{model_index}.zarr"
    idata.to_datatree().to_zarr(str(out_zarr), mode="w")  # overwrite

    # Robust plotting: drop (near-)constant vars and use hist with fewer bins
    plot_traces = True
    if plot_traces:
        # variance across chain & draw
        var_all = np.nanvar(arr_denorm, axis=(0, 1))
        keep = var_all > 1e-12
        if np.any(keep):
            var_names = [dataset.X_keys[i] for i in np.flatnonzero(keep)]
            az.plot_trace(
                idata, var_names=var_names, hist_kwargs={"bins": 50}
            )  # <-- key fix: kind/hist_kwargs at top level
            out_png = out_dir / f"X_posterior_model_{model_index}.trace.png"
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close("all")
        else:
            print("All parameters are (near) constant; skipping trace plot.")


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
