from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# -----------------------------
# Small helper
# -----------------------------


def _to_tensor(x, device: str) -> Tensor:
    return (
        x
        if isinstance(x, torch.Tensor)
        else torch.tensor(x, dtype=torch.float32, device=device)
    )


class ChainInitDataset(Dataset):
    def __init__(self, inits: torch.Tensor):
        assert inits.ndim == 2
        self.inits = inits  # (n_chains, dim)

    def __len__(self):
        return self.inits.shape[0]

    def __getitem__(self, i: int):
        return i, self.inits[i]


# -----------------------------
# Lightning MALA Sampler Module
# -----------------------------
class MALASamplerModule(pl.LightningModule):
    """
    Manifold MALA (mMALA) implemented as a LightningModule suitable for multi-device
    `Trainer.predict` runs. It **does not** train parameters; it runs sampling chains.

    Parameters
    ----------
    emulator : pl.LightningModule
        Model mapping X -> log10(Y_pred). The sampler evaluates `10**emulator(X, add_mean=True)`.
    X_min, X_max : array-like or Tensor
        Element-wise bounds for Beta prior support.
    Y_target, sigma_hat : array-like or Tensor
        Target vector and per-node std (same length as emulator output after masking).
    alpha : float, default 0.01
        Likelihood weight in the negative log-posterior V(X).
    alpha_b, beta_b : float, default 3.0
        Beta prior parameters applied to normalized parameters.
    nu : float, default 1.0
        Degrees of freedom for Student-t likelihood.
    metric_mode : {"manifold", "current"}, default "manifold"
        Whether to use H(x') in the reverse proposal or reuse H(x).
    hess_refresh : int, default 1
        Recompute local geometry H(x) every N steps (>=1).
    delayed_accept : bool, default False
        Use two-stage delayed acceptance to avoid computing H(x') on obviously bad proposals.
    adapt_method : {"dual", "ema"}, default "dual"
        Step-size adaptation during burn-in (dual-averaging or simple EMA).
    h0 : float, default 0.1
        Initial step size.
    h_min, h_max : float, default (1e-3, 1.0)
        Min/max clamps for step size.
    acc_target : float, default 0.25
        Target acceptance probability during adaptation.
    dual_t0, dual_kappa, dual_gamma : floats
        Dual-averaging hyperparameters (see NUTS/Stan references).
    k_adapt, beta : floats
        EMA adaptation hyperparameters (only if adapt_method="ema").
    burn, samples : int
        Burn-in steps and number of samples to keep per chain.
    print_interval : int
        Progress logging cadence (in steps).
    """

    def __init__(
        self,
        emulator: pl.LightningModule,
        X_min,
        X_max,
        Y_target,
        sigma_hat,
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
        acc_target: float = 0.25,
        dual_t0: float = 10.0,
        dual_kappa: float = 0.75,
        dual_gamma: float = 0.05,
        k_adapt: float = 0.01,
        beta: float = 0.99,
        burn: int = 1000,
        samples: int = 2000,
        show_progress: bool = True,
        pbar_update_every: int = 10,
        seed: int | None = None,
        **kwargs,
    ):

        super().__init__()
        self.save_hyperparameters(ignore=["emulator"])  # everything except the model

        # Make sure emulator is a proper submodule so it moves with .to(device)
        if isinstance(emulator, torch.nn.Module):
            self.emulator = emulator.eval()
        else:
            # fall back; Lightning won’t move this automatically
            raise TypeError("`emulator` must be an nn.Module / LightningModule")

        # Register everything that must live on the same device as the model
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

        # Scalars can also be buffers so they follow device/dtype automatically
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

        # Numerics
        self._eps_beta = 1e-7
        self._eps_eig = 1e-6
        self.hessian_counter = 0
        self.show_progress = show_progress
        self.pbar_update_every = int(pbar_update_every)
        self._pbar = None
        self._rank = 0
        self._total_steps = 0
        self.burn = burn
        self.samples = samples
        self.delayed_accept = delayed_accept
        self.metric_mode = metric_mode
        self._step_count = 0
        self.hess_refresh = hess_refresh
        self._base_seed = 0 if seed is None else int(seed)

    def configure_optimizers(self):
        return None  # no training here

    # ------------- Core maths -------------
    def forward(self, X: Tensor) -> Tensor:
        """Emulator wrapper; expects model(X, add_mean=True) -> log10(Y)."""
        return 10.0 ** self.emulator(X, add_mean=True)

    def V(self, X: Tensor) -> Tensor:
        """Negative log-posterior (scalar)."""
        Y_pred = self.forward(X)
        r = Y_pred - self.Y_target
        t = r / self.sigma_hat
        nu = self.nu
        log_like = torch.sum(
            torch.lgamma((nu + 1.0) * 0.5)
            - torch.lgamma(nu * 0.5)
            - 0.5 * torch.log(self._two_pi * (nu / 2.0))
            - torch.log(self.sigma_hat)
            - 0.5 * (nu + 1.0) * torch.log1p((t * t) / nu)
        )
        X_bar = torch.clamp(
            (X - self.X_min) / (self.X_max - self.X_min),
            self._eps_beta,
            1 - self._eps_beta,
        )
        log_prior = torch.sum(
            (self.alpha_b - 1.0) * torch.log(X_bar)
            + (self.beta_b - 1.0) * torch.log(1.0 - X_bar)
            + torch.lgamma(self.alpha_b + self.beta_b)
            - torch.lgamma(self.alpha_b)
            - torch.lgamma(self.beta_b)
        )
        return -(self.alpha * log_like + log_prior)

    @torch.enable_grad()
    def _local_geometry(self, X: torch.Tensor):
        log_pi = self.V(X)  # V is NEG log posterior
        self.hessian_counter += 1

        g = torch.autograd.grad(log_pi, X, retain_graph=True, create_graph=False)[0]
        H = torch.autograd.functional.hessian(
            self.V, X, vectorize=False, create_graph=False
        )
        H = 0.5 * (H + H.T)

        _, S, V = torch.svd_lowrank(H, q=q)
        lamda = S**2 / (n_grid_points)

        lam, Q = torch.linalg.eig(H)
        lam = lam.real
        Q = Q.real
        lam_p = torch.sqrt(lam * lam + self._eps_eig)
        Hpos = Q @ torch.diag(lam_p) @ Q.T
        Hinv = Q @ torch.diag(1.0 / lam_p) @ Q.T
        log_det_Hinv = torch.sum(torch.log(1.0 / lam_p))
        return log_pi, g, Hpos, Hinv, log_det_Hinv

    @staticmethod
    def _proposal_logpdf(y, mu, H, log_det_Hinv, h, two_pi):
        d = y.numel()
        delta = (y - mu).unsqueeze(-1)
        quad = (delta.transpose(0, 1) @ (H @ delta)).squeeze() / (2.0 * h)
        logdet_Sigma = d * np.log(2.0 * h) + log_det_Hinv
        return -0.5 * (d * torch.log(two_pi) + logdet_Sigma + quad)

    def _mala_step(self, X: torch.Tensor, h: float, local=None):
        # refresh metric at x if needed
        if (
            (local is None)
            or (self.hess_refresh == 1)
            or (self._step_count % self.hess_refresh == 0)
        ):
            local = self._local_geometry(X)
        log_pi, g, H, Hinv, log_det_Hinv = local

        # Propose
        with torch.no_grad():
            L = torch.linalg.cholesky(2.0 * h * Hinv)
            eps = torch.randn_like(X)
            Xp = (X + L @ eps).detach()
        Xp.requires_grad_(True)

        # Target at proposal
        log_pi_p = self.V(Xp)

        # forward q(x'|x) using H(x)
        logq_f = self._proposal_logpdf(Xp, X, H, log_det_Hinv, h, self._two_pi)

        if not self.delayed_accept or self.metric_mode == "current":
            # reverse q(x|x') also using H(x)  (CURRENT metric variant)
            logq_r = self._proposal_logpdf(X, Xp, H, log_det_Hinv, h, self._two_pi)
            # log α = -V(x') + log q(x|x') + V(x) - log q(x'|x)
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
        # cheap reverse with H(x)
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

    # --------------------------- MAP via L-BFGS ---------------------------
    def find_MAP(
        self,
        X: torch.Tensor,
        n_iters: int = 25,
        lr: float = 0.1,
    ) -> torch.Tensor:
        """Find a MAP estimate via L-BFGS on the negative log-posterior V(X)."""

        X = X.detach().to(self.device, dtype=torch.float32).requires_grad_(True)

        def closure():
            self.zero_grad(set_to_none=True)
            loss = self.V(X)  # V is negative log posterior
            loss.backward()
            return loss

        opt = torch.optim.LBFGS([X], lr=lr, max_iter=25, line_search_fn="strong_wolfe")

        last_val = float("nan")
        for i in range(n_iters):
            val = opt.step(closure)  # runs closure + line search internally
            last_val = float(val.detach())
            self.zero_grad(set_to_none=True)

        return X

    # ------------- Predict loop (one chain per batch item) -------------
    def predict_step(self, batch: Tensor, batch_idx: int):
        chain_id, X0 = batch
        chain_id = (
            int(chain_id) if isinstance(chain_id, torch.Tensor) else int(chain_id)
        )
        if X0.dim() > 1 and X0.size(0) == 1:
            X0 = X0.squeeze(0)
        X = X0.detach().requires_grad_(True)

        burn, samples = int(self.hparams.burn), int(self.hparams.samples)
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
        local = None

        # init adaptation state
        acc_ema = acc_target
        if adapt_method == "dual":
            n = 0
            log_h = math.log(max(h, 1e-12))
            mu = math.log(10.0 * h)
            Hbar = 0.0
            log_hbar = log_h

        total = burn + samples
        for t in range(total):
            X, local, s, a = self._mala_step(X, h, local)

            if t >= burn:
                kept.append(X.detach())

            # adapt only during burn-in
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

            if local is not None:
                try:
                    logP = -float(local[0].detach().item())
                except Exception:
                    logP = float("nan")
            else:
                logP = float("nan")
            self._update_bar(t, h, logP)

        samples = torch.stack(kept).cpu()
        return {"chain": chain_id, "samples": samples}

    def on_predict_start(self) -> None:
        # one chain per process -> rank == chain id
        self._rank = int(getattr(self.trainer, "global_rank", 0))

        s = self._base_seed + 1000 * self._rank + 12345
        torch.manual_seed(s)
        np.random.seed(s % (2**32))

        self._total_steps = self.burn + self.samples

        # Pretty, stable single-line bar per rank
        bar_fmt = (
            f"chain {self._rank}: "
            "{percentage:>3.0f}%|{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )

        # If stdout isn’t a TTY (e.g. piping to a file), suppress bars
        disable_bars = not sys.stdout.isatty() or not self.show_progress

        self._pbar = tqdm(
            total=self._total_steps,
            position=self._rank,  # each rank gets its own line
            leave=True,
            ncols=120,  # fixed width avoids flicker
            dynamic_ncols=False,
            ascii=True,  # safe in multi-proc terminals
            bar_format=bar_fmt,
            mininterval=0.25,  # throttle refresh rate
            disable=disable_bars,
        )

    def on_predict_end(self) -> None:
        p = self._pbar
        self._pbar = None
        if p is not None:
            try:
                # final refresh; then close without tripping tqdm internals
                p.set_postfix_str("done", refresh=True)
                p.refresh()
            except Exception:
                pass
            try:
                p.disable = True
                p.close()
            except Exception:
                pass

    def _update_bar(self, t: int, h: float, logp: float) -> None:
        """Update tqdm bar every `pbar_update_every` steps."""
        if self._pbar is None:
            return

        # show post-burn sample index; ‘—’ during burn
        sample_str = "—" if t < self.burn else f"{t - self.burn + 1}"

        # update postfix only occasionally to reduce contention
        if (t % self.pbar_update_every) == 0 or (t + 1) == self._total_steps:
            # single string postfix is more robust than dict
            self._pbar.set_postfix_str(
                f"h={h:.3f}  logp={logp:.3f}  sample={sample_str}", refresh=True
            )

        self._pbar.update(1)


# -----------------------------
# Example usage (script)
# -----------------------------
if __name__ == "__main__":
    # Dummy emulator for illustration only
    class DummyEmu(pl.LightningModule):
        def forward(self, X: Tensor, add_mean: bool = True) -> Tensor:
            # log10 of a simple quadratic around 0.5
            y = -((X - 0.5) ** 2).sum(dim=-1, keepdim=True)
            return y  # already log10-space in this toy example

    dim = 5
    emulator = DummyEmu()
    X_min = torch.zeros(dim)
    X_max = torch.ones(dim)
    Y_target = torch.ones(1)  # shape must match emulator output
    sigma_hat = torch.ones(1)

    sampler = MALASamplerModule(
        emulator,
        X_min,
        X_max,
        Y_target,
        sigma_hat,
        metric_mode="manifold",
        delayed_accept=False,
        burn=200,
        samples=500,
    )

    n_chains = 4
    X0 = torch.rand(n_chains, dim)
    dl = DataLoader(ChainInitDataset(X0), batch_size=1)

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        max_epochs=1,
    )
    outs = trainer.predict(sampler, dl)
    # outs: list of length n_chains, each (samples, dim)
    all_chains = torch.stack(
        [o.squeeze(0) if o.ndim == 3 else o for o in outs]
    )  # (chains, samples, dim)
    print(all_chains.shape)
