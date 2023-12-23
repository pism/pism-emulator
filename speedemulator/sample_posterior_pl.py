#!/bin/env python3

import os
import time
from argparse import ArgumentParser
from os.path import join
from typing import Union

import arviz as az
import lightning as pl
import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
import torch
from lightning.pytorch.callbacks import Timer
from pismemulator.datasets import PISMDataset
from pismemulator.nnemulator import NNEmulator
from pismemulator.utils import param_keys_dict as keys_dict
from scipy.stats import beta
from torch.utils.data import DataLoader, TensorDataset


class mMALA(pl.LightningModule):  # type: ignore
    def __init__(
        self,
        emulator: pl.LightningModule,  # type: ignore
        X_0,
        X_min: Union[float, torch.Tensor],
        X_max: Union[float, torch.Tensor],
        Y_target: Union[np.ndarray, torch.Tensor],
        sigma_hat: Union[np.ndarray, torch.Tensor],
        hparams,
        X_mean: Union[float, np.ndarray, torch.Tensor] = 1.0,
        X_std: Union[float, np.ndarray, torch.Tensor] = 1.0,
        X_keys: list = [],
        alpha: Union[float, torch.Tensor] = 0.01,
        alpha_b: Union[float, torch.Tensor] = 3.0,
        beta_b: Union[float, torch.Tensor] = 3.0,
        nu: Union[float, torch.Tensor] = 1.0,
        h_init: float = 0.1,
        h_max: float = 1,
        num_steps: int = 10000,
        beta: float = 0.99,
        accept_target: float = 0.25,
        save_interval: int = 1000,
        save_dir=".",
        save_format="csv",
        burn_in: int = 500,
        pretrain: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        dataloader = DataLoader(
            TensorDataset(torch.zeros_like(X_0, dtype=torch.float32))
        )  # bogus dataloader
        self.emulator = emulator.eval()
        self.register_buffer(
            "X_min",
            torch.tensor(X_min, dtype=torch.float32)
            if not isinstance(X_min, torch.Tensor)
            else X_min,
        )
        self.register_buffer(
            "X_max",
            torch.tensor(X_max, dtype=torch.float32)
            if not isinstance(X_max, torch.Tensor)
            else X_max,
        )
        self.register_buffer(
            "X_mean",
            torch.tensor(X_mean, dtype=torch.float32)
            if not isinstance(X_mean, torch.Tensor)
            else X_mean,
        )
        self.register_buffer(
            "X_std",
            torch.tensor(X_std, dtype=torch.float32)
            if not isinstance(X_std, torch.Tensor)
            else X_std,
        )
        self.X_keys = X_keys
        self.register_buffer(
            "Y_target",
            torch.tensor(Y_target, dtype=torch.float32)
            if not isinstance(Y_target, torch.Tensor)
            else Y_target,
        )
        self.register_buffer(
            "sigma_hat",
            torch.tensor(sigma_hat, dtype=torch.float32)
            if not isinstance(sigma_hat, torch.Tensor)
            else sigma_hat,
        )
        self.register_buffer(
            "alpha",
            torch.tensor(alpha, dtype=torch.float32)
            if not isinstance(alpha, torch.Tensor)
            else alpha,
        )
        self.register_buffer(
            "beta",
            torch.tensor(beta, dtype=torch.float32)
            if not isinstance(beta, torch.Tensor)
            else beta,
        )
        self.register_buffer(
            "alpha_b",
            torch.tensor(alpha_b, dtype=torch.float32)
            if not isinstance(alpha_b, torch.Tensor)
            else alpha_b,
        )
        self.register_buffer(
            "beta_b",
            torch.tensor(beta_b, dtype=torch.float32)
            if not isinstance(beta_b, torch.Tensor)
            else beta_b,
        )
        self.register_buffer(
            "nu",
            torch.tensor(nu, dtype=torch.float32)
            if not isinstance(nu, torch.Tensor)
            else nu,
        )
        self.h_init = h_init
        self.h = h_init
        self.h_max = h_max
        self.accept = accept_target
        self.accept_target = accept_target
        self.X = torch.nn.Parameter(
            torch.tensor(X_0, dtype=torch.float32)
            if not isinstance(X_0, torch.Tensor)
            else X_0,
            requires_grad=True,
        )
        self.parameters = torch.nn.ParameterList(
            [torch.nn.Parameter(X_0[i], requires_grad=True) for i in range(len(X_0))]
        )

        # self.params = torch.nn.ParameterDict(OrderedDict(zip(X_keys, X_0)))
        if pretrain:
            self.pretrain()
        self.local_data = self.get_log_like_gradient_and_hessian(
            self.X, compute_hessian=True
        )
        self.X_posterior = []

    def draw_sample(self, mu, cov, eps=1e-10):
        L = torch.linalg.cholesky(
            cov + eps * torch.eye(cov.shape[0], device=self.device)
        )
        return mu + L.to(self.device) @ torch.randn(L.shape[0], device=self.device)

    def get_proposal_likelihood(self, Y, mu, inverse_cov, log_det_cov):
        Y = Y.to(self.device)
        mu = mu.to(self.device)
        inverse_cov = inverse_cov.to(self.device)
        return -0.5 * log_det_cov - 0.5 * (Y - mu) @ inverse_cov @ (Y - mu)

    def forward(self, X=None):
        Y_pred = 10 ** self.emulator(X, add_mean=True)
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
            (alpha_b - 1) * torch.log(X_bar) + (beta_b - 1) * torch.log(1 - X_bar)
        )

        log_prob = -(self.alpha * log_likelihood + log_prior)
        return log_prob

    def get_log_like_gradient_and_hessian(self, X, eps=1e-2, compute_hessian=False):
        log_pi = self.forward(X)
        if compute_hessian:
            g = torch.autograd.grad(log_pi, X, retain_graph=True, create_graph=True)[0]
            H = torch.autograd.functional.hessian(self.forward, X)
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

    def pretrain(self):
        print("***********************************************")
        print("Finding MAP point")
        print("***********************************************")
        # Line search distances
        n_iters = 51
        alphas = torch.logspace(-4, 0, 11)
        # Find MAP point
        X = self.X
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

        print(f"\nFinal iter: {i:d}, log(P): {log_pi:.1f}\n")
        print(
            "".join(
                [
                    f"{key} = {(val * std + mean):.3f}\n"
                    for key, val, std, mean in zip(
                        self.X_keys,
                        X.data.cpu().numpy(),
                        self.X_std,
                        self.X_mean,
                    )
                ]
            )
        )
        self.X = X

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        return optimizer

    def tune_step_size(self):
        accept = self.accept
        accept_target = self.accept_target
        h_max = self.h_max
        k = 0.01
        h = min(self.h * (1 + k * np.sign(accept - accept_target)), h_max)
        self.h = h
        return h

    def training_step(self, batch, batch_idx):
        X = self.X
        h = self.tune_step_size()
        X, s = self.sample(X, h)
        self.X = X
        self.X_posterior.append(X.cpu().detach().numpy())
        self.accept = self.beta * self.accept + (1 - self.beta) * s
        loss = self.forward(X)
        return loss

    def predict_step(self, batch, batch_idx):
        X = self.X
        h = self.tune_step_size()
        X, s = self.sample(X, h)
        self.X = X
        self.X_posterior.append(X.cpu().detach().numpy())
        self.accept = self.beta * self.accept + (1 - self.beta) * s
        print(X * self.X_std + self.X_mean)

    def training_epoch_end(self, outputs):
        self.log(
            "h",
            self.h,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "accept",
            self.accept,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def sample(self, X, h):
        local_data = self.local_data
        if local_data is not None:
            pass
        else:
            local_data = self.get_log_like_gradient_and_hessian(X, compute_hessian=True)

        log_pi, g, H, Hinv, log_det_Hinv = local_data
        X_ = self.draw_sample(X, 2 * h * Hinv.to(self.device)).detach()

        log_pi_ = self.get_log_like_gradient_and_hessian(X_, compute_hessian=False)
        logq = self.get_proposal_likelihood(
            X_.to(self.device),
            X.to(self.device),
            H.to(self.device) / (2 * h),
            log_det_Hinv,
        )
        logq_ = self.get_proposal_likelihood(
            X.to(self.device),
            X_.to(self.device),
            H.to(self.device) / (2 * h),
            log_det_Hinv,
        )

        # alpha = min(1, P * Q_ / (P_ * Q))
        log_alpha = -log_pi_ + logq_ + log_pi - logq
        alpha = torch.exp(min(log_alpha, torch.tensor([0.0]).to(self.device)))
        u = torch.rand(1, dtype=torch.float32).to(self.device)
        if u <= alpha.to(self.device) and log_alpha.to(self.device) != torch.inf:
            X.data = X_.data
            local_data = self.get_log_like_gradient_and_hessian(X, compute_hessian=True)
            self.local_data = local_data
            s = 1
        else:
            s = 0
        return X, s


if __name__ == "__main__":
    __spec__ = None

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", default=False, action="store_true")
    parser.add_argument("--data_dir", default="../tests/training_data")
    parser.add_argument("--emulator_dir", default="emulator_ensemble")
    parser.add_argument("--model_index", type=int, default=0)
    parser.add_argument("--out_format", choices=["csv", "parquet"], default="parquet")
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=1_000)
    parser.add_argument(
        "--samples_file", default="../data/samples/velocity_calibration_samples_100.csv"
    )
    parser.add_argument(
        "--target_file",
        default="../data/observed_speeds/greenland_vel_mosaic250_v1_g9000m.nc",
    )
    parser.add_argument("--thinning_factor", type=int, default=1)

    parser = NNEmulator.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    hparams = vars(args)

    checkpoint = args.checkpoint
    data_dir = args.data_dir
    emulator_dir = args.emulator_dir
    model_index = args.model_index
    burn = args.burn
    out_format = args.out_format
    n_samples = args.samples
    samples_file = args.samples_file
    target_file = args.target_file
    thinning_factor = args.thinning_factor

    dataset = PISMDataset(
        data_dir=data_dir,
        samples_file=samples_file,
        target_file=target_file,
        thinning_factor=thinning_factor,
        target_corr_threshold=0,
    )

    X = dataset.X
    X_min = X.cpu().numpy().min(axis=0) - 1e-3
    X_max = X.cpu().numpy().max(axis=0) + 1e-3
    X_mean = dataset.X_mean
    X_std = dataset.X_std
    X_keys = dataset.X_keys
    n_parameters = dataset.n_parameters

    torch.manual_seed(0)
    np.random.seed(0)
    emulator_file = join(emulator_dir, "emulator", f"emulator_{model_index}.h5")

    state_dict = torch.load(emulator_file)
    e = NNEmulator(
        state_dict["l_1.weight"].shape[1],
        state_dict["V_hat"].shape[1],
        state_dict["V_hat"],
        state_dict["F_mean"],
        state_dict["area"],
        hparams,
    )
    e.load_state_dict(state_dict)

    Y_target = dataset.Y_target
    if dataset.target_has_error:
        sigma = dataset.Y_target_error
        sigma[sigma < 10] = 10
    else:
        sigma = 10

    rho = 1.0 / (1e4**2)
    point_area = (dataset.grid_resolution * thinning_factor) ** 2
    K = point_area * rho
    sigma_hat = np.sqrt(sigma**2 / K**2)

    # Eq 23 in SI
    # this is 2.0 in the paper
    alpha_b = 3.0
    beta_b = 3.0
    X_prior = (
        beta.rvs(alpha_b, beta_b, size=(100000, n_parameters)) * (X_max - X_min) + X_min
    )
    # Initial condition for MAP. Note that using 0 yields similar results
    X_0 = torch.tensor(X_prior.mean(axis=0), dtype=torch.float)

    mala = mMALA(
        e,
        X_0,
        X_min,
        X_max,
        Y_target,
        sigma_hat,
        hparams,
        X_mean=X_mean,
        X_std=X_std,
        X_keys=X_keys,
    )
    data_loader = DataLoader(
        TensorDataset(torch.zeros(1, n_parameters)),
        batch_size=1,
        num_workers=0,
    )  # bogus dataloader

    callbacks = []
    timer = Timer()
    callbacks.append(timer)

    start = time.process_time()
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(mala, data_loader)

    print(time.process_time() - start)
    X_posterior = (
        np.vstack([mala.X_posterior[k] for k in range(len(mala.X_posterior))])[burn::]
        * X_std.numpy()
        + X_mean.numpy()
    )
    df = pd.DataFrame(
        data=X_posterior,
        columns=dataset.X_keys,
    )
    posterior_dir = f"{emulator_dir}/posterior_samples/"
    if not os.path.isdir(posterior_dir):
        os.makedirs(posterior_dir)
    if out_format == "csv":
        df.to_csv(join(posterior_dir, f"X_posterior_model_{0}.csv.gz"))
    elif out_format == "parquet":
        df.to_parquet(join(posterior_dir, f"X_posterior_model_{0}.parquet"))
    else:
        raise NotImplementedError(f"{out_format} not implemented")

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
    fig.subplots_adjust(wspace=0.05, hspace=0.5)
    for k in range(X_posterior.shape[1]):
        ax = axs.ravel()[k]
        sns.kdeplot(
            X_posterior[:, k],
            ax=ax,
        )
        sns.despine(ax=ax, left=True, bottom=False)
        ax.set_xlabel(keys_dict[dataset.X_keys[k]])
        ax.set_ylabel(None)
        ax.axes.yaxis.set_visible(False)
    fig.tight_layout()
    d = {}
    for k, key in enumerate(X_keys):
        d[key] = X_posterior[:, k]

    trace = az.convert_to_inference_data(d)
    az.plot_trace(trace)
