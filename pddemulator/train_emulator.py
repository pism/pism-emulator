#!/bin/env python3

# Copyright (C) 2021-22 Andy Aschwanden, Douglas C Brinkerhoff
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
from argparse import ArgumentParser
from os.path import join
from pathlib import Path

from typing import Union

import lightning as pl
from lightning import LightningModule

import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, Timer
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import NullFormatter
from pyDOE import lhs
from scipy.special import gamma
from scipy.stats import dirichlet, gaussian_kde
from scipy.stats.distributions import uniform
from sklearn.metrics import mean_squared_error

from scipy.stats import beta

import time

from pismemulator.nnemulator import PDDEmulator, TorchPDDModel
from pismemulator.datamodules import PDDDataModule
from pismemulator.utils import load_hirham_climate_simple, load_hirham_climate

np.random.seed(2)


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
        model,
        temp_obs: Union[np.ndarray, torch.tensor],
        precip_obs: Union[np.ndarray, torch.tensor],
        std_dev_obs: Union[np.ndarray, torch.tensor],
        X_min: Union[float, torch.tensor],
        X_max: Union[float, torch.tensor],
        Y_target: Union[np.ndarray, torch.tensor],
        sigma_hat: Union[np.ndarray, torch.tensor],
        alpha: Union[float, torch.tensor] = 0.01,
        alpha_b: Union[float, torch.tensor] = 3.0,
        beta_b: Union[float, torch.tensor] = 3.0,
        nu: Union[float, torch.tensor] = 1.0,
        posterior_dir="./posterior",
        device="cpu",
    ):
        super().__init__()
        self.model = model.eval()
        self.temp_obs = (
            torch.tensor(temp_obs, dtype=torch.float32, device=device)
            if not isinstance(temp_obs, torch.Tensor)
            else temp_obs.to(device)
        )
        self.precip_obs = (
            torch.tensor(precip_obs, dtype=torch.float32, device=device)
            if not isinstance(precip_obs, torch.Tensor)
            else precip_obs.to(device)
        )
        self.std_dev_obs = (
            torch.tensor(std_dev_obs, dtype=torch.float32, device=device)
            if not isinstance(std_dev_obs, torch.Tensor)
            else std_dev_obs.to(device)
        )
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
        self.posterior_dir = posterior_dir
        self.hessian_counter = 0

    def find_MAP(
        self,
        X: torch.tensor,
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
                            f"{key}: {val:.3f}\n"
                            for key, val in zip(
                                X_keys,
                                X.data.cpu().numpy(),
                            )
                        ]
                    )
                )
        print(f"\nFinal iter: {i:d}, log(P): {log_pi:.1f}\n")
        print(
            "".join(
                [
                    f"{key}: {val:.3f}\n"
                    for key, val in zip(
                        X_keys,
                        X.data.cpu().numpy(),
                    )
                ]
            )
        )
        return X

    def V(
        self,
        X,
    ):
        f_snow = X[0]
        f_ice = X[1]
        refreeze = X[2]
        Xp = torch.hstack(
            (
                self.temp_obs.T,
                self.precip_obs.T,
                self.std_dev_obs.T,
                torch.tile(f_snow, (self.temp_obs.shape[1], 1)),
                torch.tile(f_ice, (self.temp_obs.shape[1], 1)),
                torch.tile(refreeze, (self.temp_obs.shape[1], 1)),
            )
        )
        print(Xp.shape)
        Y_pred = self.model(Xp)

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
            # + torch.lgamma(alpha_b + beta_b)
            # - torch.lgamma(alpha_b)
            # - torch.lgamma(beta_b)
        )
        return -(self.alpha * log_likelihood + log_prior)

    def get_log_like_gradient_and_hessian(self, X, eps=1e-24, compute_hessian=False):
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
        L = torch.linalg.cholesky(cov + eps * torch.eye(cov.shape[0], device=device))
        return mu + L @ torch.randn(L.shape[0], device=device)

    def get_proposal_likelihood(self, Y, mu, inverse_cov, log_det_cov):
        # - 0.5 * log_det_Hinv - 0.5 * (Y - mu) @ H / (2*h) * (Y - mu)
        return -0.5 * log_det_cov - 0.5 * (Y - mu) @ inverse_cov @ (Y - mu)

    def MALA_step(self, X, h, local_data=None):
        if local_data is not None:
            pass
        else:
            local_data = self.get_log_like_gradient_and_hessian(X, compute_hessian=True)

        log_pi, g, H, Hinv, log_det_Hinv = local_data
        X_ = self.draw_sample(X, 2 * h * Hinv).detach()
        X_.requires_grad = True

        log_pi_ = self.get_log_like_gradient_and_hessian(X_, compute_hessian=False)
        logq = self.get_proposal_likelihood(X_, X, H / (2 * h), log_det_Hinv)
        logq_ = self.get_proposal_likelihood(X, X_, H / (2 * h), log_det_Hinv)

        # alpha = min(1, P * Q_ / (P_ * Q))
        # s = self.MetropolisHastingsAcceptance(log_pi, log_pi_, logq, logq_)
        # if s == 1:
        #     local_data = self.get_log_like_gradient_and_hessian(X, compute_hessian=True)
        log_alpha = -log_pi_ + logq_ + log_pi - logq
        alpha = torch.exp(min(log_alpha, torch.tensor([0.0], device=device)))
        u = torch.rand(1, device=device)
        if u <= alpha and log_alpha != np.inf:
            X.data = X_.data
            local_data = self.get_log_like_gradient_and_hessian(X, compute_hessian=True)
            s = 1
        else:
            s = 0

        return X, local_data, s

    def MetropolisHastingsAcceptance(self, log_pi, log_pi_, logq, logq_):
        log_alpha = -log_pi_ + logq_ + log_pi - logq
        alpha = torch.exp(min(log_alpha, torch.tensor([0.0], device=device)))
        u = torch.rand(1, device=device)
        if u <= alpha and log_alpha != np.inf:
            X.data = X_.data
            s = 1
        else:
            s = 0
        return s

    def sample(
        self,
        X,
        burn: int = 1000,
        samples: int = 10001,
        h: float = 0.1,
        h_max: float = 1.0,
        acc_target: float = 0.25,
        k: float = 0.01,
        beta: float = 0.99,
        save_interval: int = 1000,
        print_interval: int = 50,
    ):
        print(
            "****************************************************************************"
        )
        print(
            "****************************************************************************"
        )
        print("Running Metropolis-Adjusted Langevin Algorithm")
        print(
            "****************************************************************************"
        )
        print(
            "****************************************************************************"
        )

        posterior_dir = self.posterior_dir
        if not os.path.isdir(posterior_dir):
            os.makedirs(posterior_dir)

        local_data = None
        m_vars = []
        acc = acc_target
        progress = tqdm(range(samples + burn))
        for i in progress:
            X, local_data, s = self.MALA_step(X, h, local_data=local_data)
            if i >= burn:
                m_vars.append(X.detach())
            acc = beta * acc + (1 - beta) * s
            h = min(h * (1 + k * np.sign(acc - acc_target)), h_max)
            log_p = local_data[0].item()
            desc = f"sample: {(i):d}, accept rate: {acc:.2f}, step size: {h:.2f}, log(P): {log_p:.1f} "
            progress.set_description(desc=desc)

            if (i + burn % save_interval == 0) & (i >= burn):
                print("///////////////////////////////////////////////")
                print(f"Saving samples")
                print("///////////////////////////////////////////////")
                X_posterior = torch.stack(m_vars).cpu().numpy()
                df = pd.DataFrame(
                    data=X_posterior.astype("float32"),
                    columns=X_keys,
                )
                if out_format == "csv":
                    df.to_csv(join(posterior_dir, f"X_posterior.csv.gz"))
                elif out_format == "parquet":
                    df.to_parquet(join(posterior_dir, f"X_posterior.parquet"))
                else:
                    raise NotImplementedError(f"{out_format} not implemented")

        X_posterior = torch.stack(m_vars).cpu().numpy()
        return X_posterior


def draw_samples(n_samples=250, random_seed=2):
    np.random.seed(random_seed)

    distributions = {
        "f_snow": uniform(loc=1.0, scale=5.0),  # uniform between 1 and 6
        "f_ice": uniform(loc=3.0, scale=12),  # uniform between 3 and 15
        "refreeze": uniform(loc=0, scale=1.0),  # uniform between 0 and 1
    }
    # Names of all the variables
    keys = [x for x in distributions.keys()]

    # Describe the Problem
    problem = {"num_vars": len(keys), "names": keys, "bounds": [[0, 1]] * len(keys)}

    # Generate uniform samples (i.e. one unit hypercube)
    unif_sample = lhs(len(keys), n_samples)

    # To hold the transformed variables
    dist_sample = np.zeros_like(unif_sample)

    # Now transform the unit hypercube to the prescribed distributions
    # For each variable, transform with the inverse of the CDF (inv(CDF)=ppf)
    for i, key in enumerate(keys):
        dist_sample[:, i] = distributions[key].ppf(unif_sample[:, i])

    # Save to CSV file using Pandas DataFrame and to_csv method
    header = keys
    # Convert to Pandas dataframe, append column headers, output as csv
    df = pd.DataFrame(data=dist_sample, columns=header)

    return df


if __name__ == "__main__":
    __spec__ = None

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", default=False, action="store_true")
    parser.add_argument("--data_dir", default="../tests/training_data")
    parser.add_argument("--emulator_dir", default="emulator_ensemble")
    parser.add_argument("--model_index", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--thinning_factor", type=int, default=1)

    parser = PDDEmulator.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    hparams = vars(args)

    batch_size = args.batch_size
    checkpoint = args.checkpoint
    data_dir = args.data_dir
    emulator_dir = args.emulator_dir
    max_epochs = args.max_epochs
    model_index = args.model_index
    num_workers = args.num_workers
    train_size = args.train_size
    thinning_factor = args.thinning_factor
    tb_logs_dir = f"{emulator_dir}/tb_logs"

    torch.manual_seed(0)
    pl.seed_everything(0)
    np.random.seed(model_index)

    samples = 2_000
    if not os.path.isdir(emulator_dir):
        os.makedirs(emulator_dir)
        os.makedirs(os.path.join(emulator_dir, "emulator"))

    (temp, precip, a, m, r, b, rfr) = load_hirham_climate(
        thinning_factor=thinning_factor
    )

    temp = temp.reshape(1, -1)
    precip = precip.reshape(1, -1)
    a = a.reshape(1, -1)
    m = m.reshape(1, -1)
    r = r.reshape(1, -1)
    b = b.reshape(1, -1)

    std_dev = np.zeros_like(temp)
    prior_df = draw_samples(n_samples=50)

    nt = temp.shape[0]

    X_m = []
    Y_m = []
    for k, row in prior_df.iterrows():
        m_f_snow = row["f_snow"]
        m_f_ice = row["f_ice"]
        m_refreeze = row["refreeze"]

        pdd = TorchPDDModel(
            pdd_factor_snow=m_f_snow,
            pdd_factor_ice=m_f_ice,
            refreeze_snow=m_refreeze,
            refreeze_ice=0,
            temp_snow=0.0,
            temp_rain=0.0,
            n_interpolate=12,
        )
        result = pdd(temp, precip, std_dev)

        A = result["accu"]
        M = result["melt"]
        R = result["runoff"]
        B = result["smb"]
        m_Y = torch.vstack(
            (
                A,
                M,
                R,
                B,
            )
        ).T
        Y_m.append(m_Y)
        X_m.append(
            torch.from_numpy(
                np.hstack(
                    (
                        temp.T,
                        precip.T,
                        std_dev.T,
                        np.tile(m_f_snow, (temp.shape[1], 1)),
                        np.tile(m_f_ice, (temp.shape[1], 1)),
                        np.tile(m_refreeze, (temp.shape[1], 1)),
                    )
                )
            )
        )

    X = torch.vstack(X_m).type(torch.FloatTensor)
    Y = torch.vstack(Y_m).type(torch.FloatTensor)
    n_samples, n_parameters = X.shape
    n_outputs = Y.shape[1]

    # Normalize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_norm = torch.nan_to_num((X - X_mean) / X_std, 0)

    callbacks = []
    timer = Timer()
    callbacks.append(timer)

    print(f"Training model {model_index}")
    omegas = torch.Tensor(dirichlet.rvs(np.ones(n_samples))).T
    omegas = omegas.type(torch.FloatTensor)
    omegas_0 = torch.ones_like(omegas) / len(omegas)
    area = torch.ones_like(omegas)

    # Load training data
    data_loader = PDDDataModule(X_norm, Y, omegas, omegas_0, num_workers=num_workers)
    data_loader.setup()

    # Generate emulator
    e = PDDEmulator(
        n_parameters,
        n_outputs,
        hparams,
    )

    # Setup trainer
    logger = TensorBoardLogger(tb_logs_dir, name=f"Emulator {model_index}")
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        default_root_dir=emulator_dir,
        num_sanity_val_steps=0,
    )
    train_loader = data_loader.train_loader
    val_loader = data_loader.val_loader

    # Train the emulator

    trainer.fit(e, train_loader, val_loader)
    print(f"Training took {timer.time_elapsed():.0f}s")
    trainer.save_checkpoint(f"{emulator_dir}/emulator/emulator_{model_index}.ckpt")

    # Out-Of-Set validation

    # Is there a better way to re-use the device for inference?
    device = e.device.type
    e.to(device)

    # Load trained emulator
    e = PDDEmulator.load_from_checkpoint(
        f"{emulator_dir}/emulator/emulator_{model_index}.ckpt",
        n_parameters=n_parameters,
        n_outputs=n_outputs,
    )

    X_val = torch.vstack([d[0] for d in data_loader.val_data])
    Y_val = torch.vstack([d[1] for d in data_loader.val_data])

    e.eval()

    Y_pred = e(X_val).detach().cpu()
    rmse = [
        np.sqrt(
            mean_squared_error(
                Y_pred.detach().cpu().numpy()[:, i], Y_val.detach().cpu().numpy()[:, i]
            )
        )
        for i in range(Y_val.shape[1])
    ]
    print("RMSE")
    print(f"A={rmse[0]:.6f}, M={rmse[1]:.6f}", f"R={rmse[2]:.6f}, B={rmse[3]:.6f}")

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    axs[0].plot(Y_val[:, 0], Y_pred[:, 0], ".", ms=0.25, label="Accumulation")
    axs[1].plot(Y_val[:, 1], Y_pred[:, 1], ".", ms=0.25, label="Melt")
    axs[2].plot(Y_val[:, 2], Y_pred[:, 2], ".", ms=0.25, label="Runoff")
    axs[3].plot(Y_val[:, 3], Y_pred[:, 3], ".", ms=0.25, label="SMB")
    for k in range(4):
        m_max = np.ceil(np.maximum(Y_val[:, k].max(), Y_pred[:, k].max()))
        m_min = np.floor(np.minimum(Y_val[:, k].min(), Y_pred[:, k].min()))
        axs[k].set_xlim(m_min, m_max)
        axs[k].set_ylim(m_min, m_max)
    # axs[0].axis("equal")
    # axs[1].axis("equal")
    # axs[2].axis("equal")
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()
    fig.savefig(f"{emulator_dir}/validation.pdf")

    # Create observations using the forward model
    obs_df = draw_samples(n_samples=100, random_seed=4)
    temp_obs, precip_obs, _, _, _, _, _ = load_hirham_climate(thinning_factor=100)
    std_dev_obs = np.zeros_like(temp_obs)

    f_snow_obs = 3.44
    f_ice_obs = 7.79
    refreeze_obs = 0.0
    f_true = [f_snow_obs, f_ice_obs, refreeze_obs]

    pdd = TorchPDDModel(
        pdd_factor_snow=f_snow_obs,
        pdd_factor_ice=f_ice_obs,
        refreeze_snow=refreeze_obs,
        refreeze_ice=0,
        temp_snow=0.0,
        temp_rain=0.0,
    )
    result = pdd(temp_obs, precip_obs, std_dev_obs)

    A_obs = result["accu"]
    M_obs = result["melt"]
    R_obs = result["runoff"]
    B_obs = result["smb"]

    Y_obs = torch.vstack((A_obs, M_obs, R_obs, B_obs)).T.type(torch.FloatTensor)

    # Create observations using the forward model
    mcmc_df = draw_samples(n_samples=1_000, random_seed=5)
    temp_prior, precip_prior, _, _, _, _, _ = load_hirham_climate(
        thinning_factor=thinning_factor
    )
    std_dev_prior = np.zeros_like(temp_prior)

    # Y_obs = torch.vstack(
    #     (
    #         torch.from_numpy(a_obs),
    #         torch.from_numpy(m_obs),
    #         torch.from_numpy(r_obs),
    #     )
    # ).T.type(torch.FloatTensor)

    X_m = []
    for k, row in mcmc_df.iterrows():
        m_f_snow = row["f_snow"]
        m_f_ice = row["f_ice"]
        m_refreeze = row["refreeze"]
        X_m.append(
            torch.from_numpy(
                np.hstack(
                    (
                        np.tile(m_f_snow, (temp_prior.shape[1], 1)),
                        np.tile(m_f_ice, (temp_prior.shape[1], 1)),
                        np.tile(m_refreeze, (temp_prior.shape[1], 1)),
                    )
                )
            )
        )

    X_prior = torch.vstack(X_m).type(torch.FloatTensor)
    X_min = X_prior.cpu().numpy().min(axis=0)
    X_max = X_prior.cpu().numpy().max(axis=0)

    sigma_hat = torch.tensor([0.1, 0.1, 0.1, 0.1])
    sh = torch.ones_like(Y_obs)
    sigma_hat = sh * torch.tensor([0.5, 0.5, 0.1, 0.1])
    X_keys = ["f_snow", "f_ice", "refreeze"]

    burn = 1000
    alpha = 100
    alpha_b = 3.0
    beta_b = 3.0
    X_prior = beta.rvs(alpha_b, beta_b, size=(samples, 3)) * (X_max - X_min) + X_min
    # Initial condition for MAP. Note that using 0 yields similar results
    X_0 = torch.tensor(
        X_prior.mean(axis=0), requires_grad=True, dtype=torch.float, device=device
    )

    start = time.process_time()
    sampler = MALASampler(
        e,
        torch.from_numpy(temp_obs),
        torch.from_numpy(precip_obs),
        torch.from_numpy(std_dev_obs),
        X_min,
        X_max,
        Y_obs,
        sigma_hat,
        posterior_dir=".",
        device=device,
        alpha=alpha,
    )
    X_map = sampler.find_MAP(X_0)
    X_posterior = sampler.sample(
        X_map,
        samples=samples,
        burn=burn,
        save_interval=1000,
        print_interval=100,
    )
    print(time.process_time() - start)

    # mala = MALAPDDSampler(e, emulator_dir=emulator_dir)
    # X_map = mala.find_MAP(
    #     X_P_0, temp_obs, precip_obs, std_dev_obs, Y_target, X_P_min, X_P_max
    # )

    # # To reproduce the paper, n_iters should be 10^5
    # X_posterior = mala.MALA(
    #     X_map,
    #     X_P_min,
    #     X_P_max,
    #     temp_obs,
    #     precip_obs,
    #     std_dev_obs,
    #     Y_target,
    #     n_iters=n_iters,
    #     model_index=int(model_index),
    #     save_interval=1000,
    #     print_interval=100,
    # )

    # X_std = X_P_std
    # X_mean = X_P_mean
    # frac = 1.0
    # lw = 1
    # color_prior = "b"
    # X_list = []
    # X_keys = ["f_snow", "f_ice", "refreeze"]
    # X_Prior = (
    #     X_P_prior.detach().cpu().numpy() * X_P_std[-3::].detach().cpu().numpy()
    #     + X_P_mean[-3::].detach().cpu().numpy()
    # )
    # keys_dict = {
    #     "f_ice": "$f_{\mathrm{ice}}$",
    #     "f_snow": "$f_{\mathrm{snoe}}$",
    #     "refreeze": "$r$",
    # }
    # p = Path(f"{emulator_dir}/posterior_samples/")
    # print("Loading posterior samples\n")
    # for m, m_file in enumerate(sorted(p.glob("X_posterior_model_*.csv.gz"))):
    #     print(f"  -- {m_file}")
    #     df = pd.read_csv(m_file).sample(frac=frac)
    #     if "Unnamed: 0" in df.columns:
    #         df.drop(columns=["Unnamed: 0"], inplace=True)
    #     model = m_file.name.split("_")[-1].split(".")[0]
    #     df["Model"] = int(model)
    #     X_list.append(df)

    # print(f"Merging posteriors into dataframe")
    # posterior_df = pd.concat(X_list)

    # X_posterior = posterior_df.drop(columns=["Model"]).values
    # C_0 = np.corrcoef((X_posterior - X_posterior.mean(axis=0)).T)
    # Cn_0 = (np.sign(C_0) * C_0**2 + 1) / 2.0

    # fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    # fig.subplots_adjust(hspace=0.25, wspace=0.25)
    # for i in range(3):
    #     min_val = min(X_Prior[:, i].min(), X_posterior[:, i].min())
    #     max_val = max(X_Prior[:, i].max(), X_posterior[:, i].max())
    #     bins = np.linspace(min_val, max_val, 100)
    #     X_prior_hist, b = np.histogram(X_Prior[:, i], bins, density=True)
    #     X_posterior_hist, _ = np.histogram(X_posterior[:, i], bins, density=True)
    #     b = 0.5 * (b[1:] + b[:-1])
    #     axs[i].plot(
    #         b,
    #         X_posterior_hist * 0.5,
    #         color="k",
    #         linewidth=lw,
    #         linestyle="solid",
    #     )
    #     axs[i].axvline(f_true[i], lw=3, color="orange", label="true")

    # figfile = f"{emulator_dir}/posterior.pdf"
    # print(f"Saving figure to {figfile}")
    # fig.savefig(figfile)

    # fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(5.4, 5.6))
    # fig.subplots_adjust(hspace=0.0, wspace=0.0)
    # for i in range(3):
    #     for j in range(3):
    #         if i > j:

    #             axs[i, j].scatter(
    #                 X_posterior[:, j],
    #                 X_posterior[:, i],
    #                 c="#31a354",
    #                 s=0.05,
    #                 alpha=0.01,
    #                 label="Posterior",
    #                 rasterized=True,
    #             )

    #             min_val = min(X_Prior[:, i].min(), X_posterior[:, i].min())
    #             max_val = max(X_Prior[:, i].max(), X_posterior[:, i].max())
    #             bins_y = np.linspace(min_val, max_val, 100)

    #             min_val = min(X_Prior[:, j].min(), X_posterior[:, j].min())
    #             max_val = max(X_Prior[:, j].max(), X_posterior[:, j].max())
    #             bins_x = np.linspace(min_val, max_val, 100)

    #             v = gaussian_kde(X_posterior[:, [j, i]].T)
    #             bx = 0.5 * (bins_x[1:] + bins_x[:-1])
    #             by = 0.5 * (bins_y[1:] + bins_y[:-1])
    #             Bx, By = np.meshgrid(bx, by)

    #             axs[i, j].contour(
    #                 Bx,
    #                 By,
    #                 v(np.vstack((Bx.ravel(), By.ravel()))).reshape(Bx.shape),
    #                 7,
    #                 linewidths=0.5,
    #                 colors="black",
    #             )

    #             axs[i, j].set_xlim(X_Prior[:, j].min(), X_Prior[:, j].max())
    #             axs[i, j].set_ylim(X_Prior[:, i].min(), X_Prior[:, i].max())

    #         elif i < j:
    #             patch_upper = Polygon(
    #                 np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]),
    #                 facecolor=plt.cm.seismic(Cn_0[i, j]),
    #             )
    #             axs[i, j].add_patch(patch_upper)
    #             if C_0[i, j] > -0.5:
    #                 color = "black"
    #             else:
    #                 color = "white"
    #             axs[i, j].text(
    #                 0.5,
    #                 0.5,
    #                 "{0:.2f}".format(C_0[i, j]),
    #                 fontsize=6,
    #                 horizontalalignment="center",
    #                 verticalalignment="center",
    #                 transform=axs[i, j].transAxes,
    #                 color=color,
    #             )

    #         elif i == j:

    #             min_val = min(X_Prior[:, i].min(), X_posterior[:, i].min())
    #             max_val = max(X_Prior[:, i].max(), X_posterior[:, i].max())
    #             bins = np.linspace(min_val, max_val, 30)
    #             X_prior_hist, b = np.histogram(X_Prior[:, i], bins, density=True)
    #             X_posterior_hist, _ = np.histogram(
    #                 X_posterior[:, i], bins, density=True
    #             )
    #             b = 0.5 * (b[1:] + b[:-1])

    #             axs[i, j].plot(
    #                 b,
    #                 X_prior_hist,
    #                 color=color_prior,
    #                 linewidth=lw,
    #                 label="Prior",
    #                 linestyle="solid",
    #             )

    #             all_models = posterior_df["Model"].unique()
    #             for k, m_model in enumerate(all_models):
    #                 m_df = posterior_df[posterior_df["Model"] == m_model].drop(
    #                     columns=["Model"]
    #                 )
    #                 X_model_posterior = m_df.values
    #                 X_model_posterior_hist, _ = np.histogram(
    #                     X_model_posterior[:, i], _, density=True
    #                 )
    #                 if k == 0:
    #                     axs[i, j].plot(
    #                         b,
    #                         X_model_posterior_hist * 0.5,
    #                         color="0.5",
    #                         linewidth=lw * 0.25,
    #                         linestyle="solid",
    #                         alpha=0.5,
    #                         label="Posterior (BayesBag)",
    #                     )
    #                 else:
    #                     axs[i, j].plot(
    #                         b,
    #                         X_model_posterior_hist * 0.5,
    #                         color="0.5",
    #                         linewidth=lw * 0.25,
    #                         linestyle="solid",
    #                         alpha=0.5,
    #                     )

    #             axs[i, j].plot(
    #                 b,
    #                 X_posterior_hist,
    #                 color="black",
    #                 linewidth=lw,
    #                 linestyle="solid",
    #                 label="Posterior",
    #             )

    #             axs[i, j].set_xlim(min_val, max_val)

    #         else:
    #             axs[i, j].remove()

    # for i, ax in enumerate(axs[:, 0]):
    #     ax.set_ylabel(keys_dict[X_keys[i]])

    # for j, ax in enumerate(axs[-1, :]):
    #     ax.set_xlabel(keys_dict[X_keys[j]])
    #     plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    #     plt.setp(ax.xaxis.get_minorticklabels(), rotation=45)
    #     if j > 0:
    #         ax.tick_params(axis="y", which="both", length=0)
    #         ax.yaxis.set_minor_formatter(NullFormatter())
    #         ax.yaxis.set_major_formatter(NullFormatter())

    # for ax in axs[:-1, 0].ravel():
    #     ax.xaxis.set_major_formatter(NullFormatter())
    #     ax.xaxis.set_minor_formatter(NullFormatter())
    #     ax.tick_params(axis="x", which="both", length=0)

    # for ax in axs[:-1, 1:].ravel():
    #     ax.xaxis.set_major_formatter(NullFormatter())
    #     ax.xaxis.set_minor_formatter(NullFormatter())
    #     ax.yaxis.set_major_formatter(NullFormatter())
    #     ax.yaxis.set_minor_formatter(NullFormatter())
    #     ax.tick_params(axis="both", which="both", length=0)

    # l_prior = Line2D([], [], c=color_prior, lw=lw, ls="solid", label="Prior")
    # l_post = Line2D([], [], c="k", lw=lw, ls="solid", label="Posterior")
    # l_post_b = Line2D(
    #     [], [], c="0.25", lw=lw * 0.25, ls="solid", label="Posterior (BayesBag)"
    # )

    # legend = fig.legend(
    #     handles=[l_prior, l_post, l_post_b], bbox_to_anchor=(0.3, 0.955)
    # )
    # legend.get_frame().set_linewidth(0.0)
    # legend.get_frame().set_alpha(0.0)

    # figfile = f"{emulator_dir}/emulator_posterior.pdf"
    # print(f"Saving figure to {figfile}")
    # fig.savefig(figfile)
