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

from tqdm import tqdm
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
        Xp = torch.vstack((
            self.temp_obs,
            self.precip_obs,
            self.std_dev_obs,
            torch.tile(X, (temp_obs.shape[1], 1)).T
            )).T

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
        "refreeze_snow": uniform(loc=0, scale=1.0),  # uniform between 0 and 1
        "refreeze_ice": uniform(loc=0, scale=1.0),  # uniform between 0 and 1
        "temp_snow": uniform(loc=-1, scale=1.0),  # uniform between 0 and 1
        "temp_rain": uniform(loc=0.5, scale=1.5),  # uniform between 0 and 1
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

    (temp, precip, a, m, r, b, f) = load_hirham_climate(thinning_factor=thinning_factor)

    # temp = temp.reshape(1, -1)
    # precip = precip.reshape(1, -1)
    # a = a.reshape(1, -1)
    # m = m.reshape(1, -1)
    # r = r.reshape(1, -1)
    # b = b.reshape(1, -1)
    # f = f.reshape(1, -1)

    std_dev = np.zeros_like(temp)
    prior_df = draw_samples(n_samples=50)

    nt = temp.shape[0]

    X_m = []
    Y_m = []
    for k, row in prior_df.iterrows():
        m_f_snow = row["f_snow"]
        m_f_ice = row["f_ice"]
        m_refreeze_snow = row["refreeze_snow"]
        m_refreeze_ice = row["refreeze_ice"]
        m_temp_snow = row["temp_snow"]
        m_temp_rain = row["temp_rain"]
        params = np.hstack(
            [np.tile(row[k], (temp.shape[1], 1)) for k in range(len(row))]
        )

        pdd = TorchPDDModel(
            pdd_factor_snow=m_f_snow,
            pdd_factor_ice=m_f_ice,
            refreeze_snow=m_refreeze_snow,
            refreeze_ice=m_refreeze_ice,
            temp_snow=m_temp_snow,
            temp_rain=m_temp_rain,
            n_interpolate=12,
        )
        result = pdd(temp, precip, std_dev)

        A = result["accu"]
        M = result["melt"]
        R = result["runoff"]
        F = result["refreeze"]
        B = result["smb"]

        m_Y = torch.vstack(
            (
                A,
                M,
                R,
                F,
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
                        params,
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
    data_loader = PDDDataModule(X, Y, omegas, omegas_0, num_workers=num_workers)
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

    # # Out-Of-Set validation

    # # Is there a better way to re-use the device for inference?
    # device = e.device.type
    # e.to(device)

    # # Load trained emulator
    # e = PDDEmulator.load_from_checkpoint(
    #     f"{emulator_dir}/emulator/emulator_{model_index}.ckpt",
    #     n_parameters=n_parameters,
    #     n_outputs=n_outputs,
    # )

    # X_val = torch.vstack([d[0] for d in data_loader.val_data])
    # Y_val = torch.vstack([d[1] for d in data_loader.val_data])

    # e.eval()

    # Y_pred = e(X_val).detach().cpu()
    # rmse = [
    #     np.sqrt(
    #         mean_squared_error(
    #             Y_pred.detach().cpu().numpy()[:, i], Y_val.detach().cpu().numpy()[:, i]
    #         )
    #     )
    #     for i in range(Y_val.shape[1])
    # ]
    # print("RMSE")
    # print(f"A={rmse[0]:.6f}, M={rmse[1]:.6f}", f"R={rmse[2]:.6f}, B={rmse[3]:.6f}")

    # fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
    # fig.subplots_adjust(hspace=0.25, wspace=0.25)
    # axs[0].plot(Y_val[:, 0], Y_pred[:, 0], ".", ms=0.25, label="Accumulation")
    # axs[1].plot(Y_val[:, 1], Y_pred[:, 1], ".", ms=0.25, label="Melt")
    # axs[2].plot(Y_val[:, 2], Y_pred[:, 2], ".", ms=0.25, label="Runoff")
    # axs[3].plot(Y_val[:, 3], Y_pred[:, 3], ".", ms=0.25, label="SMB")
    # for k in range(4):
    #     m_max = np.ceil(np.maximum(Y_val[:, k].max(), Y_pred[:, k].max()))
    #     m_min = np.floor(np.minimum(Y_val[:, k].min(), Y_pred[:, k].min()))
    #     axs[k].set_xlim(m_min, m_max)
    #     axs[k].set_ylim(m_min, m_max)
    # # axs[0].axis("equal")
    # # axs[1].axis("equal")
    # # axs[2].axis("equal")
    # axs[0].legend()
    # axs[1].legend()
    # axs[2].legend()
    # axs[3].legend()
    # fig.savefig(f"{emulator_dir}/validation.pdf")

    # # Create observations using the forward model
    # obs_df = draw_samples(n_samples=100, random_seed=4)
    # temp_obs, precip_obs, a_obs, m_obs, r_obs, f_obs, b_obs = load_hirham_climate(thinning_factor=100)
    # std_dev_obs = np.zeros_like(temp_obs)

    # f_snow_obs = 3.44
    # f_ice_obs = 7.79
    # refreeze_snow_obs = 0.4
    # refreeze_ice_obs = 0.0
    # temp_snow_obs = 0.0
    # temp_rain_obs = 1.0
    # f_true = [f_snow_obs, f_ice_obs, refreeze_snow_obs, refreeze_ice_obs, temp_snow_obs, temp_rain_obs]

    # pdd = TorchPDDModel(
    #     pdd_factor_snow=f_snow_obs,
    #     pdd_factor_ice=f_ice_obs,
    #     refreeze_snow=refreeze_snow_obs,
    #     refreeze_ice=refreeze_ice_obs,
    #     temp_snow=temp_snow_obs,
    #     temp_rain=temp_rain_obs,
    # )
    # result = pdd(temp_obs, precip_obs, std_dev_obs)

    # A_obs = result["accu"]
    # M_obs = result["melt"]
    # R_obs = result["runoff"]
    # F_obs = result["refreeze"]
    # B_obs = result["smb"]

    # Y_obs = torch.vstack((A_obs, M_obs, R_obs, F_obs, B_obs)).T.type(torch.FloatTensor)
    # # Y_obs = torch.vstack((torch.from_numpy(a_obs), torch.from_numpy(m_obs), torch.from_numpy(r_obs), torch.from_numpy(f_obs), torch.from_numpy(b_obs))).T.type(torch.FloatTensor)

    # # Create observations using the forward model
    # mcmc_df = draw_samples(n_samples=1_000, random_seed=5)

    # X_prior = torch.from_numpy(prior_df.values).type(torch.FloatTensor)
    # X_min = X_prior.cpu().numpy().min(axis=0)
    # X_max = X_prior.cpu().numpy().max(axis=0)

    # sh = torch.ones_like(Y_obs)
    # sigma_hat = sh * torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01])
    # X_keys = ["f_snow", "f_ice", "refreeze_snow", "refreeze_ice", "temp_snow", "temp_rain"]

    # burn = 1000
    # alpha = 1
    # alpha_b = 3.0
    # beta_b = 3.0
    # X_prior = beta.rvs(alpha_b, beta_b, size=(samples, X_prior.shape[-1])) * (X_max - X_min) + X_min
    # # Initial condition for MAP. Note that using 0 yields similar results
    # X_0 = torch.tensor(
    #     X_prior.mean(axis=0), requires_grad=True, dtype=torch.float, device=device
    # )

    # start = time.process_time()
    # sampler = MALASampler(
    #     e,
    #     torch.from_numpy(temp_obs),
    #     torch.from_numpy(precip_obs),
    #     torch.from_numpy(std_dev_obs),
    #     X_min,
    #     X_max,
    #     Y_obs,
    #     sigma_hat,
    #     posterior_dir=".",
    #     device=device,
    #     alpha=alpha,
    # )
    # X_map = sampler.find_MAP(X_0)
    # X_posterior = sampler.sample(
    #     X_map,
    #     samples=samples,
    #     burn=burn,
    #     save_interval=1000,
    #     print_interval=100,
    # )
    # print(time.process_time() - start)
