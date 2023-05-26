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

import contextlib

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
import joblib
from joblib import Parallel, delayed

# from lightning.pytorch.tuner import Tuner
from pyDOE import lhs
from scipy.special import gamma
from scipy.stats import dirichlet, gaussian_kde
from scipy.stats.distributions import uniform
from sklearn.metrics import mean_squared_error

from tqdm import tqdm
from scipy.stats import beta

import time

from pismemulator.nnemulator import TorchPDDModel as PDDModel
from pismemulator.datamodules import PDDDataModule
from pismemulator.utils import load_hirham_climate_w_std_dev

torch.autograd.set_detect_anomaly(True)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


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
        model: PDDModel
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
        emulator_dir="./emulator",
        device="cpu",
    ):
        super().__init__()
        self.model = model
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
        self.emulator_dir = emulator_dir
        self.hessian_counter = 0

    def find_MAP(
        self,
        X: torch.tensor,
        n_iters: int = 26,
        verbose: bool = False,
        print_interval: int = 10,
    ):
        print("***********************************************")
        print("Finding Maximum A-Posterior (MAP) point")
        print("***********************************************")
        # Line search distances
        alphas = torch.logspace(-4, 0, 11)
        # Find MAP point
        for i in tqdm(range(n_iters)):
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

    def neg_log_prob(
        self,
        X,
    ):
        self.model.f_snow = X[0]
        self.model.f_ice = X[1]
        self.model.refreeze_snow = X[2]
        self.model.refreeze_ice = X[3]
        self.model.temp_snow = X[4]
        self.model.temp_rain = X[5]
        result = self.model.forward(
            self.temp_obs,
            self.precip_obs,
            self.std_dev_obs,
        )

        A = result["accu"]
        M = result["melt"]
        R = result["runoff"]
        F = result["refreeze"]
        B = result["smb"]

        Y_pred = torch.vstack((A, M, R, F, B)).T.type(torch.FloatTensor).to(device)

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

    def get_log_like_gradient_and_hessian(self, X, eps=1e-24, compute_hessian=False):
        log_pi = self.neg_log_prob(X)
        if compute_hessian:
            self.hessian_counter += 1
            g = torch.autograd.grad(log_pi, X, retain_graph=True, create_graph=True)[0]
            H = torch.autograd.functional.hessian(
                self.neg_log_prob, X, create_graph=True
            )
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
        """
        Log-likelihood for a multivariate Normal distribution

        """
        # - 0.5 * log_det_Hinv - 0.5 * (Y - mu) @ H / (2*h) * (Y - mu)
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

        logq = (
            torch.distributions.MultivariateNormal(X_, covariance_matrix=Hinv / (2 * h))
            .log_prob(X)
            .sum()
        )
        logq_ = (
            torch.distributions.MultivariateNormal(X, covariance_matrix=Hinv / (2 * h))
            .log_prob(X_)
            .sum()
        )

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
        chain: int = 0,
        samples: int = 10001,
        h: float = 0.1,
        h_max: float = 1.0,
        acc_target: float = 0.25,
        k: float = 0.01,
        beta: float = 0.99,
        save_interval: int = 1000,
        print_interval: int = 50,
        validate: bool = False,
    ):
        if validate:
            posterior_dir = f"{self.emulator_dir}/posterior_samples_validate/"
        else:
            posterior_dir = f"{self.emulator_dir}/posterior_samples/"
        if not os.path.isdir(posterior_dir):
            os.makedirs(posterior_dir)

        local_data = None
        m_vars = []
        acc = acc_target
        progress = tqdm(range(samples + burn), position=chain, leave=False)
        for i in progress:
            X, local_data, s = self.MALA_step(X, h, local_data=local_data)
            m_vars.append(X.detach())
            acc = beta * acc + (1 - beta) * s
            h = min(h * (1 + k * np.sign(acc - acc_target)), h_max)
            log_p = local_data[0].item()
            desc = f"chain {chain}:  accept rate: {acc:.2f}, step size: {h:.2f}, log(P): {log_p:.1f}"
            progress.set_description(desc=desc)
            if ((i + burn) % save_interval == 0) & (i >= burn):
                X_posterior = torch.stack(m_vars).cpu().numpy()[burn::]
                df = pd.DataFrame(
                    data=X_posterior.astype("float32"),
                    columns=X_keys,
                )
                df["Committee Member"] = model_index
                df.to_parquet(
                    join(
                        posterior_dir,
                        f"X_posterior_model_{model_index}_chain_{chain}.parquet",
                    )
                )

        X_posterior = torch.stack(m_vars).cpu().numpy()
        return X_posterior


def draw_samples(n_samples=250, random_seed=2):
    np.random.seed(random_seed)

    distributions = {
        "f_snow": uniform(loc=1.0, scale=5.0),  # uniform between 1 and 6
        "f_ice": uniform(loc=3.0, scale=12),  # uniform between 3 and 15
        "refreeze_snow": uniform(loc=0.0, scale=1.0),  # uniform between 0 and 1
        "refreeze_ice": uniform(loc=0.0, scale=1.0),  # uniform between 0 and 1
        "temp_snow": uniform(loc=-1.0, scale=2.0),  # uniform between 0 and 1
        "temp_rain": uniform(loc=1.0, scale=2.0),  # uniform between 0 and 1
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
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--emulator_dir", default="emulator_ensemble")
    parser.add_argument("--model_index", type=int, default=0)
    parser.add_argument("--n_interpolate", type=int, default=52)
    parser.add_argument("--chains", type=int, default=5)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--burn", type=int, default=1_000)
    parser.add_argument("--thinning_factor", type=int, default=100)
    parser.add_argument("--validate", action="store_true", default=False)
    parser.add_argument(
        "--training_file", type=str, default="DMI-HIRHAM5_1980_2020_MMS.nc"
    )
    parser.add_argument("--use_obs_sd", default=False, action="store_true")

    args = parser.parse_args()
    hparams = vars(args)

    alpha = args.alpha
    burn = args.burn
    device = args.device
    emulator_dir = args.emulator_dir
    model_index = args.model_index
    n_interpolate = args.n_interpolate
    n_chains = args.chains
    samples = args.samples
    thinning_factor = args.thinning_factor
    training_file = args.training_file
    use_observed_std_dev = args.use_obs_sd
    validate = args.validate

    if not os.path.isdir(emulator_dir):
        os.makedirs(emulator_dir)
        os.makedirs(os.path.join(emulator_dir, "emulator"))

    n_parameters = 12 * 3 + 6
    n_outputs = 5
    posteriors = []

    torch.manual_seed(0)
    pl.seed_everything(0)
    np.random.seed(model_index)

    prior_df = draw_samples(n_samples=250)

    # Create observations using the forward model
    obs_df = draw_samples(n_samples=100, random_seed=4)
    (
        temp,
        precip,
        std_dev,
        snow_depth,
        accumulation,
        melt,
        runoff,
        refreeze,
        smb,
    ) = load_hirham_climate_w_std_dev(training_file, thinning_factor=thinning_factor)
    if not use_observed_std_dev:
        std_dev = np.zeros_like(temp)

    if validate:
        f_snow_val = 3.2
        f_ice_val = 8.5
        refreeze_snow_val = 0.6
        refreeze_ice_val = 0.2
        temp_snow_val = 0.0
        temp_rain_val = 2.0
        f_true = [
            f_snow_val,
            f_ice_val,
            refreeze_snow_val,
            refreeze_ice_val,
            temp_snow_val,
            temp_rain_val,
        ]

        pdd = PDDModel(
            pdd_factor_snow=f_snow_val,
            pdd_factor_ice=f_ice_val,
            refreeze_snow=refreeze_snow_val,
            refreeze_ice=refreeze_ice_val,
            temp_snow=temp_snow_val,
            temp_rain=temp_rain_val,
            n_interpolate=n_interpolate,
        )
        result = pdd(temp, precip, std_dev)

        A = result["accu"]
        M = result["melt"]
        R = result["runoff"]
        F = result["refreeze"]
        B = result["smb"]

        Y_obs = torch.vstack((A, M, R, F, B)).T.type(torch.FloatTensor).to(device)
    else:
        Y_obs = (
            torch.vstack(
                (
                    torch.from_numpy(accumulation),
                    torch.from_numpy(melt),
                    torch.from_numpy(runoff),
                    torch.from_numpy(refreeze),
                    torch.from_numpy(smb),
                )
            )
            .T.type(torch.FloatTensor)
            .to(device)
        )

    X_prior = torch.from_numpy(prior_df.values).type(torch.FloatTensor)
    X_min = X_prior.cpu().numpy().min(axis=0)
    X_max = X_prior.cpu().numpy().max(axis=0)

    sh = torch.ones_like(Y_obs)
    sigma_hat = sh * torch.tensor([0.001, 0.001, 0.001, 0.001, 0.001]).to(device)
    X_keys = [
        "f_snow",
        "f_ice",
        "refreeze_snow",
        "refreeze_ice",
        "temp_snow",
        "temp_rain",
    ]

    alpha_b = 3.0
    beta_b = 3.0
    X_prior = (
        beta.rvs(alpha_b, beta_b, size=(10_000, X_prior.shape[-1])) * (X_max - X_min)
        + X_min
    )
    # Initial condition for MAP. Note that using 0 yields similar results
    X_0 = torch.tensor(
        X_prior.mean(axis=0), requires_grad=True, dtype=torch.float, device=device
    )

    pdd = PDDModel()

    start = time.process_time()
    sampler = MALASampler(
        pdd,
        torch.from_numpy(temp),
        torch.from_numpy(precip),
        torch.from_numpy(std_dev),
        X_min,
        X_max,
        Y_obs,
        sigma_hat,
        emulator_dir=emulator_dir,
        device=device,
        alpha=alpha,
    )
    X_map = sampler.find_MAP(X_0)

    result = Parallel(n_jobs=n_chains)(
        delayed(sampler.sample)(
            X_map,
            samples=samples,
            burn=burn,
            chain=c,
            save_interval=1000,
            print_interval=100,
            validate=validate,
        )
        for c in range(n_chains)
    )

    elapsed_time = time.process_time() - start
    print(f"Sampling took {elapsed_time:.0f}s")
