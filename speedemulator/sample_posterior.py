#!/bin/env python3

import os
import time
from argparse import ArgumentParser
from os.path import join
from typing import Union

import arviz as az
import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
import torch
from lightning import LightningModule
from scipy.stats import beta
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm

from pismemulator.datasets import PISMDataset
from pismemulator.nnemulator import NNEmulator
from pismemulator.utils import param_keys_dict as keys_dict


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

    def torch_find_MAP(
        self,
        X: torch.tensor,
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
                        dataset.X_keys,
                        X.data.cpu().numpy(),
                        dataset.X_std,
                        dataset.X_mean,
                    )
                ]
            )
        )
        return X

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
                            f"{key}: {(val * std + mean):.3f}\n"
                            for key, val, std, mean in zip(
                                dataset.X_keys,
                                X.data.cpu().numpy(),
                                dataset.X_std,
                                dataset.X_mean,
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
                        dataset.X_keys,
                        X.data.cpu().numpy(),
                        dataset.X_std,
                        dataset.X_mean,
                    )
                ]
            )
        )
        return X

    def V(
        self,
        X,
    ):

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
            # + torch.lgamma(alpha_b + beta_b)
            # - torch.lgamma(alpha_b)
            # - torch.lgamma(beta_b)
        )
        print(log_likelihood, log_prior)
        log_likelihood = torch.distributions.StudentT(nu).log_prob(t).sum()
        log_prior = torch.distributions.Beta(alpha_b, beta_b).log_prob(X_bar).sum()
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
        model_index: int = 0,
        save_interval: int = 1000,
        print_interval: int = 50,
    ):
        print(
            "****************************************************************************"
        )
        print(
            "****************************************************************************"
        )
        print(
            "Running Metropolis-Adjusted Langevin Algorithm for model {0}".format(
                model_index
            )
        )
        print(
            "****************************************************************************"
        )
        print(
            "****************************************************************************"
        )

        posterior_dir = f"{self.emulator_dir}/posterior_samples/"
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
                print(f"Saving samples for model {model_index}")
                print("///////////////////////////////////////////////")
                X_posterior = torch.stack(m_vars).cpu().numpy()
                df = pd.DataFrame(
                    data=X_posterior.astype("float32") * dataset.X_std.cpu().numpy()
                    + dataset.X_mean.cpu().numpy(),
                    columns=dataset.X_keys,
                )
                if out_format == "csv":
                    df.to_csv(join(posterior_dir, f"X_posterior_model_{0}.csv.gz"))
                elif out_format == "parquet":
                    df.to_parquet(join(posterior_dir, f"X_posterior_model_{0}.parquet"))
                else:
                    raise NotImplementedError(f"{out_format} not implemented")

        X_posterior = torch.stack(m_vars).cpu().numpy()
        return X_posterior


if __name__ == "__main__":
    __spec__ = None

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", default=False, action="store_true")
    parser.add_argument("--data_dir", default="../tests/training_data")
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
    parser.add_argument("--thinning_factor", type=int, default=1)

    parser = NNEmulator.add_model_specific_args(parser)
    args = parser.parse_args()
    hparams = vars(args)

    checkpoint = args.checkpoint
    data_dir = args.data_dir
    device = args.device
    emulator_dir = args.emulator_dir
    alpha = args.alpha
    model_index = args.model_index
    samples = args.samples
    burn = args.burn
    out_format = args.out_format
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
    e.to(device)

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
        beta.rvs(alpha_b, beta_b, size=(samples, n_parameters)) * (X_max - X_min)
        + X_min
    )
    # Initial condition for MAP. Note that using 0 yields similar results
    X_0 = torch.tensor(
        X_prior.mean(axis=0), requires_grad=True, dtype=torch.float, device=device
    )

    start = time.process_time()
    sampler = MALASampler(
        e,
        X_min,
        X_max,
        Y_target,
        sigma_hat,
        emulator_dir=emulator_dir,
        device=device,
        alpha=alpha,
    )
    X_map = sampler.find_MAP(X_0)
    X_posterior = sampler.sample(
        X_map,
        samples=samples,
        model_index=int(model_index),
        burn=burn,
        save_interval=1000,
        print_interval=100,
    )
    print(time.process_time() - start)
    X_posterior *= dataset.X_std.cpu().numpy()
    X_posterior += dataset.X_mean.cpu().numpy()

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
    for k, key in enumerate(dataset.X_keys):
        d[key] = X_posterior[:, k]

    trace = az.convert_to_inference_data(d)
    az.plot_trace(trace)
