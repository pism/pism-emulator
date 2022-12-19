#!/bin/env python3

import os
import time
from argparse import ArgumentParser
from os.path import join

import numpy as np
import pandas as pd
import torch
from scipy.special import gamma
from scipy.stats import beta

from pismemulator.nnemulator import NNEmulator, PISMDataset


def torch_find_MAP(X, X_min, X_max, Y_target, model):
    Y_pred = 10 ** model(X, add_mean=True)
    r = Y_pred - Y_target

    t = r / sigma_hat
    X_bar = torch.tensor(X, requires_grad=True)

    learning_rate = 0.1
    for it in range(51):
        X_bar.data = (X_bar.data - X_min) / (X_max - X_min)
        likelihood_dist = torch.distributions.StudentT(nu)
        log_likelihood = (
            likelihood_dist.log_prob(t**2).sum() - np.log(sigma_hat).sum()
        )
        prior_dist = torch.distributions.Beta(alpha_b, beta_b)
        log_prior = (
            prior_dist.log_prob(X_bar).sum()
            * torch.lgamma(torch.tensor(alpha_b + beta_b))
            / (torch.lgamma(torch.tensor(alpha_b)) * torch.lgamma(torch.tensor(beta_b)))
        )
        print(log_likelihood, log_prior)
        NLL = -(log_likelihood + log_prior)
        NLL.backward()

        if it % 10 == 0:
            print(f"iter: {it:d}, log(P): {NLL:.1f}\n")
            print(
                "".join(
                    [
                        f"{key}: {(val * std + mean):.3f}\n"
                        for key, val, std, mean in zip(
                            dataset.X_keys,
                            X_bar.data.cpu().numpy(),
                            dataset.X_std,
                            dataset.X_mean,
                        )
                    ]
                )
            )

        X_bar.data -= learning_rate * X_bar.grad.data
        X_bar.grad.data.zero_()
    return X_bar


class MALASampler(object):
    """
    MALA Sampler

    Author: Douglas C Brinkerhoff, University of Montana
    """

    def __init__(
        self,
        model,
        X_min,
        X_max,
        Y_target,
        sigma_hat,
        alpha=0.01,
        alpha_b=3.0,
        beta_b=3.0,
        nu=1.0,
        emulator_dir="./emulator",
        device="cpu",
    ):
        super().__init__()
        self.model = model.eval()
        self.X_min = X_min
        self.X_max = X_max
        self.Y_target = Y_target
        self.sigma_hat = torch.tensor(sigma_hat, device=device)
        self.alpha = torch.tensor(alpha, device=device)
        self.alpha_b = torch.tensor(alpha_b, device=device)
        self.beta_b = torch.tensor(beta_b, device=device)
        self.nu = torch.tensor(nu, device=device)
        self.emulator_dir = emulator_dir

    def find_MAP(
        self,
        X,
        n_iters=51,
        print_interval=10,
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
            if i % print_interval == 0:
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
                print("===============================================")
        return X

    def V(
        self,
        X,
    ):
        X_bar = (X - self.X_min) / (self.X_max - self.X_min)

        Y_pred = 10 ** self.model(X, add_mean=True)
        r = Y_pred - self.Y_target
        sigma_hat = self.sigma_hat
        t = r / sigma_hat
        nu = self.nu

        # Likelihood
        log_likelihood = torch.sum(
            torch.lgamma((nu + 1) / 2.0)
            - torch.lgamma(nu / 2.0)
            - torch.log(torch.sqrt(torch.pi * nu) * sigma_hat)
            - (nu + 1) / 2.0 * torch.log(1 + 1.0 / nu * t**2)
        )

        # Prior
        log_prior = torch.sum(
            (self.alpha_b - 1) * torch.log(X_bar)
            + (self.beta_b - 1) * torch.log(1 - X_bar)
        )
        return -(self.alpha * log_likelihood + log_prior)

    def get_log_like_gradient_and_hessian(self, X, eps=1e-2, compute_hessian=False):

        log_pi = self.V(X)
        if compute_hessian:
            g = torch.autograd.grad(log_pi, X, retain_graph=True, create_graph=True)[0]
            H = torch.stack(
                [torch.autograd.grad(e, X, retain_graph=True)[0] for e in g]
            )
            lamda, Q = torch.linalg.eig(H)
            lamda, Q = lamda.type(torch.float), Q.type(torch.float)
            lamda_prime = torch.sqrt(lamda**2 + eps)
            lamda_prime_inv = 1.0 / torch.sqrt(lamda**2 + eps)
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
        return -0.5 * log_det_cov - 0.5 * (Y - mu) @ inverse_cov @ (Y - mu)

    def MALA_step(self, X, h, local_data=None):
        if local_data is not None:
            pass
        else:
            local_data = self.get_log_like_gradient_and_hessian(X, compute_hessian=True)

        log_pi, _, H, Hinv, log_det_Hinv = local_data

        X_ = self.draw_sample(X, 2 * h * Hinv).detach()
        X_.requires_grad = True

        log_pi_ = self.get_log_like_gradient_and_hessian(X_, compute_hessian=False)

        logq = self.get_proposal_likelihood(X_, X, H / (2 * h), log_det_Hinv)
        logq_ = self.get_proposal_likelihood(X, X_, H / (2 * h), log_det_Hinv)

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

    def MALA(
        self,
        X,
        burn: int = 1000,
        n_samples: int = 10001,
        h: float = 0.1,
        h_max: float = 1.0,
        acc_target=0.25,
        k=0.01,
        beta=0.99,
        model_index=0,
        save_interval=1000,
        print_interval=50,
    ):
        print(
            "********************************************************************************"
        )
        print(
            "********************************************************************************"
        )
        print(
            "Running Metropolis-Adjusted Langevin Algorithm for model index {0}".format(
                model_index
            )
        )
        print(
            "********************************************************************************"
        )
        print(
            "********************************************************************************"
        )

        posterior_dir = f"{self.emulator_dir}/posterior_samples/"
        if not os.path.isdir(posterior_dir):
            os.makedirs(posterior_dir)

        local_data = None
        m_vars = []
        acc = acc_target
        for i in range(n_samples + burn):
            X, local_data, s = self.MALA_step(X, h, local_data=local_data)
            m_vars.append(X.detach())
            acc = beta * acc + (1 - beta) * s
            h = min(h * (1 + k * np.sign(acc - acc_target)), h_max)
            if i % print_interval == 0:
                print("===============================================")
                print(
                    "sample: {0:d}, acc. rate: {1:4.2f}, log(P): {2:6.1f}".format(
                        i, acc, local_data[0].item()
                    )
                )
                print(
                    " ".join(
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
                print("===============================================")

            if (i % save_interval == 0) & (i >= burn):
                print("///////////////////////////////////////////////")
                print("Saving samples for model {0}".format(model_index))
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
    parser.add_argument("--data_dir", default="../tests/training_data")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--emulator_dir", default="emulator_ensemble")
    parser.add_argument("--model_index", type=int, default=0)
    parser.add_argument("--num_posterior_samples", type=int, default=100000)
    parser.add_argument("--num_iterations", type=int, default=100000)
    parser.add_argument("--out_format", choices=["csv", "parquet"], default="parquet")
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

    data_dir = args.data_dir
    device = args.device
    emulator_dir = args.emulator_dir
    model_index = args.model_index
    n_posterior_samples = args.num_posterior_samples
    n_samples = args.num_iterations
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

    # alpha = 0.01

    nu = 1.0

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
        beta.rvs(alpha_b, beta_b, size=(n_posterior_samples, n_parameters))
        * (X_max - X_min)
        + X_min
    )
    # Initial condition for MAP. Note that using 0 yields similar results
    X_0 = torch.tensor(
        X_prior.mean(axis=0), requires_grad=True, dtype=torch.float, device=device
    )
    X_min = torch.tensor(X_min, dtype=torch.float32, device=device)
    X_max = torch.tensor(X_max, dtype=torch.float32, device=device)

    # Needs
    # alpha_b, beta_b: float
    # alpha: float
    # nu: float
    # gamma
    # sigma_hat
    Y_target = dataset.Y_target.to(device)

    start = time.process_time()
    mala = MALASampler(
        e, X_min, X_max, Y_target, sigma_hat, emulator_dir=emulator_dir, device=device
    )
    X_map = mala.find_MAP(X_0)
    # To reproduce the paper, n_samples should be 10^5
    n_samples = 1000
    X_posterior = mala.MALA(
        X_map,
        n_samples=n_samples,
        model_index=int(model_index),
        save_interval=1000,
        print_interval=100,
    )
    print(time.process_time() - start)
