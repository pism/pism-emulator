#!/bin/env python3

from argparse import ArgumentParser

from glob import glob
import numpy as np
import os
from os.path import join
from scipy.special import gamma
from scipy.stats import beta

import torch
import torch.multiprocessing as mp

from pismemulator.nnemulator import NNEmulator, PISMDataset

import pandas as pd
import pylab as plt
import seaborn as sns


class MALASampler(object):
    """
    MALA Sampler
    """

    def __init__(self, model, alpha_b=3.0, beta_b=3.0, alpha=0.01, emulator_dir="./emulator"):
        super().__init__()
        self.model = model.eval()
        self.alpha = alpha
        self.alpha_b = alpha_b
        self.beta_b = beta_b
        self.emulator_dir = emulator_dir

    def find_MAP(self, X, Y_target, X_min, X_max, n_iters=50, print_interval=10):
        print("***********************************************")
        print("***********************************************")
        print("Finding MAP point")
        print("***********************************************")
        print("***********************************************")
        # Line search distances
        alphas = np.logspace(-4, 0, 11)
        # Find MAP point
        for i in range(n_iters):
            log_pi, g, H, Hinv, log_det_Hinv = self.get_log_like_gradient_and_hessian(
                X, Y_target, X_min, X_max, compute_hessian=True
            )
            p = Hinv @ -g
            alpha_index = np.nanargmin(
                [
                    self.get_log_like_gradient_and_hessian(
                        X + alpha * p, Y_target, X_min, X_max, compute_hessian=False
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    for alpha in alphas
                ]
            )
            mu = X + alphas[alpha_index] * p
            X.data = mu.data
            if i % print_interval == 0:
                print("===============================================")
                print(f"iter: {i:d}, log(P): {log_pi:.1f}\n")
                print("".join([f"{i}: {(10**j):.3f}\n" for i, j in zip(dataset.X_keys, X.data.cpu().numpy())]))
                print("===============================================")
        return X

    def V(self, X, Y_target, X_bar):
        # model result is in log space
        Y_pred = 10 ** self.model(X, add_mean=True)
        r = Y_pred - Y_target
        L1 = torch.sum(
            np.log(gamma((nu + 1) / 2.0))
            - np.log(gamma(nu / 2.0))
            - np.log(np.sqrt(np.pi * nu) * sigma_hat)
            - (nu + 1) / 2.0 * torch.log(1 + 1.0 / nu * (r / sigma_hat) ** 2)
        )
        L2 = torch.sum((self.alpha_b - 1) * torch.log(X_bar) + (self.beta_b - 1) * torch.log(1 - X_bar))

        return -(self.alpha * L1 + L2)

    def get_log_like_gradient_and_hessian(self, X, Y_target, X_min, X_max, eps=1e-2, compute_hessian=False):

        X_bar = (X - X_min) / (X_max - X_min)
        log_pi = self.V(X, Y_target, X_bar)
        if compute_hessian:
            g = torch.autograd.grad(log_pi, X, retain_graph=True, create_graph=True)[0]
            H = torch.stack([torch.autograd.grad(e, X, retain_graph=True)[0] for e in g])
            lamda, Q = torch.eig(H, eigenvectors=True)
            lamda_prime = torch.sqrt(lamda[:, 0] ** 2 + eps)
            lamda_prime_inv = 1.0 / torch.sqrt(lamda[:, 0] ** 2 + eps)
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

    def MALA_step(self, X, Y_target, X_min, X_max, h, local_data=None):
        if local_data is not None:
            pass
        else:
            local_data = self.get_log_like_gradient_and_hessian(X, Y_target, X_min, X_max, compute_hessian=True)

        log_pi, g, H, Hinv, log_det_Hinv = local_data

        X_ = self.draw_sample(X, 2 * h * Hinv).detach()
        X_.requires_grad = True

        log_pi_ = self.get_log_like_gradient_and_hessian(X_, Y_target, X_min, X_max, compute_hessian=False)

        logq = self.get_proposal_likelihood(X_, X, H / (2 * h), log_det_Hinv)
        logq_ = self.get_proposal_likelihood(X, X_, H / (2 * h), log_det_Hinv)

        log_alpha = -log_pi_ + logq_ + log_pi - logq
        alpha = torch.exp(min(log_alpha, torch.tensor([0.0], device=device)))
        u = torch.rand(1, device=device)
        if u <= alpha and log_alpha != np.inf:
            X.data = X_.data
            local_data = self.get_log_like_gradient_and_hessian(X, Y_target, X_min, X_max, compute_hessian=True)
            s = 1
        else:
            s = 0
        return X, local_data, s

    def MALA(
        self,
        X,
        Y_target,
        n_iters=10001,
        h=0.1,
        h_max=1.0,
        acc_target=0.25,
        k=0.01,
        beta=0.99,
        model_index=0,
        save_interval=1000,
        print_interval=50,
    ):
        print("***********************************************")
        print("***********************************************")
        print("Running Metropolis-Adjusted Langevin Algorithm for model index {0}".format(model_index))
        print("***********************************************")
        print("***********************************************")

        posterior_dir = f"{self.emulator_dir}/posterior_samples/"
        if not os.path.isdir(posterior_dir):
            os.makedirs(posterior_dir)

        local_data = None
        m_vars = []
        acc = acc_target
        print(n_iters)
        for i in range(n_iters):
            X, local_data, s = self.MALA_step(X, Y_target, X_min, X_max, h, local_data=local_data)
            m_vars.append(X.detach())
            acc = beta * acc + (1 - beta) * s
            h = min(h * (1 + k * np.sign(acc - acc_target)), h_max)
            if i % print_interval == 0:
                print("===============================================")
                print("sample: {0:d}, acc. rate: {1:4.2f}, log(P): {2:6.1f}".format(i, acc, local_data[0].item()))
                print(
                    " ".join(
                        [
                            f"{i}: {(j * dataset.X_std + dataset.X_mean):.3f}\n"
                            for i, j in zip(dataset.X_keys, X.data.cpu().numpy())
                        ]
                    )
                )
                print("===============================================")

            if i % save_interval == 0:
                print("///////////////////////////////////////////////")
                print("Saving samples for model {0:03d}".format(model_index))
                print("///////////////////////////////////////////////")
                X_posterior = torch.stack(m_vars).cpu().numpy()
                np.save(
                    open(posterior_dir + "X_posterior_model_{0:03d}.npy".format(model_index), "wb"),
                    X_posterior.astype("float32"),
                )
                df = pd.DataFrame(
                    data=X_posterior.astype("float32") * dataset.X_std + dataset.X_mean, columns=Dataset.X_keys
                )
                df.to_csv(posterior_dir + "X_posterior_model_{0:03d}.csv".format(model_index), compression="infer")
        X_posterior = torch.stack(m_vars).cpu().numpy()
        return X_posterior


if __name__ == "__main__":
    __spec__ = None

    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="../tests/training_data")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--emulator_dir", default="emulator_ensemble")
    parser.add_argument("--model_index", type=str, default=0)
    parser.add_argument("--num_posterior_samples", type=int, default=100000)
    parser.add_argument("--samples_file", default="../data/samples/velocity_calibration_samples_50.csv")
    parser.add_argument("--target_file", default="../data/validation/greenland_vel_mosaic250_v1_g9000m.nc")
    parser.add_argument("--thinning_factor", type=int, default=1)

    parser = NNEmulator.add_model_specific_args(parser)
    args = parser.parse_args()
    hparams = vars(args)

    data_dir = args.data_dir
    device = args.device
    emulator_dir = args.emulator_dir
    model_index = args.model_index
    n_posterior_samples = args.num_posterior_samples
    samples_file = args.samples_file
    target_file = args.target_file
    thinning_factor = args.thinning_factor

    dataset = PISMDataset(
        data_dir=data_dir,
        samples_file=samples_file,
        target_file=target_file,
        thinning_factor=thinning_factor,
    )

    X = dataset.X
    F = dataset.Y
    n_grid_points = dataset.n_grid_points
    n_parameters = dataset.n_parameters
    n_samples = dataset.n_samples

    torch.manual_seed(0)
    np.random.seed(0)
    emulator_file = join(emulator_dir, f"emulator_{model_index}.h5")

    state_dict = torch.load(emulator_file)
    e = NNEmulator(
        state_dict["l_1.weight"].shape[1],
        state_dict["V_hat"].shape[1],
        state_dict["V_hat"],
        state_dict["F_mean"],
        dataset.normed_area,
        hparams,
    )
    e.load_state_dict(state_dict)
    e.to(device)

    # alpha = 0.01

    nu = 1.0

    sigma = 10
    rho = 1.0 / (1e4 ** 2)
    point_area = (dataset.grid_resolution * thinning_factor) ** 2
    K = point_area * rho
    sigma_hat = np.sqrt(sigma ** 2 / K ** 2)

    X_min = X.cpu().numpy().min(axis=0) - 1e-3
    X_max = X.cpu().numpy().max(axis=0) + 1e-3
    # Eq 52
    # this is 2.0 in the paper
    alpha_b = 3.0
    beta_b = 3.0
    X_prior = beta.rvs(alpha_b, beta_b, size=(n_posterior_samples, n_parameters)) * (X_max - X_min) + X_min
    X_0 = torch.tensor(X_prior.mean(axis=0), requires_grad=True, dtype=torch.float, device=device)
    # This is required for
    # X_bar = (X - X_min) / (X_max - X_min)
    # to work

    X_min = torch.tensor(X_min, dtype=torch.float32, device=device)
    X_max = torch.tensor(X_max, dtype=torch.float32, device=device)

    # Needs
    # alpha_b, beta_b: float
    # alpha: float
    # nu: float
    # gamma
    # sigma_hat
    U_target = dataset.Y_target

    mala = MALASampler(e, emulator_dir=emulator_dir)
    X_map = mala.find_MAP(X_0, U_target, X_min, X_max)
    # To reproduce the paper, n_iters should be 10^5
    X_posterior = mala.MALA(
        X_map,
        U_target,
        n_iters=n_posterior_samples,
        model_index=int(model_index),
        save_interval=1000,
        print_interval=100,
    )
