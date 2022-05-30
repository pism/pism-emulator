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

from argparse import ArgumentParser

import numpy as np
import os
import pandas as pd
from pyDOE import lhs
from scipy.stats import dirichlet
from scipy.stats.distributions import uniform
from scipy.special import gamma


import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pismemulator.nnemulator import PDDEmulator, TorchPDDModel, PDDDataModule
from pismemulator.utils import load_hirham_climate


class MALASampler(object):
    """
    MALA Sampler

    Author: Douglas C Brinkerhoff, University of Montana
    """

    def __init__(
        self,
        model,
        alpha_b=3.0,
        beta_b=3.0,
        alpha=0.01,
        nu=1,
        emulator_dir="./emulator",
    ):
        super().__init__()
        self.model = model.eval()
        self.alpha = alpha
        self.alpha_b = alpha_b
        self.beta_b = beta_b
        self.nu = nu
        self.emulator_dir = emulator_dir

    def find_MAP(self, X, X_I, Y_target, X_min, X_max, n_iters=50, print_interval=10):
        print("***********************************************")
        print("***********************************************")
        print("Finding MAP point")
        print("***********************************************")
        print("***********************************************")
        # Line search distances
        alphas = np.logspace(-4, 0, 11)
        # Find MAP point
        for i in range(n_iters):
            log_pi, g, _, Hinv, log_det_Hinv = self.get_log_like_gradient_and_hessian(
                X, X_I, Y_target, X_min, X_max, compute_hessian=True
            )
            p = Hinv @ -g
            alpha_index = np.nanargmin(
                [
                    self.get_log_like_gradient_and_hessian(
                        X + alpha * p,
                        X_I,
                        Y_target,
                        X_min,
                        X_max,
                        compute_hessian=False,
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
                print(
                    "".join(
                        [
                            f"{key}: {(val * std + mean):.3f}\n"
                            for key, val, std, mean in zip(
                                X_P_keys,
                                X.data.cpu().numpy(),
                                X_P_std,
                                X_P_mean,
                            )
                        ]
                    )
                )

                print("===============================================")
        return X

    def V(self, X, X_I, Y_target, X_bar):
        # model result is in log space
        X_IP = torch.hstack((X, X_I))
        Y_pred = self.model(X_IP)
        r = Y_pred - Y_target
        L1 = torch.sum(
            np.log(gamma((self.nu + 1) / 2.0))
            - np.log(gamma(self.nu / 2.0))
            - np.log(np.sqrt(np.pi * nu) * sigma_hat)
            - (self.nu + 1) / 2.0 * torch.log(1 + 1.0 / self.nu * (r / sigma_hat) ** 2)
        )
        L2 = torch.sum(
            (self.alpha_b - 1) * torch.log(X_bar)
            + (self.beta_b - 1) * torch.log(1 - X_bar)
        )

        return -(self.alpha * L1 + L2)

    def get_log_like_gradient_and_hessian(
        self, X, X_I, Y_target, X_min, X_max, eps=1e-2, compute_hessian=False
    ):

        X_bar = (X - X_min) / (X_max - X_min)
        log_pi = self.V(X, X_I, Y_target, X_bar)
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

    def MALA_step(self, X, X_I, Y_target, X_min, X_max, h, local_data=None):
        if local_data is not None:
            pass
        else:
            local_data = self.get_log_like_gradient_and_hessian(
                X, X_I, Y_target, X_min, X_max, compute_hessian=True
            )

        log_pi, _, H, Hinv, log_det_Hinv = local_data

        X_ = self.draw_sample(X, 2 * h * Hinv).detach()
        X_.requires_grad = True

        log_pi_ = self.get_log_like_gradient_and_hessian(
            X_, X_I, Y_target, X_min, X_max, compute_hessian=False
        )

        logq = self.get_proposal_likelihood(X_, X, H / (2 * h), log_det_Hinv)
        logq_ = self.get_proposal_likelihood(X, X_, H / (2 * h), log_det_Hinv)

        log_alpha = -log_pi_ + logq_ + log_pi - logq
        alpha = torch.exp(min(log_alpha, torch.tensor([0.0], device=device)))
        u = torch.rand(1, device=device)
        if u <= alpha and log_alpha != np.inf:
            X.data = X_.data
            local_data = self.get_log_like_gradient_and_hessian(
                X, X_I, Y_target, X_min, X_max, compute_hessian=True
            )
            s = 1
        else:
            s = 0
        return X, local_data, s

    def MALA(
        self,
        X,
        X_I,
        X_min,
        X_max,
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
        print(
            "Running Metropolis-Adjusted Langevin Algorithm for model index {0}".format(
                model_index
            )
        )
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
            X, local_data, s = self.MALA_step(
                X, X_I, Y_target, X_min, X_max, h, local_data=local_data
            )
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
                                X_P_keys,
                                X.data.cpu().numpy(),
                                X_P_std,
                                X_P_mean,
                            )
                        ]
                    )
                )

                print("===============================================")

            if i % save_interval == 0:
                print("///////////////////////////////////////////////")
                print("Saving samples for model {0}".format(model_index))
                print("///////////////////////////////////////////////")
                X_posterior = torch.stack(m_vars).cpu().numpy()
                df = pd.DataFrame(
                    data=X_posterior.astype("float32") * X_P_std.cpu().numpy()
                    + X_P_mean.cpu().numpy(),
                    columns=X_P_keys,
                )
                df.to_csv(
                    posterior_dir + "X_posterior_model_{0}.csv.gz".format(model_index),
                    compression="infer",
                )
        X_posterior = torch.stack(m_vars).cpu().numpy()
        return X_posterior


def draw_samples(n_samples=250, random_seed=2):
    np.random.seed(random_seed)

    distributions = {
        "f_snow": uniform(loc=1.0, scale=4.0),  # uniform between 2 and 6
        "f_ice": uniform(loc=3.0, scale=9),  # uniform between 3 and 12
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
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=5)
    parser.add_argument("--train_size", type=float, default=1.0)
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

    if not os.path.isdir(emulator_dir):
        os.makedirs(emulator_dir)
        os.makedirs(os.path.join(emulator_dir, "emulator"))

    if not os.path.isdir(emulator_dir):
        os.makedirs(emulator_dir)
        os.makedirs(os.path.join(emulator_dir, "emulator"))

    temp, precip, _, _, _, _ = load_hirham_climate(thinning_factor=200)
    std_dev = np.zeros_like(temp)

    prior_df = draw_samples(n_samples=100)

    X = []
    Y = []
    for k, row in prior_df.iterrows():
        m_f_snow = row["f_snow"]
        m_f_ice = row["f_ice"]
        m_refreeze = row["refreeze"]

        pdd = TorchPDDModel(
            pdd_factor_snow=m_f_snow,
            pdd_factor_ice=m_f_ice,
            refreeze_snow=m_refreeze,
            refreeze_ice=m_refreeze,
        )
        result = pdd(temp, precip, std_dev)

        M_train = result["melt"]
        A_train = result["accu"]
        R_train = result["refreeze"]
        m_Y = torch.vstack(
            (
                M_train,
                A_train,
                R_train,
            )
        ).T
        Y.append(m_Y)
        X.append(
            torch.from_numpy(
                np.hstack(
                    (
                        temp.T,
                        precip.T,
                        np.tile(m_f_snow, (temp.shape[1], 1)),
                        np.tile(m_f_ice, (temp.shape[1], 1)),
                        np.tile(m_refreeze, (temp.shape[1], 1)),
                    )
                )
            )
        )

    X_train = torch.vstack(X).type(torch.FloatTensor)
    Y_train = torch.vstack(Y).type(torch.FloatTensor)
    n_samples, n_parameters = X_train.shape
    n_outputs = Y_train.shape[1]

    # Normalize
    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    X_train_norm = (X_train - X_train_mean) / X_train_std

    callbacks = []

    print(f"Training model {model_index}")
    omegas = torch.Tensor(dirichlet.rvs(np.ones(n_samples))).T
    omegas = omegas.type(torch.FloatTensor)
    omegas_0 = torch.ones_like(omegas) / len(omegas)
    area = torch.ones_like(omegas)
    train_size = 1.0
    num_workers = 8
    hparams = {"n_layers": 5, "n_hidden": 128, "batch_size": 128, "learning_rate": 0.01}

    if train_size == 1.0:
        data_loader = PDDDataModule(
            X_train_norm, Y_train, omegas, omegas_0, num_workers=num_workers
        )
    else:
        data_loader = PDDDataModule(
            X_train_norm,
            Y_train,
            omegas,
            omegas_0,
            train_size=train_size,
            num_workers=num_workers,
        )

    data_loader.setup()

    e = PDDEmulator(
        n_parameters,
        n_outputs,
        hparams,
    )

    logger = TensorBoardLogger(tb_logs_dir, name=f"Emulator {model_index}")
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    if train_size == 1.0:
        train_loader = data_loader.train_all_loader
        val_loader = data_loader.val_all_loader
    else:
        train_loader = data_loader.train_loader
        val_loader = data_loader.val_loader

    trainer.fit(e, train_loader, val_loader)
    trainer.save_checkpoint(f"{emulator_dir}/emulator/emulator_{model_index}.ckpt")
    torch.save(e.state_dict(), f"{emulator_dir}/emulator/emulator_{model_index}.h5")

    # Out-Of-Set validation
    val_df = draw_samples(n_samples=100, random_seed=3)

    X = []
    Y = []
    for k, row in val_df.iterrows():
        m_f_snow = row["f_snow"]
        m_f_ice = row["f_ice"]
        m_refreeze = row["refreeze"]

        pdd = TorchPDDModel(
            pdd_factor_snow=m_f_snow,
            pdd_factor_ice=m_f_ice,
            refreeze_snow=m_refreeze,
            refreeze_ice=m_refreeze,
        )
        result = pdd(temp, precip, std_dev)

        M_val = result["melt"]
        A_val = result["accu"]
        R_val = result["refreeze"]
        m_Y = torch.vstack(
            (
                M_val,
                A_val,
                R_val,
            )
        ).T
        Y.append(m_Y)
        X.append(
            torch.from_numpy(
                np.hstack(
                    (
                        temp.T,
                        precip.T,
                        np.tile(m_f_snow, (temp.shape[1], 1)),
                        np.tile(m_f_ice, (temp.shape[1], 1)),
                        np.tile(m_refreeze, (temp.shape[1], 1)),
                    )
                )
            )
        )

    X_val = torch.vstack(X).type(torch.FloatTensor)
    Y_val = torch.vstack(Y).type(torch.FloatTensor)

    X_val_mean = X_val.mean(axis=0)
    X_val_std = X_val.std(axis=0)
    X_val_norm = (X_val - X_val_mean) / X_val_std

    from sklearn.metrics import mean_squared_error

    e.eval()
    Y_pred = e(X_val_norm).detach().cpu()
    rmse = [
        np.sqrt(
            mean_squared_error(
                Y_pred.detach().cpu().numpy()[:, i], Y_val.detach().cpu().numpy()[:, i]
            )
        )
        for i in range(Y_val.shape[1])
    ]
    print("RMSE")
    print(f"A={rmse[0]:.4f}, M={rmse[1]:.4f}, R={rmse[2]:.4f}")

    nu = 1
    mcmc_df = draw_samples(n_samples=200, random_seed=4)
    temp_test, precip_test, _, _, _, _ = load_hirham_climate(thinning_factor=25)
    std_dev_test = np.zeros_like(temp_test)

    f_snow_test = 3.0
    f_ice_test = 8.0
    refreeze_test = 0.0
    pdd = TorchPDDModel(
        pdd_factor_snow=f_snow_test,
        pdd_factor_ice=f_ice_test,
        refreeze_snow=refreeze_test,
        refreeze_ice=refreeze_test,
    )
    result = pdd(temp_test, precip_test, std_dev_test)

    M_test = result["melt"]
    A_test = result["accu"]
    R_test = result["refreeze"]

    Y_test = torch.vstack((M_test, A_test, R_test)).T.type(torch.FloatTensor)

    device = "cpu"
    X = []
    Y = []
    for k, row in mcmc_df.iterrows():
        m_f_snow = row["f_snow"]
        m_f_ice = row["f_ice"]
        m_refreeze = row["refreeze"]
        X.append(
            torch.from_numpy(
                np.hstack(
                    (
                        temp_test.T,
                        precip_test.T,
                        np.tile(m_f_snow, (temp_test.shape[1], 1)),
                        np.tile(m_f_ice, (temp_test.shape[1], 1)),
                        np.tile(m_refreeze, (temp_test.shape[1], 1)),
                    )
                )
            )
        )

    X_test = torch.vstack(X).type(torch.FloatTensor)
    X_test_mean = X_test.nanmean(axis=0)
    X_test_std = X_test.std(axis=0)

    X_test_norm = torch.nan_to_num((X_test - X_test_mean) / X_test_std)

    X_P_mean = X_test_mean[-3::].to(device)
    X_P_std = X_test_std[-3::].to(device)
    X_P_prior = X_test_norm[:, -3::].to(device)
    X_I_prior = X_test_norm[:, :-3].to(device)

    X_min = X_test_norm.cpu().numpy().min(axis=0)
    X_max = X_test_norm.cpu().numpy().max(axis=0)

    sigma_hat = 0.01

    X_min = torch.tensor(X_min, dtype=torch.float32, device=device)
    X_max = torch.tensor(X_max, dtype=torch.float32, device=device)

    # Needs
    # alpha_b, beta_b: float
    # alpha: float
    # nu: float
    # gamma
    # sigma_hat
    X_P_0 = torch.tensor(
        X_P_prior.mean(axis=0), requires_grad=True, dtype=torch.float, device=device
    )

    X_I_0 = torch.tensor(
        X_I_prior.mean(axis=0), requires_grad=True, dtype=torch.float, device=device
    )

    X_P_min = X_min[-3:]
    X_P_max = X_max[-3:]

    Y_target = Y_test.to(device)

    n_iters = 20000
    X_P_keys = ["f_snow", "f_ice", "refreeze"]
    mala = MALASampler(e, emulator_dir=emulator_dir)
    X_map = mala.find_MAP(X_P_0, X_I_0, Y_target, X_P_min, X_P_max)

    # To reproduce the paper, n_iters should be 10^5
    X_posterior = mala.MALA(
        X_map,
        X_I_0,
        X_P_min,
        X_P_max,
        Y_target,
        n_iters=n_iters,
        model_index=int(model_index),
        save_interval=1000,
        print_interval=100,
    )
