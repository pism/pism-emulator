# Copyright (C) 2022 Andy Aschwanden, Douglas C Brinkerhoff
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


import future, sys, os, datetime, argparse

# print(os.path.dirname(sys.executable))
import torch
import numpy as np
import matplotlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Union

# matplotlib.rcParams["figure.figsize"] = [10, 10]

import torch
from torch.nn import Module, Parameter, Sequential
from torch.nn import Linear, Tanh, ReLU, CELU
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical, Normal
import lightning as pl

from joblib import Parallel, delayed

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)


from pismemulator.probmodel import ProbModel


class GMM(ProbModel):
    def __init__(self):

        dataloader = DataLoader(TensorDataset(torch.zeros(1, 2)))  # bogus dataloader
        ProbModel.__init__(self, dataloader)

        self.means = FloatTensor([[-1, -1.25], [-1, 1.25], [1.5, 1]])
        # self.means = FloatTensor([[-1,-1.25]])
        self.num_dists = self.means.shape[0]
        I = FloatTensor([[1, 0], [0, 1]])
        I_compl = FloatTensor([[0, 1], [1, 0]])
        self.covars = [I * 0.5, I * 0.5, I * 0.5 + I_compl * 0.3]
        # self.covars = [I * 0.9, I * 0.9, I * 0.9 + I_compl * 0.3]
        self.weights = [0.4, 0.2, 0.4]
        self.dists = []

        for mean, covar in zip(self.means, self.covars):
            self.dists.append(MultivariateNormal(mean, covar))

        self.X_grid = None
        self.Y_grid = None
        self.surface = None

        self.param = torch.nn.Parameter(self.sample())

    def forward(self, x=None):

        log_probs = torch.stack(
            [
                weight * torch.exp(dist.log_prob(x))
                for dist, weight in zip(self.dists, self.weights)
            ],
            dim=1,
        )
        log_prob = torch.log(torch.sum(log_probs, dim=1))

        return log_prob

    def log_prob(self, *x):

        log_probs = torch.stack(
            [
                weight * torch.exp(dist.log_prob(self.param))
                for dist, weight in zip(self.dists, self.weights)
            ],
            dim=1,
        )
        log_prob = torch.log(torch.sum(log_probs, dim=1))

        return {"log_prob": log_prob}

    def prob(self, x):

        log_probs = torch.stack(
            [
                weight * torch.exp(dist.log_prob(x))
                for dist, weight in zip(self.dists, self.weights)
            ],
            dim=1,
        )
        log_prob = torch.sum(log_probs, dim=1)
        return log_prob

    def sample(self, _shape=(1,)):

        probs = torch.ones(self.num_dists) / self.num_dists
        categorical = Categorical(probs)
        sampled_dists = categorical.sample(_shape)

        samples = []
        for sampled_dist in sampled_dists:
            sample = self.dists[sampled_dist].sample((1,))
            samples.append(sample)

        samples = torch.cat(samples)

        return samples

    def reset_parameters(self):

        self.param.data = self.sample()

    def generate_surface(self, plot_min=-3, plot_max=3, plot_res=500, plot=False):

        # print('in surface')

        x = np.linspace(plot_min, plot_max, plot_res)
        y = np.linspace(plot_min, plot_max, plot_res)
        X, Y = np.meshgrid(x, y)

        self.X_grid = X
        self.Y_grid = Y

        points = FloatTensor(np.stack((X.ravel(), Y.ravel())).T)  # .requires_grad_()

        probs = self.prob(points).view(plot_res, plot_res)
        self.surface = probs.numpy()

        area = ((plot_max - plot_min) / plot_res) ** 2
        sum_px = (
            probs.sum() * area
        )  # analogous to integrating cubes: volume is probs are the height times the area

        fig = plt.figure(figsize=(10, 10))

        contour = plt.contourf(self.X_grid, self.Y_grid, self.surface, levels=20)
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.grid()
        cbar = fig.colorbar(contour)
        if plot:
            plt.show()

        return fig


class StudentT(ProbModel):
    def __init__(
        self,
        emulator: pl.LightningModule,
        X_0,
        X_min: Union[float, torch.tensor],
        X_max: Union[float, torch.tensor],
        Y_target: Union[np.ndarray, torch.tensor],
        sigma_hat: Union[np.ndarray, torch.tensor],
        X_mean: Union[float, np.ndarray, torch.tensor] = 1.0,
        X_std: Union[float, np.ndarray, torch.tensor] = 1.0,
        X_keys: list = [],
        alpha: Union[float, torch.tensor] = 0.01,
        alpha_b: Union[float, torch.tensor] = 3.0,
        beta_b: Union[float, torch.tensor] = 3.0,
        nu: Union[float, torch.tensor] = 1.0,
        device="cpu",
    ):
        dataloader = DataLoader(
            TensorDataset(torch.zeros_like(X_0))
        )  # bogus dataloader
        ProbModel.__init__(self, dataloader)
        self.emulator = emulator.eval()
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
        self.X_mean = (
            torch.tensor(X_mean, dtype=torch.float32, device=device)
            if not isinstance(X_mean, torch.Tensor)
            else X_mean.to(device)
        )
        self.X_std = (
            torch.tensor(X_std, dtype=torch.float32, device=device)
            if not isinstance(X_std, torch.Tensor)
            else X_std.to(device)
        )
        self.X_keys = X_keys
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
        self.X = torch.nn.Parameter(torch.tensor(X_0), requires_grad=True)

    def reset_parameters(self):

        samples = self.sample()
        print("Resetting: new samples", samples)
        self.X.data = samples

    def sample(self):
        h: float = 0.1
        log_pi, _, H, Hinv, log_det_Hinv = self.get_log_like_gradient_and_hessian(
            self.X, compute_hessian=True
        )

        samples = self.draw_sample(self.X, 2 * h * Hinv).detach()
        print("Sampling new", samples)

        return samples

    def draw_sample(self, mu, cov, eps=1e-10):
        L = torch.linalg.cholesky(cov + eps * torch.eye(cov.shape[0]))
        return mu + L @ torch.randn(L.shape[0])

    def get_proposal_likelihood(self, Y, mu, inverse_cov, log_det_cov):
        return -0.5 * log_det_cov - 0.5 * (Y - mu) @ inverse_cov @ (Y - mu)

    def log_prob(self, *x):
        X = self.X
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
        return {
            "log_prob": log_prob,
        }

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
            H = torch.stack(
                [torch.autograd.grad(e, X, retain_graph=True)[0] for e in g]
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
                    f"{(val * std + mean):.3f}\n"
                    for val, std, mean in zip(
                        X.data.cpu().numpy(),
                        self.X_std,
                        self.X_mean,
                    )
                ]
            )
        )
        self.X = X


class LinReg(ProbModel):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, log_noise: float = 0):

        self.data = x
        self.target = y

        dataloader = DataLoader(
            TensorDataset(self.data, self.target),
            shuffle=True,
            batch_size=self.data.shape[0],
        )

        self.dataloader = dataloader

        ProbModel.__init__(self, dataloader)
        self.m = Parameter(FloatTensor(1 * torch.randn((1,))))
        self.b = Parameter(FloatTensor(1 * torch.randn((1,))))
        # self.log_noise = Parameter(FloatTensor([-1.]))
        self.log_noise = FloatTensor([log_noise])

    def reset_parameters(self):
        torch.nn.init.normal_(self.m, std=0.1)
        torch.nn.init.normal_(self.b, std=0.11)

    def sample(self):
        self.reset_parameters()

    def forward(self, x):

        return self.m * x + self.b

    def log_prob(self, data, target):

        mu = self.forward(data)
        sigma = F.softplus(self.log_noise)
        log_prob = Normal(mu, sigma).log_prob(target).mean()

        return {"log_prob": log_prob}

    @torch.no_grad()
    def predict(self, chain):

        x_min = 2 * self.data.min()
        x_max = 2 * self.data.max()
        data = torch.arange(x_min, x_max).reshape(-1, 1)

        pred = []
        for model_state_dict in chain.samples:
            self.load_state_dict(model_state_dict)
            # data.append(self.data)
            pred_i = self.forward(data)
            pred.append(pred_i)

        pred = torch.stack(pred)
        # data = torch.stack(data)

        mu = pred.mean(dim=0).squeeze()
        std = pred.std(dim=0).squeeze()

        # print(f'{data.shape=}')
        # print(f'{pred.shape=}')

        plt.plot(data, mu, alpha=1.0, color="red")
        plt.fill_between(data.squeeze(), mu + std, mu - std, color="red", alpha=0.25)
        plt.fill_between(
            data.squeeze(), mu + 2 * std, mu - 2 * std, color="red", alpha=0.10
        )
        plt.fill_between(
            data.squeeze(), mu + 3 * std, mu - 3 * std, color="red", alpha=0.05
        )
        plt.scatter(self.data, self.target, alpha=1, s=1, color="blue")
        plt.ylim(pred.min(), pred.max())
        plt.xlim(x_min, x_max)
        plt.show()


class RegressionNNHomo(ProbModel):
    def __init__(self, x, y, batch_size=1):

        self.data = x
        self.target = y

        # dataloader = DataLoader(TensorDataset(self.data, self.target), shuffle=True, batch_size=self.data.shape[0], drop_last=False)
        dataloader = DataLoader(
            TensorDataset(self.data, self.target),
            shuffle=True,
            batch_size=batch_size,
            drop_last=False,
        )

        ProbModel.__init__(self, dataloader)

        num_hidden = 50
        self.model = Sequential(
            Linear(1, num_hidden),
            ReLU(),
            Linear(num_hidden, num_hidden),
            ReLU(),
            # Linear(num_hidden, num_hidden),
            # ReLU(),
            # Linear(num_hidden, num_hidden),
            # ReLU(),
            Linear(num_hidden, 1),
        )

        self.log_std = Parameter(FloatTensor([-1]))

    def reset_parameters(self):
        for module in self.model.modules():
            if isinstance(module, Linear):
                module.reset_parameters()

        self.log_std.data = FloatTensor([3.0])

    def sample(self):
        self.reset_parameters()

    def forward(self, x):
        pred = self.model(x)
        return pred

    def log_prob(self, data, target):

        # if data is None and target is None:
        # 	data, target = next(self.dataloader.__iter__())

        mu = self.forward(data)
        mse = F.mse_loss(mu, target)

        log_prob = Normal(mu, F.softplus(self.log_std)).log_prob(target).mean() * len(
            self.dataloader.dataset
        )

        return {"log_prob": log_prob, "MSE": mse.detach_()}

    def pretrain(self):

        num_epochs = 200
        optim = torch.optim.Adam(self.parameters(), lr=0.01)

        # print(f"{F.softplus(self.log_std)=}")

        progress = tqdm(range(num_epochs))
        for epoch in progress:
            for batch_i, (data, target) in enumerate(self.dataloader):
                optim.zero_grad()
                mu = self.forward(data)
                loss = -Normal(mu, F.softplus(self.log_std)).log_prob(target).mean()
                mse_loss = F.mse_loss(mu, target)
                loss.backward()
                optim.step()

                desc = f"Pretraining: MSE:{mse_loss:.3f}"
                progress.set_description(desc)

        # print(f"{F.softplus(self.log_std)=}")

    @torch.no_grad()
    def predict(self, chains, plot=False):

        x_min = 2 * self.data.min()
        x_max = 2 * self.data.max()
        data = torch.linspace(x_min, x_max).reshape(-1, 1)

        def parallel_predict(parallel_chain):
            parallel_pred = []
            for model_state_dict in parallel_chain.samples[::50]:
                self.load_state_dict(model_state_dict)
                pred_mu_i = self.forward(data)
                parallel_pred.append(pred_mu_i)
            try:
                parallel_pred_mu = torch.stack(
                    parallel_pred
                )  # list [ pred_0, pred_1, ... pred_N] -> Tensor([pred_0, pred_1, ... pred_N])
                return parallel_pred_mu
            except:
                pass

        parallel_pred = Parallel(n_jobs=len(chains))(
            delayed(parallel_predict)(chain) for chain in chains
        )

        pred = [
            parallel_pred_i
            for parallel_pred_i in parallel_pred
            if parallel_pred_i is not None
        ]  # flatten [ [pred_chain_0], [pred_chain_1] ... [pred_chain_N] ]
        # pred_log_std = [parallel_pred_i for parallel_pred_i in parallel_pred_log_std if parallel_pred_i is not None] # flatten [ [pred_chain_0], [pred_chain_1] ... [pred_chain_N] ]

        pred = torch.cat(
            pred
        ).squeeze()  # cat list of tensors to single prediciton tensor with samples in first dim
        std = F.softplus(self.log_std)

        epistemic = pred.std(dim=0)
        aleatoric = std
        total_std = (epistemic**2 + aleatoric**2) ** 0.5

        mu = pred.mean(dim=0)
        std = std.mean(dim=0)

        data.squeeze_()

        if plot:
            fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
            axs = axs.flatten()

            axs[0].scatter(self.data, self.target, alpha=1, s=1, color="blue")
            axs[0].plot(data.squeeze(), mu, alpha=1.0, color="red")
            axs[0].fill_between(
                data, mu + total_std, mu - total_std, color="red", alpha=0.25
            )
            axs[0].fill_between(
                data, mu + 2 * total_std, mu - 2 * total_std, color="red", alpha=0.10
            )
            axs[0].fill_between(
                data, mu + 3 * total_std, mu - 3 * total_std, color="red", alpha=0.05
            )

            [axs[1].plot(data, pred, alpha=0.1, color="red") for pred in pred]
            axs[1].scatter(self.data, self.target, alpha=1, s=1, color="blue")

            axs[2].scatter(self.data, self.target, alpha=1, s=1, color="blue")
            axs[2].plot(data, mu, color="red")
            axs[2].fill_between(
                data,
                mu - aleatoric,
                mu + aleatoric,
                color="red",
                alpha=0.25,
                label="Aleatoric",
            )
            axs[2].legend()

            axs[3].scatter(self.data, self.target, alpha=1, s=1, color="blue")
            axs[3].plot(data, mu, color="red")
            axs[3].fill_between(
                data,
                mu - epistemic,
                mu + epistemic,
                color="red",
                alpha=0.25,
                label="Epistemic",
            )
            axs[3].legend()

            plt.ylim(2 * self.target.min(), 2 * self.target.max())
            plt.xlim(x_min, x_max)
            plt.show()


class RegressionNNHetero(ProbModel):
    def __init__(self, x, y, batch_size=1):

        self.data = x
        self.target = y

        dataloader = DataLoader(
            TensorDataset(self.data, self.target),
            shuffle=True,
            batch_size=self.data.shape[0],
            drop_last=False,
        )
        # dataloader = DataLoader(TensorDataset(x, y), shuffle=True, batch_size=batch_size, drop_last=False)

        ProbModel.__init__(self, dataloader)

        num_hidden = 50
        self.model = Sequential(
            Linear(1, num_hidden),
            ReLU(),
            Linear(num_hidden, num_hidden),
            ReLU(),
            Linear(num_hidden, num_hidden),
            ReLU(),
            # Linear(num_hidden, num_hidden),
            # ReLU(),
            Linear(num_hidden, 2),
        )

    def reset_parameters(self):
        for module in self.model.modules():
            if isinstance(module, Linear):
                module.reset_parameters()

    def sample(self):
        self.reset_parameters()

    def forward(self, x):
        pred = self.model(x)
        mu, log_std = torch.chunk(pred, chunks=2, dim=-1)
        return mu, log_std

    def log_prob(self, data, target):

        # if data is None and target is None:
        # 	data, target = next(self.dataloader.__iter__())

        mu, log_std = self.forward(data)
        mse = F.mse_loss(mu, target)

        log_prob = Normal(mu, F.softplus(log_std)).log_prob(target).mean() * len(
            self.dataloader.dataset
        )

        return {"log_prob": log_prob, "MSE": mse.detach_()}

    def pretrain(self):

        num_epochs = 100
        optim = torch.optim.Adam(self.parameters(), lr=0.001)

        progress = tqdm(range(num_epochs))
        for epoch in progress:
            for batch_i, (data, target) in enumerate(self.dataloader):
                optim.zero_grad()
                mu, log_std = self.forward(data)
                loss = -Normal(mu, F.softplus(log_std)).log_prob(target).mean()
                mse_loss = F.mse_loss(mu, target)
                loss.backward()
                optim.step()

                desc = f"Pretraining: MSE:{mse_loss:.3f}"
                progress.set_description(desc)

    @torch.no_grad()
    def predict(self, chains, plot=False):

        x_min = 2 * self.data.min()
        x_max = 2 * self.data.max()
        data = torch.linspace(x_min, x_max, 100).reshape(-1, 1)

        def parallel_predict(parallel_chain):
            parallel_pred_mu = []
            parallel_pred_log_std = []
            for model_state_dict in parallel_chain.samples[::50]:
                self.load_state_dict(model_state_dict)
                pred_mu_i, pred_log_std_i = self.forward(data)
                parallel_pred_mu.append(pred_mu_i)
                parallel_pred_log_std.append(pred_log_std_i)

            try:
                parallel_pred_mu = torch.stack(
                    parallel_pred_mu
                )  # list [ pred_0, pred_1, ... pred_N] -> Tensor([pred_0, pred_1, ... pred_N])
                parallel_pred_log_std = torch.stack(
                    parallel_pred_log_std
                )  # list [ pred_0, pred_1, ... pred_N] -> Tensor([pred_0, pred_1, ... pred_N])
                return parallel_pred_mu, parallel_pred_log_std
            except:
                pass

        parallel_pred_mu, parallel_pred_log_std = zip(
            *Parallel(n_jobs=len(chains))(
                delayed(parallel_predict)(chain) for chain in chains
            )
        )

        pred_mu = [
            parallel_pred_i
            for parallel_pred_i in parallel_pred_mu
            if parallel_pred_i is not None
        ]  # flatten [ [pred_chain_0], [pred_chain_1] ... [pred_chain_N] ]
        pred_log_std = [
            parallel_pred_i
            for parallel_pred_i in parallel_pred_log_std
            if parallel_pred_i is not None
        ]  # flatten [ [pred_chain_0], [pred_chain_1] ... [pred_chain_N] ]

        pred_mu = torch.cat(
            pred_mu
        ).squeeze()  # cat list of tensors to single prediciton tensor with samples in first dim
        pred_log_std = torch.cat(
            pred_log_std
        ).squeeze()  # cat list of tensors to single prediciton tensor with samples in first dim

        mu = pred_mu.squeeze()
        std = F.softplus(pred_log_std).squeeze()

        epistemic = mu.std(dim=0)
        aleatoric = (std**2).mean(dim=0) ** 0.5
        total_std = (epistemic**2 + aleatoric**2) ** 0.5

        mu = mu.mean(dim=0)
        std = std.mean(dim=0)

        data.squeeze_()

        if plot:

            fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
            axs = axs.flatten()

            axs[0].scatter(self.data, self.target, alpha=1, s=1, color="blue")
            axs[0].plot(data.squeeze(), mu, alpha=1.0, color="red")
            axs[0].fill_between(
                data, mu + total_std, mu - total_std, color="red", alpha=0.25
            )
            axs[0].fill_between(
                data, mu + 2 * total_std, mu - 2 * total_std, color="red", alpha=0.10
            )
            axs[0].fill_between(
                data, mu + 3 * total_std, mu - 3 * total_std, color="red", alpha=0.05
            )

            [axs[1].plot(data, pred, alpha=0.1, color="red") for pred in pred_mu]
            axs[1].scatter(self.data, self.target, alpha=1, s=1, color="blue")

            axs[2].scatter(self.data, self.target, alpha=1, s=1, color="blue")
            axs[2].plot(data, mu, color="red")
            axs[2].fill_between(
                data,
                mu - aleatoric,
                mu + aleatoric,
                color="red",
                alpha=0.25,
                label="Aleatoric",
            )
            axs[2].legend()

            axs[3].scatter(self.data, self.target, alpha=1, s=1, color="blue")
            axs[3].plot(data, mu, color="red")
            axs[3].fill_between(
                data,
                mu - epistemic,
                mu + epistemic,
                color="red",
                alpha=0.25,
                label="Epistemic",
            )
            axs[3].legend()

            plt.ylim(2 * self.target.min(), 2 * self.target.max())
            plt.xlim(x_min, x_max)
            plt.show()

        return data, mu, std


if __name__ == "__main__":

    pass
