#!/usr/bin/env python

# Copyright (C) 2019-2020 Rachel Chen, Andy Aschwanden
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

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import GPy as gpy
import torch
import gpytorch
import numpy as np
import os
import pandas as pd
import sys
import pylab as plt
import seaborn as sns

from pismemulator.utils import prepare_data
from pismemulator.utils import kl_divergence
from pismemulator.utils import rmsd


default_output_directory = "emulator_results"

if __name__ == "__main__":

    __spec__ = None

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Validate Regression Methods."
    parser.add_argument(
        "-n",
        dest="n_samples_validation",
        choices=[20, 170, 240, 320, 400],
        type=int,
        help="Number of validation samples.",
        default=400,
    )
    parser.add_argument(
        "--n_lhs_samples",
        dest="n_lhs_samples",
        choices=[100, 200, 300, 400, 500],
        type=int,
        help="Size of LHS ensemble.",
        default=500,
    )
    parser.add_argument(
        "--output_dir",
        dest="odir",
        help=f"Output directory. Default = {default_output_directory}.",
        default=default_output_directory,
    )

    options = parser.parse_args()
    n_lhs_samples = options.n_lhs_samples
    n_samples_validation = options.n_samples_validation
    odir = options.odir
    rcp = 45
    year = 2100

    if not os.path.exists(odir):
        os.makedirs(odir)

    # Load the AS19 data. Samples are the M x D parameter matrix
    # Response is the 1 X D sea level rise response (cm SLE)
    samples_file = "../data/samples/lhs_samples_{}.csv".format(n_lhs_samples)
    response_file = "../data/validation/dgmsl_rcp_{}_year_{}_lhs_{}.csv".format(rcp, year, n_lhs_samples)
    samples, response = prepare_data(samples_file, response_file)
    Y = response.values

    # Load the "True" data.
    s_true, r_true = prepare_data(
        "../data/samples/saltelli_samples_{}.csv".format(n_samples_validation),
        "../data/validation/dgmsl_rcp_45_year_2100_s{}.csv".format(n_samples_validation),
    )
    X_true = s_true.values
    Y_true = r_true.values

    X = samples.values
    Y = response.values
    n = X.shape[1]

    varnames = samples.columns
    X_new = X_true
    Y_new = Y_true

    X_n_std = (X - X.mean(axis=0)) / X.std()
    Y_n_std = (Y - Y.mean(axis=0)) / Y.std()
    X_n_std_new = (X_new - X_new.mean(axis=0)) / X_new.std()
    Y_n_std_new = (Y_new - Y_new.mean(axis=0)) / Y_new.std()

    X_n = X - X.mean(axis=0)
    Y_n = Y - Y.mean(axis=0)
    X_n_new = X_new - X_new.mean(axis=0)
    Y_n_new = Y_new - Y_new.mean(axis=0)

    covs_name = ["exp", "mat52", "mat32"]
    covs_gpy = [gpy.kern.Exponential, gpy.kern.Matern52, gpy.kern.Matern32]
    covs_torch = [
        gpytorch.kernels.RBFKernel(ard_num_dims=n),
        gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=n),
        gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=n),
    ]
    # We will use the simplest form of GP model, exact inference
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, cov):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(cov)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    for cov_name, cov_gpy, cov_torch in zip(covs_name, covs_gpy, covs_torch):

        # GPy
        print("\n\nTesting GPy")

        gp_kern = cov_gpy(input_dim=n, ARD=True)

        m = gpy.models.GPRegression(X, Y, gp_kern, normalizer=True)
        f = m.optimize(messages=True, max_iters=4000)
        p_gpy = m.predict(X_new)

        m = gpy.models.GPRegression(X_n, Y_n, gp_kern, normalizer=False)
        f = m.optimize(messages=True, max_iters=4000)
        p_gpy_n = m.predict(X_n_new)

        m = gpy.models.GPRegression(X_n_std, Y_n_std, gp_kern, normalizer=False)
        f = m.optimize(messages=True, max_iters=4000)
        p_gpy_n_std = m.predict(X_n_std_new)

        print("\n\nTesting GPyTorch")

        X_train = torch.tensor(X).to(torch.float)
        y_train = torch.tensor(np.squeeze(Y)).to(torch.float)
        X_test = torch.tensor(X_new).to(torch.float)
        y_test = torch.tensor(np.squeeze(Y_new)).to(torch.float)

        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(X_train, y_train, likelihood, cov_torch)

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(2000):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(X_train)
            # Calc loss and backprop gradients
            loss = -mll(output, y_train)
            loss.backward()
            if i % 20 == 0:
                print(i, loss.item(), model.covar_module.base_kernel.lengthscale[0], model.likelihood.noise.item())
            optimizer.step()

        # Plot posterior distributions (with test data on x-axis)

        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()
        with torch.no_grad():  # , gpytorch.settings.fast_pred_var():
            y_pred = likelihood(model(X_test))

        idx = np.argsort(y_test.numpy())

        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1, figsize=(12, 12))
            lower, upper = y_pred.confidence_region()
            ax.plot(y_test.numpy()[idx], y_pred.mean.numpy()[idx], "k*")
            ax.fill_between(y_test.numpy()[idx], lower.numpy()[idx], upper.numpy()[idx], alpha=0.5)

        mu = y_pred.mean.numpy()

        print("\n\nTesting GPyTorch normalized std")

        X_train = torch.tensor(X_n).to(torch.float)
        y_train = torch.tensor(np.squeeze(Y_n)).to(torch.float)
        X_test = torch.tensor(X_n_new).to(torch.float)
        y_test = torch.tensor(np.squeeze(Y_n_new)).to(torch.float)

        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(X_train, y_train, likelihood, cov_torch)

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(1000):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(X_train)
            # Calc loss and backprop gradients
            loss = -mll(output, y_train)
            loss.backward()
            if i % 20 == 0:
                print(i, loss.item(), model.covar_module.base_kernel.lengthscale[0], model.likelihood.noise.item())
            optimizer.step()

        # Plot posterior distributions (with test data on x-axis)

        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()
        with torch.no_grad():  # , gpytorch.settings.fast_pred_var():
            y_pred_norm = likelihood(model(X_test))

        mu_norm = y_pred_norm.mean.numpy()

        print("\n\nTesting GPyTorch normalized std")

        X_train = torch.tensor(X_n_std).to(torch.float)
        y_train = torch.tensor(np.squeeze(Y_n_std)).to(torch.float)
        X_test = torch.tensor(X_n_std_new).to(torch.float)
        y_test = torch.tensor(np.squeeze(Y_n_std_new)).to(torch.float)

        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(X_train, y_train, likelihood, cov_torch)

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(1000):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(X_train)
            # Calc loss and backprop gradients
            loss = -mll(output, y_train)
            loss.backward()
            if i % 20 == 0:
                print(i, loss.item(), model.covar_module.base_kernel.lengthscale[0], model.likelihood.noise.item())
            optimizer.step()

        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()
        with torch.no_grad():  # , gpytorch.settings.fast_pred_var():
            y_pred_norm_std = likelihood(model(X_test))

        mu_norm_std = y_pred_norm_std.mean.numpy()

        bins = np.arange(np.floor(Y_true.min()), np.ceil(Y_true.max()), 1.0)

        p_mean_gpy = p_gpy[0]
        p_mean_gpy_norm = p_gpy_n[0] + Y_new.mean(axis=0)
        p_mean_gpy_norm_std = p_gpy_n_std[0] * Y_new.std(axis=0) + Y_new.mean(axis=0)

        p_mean_torch = mu
        p_mean_torch_norm = mu_norm + Y_new.mean(axis=0)
        p_mean_torch_norm_std = mu_norm_std * Y_new.std(axis=0) + Y_new.mean(axis=0)

        for pred, pred_t in zip(
            [p_mean_gpy, p_mean_gpy_norm, p_mean_gpy_norm_std, p_mean_torch, p_mean_torch_norm, p_mean_torch_norm_std],
            ["GPy", "GPy-norm", "GPy-std", "Torch", "Torch-norm", "Torch-std"],
        ):
            print(f"{pred_t}: {rmsd(Y_true, pred):.4f}")

        P = np.histogram(Y_true, bins=bins, density=True)[0]
        Q_gpy = np.histogram(p_mean_gpy, bins=bins, density=True)[0]
        Q_gpy_norm = np.histogram(p_mean_gpy_norm, bins=bins, density=True)[0]
        Q_gpy_norm_std = np.histogram(p_mean_gpy_norm_std, bins=bins, density=True)[0]
        Q_torch = np.histogram(p_mean_torch, bins=bins, density=True)[0]
        Q_torch_norm = np.histogram(p_mean_torch_norm, bins=bins, density=True)[0]
        Q_torch_norm_std = np.histogram(p_mean_torch_norm_std, bins=bins, density=True)[0]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        sns.distplot(
            Y_true,
            bins=bins,
            ax=ax,
            norm_hist=True,
            kde_kws={"linewidth": 0.8},
            color="0",
            label='"True" $D_{KL}$ RMSD',
        )
        sns.distplot(
            p_mean_torch,
            bins=bins,
            ax=ax,
            norm_hist=True,
            kde_kws={"linewidth": 0.6},
            hist_kws={"histtype": "step"},
            color="#08519c",
            label=f"Torch {kl_divergence(P, Q_torch):.3f} {rmsd(Y_true, p_mean_torch):.3f}",
        )
        sns.distplot(
            p_mean_torch_norm,
            bins=bins,
            ax=ax,
            norm_hist=True,
            kde_kws={"linewidth": 0.6},
            hist_kws={"histtype": "step"},
            color="#3182bd",
            label=f"Torch-norm {kl_divergence(P, Q_torch_norm):.3f} {rmsd(Y_true, p_mean_torch_norm):.3f}",
        )
        sns.distplot(
            p_mean_torch_norm_std,
            bins=bins,
            ax=ax,
            norm_hist=True,
            kde_kws={"linewidth": 0.6},
            hist_kws={"histtype": "step"},
            color="#6baed6",
            label=f"Torch-norm-std {kl_divergence(P, Q_torch_norm_std):.3f} {rmsd(Y_true, p_mean_torch_norm_std):.3f}",
        )
        sns.distplot(
            p_mean_gpy,
            bins=bins,
            ax=ax,
            norm_hist=True,
            kde_kws={"linewidth": 0.6},
            hist_kws={"histtype": "step"},
            color="#54278f",
            label=f"GPy {kl_divergence(P, Q_gpy):.3f} {rmsd(Y_true, p_mean_gpy):.3f}",
        )
        sns.distplot(
            p_mean_gpy_norm,
            bins=bins,
            color="#756bb1",
            kde_kws={"linewidth": 0.6},
            hist_kws={"histtype": "step"},
            ax=ax,
            norm_hist=True,
            label=f"GPy-norm {kl_divergence(P, Q_gpy_norm):.3f} {rmsd(Y_true, p_mean_gpy_norm):.3f}",
        )
        sns.distplot(
            p_mean_gpy_norm_std,
            bins=bins,
            color="#9e9ac8",
            kde_kws={"linewidth": 0.6},
            hist_kws={"histtype": "step"},
            ax=ax,
            norm_hist=True,
            label=f"GPy-norm-std {kl_divergence(P, Q_gpy_norm_std):.3f} {rmsd(Y_true, p_mean_gpy_norm_std):.3f}",
        )

        plt.legend()
        fig.savefig(f"gpytorch_vs_gpy_{cov_name}.pdf")
