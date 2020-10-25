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

    # GPy
    print("\n\nTesting GPy")

    gp_cov = gpy.kern.Exponential
    gp_kern = gp_cov(input_dim=n, ARD=True)
    m = gpy.models.GPRegression(X, Y, gp_kern, normalizer=True)
    f = m.optimize(messages=True, max_iters=4000)

    p = m.predict(X_new)

    if f.status == "Converged":
        df = pd.DataFrame(
            data=np.hstack(
                [
                    p[0].reshape(-1, 1),
                    p[1].reshape(-1, 1),
                ]
            ),
            columns=["Y_mean", "Y_var"],
        )
        outfile = f"test_gpy.csv"
        df.to_csv(outfile, index_label="id")

    print(p[0])

    print("\n\nTesting GPyTorch")
    training_iter = 300
    X_train = torch.tensor(X).to(torch.float)
    y_train = torch.tensor(np.squeeze(Y)).to(torch.float)
    X_test = torch.tensor(X_new).to(torch.float)
    y_test = torch.tensor(np.squeeze(Y_new)).to(torch.float)

    # We will use the simplest form of GP model, exact inference
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=n))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X_train, y_train, likelihood)

    # Plot test set predictions prior to training

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
        y_pred = likelihood(model(X_test))

    idx = np.argsort(y_test.numpy())

    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(12, 12))
        lower, upper = y_pred.confidence_region()
        ax.plot(y_test.numpy()[idx], y_pred.mean.numpy()[idx], "k*")
        ax.fill_between(y_test.numpy()[idx], lower.numpy()[idx], upper.numpy()[idx], alpha=0.5)

    mu = y_pred.mean.numpy()

    print("\n\nTesting GPyTorch normalized ")

    X_n = (X - X.mean(axis=0)) / X.std()
    Y_n = (Y - Y.mean(axis=0)) / Y.std()
    X_n_new = (X_new - X_new.mean(axis=0)) / X_new.std()
    Y_n_new = (Y_new - Y_new.mean(axis=0)) / Y_new.std()

    X_train = torch.tensor(X_n).to(torch.float)
    y_train = torch.tensor(np.squeeze(Y_n)).to(torch.float)
    X_test = torch.tensor(X_n_new).to(torch.float)
    y_test = torch.tensor(np.squeeze(Y_n_new)).to(torch.float)

    # We will use the simplest form of GP model, exact inference
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=n))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X_train, y_train, likelihood)

    # Plot test set predictions prior to training

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
        y_pred = likelihood(model(X_test))

    idx = np.argsort(y_test.numpy())

    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(12, 12))
        lower, upper = y_pred.confidence_region()
        ax.plot(y_test.numpy()[idx], y_pred.mean.numpy()[idx], "k*")
        ax.fill_between(y_test.numpy()[idx], lower.numpy()[idx], upper.numpy()[idx], alpha=0.5)

    mu_norm = y_pred.mean.numpy() * Y_new.std(axis=0) + Y_new.mean(axis=0)

    bins = np.arange(0, 60, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.distplot(mu, ax=ax, label="GPyTorch")
    sns.distplot(mu_norm, ax=ax, label="GPyTorch-norm")
    sns.distplot(p[0], ax=ax, label="GPy")

    sns.distplot(
        Y_true,
        bins=bins,
        hist=False,
        hist_kws={"alpha": 0.3},
        norm_hist=True,
        kde=True,
        kde_kws={"shade": False, "alpha": 1.0, "linewidth": 0.8},
        ax=ax,
        color="k",
        label='"True"',
    )
    plt.legend()
    fig.savefig("gpytorch_vs_gpy.pdf")
