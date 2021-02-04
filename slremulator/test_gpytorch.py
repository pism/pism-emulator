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
from pismemulator.utils import stepwise_bic


def generate_kernel(varlist, kernel, varnames):
    """
    Generate a kernel based on a list of model terms

    Currently only supports GPy but could be easily extended
    to gpflow.

    :param varlist: list of strings containing model terms.
    :param kernel: GPy.kern instance
    :param var_names: list of strings containing variable names

    :return kernel: GPy.kern instance

    Example:

    To generate a kernel that represents the model

    Y ~ X1 + X2 + X1*X2 with an exponential kernel

    use

    varlist = ["X1", "X2", "X1*X2"]
    kernel = GPy.kern.Exponential
    varnames = ["X1", "X2"]

    """

    params_dict = {k: v for v, k in enumerate(varnames)}

    first_order = [params_dict[x] for x in varlist if "*" not in x]
    interactions = [x.split("*") for x in varlist if "*" in x]
    nfirst = len(first_order)

    kern = kernel(ard_num_dims=nfirst, active_dims=first_order)

    mult = []
    active_intx = []
    for i in range(len(interactions)):
        mult.append([])
        for j in list(range(len(interactions[i]))):
            active_dims = [params_dict[interactions[i][j]]]
            mult[i].append(kernel(ard_num_dims=1, active_dims=active_dims))
        active_intx.append(np.prod(mult[i]))
    for k in active_intx:
        kern += k

    return kern


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

    print("\n\nTesting GPyTorch")

    X_train = torch.tensor(X).to(torch.float)
    y_train = torch.tensor(np.squeeze(Y)).to(torch.float)
    X_test = torch.tensor(X_new).to(torch.float)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    cov = gpytorch.kernels.RBFKernel
    steplist = stepwise_bic(X, Y, varnames=varnames)
    cov_func = generate_kernel(steplist, kernel=cov, varnames=varnames)

    model = ExactGPModel(X_train, y_train, likelihood, cov())

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
            print(i, loss.item(), model.likelihood.noise.item())
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
    var = y_pred.variance.numpy()

    model = ExactGPModel(X_train, y_train, likelihood, cov_func)

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
            print(i, loss.item(), model.likelihood.noise.item())
        optimizer.step()

    # Plot posterior distributions (with test data on x-axis)

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    with torch.no_grad():  # , gpytorch.settings.fast_pred_var():
        y_pred = likelihood(model(X_test))


    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
