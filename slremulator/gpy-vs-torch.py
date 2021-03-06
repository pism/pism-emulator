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
from pismemulator.emulate import generate_kernel as generate_gpy_kernel


from sklearn.metrics import mean_squared_error, r2_score


def r2_f(df):

    return r2_score(df["Y_pred_mean"], df["Y_true"])


def s_res(df):
    """
    Standardized residual
    """

    df["sres"] = (df["Y_pred_mean"] - df["Y_true"]) / df["Y_pred_var"]
    return df["sres"]


def rmsd_f(df):

    return rmsd(df["Y_pred_mean"], df["Y_true"])


def generate_torch_kernel(varlist, kernel, varnames):
    """
    Generate a kernel based on a list of model terms

    Same as GPy but for GPyTorch

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

    # The training data set is the 500 member ensemble from AS2019
    X = samples.values
    Y = response.values
    m_train, n_train = X.shape

    # The test data set is the 5200 member ensemble
    varnames = samples.columns
    X_new = X_true
    Y_new = Y_true
    m_test, n_test = X_true.shape

    # Perfom model selection with BIC
    steplist = stepwise_bic(X, Y, varnames=varnames)

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

    X_train = torch.tensor(X).to(torch.float)
    y_train = torch.tensor(np.squeeze(Y)).to(torch.float)
    X_test = torch.tensor(X_new).to(torch.float)
    y_test = torch.tensor(np.squeeze(Y_new)).to(torch.float)

    # GPy
    print("\n\nTesting GPy")

    gpy_kern = gpy.kern.RBF

    # Without BIC model selection
    m = gpy.models.GPRegression(X, Y, gpy_kern(input_dim=n_train, ARD=True), normalizer=True)
    f = m.optimize(messages=True, max_iters=4000)
    p_gpy = m.predict(X_new)
    gpy_mean, gpy_var = p_gpy

    # With BIC model selection
    gpy_kern_bic = generate_gpy_kernel(steplist, kernel=gpy_kern, varnames=varnames)
    m = gpy.models.GPRegression(X, Y, gpy_kern_bic, normalizer=True)
    f = m.optimize(messages=True, max_iters=4000)
    p_gpy_bic = m.predict(X_new)
    gpy_mean_bic, gpy_var_bic = p_gpy_bic

    print("\n\nTesting GPyTorch")

    torch_kern = gpytorch.kernels.RBFKernel
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # Without BIC model selection
    model = ExactGPModel(X_train, y_train, likelihood, torch_kern(ard_num_dims=n_train))

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
            print(i, loss.item(), model.covar_module.base_kernel.lengthscale, model.likelihood.noise.item())
        optimizer.step()

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    with torch.no_grad():  # , gpytorch.settings.fast_pred_var():
        y_pred = likelihood(model(X_test))
        torch_mean = y_pred.mean.numpy()
        torch_var = y_pred.variance.numpy()

    # With BIC model selection
    torch_kern_bic = generate_torch_kernel(steplist, kernel=torch_kern, varnames=varnames)

    model = ExactGPModel(X_train, y_train, likelihood, torch_kern_bic)

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
            print(i, loss.item(), model.covar_module.base_kernel.lengthscale, model.likelihood.noise.item())
        optimizer.step()

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    with torch.no_grad():  # , gpytorch.settings.fast_pred_var():
        y_pred = likelihood(model(X_test))
        torch_mean_bic = y_pred.mean.numpy()
        torch_var_bic = y_pred.variance.numpy()

    bins = np.arange(np.floor(Y_true.min()), np.ceil(Y_true.max()), 1.0)

    dfs = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name, mean, var in zip(
        ["GPy", "GPy-BIC", "Torch", "Torch-BIC"],
        [gpy_mean, gpy_mean_bic, torch_mean, torch_mean_bic],
        [gpy_var, gpy_var_bic, torch_var, torch_var_bic],
    ):
        dfs.append(
            pd.DataFrame(
                data=np.hstack(
                    [
                        Y_true.reshape(-1, 1),
                        mean.reshape(-1, 1),
                        var.reshape(-1, 1),
                        np.repeat(name, m_test).reshape(-1, 1),
                    ]
                ),
                columns=["Y_true", "Y_pred_mean", "Y_pred_var", "Model"],
            )
        )
        sns.distplot(mean, bins=bins, hist=False, kde=True, ax=ax, label=name)
    sns.distplot(Y_true, bins=bins, color="k", ax=ax, label="True")
    plt.legend()
    fig.savefig("gpy_vs_torch_rbf.pdf")
    df = pd.concat(dfs)
    df = df.astype({"Y_true": float, "Y_pred_mean": float, "Y_pred_var": float, "Model": str})

    print("RMSE")
    print(df.groupby(by=["Model"]).apply(rmsd_f))
