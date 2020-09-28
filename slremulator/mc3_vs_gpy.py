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
import pymc3 as pm
from math import sqrt
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
    # PyMC3
    print("\n\nTesting PyMC3")

    with pm.Model() as model:
        ls = pm.Normal("ls", 100, 75, shape=n)
        nu = pm.Normal("nu", 50, 25)
        mc3_cov = pm.gp.cov.Exponential
        mc3_kern = nu * mc3_cov(input_dim=n, ls=ls)

        gp = pm.gp.Marginal(cov_func=mc3_kern)
        y_ = gp.marginal_likelihood("y", X=X, y=Y.squeeze(), noise=0)

        mp = pm.find_MAP(return_raw=True)
        f_pred = gp.conditional("f_pred", X_new)
        mu, var = gp.predict(X_new.squeeze(), point=mp, diag=True)

        if mp[1]["success"]:
            df = pd.DataFrame(
                data=np.hstack(
                    [
                        mu.reshape(-1, 1),
                        var.reshape(-1, 1),
                    ]
                ),
                columns=["Y_mean", "Y_var"],
            )
            outfile = f"test_mc3.csv"
            df.to_csv(outfile, index_label="id")

    print(mu)

    bins = np.arange(0, 60, 1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.distplot(mu, ax=ax, label="pymc3")
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
    fig.savefig("gpy_vs_pymc3.pdf")
