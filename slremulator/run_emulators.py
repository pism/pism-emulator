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
import GPy as gp
from math import sqrt
import numpy as np
import os
import pandas as pd
import sys

from pismemulator.utils import prepare_data
from pismemulator.emulate import emulate_gp, emulate_sklearn

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

    n = len(Y_true)

    # Dictionary of GP kernels
    gp_methods = {
        "exp": gp.kern.Exponential,
        "expquad": gp.kern.ExpQuad,
        "mat32": gp.kern.Matern32,
        "mat52": gp.kern.Matern52,
    }

    # List of regression methods
    sklearn_methods = ("lasso", "lasso-lars", "ridge")

    for method in sklearn_methods:
        p = emulate_sklearn(samples, response, X_true, method=method)
        df = pd.DataFrame(
            data=np.hstack(
                [p.reshape(-1, 1), np.repeat(method, n).reshape(-1, 1), np.repeat(n_lhs_samples, n).reshape(-1, 1)]
            ),
            columns=["Y_mean", "method", "n_lhs"],
        )
        outfile = f"{odir}/dgmsl_rcp_{rcp}_{year}_{method}_lhs_{n_lhs_samples}.csv"
        df.to_csv(outfile, index_label="id")

    for method in gp_methods.keys():
        for stepwise in (True, False):
            if stepwise:
                m = method + "-step"
            else:
                m = method
            p, status = emulate_gp(
                samples,
                response,
                X_true,
                kernel=gp_methods[method],
                stepwise=stepwise,
                optimizer_options={"max_iters": 4000},
                regressor_options={"normalizer": True},
            )
            if status == "Converged":
                df = pd.DataFrame(
                    data=np.hstack(
                        [
                            p[0],
                            p[1],
                            np.repeat(m, n).reshape(-1, 1),
                            np.repeat(n_lhs_samples, n).reshape(-1, 1),
                            np.repeat(rcp, n).reshape(-1, 1),
                            np.repeat(year, n).reshape(-1, 1),
                        ]
                    ),
                    columns=["Y_mean", "Y_var", "method", "n_lhs", "rcp", "year"],
                )
                outfile = f"{odir}/dgmsl_rcp_{rcp}_{year}_{m}_lhs_{n_lhs_samples}.csv"
                df.to_csv(outfile, index_label="id")
