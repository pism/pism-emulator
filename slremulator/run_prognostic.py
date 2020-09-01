#!/usr/bin/env python

# Copyright (C) 2019 Rachel Chen, Andy Aschwanden
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
import pylab as plt
import seaborn as sns
import sys

from pismemulator.utils import draw_samples
from pismemulator.utils import get_as19_distributions
from pismemulator.utils import prepare_data
from pismemulator.utils import set_size

from pismemulator.emulate import emulate_gp, emulate_sklearn


if __name__ == "__main__":

    __spec__ = None

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Run Prognostic Simulations."
    parser.add_argument(
        "-s",
        "--samples_file",
        dest="samples_file",
        help="File that has all combinations for ensemble study",
        default="../data/samples/samples.csv",
    )

    options = parser.parse_args()
    samples_file = options.samples_file

    options = parser.parse_args()
    rcp = 45

    distributions = get_as19_distributions()
    X_true = draw_samples(distributions, n_samples=2000, method="saltelli")

    gp_methods = {
        "exp": gp.kern.Exponential,
        "expquad": gp.kern.ExpQuad,
        "mat32": gp.kern.Matern32,
        "mat52": gp.kern.Matern52,
    }

    sklearn_methods = ("lasso", "lasso-lars", "ridge")

    emulators_gp = {}
    emulators_sklearn = {}

    start_year = 2099
    end_year = 2100
    for year in range(start_year, end_year + 1):

        response_file = "../data/response/dgmsl_rcp_{}_year_{}.csv".format(rcp, year)
        samples, response = prepare_data(samples_file, response_file)
        Y = response.values

        for method in sklearn_methods:
            p = emulate_sklearn(samples, response, X_true, method=method)
            Y_p = np.squeeze(p.reshape(-1, 1))
            my = "_".join([method, str(year)])
            emulators_sklearn[my] = {
                "Y_p": Y_p,
                "Year": np.zeros_like(Y_p) + year,
                "Method": [method for x in range(len(Y_p))],
            }

        for method in gp_methods.keys():
            stepwise = False
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
            )
            if status == "Converged":
                Y_p = np.squeeze(p[0])
                my = "_".join([m, str(year)])
                emulators_gp[my] = {
                    "Y_p": Y_p,
                    "Year": np.zeros_like(Y_p) + year,
                    "Method": [m for x in range(len(Y_p))],
                }

        emulators = {**emulators_sklearn, **emulators_gp}
        dfs = [pd.DataFrame.from_dict(emulators[x]) for x in emulators]
        df = pd.concat(dfs)
        df = df.reset_index()
        df.to_csv("prognostic_dgmsl_{}.csv".format(year), index_label="id")
