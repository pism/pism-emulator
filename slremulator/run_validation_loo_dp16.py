#!/usr/bin/env python

# Copyright (C) 2019-20 Andy Aschwanden, Rachel Chen
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
from functools import partial
import GPy as gp
from io import StringIO
from math import sqrt
import multiprocessing
from multiprocessing import Pool
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
import sys

from pismemulator.utils import prepare_data
from pismemulator.emulate import emulate_gp, emulate_sklearn, gp_loo_mp

multiprocessing.set_start_method("forkserver", force=True)

default_output_directory = "loo_results_dp16"


if __name__ == "__main__":

    __spec__ = None

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Validate Regression Methods."
    parser.add_argument(
        "--n_procs", dest="n_procs", type=int, help="""number of cores/processors. default=4.""", default=1
    )
    parser.add_argument(
        "-m",
        "--method",
        dest="method",
        help="""Method.""",
        choices=["exp", "expquad", "mat32", "mat52"],
        default="exp",
    )
    parser.add_argument(
        "--normalizer", dest="normalizer", action="store_true", help=f"Normalize regressor.", default=False,
    )
    parser.add_argument(
        "-s",
        "--simulation",
        dest="simulation",
        help="""Which simulation.""",
        choices=["LIG", "PLIO", "RCP45_pres", "RCP26_2100", "RCP45_2100", "RCP85_2100"],
        default="RCP85_2100",
    )
    parser.add_argument(
        "--output_dir",
        dest="odir",
        help=f"Output directory. Default = {default_output_directory}.",
        default=default_output_directory,
    )
    parser.add_argument(
        "--step_bic",
        dest="step",
        action="store_true",
        help=f"Use stepBIC for determining the mean function.",
        default=False,
    )
    parser.add_argument(
        "--test", dest="test", action="store_true", help="""Test script by running only 8 LOO""", default=False
    )

    options = parser.parse_args()
    method = options.method
    normalizer = options.normalizer
    n_procs = options.n_procs
    odir = options.odir
    simulation = options.simulation
    stepwise = options.step
    test = options.test

    if not os.path.exists(odir):
        os.makedirs(odir)

    print("-----------------------------------------------------------")
    print("Running leave-one-out validation")
    print("-----------------------------------------------------------")

    dp16_df = pd.read_csv("../data/dp16/dp16.csv")
    samples = dp16_df[["OCFAC", "CREVLIQ", "VCLIF", "BIAS"]]
    response = dp16_df[[simulation]]

    ids_loo = samples.index
    if test:
        ids_loo = samples.index[0:8]

    gp_methods = {
        "exp": gp.kern.Exponential,
        "expquad": gp.kern.ExpQuad,
        "mat32": gp.kern.Matern32,
        "mat52": gp.kern.Matern52,
    }

    if n_procs > 1:
        with Pool(n_procs) as pool:
            if stepwise:
                m = method + "-step"
            else:
                m = method
            results = pool.map(
                partial(
                    gp_loo_mp,
                    samples=samples,
                    response=response,
                    kernel=gp_methods[method],
                    stepwise=stepwise,
                    regressor_options={"normalizer": normalizer},
                ),
                ids_loo,
            )
    else:
        if stepwise:
            m = method + "-step"
        else:
            m = method
        results = []
        for idx in ids_loo:
            print("Running sample {}".format(idx))
            result = gp_loo_mp(
                idx,
                samples=samples,
                response=response,
                kernel=gp_methods[method],
                stepwise=stepwise,
                regressor_options={"normalizer": normalizer},
            )
            results.append(result)

    results = [i for i in results if i is not None]
    results = np.round(np.squeeze(results), decimals=8)
    n = results.shape[0]
    df = pd.DataFrame(
        data=np.hstack((results, np.repeat(m, n).reshape(-1, 1), np.repeat(simulation, n).reshape(-1, 1),)),
        columns=["Y_mean", "Y_var", "distance", "loo_id", "method", "simulation"],
    )
    if not test:
        df.to_csv(f"{odir}/loo_{m}_{simulation}.csv", index_label="id")
    else:
        df.to_csv(f"{odir}/test_loo_{m}_{simulation}.csv", index_label="id")
