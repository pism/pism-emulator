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
from functools import partial
import GPy as gp
from math import sqrt
import multiprocessing
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import sys

from pismemulator.utils import draw_samples, distributions_as19, prepare_data
from pismemulator.gpemulator import emulate_gp, gp_response_mp

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

    parser.add_argument("--n_samples", dest="n_samples", type=int, help="""Number of new samples.""", default=100)
    parser.add_argument(
        "--method",
        dest="sampling_method",
        choices=["lhs", "saltelli"],
        help="""Type of method to draw samples.""",
        default="saltelli",
    )
    parser.add_argument(
        "-a", "--start_year", dest="start_year", type=int, help="""Start year. default=2010.""", default=2010
    )
    parser.add_argument(
        "-e", "--end_year", dest="end_year", type=int, help="""End year. default=2300.""", default=2300
    )
    parser.add_argument("--test", dest="test", action="store_true", help="""Only run test.""", default=False)
    parser.add_argument(
        "--n_procs", dest="n_procs", type=int, help="""number of cores/processors. default=4.""", default=4
    )
    options = parser.parse_args()
    n_samples = options.n_samples
    n_lhs_samples = options.n_lhs_samples
    n_samples_validation = options.n_samples_validation
    n_procs = options.n_procs
    odir = options.odir
    sampling_method = options.sampling_method
    rcp = 45
    start_year = options.start_year
    end_year = options.end_year

    if not os.path.exists(odir):
        os.makedirs(odir)

    # Load the AS19 data. Samples are the M x D parameter matrix
    # Response is the 1 X D sea level rise response (cm SLE)
    samples_file = "../data/samples/lhs_samples_{}.csv".format(n_lhs_samples)

    # Dictionary of GP kernels
    gp_methods = {
        "expquad": gp.kern.ExpQuad,
        "mat52": gp.kern.Matern52,
    }

    d_as19 = distributions_as19()

    X_new = draw_samples(d_as19, n_samples, method=sampling_method)

    if options.test:
        response_files = ["../data/response/dgmsl_rcp_{}_year_{}.csv".format(rcp, year) for year in range(2010, 2012)]
    else:
        response_files = [
            "../data/response/dgmsl_rcp_{}_year_{}.csv".format(rcp, year) for year in range(start_year, end_year)
        ]

    stepwise = False
    for method in gp_methods.keys():

        if n_procs > 1:
            multiprocessing.set_start_method("forkserver", force=True)
            with Pool(n_procs) as pool:
                results = pool.map(
                    partial(gp_response_mp, samples_file, X_new, kernel=gp_methods[method], stepwise=stepwise),
                    response_files,
                )

            # p, status = emulate_gp(
            #     samples,
            #     response,
            #     X_true,
            #     kernel=gp_methods[method],
            #     stepwise=stepwise,
            #     optimizer_options={"max_iters": 4000},
            #     regressor_options={"normalizer": True},
            # )
            # if status == "Converged":
            #     df = pd.DataFrame(
            #         data=np.hstack(
            #             [
            #                 p[0],
            #                 p[1],
            #                 np.repeat(m, n).reshape(-1, 1),
            #                 np.repeat(n_lhs_samples, n).reshape(-1, 1),
            #                 np.repeat(rcp, n).reshape(-1, 1),
            #                 np.repeat(year, n).reshape(-1, 1),
            #             ]
            #         ),
            #         columns=["Y_mean", "Y_var", "method", "n_lhs", "rcp", "year"],
            #     )
            #     outfile = f"{odir}/dgmsl_rcp_{rcp}_{year}_{m}_lhs_{n_lhs_samples}.csv"
            #     df.to_csv(outfile, index_label="id")
