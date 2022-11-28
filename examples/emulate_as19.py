#!/usr/bin/env python

# Copyright (C) 2019 Andy Aschwanden, Rachel Chen
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

# This scripts emulates the "true" ensemble using the 500 member ensembe
# of Aschwanden et al. (2019) as training data

import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import GPy as gp
import numpy as np
import pylab as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy import stats

from pismemulator.emulate import emulate_gp
from pismemulator.utils import (
    gelman_rubin,
    golden_ratio,
    kl_divergence,
    prepare_data,
    rmsd,
    set_size,
)

if __name__ == "__main__":

    __spec__ = None

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Validate Regression Methods."
    parser.add_argument(
        "-s",
        "--samples_file",
        dest="samples_file",
        help="File that has all combinations for ensemble study",
        default="samples/samples.csv",
    )
    parser.add_argument(
        "-n",
        dest="n_samples_validation",
        choices=[20, 170, 240, 320],
        type=int,
        help="Number of validation samples.",
        default=320,
    )
    parser.add_argument(
        "-b",
        "--bin_width",
        dest="bin_width",
        help="Width of histogram bins. Default=1cm",
        type=float,
        default=1.0,
    )

    options = parser.parse_args()
    bin_width = options.bin_width
    samples_file = options.samples_file
    n_samples_validation = options.n_samples_validation
    rcp = 45
    year = 2100
    response_file = "response/dgmsl_rcp_{}_year_{}.csv".format(rcp, year)

    print("-----------------------------------------------------------")
    print("Running validation of the Aschwanden et al. (2019) ensemble")
    print("-----------------------------------------------------------")

    # Load the AS19 data. Samples are the M x D parameter matrix
    # Response is the 1 X D sea level rise response (cm SLE)
    samples, response = prepare_data(samples_file, response_file)
    Y = response.values

    # Load the "True" data.
    X_true, Y_true = prepare_data(
        "samples/saltelli_samples_{}.csv".format(n_samples_validation),
        "validation/dgmsl_rcp_45_year_2100_s{}.csv".format(n_samples_validation),
        return_numpy=True,
    )

    # Histogram and PDF of "true" ensemble estimated using a Gaussian KDE
    p = Y_true
    bins = np.arange(np.floor(p.min()), np.ceil(p.max()), bin_width)
    P_hist = np.histogram(p, bins=bins, density=True)[0]
    p_kernel = stats.gaussian_kde(np.squeeze(p), bw_method=2 / p.std(ddof=1))
    P_kde = p_kernel.evaluate(bins[:-1])

    # Histogram and PDF of AS19 ensemble estimated using a Gaussian KDE
    q = response.values
    Q_hist = np.histogram(q, bins=bins, density=True)[0]
    q_kernel = stats.gaussian_kde(np.squeeze(q), bw_method=2 / q.std(ddof=1))
    Q_kde = p_kernel.evaluate(bins)

    # Use the "best" emulator and train the emlator with the AS19 samples / response, then
    # run the emlator
    p_as19, status = emulate_gp(
        samples,
        response,
        X_true,
        kernel=gp.kern.ExpQuad,
        stepwise=False,
        optimizer_options={"max_iters": 4000},
    )

    # p_as[0] are the posterior mean
    # p_as[1] are the posterior variance

    # Histogram and PDF of the emulated AS19 ensemble estimated using a Gaussian KDE
    p_as19 = np.squeeze(p_as19[0])
    P_as19_hist = np.histogram(p_as19, bins=bins, density=True)[0]
    p_as19_kernel = stats.gaussian_kde(
        np.squeeze(p_as19), bw_method=2 / p_as19.std(ddof=1)
    )
    P_as19_kde = p_as19_kernel.evaluate(bins[:-1])

    percentiles = [5, 16, 50, 84, 95]

    # Plot the results

    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.distplot(
        p,
        bins=bins,
        hist=True,
        hist_kws={"alpha": 0.3},
        norm_hist=True,
        kde_kws={"shade": False, "alpha": 1, "linewidth": 1.5},
        ax=ax,
        color="k",
    )
    sns.distplot(
        q,
        bins=bins,
        hist=True,
        hist_kws={"alpha": 0.3},
        norm_hist=True,
        kde_kws={"shade": False, "alpha": 1, "linewidth": 1.0},
        ax=ax,
        color="#3182bd",
    )

    sns.distplot(
        p_as19,
        bins=bins,
        hist=True,
        hist_kws={"alpha": 0.3},
        norm_hist=True,
        kde_kws={"shade": False, "alpha": 1, "linewidth": 1.0},
        ax=ax,
        color="r",
    )
    sns.distplot(
        p,
        bins=bins,
        hist=False,
        hist_kws={"alpha": 0.0},
        norm_hist=True,
        kde_kws={"shade": False, "alpha": 1, "linewidth": 1.5},
        ax=ax,
        color="k",
        label='"True"',
    )
    sns.distplot(
        q,
        bins=bins,
        hist=False,
        hist_kws={"alpha": 0.0},
        norm_hist=True,
        kde_kws={"shade": False, "alpha": 1, "linewidth": 1.0},
        ax=ax,
        color="#3182bd",
        label="AS19",
    )

    sns.distplot(
        p_as19,
        bins=bins,
        hist=False,
        hist_kws={"alpha": 0.0},
        norm_hist=True,
        kde_kws={"shade": False, "alpha": 1, "linewidth": 1.0},
        ax=ax,
        color="r",
        label="AS19-GP",
    )
    for pctl in percentiles:
        pc = np.percentile(p, pctl)
        ypc = p_kernel.evaluate(pc)
        ax.plot(
            [pc, pc],
            [0, ypc],
            linewidth=1.0,
            linestyle="dotted",
            color="k",
            transform=ax.transData,
        )
        qc = np.percentile(q, pctl)
        ax.plot(
            [qc, qc],
            [0, q_kernel.evaluate(qc)],
            linewidth=1.0,
            linestyle="dotted",
            color="#3182bd",
            transform=ax.transData,
        )
        pas19c = np.percentile(p_as19, pctl)
        ax.plot(
            [pas19c, pas19c],
            [0, p_as19_kernel.evaluate(pas19c)],
            linewidth=1.0,
            linestyle="dotted",
            color="r",
            transform=ax.transData,
        )
        ax.annotate(
            r"""rmsd={:2.2f}cm""".format(rmsd(p, p_as19)),
            (0.6, 0.6),
            xytext=(0.6, 0.4),
            xycoords="axes fraction",
        )

    ax.set_ylabel("Probability (1)")
    ax.set_xlabel("Sea-level contribution (cm sea-level equivalent)")
    xmin, xmax = ax.get_xlim()
    sns.despine()
    legend = ax.legend(ncol=1)
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    set_size(3.2, 3.2 / golden_ratio)

    outfile = "true_vs_as19_predicted.pdf"
    print("Saving figure to {}".format(outfile))
    fig.savefig(outfile, bbox_inches="tight")
