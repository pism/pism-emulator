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
from glob import glob
import GPy as gp
from math import sqrt
import numpy as np
import os
import re
import pandas as pd
import pylab as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy import stats
import sys

from pismemulator.utils import golden_ratio
from pismemulator.utils import kl_divergence
from pismemulator.utils import prepare_data
from pismemulator.utils import rmsd
from pismemulator.utils import set_size

from sklearn.metrics import mean_squared_error


def rmsd_f(df):

    return rmsd(df["Y_mean"], Y_true)


def mse_f(df):

    return np.sum(np.sqrt(df["Y_var"])) / len(df)


def kldiv_f(df):
    """
    Calculate the histogram Q of df["Y_mean"] and then
    return the KL-divergence of (P, Q).
    """
    Q = np.histogram(df["Y_mean"], bins=bins, density=True)[0]
    return kl_divergence(P, Q)


def hist_f(df):
    """
    Calculate the histogram Q of df["Y_mean"].
    """
    Q = np.histogram(df["Y_mean"], bins=bins, density=True)[0]
    return Q


if __name__ == "__main__":

    __spec__ = None

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Validate Regression Methods."
    parser.add_argument(
        "-b", "--bin_width", dest="bin_width", help="Width of histogram bins. Default=1cm", default=1.0
    )
    parser.add_argument(
        "-n",
        dest="n_samples_validation",
        choices=[20, 170, 240, 320, 400],
        type=int,
        help="Number of validation samples.",
        default=400,
    )

    options = parser.parse_args()
    bin_width = options.bin_width
    n_samples_validation = options.n_samples_validation
    rcp = 45
    year = 2100

    # Load the "True" data.
    s_true, r_true = prepare_data(
        "../data/samples/saltelli_samples_{}.csv".format(n_samples_validation),
        "../data/validation/dgmsl_rcp_45_year_2100_s{}.csv".format(n_samples_validation),
    )
    X_true = s_true.values
    Y_true = r_true.values

    p = Y_true
    bins = np.arange(np.floor(p.min()), np.ceil(p.max()), bin_width)
    P = np.histogram(p, bins=bins, density=True)[0]

    colors = sns.color_palette("Paired")
    cmap = [colors[k] for k in [0, 1, 2, 3, 9]]

    qs = []
    for k, n_lhs_samples in enumerate(range(100, 600, 100)):
        response_file = "../data/validation/dgmsl_rcp_{}_year_{}_lhs_{}.csv".format(rcp, year, n_lhs_samples)

        df = pd.read_csv(response_file)
        qs.append(df.drop(df.columns[0], axis=1).values)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for k, n_lhs_samples in enumerate(range(100, 600, 100)):
        sns.distplot(
            qs[k],
            bins=bins,
            hist=True,
            hist_kws={"alpha": 0.3},
            norm_hist=True,
            kde_kws={"shade": False, "alpha": 1, "linewidth": 1.5},
            ax=ax,
            color=cmap[k],
        )
        sns.distplot(
            qs[k],
            bins=bins,
            hist=False,
            norm_hist=True,
            kde_kws={"shade": False, "alpha": 1, "linewidth": 1.5},
            ax=ax,
            color=cmap[k],
            label=f"{n_lhs_samples}",
        )

    ax.set_ylabel("Probability (1)")
    ax.set_xlabel("Sea-level contribution (cm sea-level equivalent)")
    xmin, xmax = ax.get_xlim()
    sns.despine()
    legend = ax.legend(ncol=1, title="#Samples")
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    set_size(3.2, 3.2 / golden_ratio)

    outfile = "as19_pdfs.pdf"
    print("Saving figure to {}".format(outfile))
    fig.savefig(outfile, bbox_inches="tight")

    percentiles = [5, 16, 50, 84, 95]
    q = qs[-1]
    q_kernel = stats.gaussian_kde(np.squeeze(q), bw_method=2 / q.std(ddof=1))
    Q = np.histogram(q, bins=bins, density=True)[0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.distplot(
        q,
        bins=bins,
        hist=True,
        hist_kws={"alpha": 0.3},
        norm_hist=True,
        kde=False,
        kde_kws={"shade": False, "alpha": 1.0, "linewidth": 1.5},
        ax=ax,
        color=cmap[-1],
    )
    sns.distplot(
        p,
        bins=bins,
        hist=True,
        hist_kws={"alpha": 0.3},
        norm_hist=True,
        kde=False,
        kde_kws={"shade": False, "alpha": 1.0, "linewidth": 1.5},
        ax=ax,
        color="k",
    )
    sns.distplot(
        q,
        bins=bins,
        hist=False,
        hist_kws={"alpha": 0.3},
        norm_hist=True,
        kde=True,
        kde_kws={"shade": False, "alpha": 1.0, "linewidth": 1.5},
        ax=ax,
        color=cmap[-1],
        label=f"AS19, {kl_divergence(P, Q):2.3f}",
    )
    sns.distplot(
        p,
        bins=bins,
        hist=False,
        hist_kws={"alpha": 0.3},
        norm_hist=True,
        kde=True,
        kde_kws={"shade": False, "alpha": 1.0, "linewidth": 1.5},
        ax=ax,
        color="k",
        label='"True"',
    )

    # Create a Rectangle patch
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (30, -0.01), 15, 0.02, linewidth=1, edgecolor="r", facecolor="r", alpha=0.25, transform=ax.transData
    )
    # Add the patch to the Axes
    ax.add_patch(rect)

    ax.set_ylabel("Probability (1)")
    ax.set_xlabel("Sea-level contribution (cm sea-level equivalent)")
    ax.set_xlim(xmin, xmax)
    sns.despine()
    legend = ax.legend(ncol=1)
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    set_size(3.2, 3.2 / golden_ratio)

    outfile = "as19_vs_true.pdf"
    print("Saving figure to {}".format(outfile))
    fig.savefig(outfile, bbox_inches="tight")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.distplot(
        q,
        bins=bins,
        hist=True,
        hist_kws={"alpha": 0.3},
        norm_hist=True,
        kde=True,
        kde_kws={"shade": False, "alpha": 1.0, "linewidth": 1.5},
        ax=ax,
        color=cmap[-1],
    )
    for pctl in percentiles:
        qc = np.percentile(q, pctl)
        yqc = q_kernel.evaluate(qc)
        ax.plot([qc, qc], [0, yqc], linewidth=0.75, linestyle="dotted", color="k", alpha=0.75, transform=ax.transData)
        ax.annotate(r"""{}%""".format(pctl), (qc, 0), xytext=(qc, yqc), xycoords="data")

    ax.set_ylabel("Probability (1)")
    ax.set_xlabel("Sea-level contribution (cm sea-level equivalent)")
    ax.set_xlim(xmin, xmax)
    sns.despine()

    set_size(3.2, 3.2 / golden_ratio)

    outfile = "as19_hist_kde_pctl.pdf"
    print("Saving figure to {}".format(outfile))
    fig.savefig(outfile, bbox_inches="tight")
