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

# This scripts validates the 500 member ensembe of Aschwanden et al. (2019)
# against a much larger 3096 member ensemble which is considered to closely
# approximate the "true" distribution

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import GPy as gp
from matplotlib.patches import Rectangle
import numpy as np
import os
import pylab as plt
from scipy import stats
import seaborn as sns
import sys
from pismemulator.utils import gelman_rubin
from pismemulator.utils import golden_ratio
from pismemulator.utils import kl_divergence
from pismemulator.utils import prepare_data
from pismemulator.utils import rmsd
from pismemulator.utils import set_size

from pismemulator.emulate import emulate_gp


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
        "-b", "--bin_width", dest="bin_width", help="Width of histogram bins. Default=1cm", type=float, default=1.0
    )

    options = parser.parse_args()
    bin_width = options.bin_width
    n_samples_validation = options.n_samples_validation
    rcp = 45
    year = 2100

    print("-----------------------------------------------------------")
    print("Validating the Aschwanden et al. (2019) ensemble")
    print("-----------------------------------------------------------")

    # Load the "True" data.
    X_true, Y_true = prepare_data(
        "../data/samples/saltelli_samples_{}.csv".format(n_samples_validation),
        "../data/validation/dgmsl_rcp_45_year_2100_s{}.csv".format(n_samples_validation),
        return_numpy=True,
    )

    p = Y_true
    bins = np.arange(np.floor(p.min()), np.ceil(p.max()), bin_width)
    P = np.histogram(p, bins=bins, density=True)[0]

    percentiles = [5, 16, 50, 84, 95]

    print(f"Percentiles{percentiles}\n")
    pc = np.percentile(p, percentiles)

    colors2 = sns.cubehelix_palette(8).as_hex()

    colors = sns.color_palette("Paired", 10)
    cmap = [colors[k] for k in [0, 1, 2, 3, 8, 9]]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for k, no_lhs_samples in enumerate([100, 200, 300, 400]):

        samples_file = f"../data/samples/lhs_samples_{no_lhs_samples}.csv"
        response_file = f"../data/validation/dgmsl_rcp_{rcp}_year_{year}_lhs_{no_lhs_samples}.csv"
        samples, response = prepare_data(samples_file, response_file)

        q = response.values
        Q = np.histogram(q, bins=bins, density=True)[0]
        kl_div = kl_divergence(P, Q)
        qc = np.percentile(q, percentiles)
        # print(f"LHS-{no_lhs_samples}: {qc:2.1f}")

        sns.distplot(
            q,
            bins=bins,
            hist=True,
            hist_kws={"alpha": 0.3},
            norm_hist=True,
            kde_kws={"shade": False, "alpha": 1, "linewidth": 1.0},
            ax=ax,
            color=colors[k],
        )
        sns.distplot(
            q,
            bins=bins,
            hist=False,
            hist_kws={"alpha": 0.0},
            norm_hist=True,
            kde_kws={"shade": False, "alpha": 1, "linewidth": 1.0},
            ax=ax,
            color=colors[k],
            label=f"AS19-{no_lhs_samples} ({kl_div:2.3f})",
        )

    for k, no_lhs_samples in enumerate([500]):

        samples_file = f"../data/samples/lhs_samples_{no_lhs_samples}.csv"
        response_file = f"../data/validation/dgmsl_rcp_{rcp}_year_{year}_lhs_{no_lhs_samples}.csv"
        samples, response = prepare_data(samples_file, response_file)

        q = response.values
        Q = np.histogram(q, bins=bins, density=True)[0]
        kl_div = kl_divergence(P, Q)

        qc = np.percentile(q, percentiles)
        # print(f"LHS-{no_lhs_samples}: {qc:2.1f}")

        sns.distplot(
            q,
            bins=bins,
            hist=True,
            hist_kws={"alpha": 0.3},
            norm_hist=True,
            kde_kws={"shade": False, "alpha": 1, "linewidth": 1.0},
            ax=ax,
            color=colors[-1],
        )
        sns.distplot(
            q,
            bins=bins,
            hist=False,
            hist_kws={"alpha": 0.0},
            norm_hist=True,
            kde_kws={"shade": False, "alpha": 1, "linewidth": 1.0},
            ax=ax,
            color=colors[-1],
            label=f"AS19 ({kl_div:2.3f})",
        )
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
    ax.set_ylabel("Probability (1)")
    ax.set_xlabel("Sea-level contribution (cm sea-level equivalent)")
    xmin, xmax = ax.get_xlim()
    sns.despine()
    legend = ax.legend(ncol=1)
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    set_size(3.2, 3.2 / golden_ratio)

    outfile = "compare_as19_all_vs_true.pdf"
    print("Saving figure to {}".format(outfile))
    fig.savefig(outfile, bbox_inches="tight")

    fig = plt.figure()
    ax_h = fig.add_subplot(111)

    sns.distplot(
        p,
        bins=bins,
        hist=True,
        hist_kws={"alpha": 0.4},
        norm_hist=True,
        kde_kws={"shade": False, "alpha": 1, "linewidth": 1.0},
        ax=ax_h,
        color="k",
    )
    sns.distplot(
        p,
        bins=bins,
        hist=False,
        hist_kws={"alpha": 0.0},
        norm_hist=True,
        kde_kws={"shade": False, "alpha": 1, "linewidth": 1.0},
        ax=ax_h,
        color="k",
        label="True",
    )

    for k, no_lhs_samples in enumerate([500]):

        samples_file = f"../data/samples/lhs_samples_{no_lhs_samples}.csv"
        response_file = f"../data/validation/dgmsl_rcp_{rcp}_year_{year}_lhs_{no_lhs_samples}.csv"
        samples, response = prepare_data(samples_file, response_file)

        q = response.values
        Q = np.histogram(q, bins=bins, density=True)[0]
        kl_div = kl_divergence(P, Q)

        sns.distplot(
            q,
            bins=bins,
            hist=True,
            hist_kws={"alpha": 0.4},
            norm_hist=True,
            kde_kws={"shade": False, "alpha": 1, "linewidth": 1.0},
            ax=ax_h,
            color=colors[-1],
        )
        sns.distplot(
            q,
            bins=bins,
            hist=False,
            norm_hist=True,
            kde_kws={"shade": False, "alpha": 1, "linewidth": 1.0},
            ax=ax_h,
            color=colors[-1],
            label=f"AS19",
        )

    pctls = [5, 16, 50, 84, 95]
    p_pctls = np.percentile(p, pctls)
    q_pctls = np.percentile(q, pctls)

    print(f"True 50th percentile at {p_pctls[2]:2.0f} cm SLE")
    print(f"AS19 50th percentile at {q_pctls[2]:2.0f} cm SLE")
    print(f"Increase 50th percentile  {np.abs((q_pctls[2]-p_pctls[2]))/q_pctls[2]*100:2.2f}\n")

    print(f"True 95th percentile at {p_pctls[-1]:2.0f} cm SLE")
    print(f"AS19 95th percentile at {q_pctls[-1]:2.0f} cm SLE")
    print(f"Increase 95th percentile  {np.abs((q_pctls[-1]-p_pctls[-1]))/q_pctls[-1]*100:2.2f}\n")
    ax_h.vlines(p_pctls, 0, 0.05, linestyle="dotted", linewidth=0.5, color="k")
    ax_h.vlines(q_pctls, 0, 0.05, linestyle="dotted", linewidth=0.5, color=colors[-1])

    ax_h.set_ylabel("Probability (1)")
    ax_h.set_xlabel("Sea-level contribution (cm sea-level equivalent)")
    xmin, xmax = ax_h.get_xlim()
    ax_h.set_xlim(-10, 50)
    sns.despine(ax=ax_h)
    legend = ax_h.legend(ncol=1)
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    set_size(3.2, 3.2 / golden_ratio)

    outfile = "as19_vs_true.pdf"
    print("Saving figure to {}".format(outfile))
    fig.savefig(outfile, bbox_inches="tight")

    P1 = 1 - np.cumsum(P)
    Q1 = 1 - np.cumsum(Q)

    m_thr = 30
    p_P_thr = P1[bins[:-1] > m_thr][0] * 100
    p_Q_thr = Q1[bins[:-1] > m_thr][0] * 100

    print(f"")
    fig = plt.figure()
    ax_h = fig.add_subplot(111)

    ax_h.plot(bins[:-1], P1, color="k")
    ax_h.plot(bins[:-1], Q1, color=colors[-1])
    pctls = [5, 50, 95]
    p_pctls = np.percentile(p, pctls)
    q_pctls = np.percentile(q, pctls)
    print(f"True 95th percentile at {p_pctls[-1]:2.0f} cm SLE")
    print(f"AS19 95th percentile at {q_pctls[-1]:2.0f} cm SLE")
    print(f"Increase 95th percentile  {np.abs((q_pctls[-1]-p_pctls[-1]))/q_pctls[-1]*100:2.2f}")

    ax_h.set_ylabel("1 - Cumulative Probability (1)")
    ax_h.set_xlabel("Sea-level contribution (cm sea-level equivalent)")
    xmin, xmax = ax_h.get_xlim()
    ax_h.set_xlim(-10, 50)
    sns.despine(ax=ax_h)
    legend = ax_h.legend(ncol=1)
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    set_size(3.2, 3.2 / golden_ratio)

    outfile = "as19_vs_true_cum.pdf"
    print("Saving figure to {}".format(outfile))
    fig.savefig(outfile, bbox_inches="tight")
