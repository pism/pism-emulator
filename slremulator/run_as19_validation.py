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
        "-s",
        "--samples_file",
        dest="samples_file",
        help="File that has all combinations for ensemble study",
        default="../data/samples/lhs_samples_500.csv",
    )
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
    samples_file = options.samples_file
    n_samples_validation = options.n_samples_validation
    rcp = 45
    year = 2100
    response_file = "../data/response/dgmsl_rcp_{}_year_{}.csv".format(rcp, year)

    print("-----------------------------------------------------------")
    print("Running validation of the Aschwanden et al. (2019) ensemble")
    print("-----------------------------------------------------------")

    # Load the AS19 data. Samples are the M x D parameter matrix
    # Response is the 1 X D sea level rise response (cm SLE)
    samples, response = prepare_data(samples_file, response_file)
    Y = response.values

    # Load the "True" data.
    X_true, Y_true = prepare_data(
        "../data/samples/saltelli_samples_{}.csv".format(n_samples_validation),
        "../data/validation/dgmsl_rcp_45_year_2100_s{}.csv".format(n_samples_validation),
        return_numpy=True,
    )

    p = Y_true
    bins = np.arange(np.floor(p.min()), np.ceil(p.max()), bin_width)
    P_hist = np.histogram(p, bins=bins, density=True)[0]
    p_kernel = stats.gaussian_kde(np.squeeze(p), bw_method=2 / p.std(ddof=1))
    P_kde = p_kernel.evaluate(bins[:-1])

    q = response.values
    Q_hist = np.histogram(q, bins=bins, density=True)[0]
    q_kernel = stats.gaussian_kde(np.squeeze(q), bw_method=2 / q.std(ddof=1))
    Q_kde = p_kernel.evaluate(bins)

    # Use the "best" emulator and train the emlator with the AS19 samples / response, then
    # run the emlator
    p_as19, status = emulate_gp(
        samples, response, X_true, kernel=gp.kern.ExpQuad, stepwise=False, optimizer_options={"max_iters": 4000}
    )
    p_as19 = np.squeeze(p_as19[0])
    P_as19_hist = np.histogram(p_as19, bins=bins, density=True)[0]
    p_as19_kernel = stats.gaussian_kde(np.squeeze(p_as19), bw_method=2 / p_as19.std(ddof=1))
    P_as19_kde = p_as19_kernel.evaluate(bins[:-1])

    colors = sns.cubehelix_palette(6, start=0.5, rot=-0.75).as_hex()
    colors2 = sns.cubehelix_palette(8).as_hex()

    percentiles = [5, 16, 50, 84, 95]
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
    for pctl in percentiles:
        pc = np.percentile(p, pctl)
        ypc = p_kernel.evaluate(pc)
        ax.plot([pc, pc], [0, ypc], linewidth=1.0, linestyle="dotted", color="k", transform=ax.transData)
        qc = np.percentile(q, pctl)
        ax.plot(
            [qc, qc],
            [0, q_kernel.evaluate(qc)],
            linewidth=1.0,
            linestyle="dotted",
            color="#3182bd",
            transform=ax.transData,
        )
    ax.set_ylabel("Probability (1)")
    ax.set_xlabel("Sea-level contribution (cm sea-level equivalent)")
    xmin, xmax = ax.get_xlim()
    sns.despine()
    legend = ax.legend(ncol=1)
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    set_size(3.2, 3.2 / golden_ratio)

    outfile = "true_vs_as19_clean.pdf"
    print("Saving figure to {}".format(outfile))
    fig.savefig(outfile, bbox_inches="tight")

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
    for pctl in percentiles:
        pc = np.percentile(p, pctl)
        ypc = p_kernel.evaluate(pc)
        ax.plot([pc, pc], [0, ypc], linewidth=1.0, linestyle="dotted", color="k", transform=ax.transData)
        qc = np.percentile(q, pctl)
        ax.plot(
            [qc, qc],
            [0, q_kernel.evaluate(qc)],
            linewidth=1.0,
            linestyle="dotted",
            color="#3182bd",
            transform=ax.transData,
        )
        ax.arrow(qc, 0.01, pc - qc, 0, width=0.00001, head_width=0.001, transform=ax.transData)

    # Create a Rectangle patch
    rect = Rectangle(
        (-8, -0.01), 10, 0.02, linewidth=1, edgecolor="r", facecolor="r", alpha=0.5, transform=ax.transData
    )
    # Add the patch to the Axes
    ax.add_patch(rect)
    # Create a Rectangle patch
    rect = Rectangle(
        (30, -0.01), 15, 0.02, linewidth=1, edgecolor="r", facecolor="r", alpha=0.5, transform=ax.transData
    )
    # Add the patch to the Axes
    ax.add_patch(rect)
    ax.set_ylabel("Probability (1)")
    ax.set_xlabel("Sea-level contribution (cm sea-level equivalent)")
    xmin, xmax = ax.get_xlim()
    sns.despine()
    legend = ax.legend(ncol=1)
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    set_size(3.2, 3.2 / golden_ratio)

    outfile = "true_vs_as19_arrows.pdf"
    print("Saving figure to {}".format(outfile))
    fig.savefig(outfile, bbox_inches="tight")

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
    for pctl in percentiles:
        pc = np.percentile(p, pctl)
        ypc = p_kernel.evaluate(pc)
        ax.plot([pc, pc], [0, ypc], linewidth=1.0, linestyle="dotted", color="k", transform=ax.transData)
        qc = np.percentile(q, pctl)
        ax.plot(
            [qc, qc],
            [0, q_kernel.evaluate(qc)],
            linewidth=1.0,
            linestyle="dotted",
            color="#3182bd",
            transform=ax.transData,
        )
    ax.annotate(
        r"""$D_{{\mathrm{{KL}}}}\left(P,Q\right)$={:2.3f}
$D_{{\mathrm{{KL}}}}\left(P,\widehat P\right)$={:2.3f}
$\chi_{{P, Q}}$: {:2.2f}mm sle
$\chi_{{P, \widehat P}}$: {:2.2f}mm sle""".format(
            kl_divergence(P_hist, Q_hist),
            kl_divergence(P_hist, P_kde),
            rmsd(P_hist, Q_hist) * 10,
            rmsd(P_hist, P_kde) * 10,
        ),
        (0.7, 0.6),
        xytext=(0.7, 0.4),
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

    outfile = "true_vs_as19.pdf"
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
        kde=False,
        kde_kws={"shade": False, "alpha": 1, "linewidth": 1.0},
        ax=ax,
        color="#3182bd",
    )
    ax.set_ylabel("Probability (1)")
    ax.set_xlabel("Sea-level contribution (cm sea-level equivalent)")
    ax.set_xlim(xmin, xmax)
    sns.despine()

    set_size(3.2, 3.2 / golden_ratio)

    outfile = "as19_hist.pdf"
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
        kde=False,
        kde_kws={"shade": False, "alpha": 1, "linewidth": 1.0},
        ax=ax,
        color="#3182bd",
    )
    for pctl in percentiles:
        pc = np.percentile(p, pctl)
        ypc = p_kernel.evaluate(pc)
        ax.plot([pc, pc], [0, ypc], linewidth=0.5, linestyle="dotted", color="k", alpha=0.5, transform=ax.transData)
        ax.annotate(r"""{}%""".format(pctl), (pc, 0), xytext=(pc, ypc), xycoords="data")

    ax.set_ylabel("Probability (1)")
    ax.set_xlabel("Sea-level contribution (cm sea-level equivalent)")
    ax.set_xlim(xmin, xmax)
    sns.despine()

    set_size(3.2, 3.2 / golden_ratio)

    outfile = "as19_hist_pctl.pdf"
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
        kde_kws={"shade": False, "alpha": 1, "linewidth": 1.0},
        ax=ax,
        color="#3182bd",
    )
    sns.distplot(
        q,
        bins=bins,
        hist=False,
        hist_kws={"alpha": 0.3},
        kde=True,
        kde_kws={"shade": False, "alpha": 1, "linewidth": 1.0},
        norm_hist=True,
        ax=ax,
        color="#3182bd",
        label="PDF (KDE est.)",
    )
    ax.set_ylabel("Probability (1)")
    ax.set_xlabel("Sea-level contribution (cm sea-level equivalent)")
    ax.set_xlim(xmin, xmax)
    sns.despine()
    legend = ax.legend(ncol=1)
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    set_size(3.2, 3.2 / golden_ratio)

    outfile = "as19_hist_kde.pdf"
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
        kde_kws={"shade": False, "alpha": 1, "linewidth": 1.0},
        ax=ax,
        color="#3182bd",
    )
    sns.distplot(
        q,
        bins=bins,
        hist=False,
        hist_kws={"alpha": 0.3},
        kde=True,
        kde_kws={"shade": False, "alpha": 1, "linewidth": 1.0},
        norm_hist=True,
        ax=ax,
        color="#3182bd",
        label="PDF (KDE est.)",
    )
    for pctl in percentiles:
        qc = np.percentile(q, pctl)
        yqc = p_kernel.evaluate(qc)
        ax.plot(
            [qc, qc], [0, yqc], linewidth=1.0, linestyle="dotted", color="#3182bd", alpha=1.0, transform=ax.transData
        )
        ax.annotate(r"""{}%""".format(pctl), (qc, yqc), xytext=(qc, yqc), xycoords="data")

    ax.set_ylabel("Probability (1)")
    ax.set_xlabel("Sea-level contribution (cm sea-level equivalent)")
    ax.set_xlim(xmin, xmax)
    sns.despine()
    legend = ax.legend(ncol=1)
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    set_size(3.2, 3.2 / golden_ratio)

    outfile = "as19_hist_kde_pctl.pdf"
    print("Saving figure to {}".format(outfile))
    fig.savefig(outfile, bbox_inches="tight")

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
        ax.plot([pc, pc], [0, ypc], linewidth=1.0, linestyle="dotted", color="k", transform=ax.transData)
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
            r"""rmsd={:2.2f}cm""".format(rmsd(p, p_as19)), (0.6, 0.6), xytext=(0.6, 0.4), xycoords="axes fraction"
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
