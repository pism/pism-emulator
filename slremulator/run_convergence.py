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

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import os
import pylab as plt
from scipy import stats
import seaborn as sns
import sys
from pismemulator.utils import gelman_rubin
from pismemulator.utils import kl_divergence
from pismemulator.utils import prepare_data
from pismemulator.utils import set_size


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
    saltelli_multiplier = 13

    golden_ratio = (1 + np.sqrt(5)) / 2

    X_true, Y_true = prepare_data(
        "../data/samples/saltelli_samples_{}.csv".format(n_samples_validation),
        "../data/validation/dgmsl_rcp_45_year_2100_s{}.csv".format(n_samples_validation),
        return_numpy=True,
    )

    s_true, r_true = prepare_data(
        "../data/samples/saltelli_samples_{}.csv".format(n_samples_validation),
        "../data/validation/dgmsl_rcp_45_year_2100_s{}.csv".format(n_samples_validation),
    )
    p = r_true.values
    bins = np.arange(np.floor(p.min()), np.ceil(p.max()), bin_width)
    P_hist = np.histogram(p, bins=bins, density=True)[0]

    s_as19, r_as19 = prepare_data(samples_file, response_file)
    Y = r_as19.values

    q_as19 = r_as19.values
    Q_as19_hist = np.histogram(q_as19, bins=bins, density=True)[0]
    q_as19_kernel = stats.gaussian_kde(np.squeeze(q_as19), bw_method=2 / q_as19.std(ddof=1))
    Q_as19_kde = q_as19_kernel.evaluate(bins)

    ny = len(r_true)
    m_range = range(10 * saltelli_multiplier, ny, 1 * saltelli_multiplier)
    kl_div_hist = np.zeros(len(m_range))
    kl_div_hist_rel = np.zeros(len(m_range))
    gr = np.zeros(len(m_range))
    m_samples = np.zeros(len(m_range))
    for k, idx in enumerate(m_range):
        q = r_true[0:idx].values
        if k > 0:
            q_m1 = r_true[0 : idx - 1].values
            Q_hist_m1 = np.histogram(q_m1, bins=bins, density=True)[0]
            kl_div_hist_rel[k - 1] = kl_divergence(Q_hist, Q_hist_m1)
        m_samples[k] = len(q)
        Q_hist = np.histogram(q, bins=bins, density=True)[0]
        kl_div_hist[k] = kl_divergence(P_hist, Q_hist)
        gr[k] = gelman_rubin(p, q)

    # Number of samples needed for KL-div < threshold
    kl_div_threshold = 1e-2
    res = list(filter(lambda i: i < kl_div_threshold, kl_div_hist))[0]
    samples_threshold = m_samples[list(kl_div_hist).index(res)]
    print(f"\nSamples needed for KL-divergence < {kl_div_threshold}: {samples_threshold:2.0f}\n")
    # p = Y_true
    # bins = np.arange(np.floor(p.min()), np.ceil(p.max()), bin_width)
    # p_kernel = stats.gaussian_kde(np.squeeze(p), bw_method=2 / p.std(ddof=1))
    # P_kde = p_kernel.evaluate(bins)

    colors = sns.cubehelix_palette(10, start=0.5, rot=-0.75).as_hex()
    colors2 = sns.cubehelix_palette(10).as_hex()
    colors = sns.color_palette("Paired")
    colors = sns.color_palette("GnBu_d", 5)[::-1]

    m_range = range(80 * saltelli_multiplier, ny, 80 * saltelli_multiplier)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.distplot(
        p,
        bins=bins,
        hist=True,
        hist_kws={"alpha": 0.2},
        kde_kws={"shade": False, "alpha": 1, "linewidth": 1.0},
        color="0",
        ax=ax,
    )

    for k, idx in enumerate(m_range):
        q = r_true[0:idx].values
        Q_hist = np.histogram(q, bins=bins, density=True)[0]
        sns.distplot(q, hist=False, hist_kws={"alpha": 0.2}, bins=bins, kde_kws={"lw": 1.0}, color=colors[k], ax=ax)
        sns.distplot(
            q,
            hist=False,
            hist_kws={"alpha": 0.2},
            bins=bins,
            kde_kws={"lw": 1.0},
            color=colors[k],
            ax=ax,
            label="{}, {:2.4f}".format(idx, kl_divergence(P_hist, Q_hist)),
        )
        sns.distplot(
            q,
            bins=bins,
            hist=True,
            hist_kws={"alpha": 0.2},
            kde_kws={"shade": False, "alpha": 1, "linewidth": 1.25},
            color=colors[k],
            ax=ax,
        )

    sns.distplot(
        p,
        bins=bins,
        hist=False,
        hist_kws={"alpha": 0.2},
        kde_kws={"shade": False, "alpha": 1, "linewidth": 1.25},
        color="0",
        ax=ax,
        label='5200 ("True")',
    )

    sns.distplot(
        p,
        bins=bins,
        hist=False,
        hist_kws={"alpha": 0.2},
        kde_kws={"shade": False, "alpha": 1, "linewidth": 1.25},
        color="0",
        ax=ax,
    )
    ax.set_ylabel("Probability (1)")
    ax.set_xlabel("Sea-level contribution (cm sea-level equivalent)")
    sns.despine()
    legend = ax.legend(ncol=1, title="#, $D_{{\mathrm{{KL}}}}$ (nats)")
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    set_size(3.2, 3.2 / golden_ratio)

    fig.savefig("convergence.pdf", bbox_inches="tight")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(m_samples, kl_div_hist, ".", color="k", label='"True"')
    ax.plot(500, kl_divergence(P_hist, Q_as19_hist), ".", color="#3182bd", label="AS19")
    ax.set_yscale("log")
    ax.set_xlabel("samples")
    ax.set_ylabel("$D_{{\mathrm{{KL}}}}$ (nats)")
    sns.despine()

    set_size(3.2, 2)

    fig.savefig("nsamples_vs_kldiv.pdf", bbox_inches="tight")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(kl_div_hist, m_samples, ".", color="0.5", label="hist")
    for k, idx in enumerate(m_range):
        q = r_true[0:idx].values
        Q_hist = np.histogram(q, bins=bins, density=True)[0]
        ax.plot(kl_divergence(P_hist, Q_hist), idx, ".", color=colors[k])
    ax.set_yscale("linear")
    ax.set_ylabel("samples")
    ax.set_xlabel("$D_{{\mathrm{{KL}}}}$ (nats)")
    sns.despine()

    set_size(3.2, 2)

    fig.savefig("kldiv_vs_nsamples.pdf", bbox_inches="tight")

    ax.set_xscale("log")
    # ax.set_yscale("log")
    fig.savefig("kldiv_vs_nsamples_log.pdf", bbox_inches="tight")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(m_samples, gr, ".", color="k")
    ax.set_yscale("linear")
    ax.set_xlabel("samples")
    ax.set_ylabel("Gelman-Rubin")
    sns.despine()

    set_size(3.2, 3.2 / golden_ratio)

    fig.savefig("gr_nsamples.pdf", bbox_inches="tight")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(m_samples, kl_div_hist_rel, ".", color="0.5", label="hist")
    ax.set_yscale("linear")
    ax.set_xlabel("samples")
    ax.set_ylabel("$D_{{\mathrm{{KL}}}}^n$-$D_{{\mathrm{{KL}}}}^{n-1}$")
    ax.set_yscale("log")
    sns.despine()

    set_size(3.2, 2)

    fig.savefig("kldiv_vs_nsamples_rel.pdf", bbox_inches="tight")
