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
from functools import reduce
from glob import glob
import GPy as gp
from math import sqrt
import numpy as np
import os
import re
import pandas as pd
import pylab as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error
import sys

from pismemulator.utils import golden_ratio
from pismemulator.utils import kl_divergence
from pismemulator.utils import prepare_data
from pismemulator.utils import rmsd
from pismemulator.utils import set_size

from sklearn.metrics import mean_squared_error

default_les_directory = "emulator_results"
default_loo_directory = "loo_results"


def distance_f(df):

    return np.sqrt(np.sum(df["distance"]))


def rmsd_f(df):

    return rmsd(df["Y_mean"], Y_true)


def nrmsd_f(df):

    Q1 = df["Y_true"].quantile(0.25)
    Q3 = df["Y_true"].quantile(0.75)
    IQR = Q3 - Q1

    return rmsd(df["Y_mean"], df["Y_true"]) / IQR


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


def load_data_frames(csv_files):
    """
    Load results and return a pandas.DataFrame
    """
    dfs = []
    for k, response_file in enumerate(csv_files):
        df = pd.read_csv(response_file, delimiter=",", squeeze=True, skipinitialspace=True).sort_values(by="id")
        dfs.append(df)

    m_df = pd.concat(dfs, sort=False)
    m_df.reset_index(inplace=True, drop=True)

    return m_df


def evaluate_loo_distance(df):
    """
    Evaluate LOO using the distance
    """

    print("\nLeave-one-out validation")
    print("-----------------------------------\n\n")

    for n in range(100, 600, 100):
        print(f"Using {n} LHS samples:")
        d_df = df[df.n_lhs == n].reset_index(drop=True)
        d_df = d_df.groupby(["method", "n_lhs"]).apply(distance_f)
        d_df = d_df.to_frame(name="distance").reset_index()
        # Make sure n_lhs is int
        d_df = d_df.astype({"n_lhs": "int"})
        print(d_df.sort_values(["distance"]))
        print("\n")


def evaluate_kldiv(df):
    """
    Evaluate the KL divergence for AS19
    """

    df = df[df.n_lhs == 500].reset_index(drop=True)
    d_df = df.groupby(["method", "n_lhs"]).apply(kldiv_f)
    d_df = d_df.to_frame(name="kldiv").reset_index()
    # Make sure n_lhs is int
    d_df = d_df.astype({"n_lhs": "int"})
    print("\nKL-div validation")
    print("-----------------------------------")
    print(d_df.sort_values(["kldiv"]))


def evaluate_loo_nrmsd(df):
    """
    Evaluate LOO using the normalized RMSD
    """

    d_df = df.groupby(["method", "n_lhs"]).apply(nrmsd_f)
    d_df = d_df.to_frame(name="rmsd").reset_index()
    # Make sure n_lhs is int
    d_df = d_df.astype({"n_lhs": "int"})
    print("\nNRMSD validation")
    print("-----------------------------------")
    print(d_df.sort_values(["rmsd"]))


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
    parser.add_argument(
        "--les_dir",
        dest="lesdir",
        help=f"Directory where the LES files are. Default = {default_les_directory}.",
        default=default_les_directory,
    )
    parser.add_argument(
        "--loo_dir",
        dest="loodir",
        help=f"Directory where the LOO files are. Default = {default_loo_directory}.",
        default=default_loo_directory,
    )

    options = parser.parse_args()
    bin_width = options.bin_width
    n_samples_validation = options.n_samples_validation
    lesdir = options.lesdir
    loodir = options.loodir
    rcp = 45
    year = 2100
    n_lhs_samples = 500

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

    as19_response_file = "../data/validation/dgmsl_rcp_{}_year_{}_lhs_{}.csv".format(rcp, year, n_lhs_samples)

    as19_df = pd.read_csv(as19_response_file)
    as19_df = as19_df.drop(as19_df.columns[0], axis=1)
    q_as19 = as19_df.values
    Q_as19 = np.histogram(q_as19, bins=bins, density=True)[0]

    samples_file = ("../data/samples/saltelli_samples_{}.csv".format(n_samples_validation),)

    les_files = glob(f"{lesdir}/dgmsl_rcp_45_2100_*_lhs_*.csv")
    les_df = load_data_frames(les_files)

    # Calculate the KL-divergence between the true histogram and each regressor. Group DataFrame
    # and then apply function `kld`
    kldiv_df = les_df.groupby(["method", "n_lhs"]).apply(kldiv_f)
    kldiv_df = kldiv_df.to_frame(name="kldiv").reset_index()

    # Calculate the RMSD between Y_true and the emulator
    rmsd_df = les_df.groupby(["method", "n_lhs"]).apply(rmsd_f)
    rmsd_df = rmsd_df.to_frame(name="rmsd").reset_index()

    # Merge all data frames
    les_stats_df = reduce(lambda left, right: pd.merge(left, right), [kldiv_df, rmsd_df])

    loo_files = glob(f"{loodir}/loo_*.csv")
    loo_df = load_data_frames(loo_files)

    lhs_dfs = []
    for n_lhs_samples in range(100, 600, 100):
        lhs_file = "../data/validation/dgmsl_rcp_{}_year_{}_lhs_{}.csv".format(rcp, year, n_lhs_samples)
        df = pd.read_csv(lhs_file)
        df["n_lhs"] = n_lhs_samples
        lhs_dfs.append(df)
    lhs_df = pd.concat(lhs_dfs, sort=False).rename(columns={"limnsw(cm)": "Y_true"})
    m_df = pd.merge(lhs_df, loo_df, on=["id", "n_lhs"])

    kendall_tau_df = m_df.groupby(["method", "n_lhs"])["Y_true"].corr(m_df["Y_mean"], method="kendall")
    kendall_tau_df.to_frame(name="tau").reset_index()

    distance_df = m_df.groupby(["method", "n_lhs"]).apply(distance_f)
    distance_df = distance_df.to_frame(name="distance").reset_index()
    # Make sure n_lhs is int
    distance_df = distance_df.astype({"n_lhs": "int"})

    evaluate_loo_distance(m_df)
    evaluate_loo_nrmsd(m_df)

    evaluate_kldiv(m_df)

    colors = sns.color_palette("Paired", 10)
    cmap = [colors[k] for k in [0, 1, 2, 3, 8, 9]]
    cmap2 = sns.color_palette("colorblind", 5)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.pointplot(
        x="method",
        y="kldiv",
        order=[
            "exp",
            "exp-step",
            "expquad",
            "expquad-step",
            "mat32",
            "mat32-step",
            "mat52",
            "mat52-step",
            "lasso",
            "lasso-lars",
            "ridge",
        ],
        hue="n_lhs",
        data=les_stats_df,
        ax=ax,
        palette=cmap,
        join=False,
        scale=0.8,
    )
    legend = ax.legend(ncol=1, loc="upper right", bbox_to_anchor=(1.2, 1))
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    ax.set_ylim(1e-3, 1e0)
    ax.set_yscale("log")
    ticklabels = ax.get_xticklabels()
    for tick in ticklabels:
        tick.set_rotation(90)
    set_size(4, 4 / golden_ratio)
    fig.savefig("n_lhs_kldiv.pdf", bbox_inches="tight")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.pointplot(
        x="method",
        y="kldiv",
        order=["exp", "expquad", "mat32", "mat52",],
        hue="method",
        data=les_stats_df,
        ax=ax,
        palette=cmap2,
        join=False,
        scale=0.8,
    )
    legend = ax.legend(ncol=1, loc="upper right", bbox_to_anchor=(1.2, 1))
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    ax.set_ylim(1e-3, 1e0)
    ax.set_yscale("log")
    ticklabels = ax.get_xticklabels()
    for tick in ticklabels:
        tick.set_rotation(90)
    set_size(4, 4 / golden_ratio)
    fig.savefig("n_lhs_500_kldiv.pdf", bbox_inches="tight")

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # sns.pointplot(
    #     x="method",
    #     y="rmsd",
    #     order=[
    #         "exp",
    #         "exp-step",
    #         "expquad",
    #         "expquad-step",
    #         "mat32",
    #         "mat32-step",
    #         "mat52",
    #         "mat52-step",
    #         "lasso",
    #         "lasso-lars",
    #         "ridge",
    #     ],
    #     hue="n_lhs",
    #     data=les_stats_df,
    #     ax=ax,
    #     palette=cmap,
    #     join=False,
    # )
    # legend = ax.legend(ncol=1, loc="upper right", bbox_to_anchor=(1.2, 1))
    # legend.get_frame().set_linewidth(0.0)
    # legend.get_frame().set_alpha(0.0)

    # ax.set_yscale("linear")
    # ticklabels = ax.get_xticklabels()
    # for tick in ticklabels:
    #     tick.set_rotation(90)
    # set_size(4, 4 / golden_ratio)
    # fig.savefig("n_lhs_rmsd.pdf", bbox_inches="tight")

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # sns.pointplot(
    #     x="method",
    #     y="kldiv",
    #     order=[
    #         "exp",
    #         "exp-step",
    #         "expquad",
    #         "expquad-step",
    #         "mat32",
    #         "mat32-step",
    #         "mat52",
    #         "mat52-step",
    #         "lasso",
    #         "lasso-lars",
    #         "ridge",
    #     ],
    #     data=les_stats_df[les_stats_df.n_lhs == 500],
    #     ax=ax,
    #     palette=[colors[k] for k in [9]],
    #     join=False,
    # )
    # legend = ax.legend(ncol=1, loc="upper right", bbox_to_anchor=(1.2, 1))
    # legend.get_frame().set_linewidth(0.0)
    # legend.get_frame().set_alpha(0.0)
    # ax.set_ylim(1e-3, 1e0)
    # ax.set_yscale("log")
    # ticklabels = ax.get_xticklabels()
    # for tick in ticklabels:
    #     tick.set_rotation(90)
    # set_size(4, 4 / golden_ratio)
    # fig.savefig("n_lhs_500_kldiv.pdf", bbox_inches="tight")

    # # Select only the AS19 LHS 500 values
    # les_stats_df_500 = les_stats_df[les_stats_df.n_lhs == 500].reset_index(drop=True)
    # # Drop all non-GP regressors
    # indexNames = les_stats_df_500[
    #     (
    #         (les_stats_df_500["method"] == "lasso")
    #         | (les_stats_df_500["method"] == "lasso-lars")
    #         | (les_stats_df_500["method"] == "ridge")
    #     )
    # ].index
    # les_stats_df_500_gp = les_stats_df_500.drop(indexNames)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(
    #     ["Exponential", "Power-Exponential", "Matern(3/2)", "Matern(5/2)"],
    #     les_stats_df_500_gp["kldiv"].values[0::2],
    #     "ok",
    #     color=colors[9],
    #     label="default",
    # )
    # ax.plot(les_stats_df_500_gp["kldiv"].values[1::2], "or", color=colors[8], label="stepBIC")
    # legend = ax.legend(ncol=1, loc="upper right")
    # legend.get_frame().set_linewidth(0.0)
    # legend.get_frame().set_alpha(0.0)
    # ax.set_ylim(1e-3, 1e-1)
    # ax.set_yscale("log")
    # ticklabels = ax.get_xticklabels()
    # for tick in ticklabels:
    #     tick.set_rotation(90)
    # set_size(4, 4 / golden_ratio)
    # fig.savefig("n_lhs_500_kldiv_gp.pdf", bbox_inches="tight")

    # fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

    # sns.pointplot(
    #     x="method",
    #     y="kldiv",
    #     order=[
    #         "exp",
    #         "exp-step",
    #         "expquad",
    #         "expquad-step",
    #         "mat32",
    #         "mat32-step",
    #         "mat52",
    #         "mat52-step",
    #         "lasso",
    #         "lasso-lars",
    #         "ridge",
    #     ],
    #     hue="n_lhs",
    #     data=les_stats_df,
    #     ax=axes[0],
    #     palette=cmap,
    #     join=False,
    # )

    # sns.pointplot(
    #     x="method",
    #     y="rmsd",
    #     order=[
    #         "exp",
    #         "exp-step",
    #         "expquad",
    #         "expquad-step",
    #         "mat32",
    #         "mat32-step",
    #         "mat52",
    #         "mat52-step",
    #         "lasso",
    #         "lasso-lars",
    #         "ridge",
    #     ],
    #     hue="n_lhs",
    #     data=rmsd_df,
    #     ax=axes[1],
    #     palette=cmap,
    #     join=False,
    # )
    # legend = axes[0].legend(ncol=1, loc="upper right", bbox_to_anchor=(1.2, 1))
    # legend.get_frame().set_linewidth(0.0)
    # legend.get_frame().set_alpha(0.0)
    # legend = axes[1].legend(ncol=1, loc="upper right", bbox_to_anchor=(1.2, 1))
    # legend.get_frame().set_linewidth(0.0)
    # legend.get_frame().set_alpha(0.0)

    # axes[0].set_ylim(1e-3, 1e-0)
    # axes[0].set_yscale("log")
    # axes[1].set_yscale("linear")

    # axes[0].set_xlabel("")
    # #    axes[0].set_xticks([])

    # axes[0].set_ylabel("$D_{\mathrm{KL}}$ (nats)")
    # axes[1].set_ylabel("rmsd (cm SLE)")

    # ticklabels = axes[1].get_xticklabels()
    # for tick in ticklabels:
    #     tick.set_rotation(90)
    # set_size(4, 6 / golden_ratio)
    # fig.savefig("n_lhs_kldiv_rmsd.pdf", bbox_inches="tight")

    # fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

    # sns.pointplot(
    #     x="method",
    #     y="kldiv",
    #     order=["exp", "expquad", "mat32", "mat32-step", "mat52", "mat52-step"],
    #     hue="n_lhs",
    #     data=les_stats_df,
    #     ax=axes[0],
    #     palette=cmap,
    #     join=False,
    # )

    # sns.pointplot(
    #     x="method",
    #     y="rmsd",
    #     order=["exp", "expquad", "mat32", "mat52"],
    #     hue="n_lhs",
    #     data=rmsd_df,
    #     ax=axes[1],
    #     palette=cmap,
    #     join=False,
    # )
    # legend = axes[0].legend(ncol=1, loc="upper right", bbox_to_anchor=(1.2, 1))
    # legend.get_frame().set_linewidth(0.0)
    # legend.get_frame().set_alpha(0.0)
    # legend = axes[1].legend(ncol=1, loc="upper right", bbox_to_anchor=(1.2, 1))
    # legend.get_frame().set_linewidth(0.0)
    # legend.get_frame().set_alpha(0.0)

    # axes[0].set_ylim(1e-3, 1e-0)
    # axes[0].set_yscale("log")
    # axes[1].set_yscale("linear")

    # axes[0].set_xlabel("")
    # #    axes[0].set_xticks([])

    # axes[0].set_ylabel("$D_{\mathrm{KL}}$ (nats)")
    # axes[1].set_ylabel("rmsd (cm SLE)")

    # ticklabels = axes[1].get_xticklabels()
    # for tick in ticklabels:
    #     tick.set_rotation(90)
    # set_size(4, 6 / golden_ratio)
    # fig.savefig("n_lhs_kldiv_rmsd_gp.pdf", bbox_inches="tight")

    pctl = [5, 16, 50, 84, 95]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    kld = kl_divergence(P, Q_as19)
    m_qc = []
    m_yqc = []
    for n, n_lhs in enumerate(range(100, 600, 100)):
        for k, method in enumerate(["exp", "expquad", "mat32", "mat52"]):
            df = les_df[les_df.n_lhs == n_lhs]
            q = df[df.method == method]["Y_mean"].values

            if k == 0:
                sns.distplot(
                    q,
                    bins=bins,
                    hist=False,
                    hist_kws={"alpha": 0.3},
                    norm_hist=True,
                    kde=True,
                    kde_kws={"shade": False, "alpha": 1.0, "linewidth": 0.6},
                    ax=ax,
                    label=f"EM-{n_lhs}",
                    color=cmap[n],
                )
            else:
                sns.distplot(
                    q,
                    bins=bins,
                    hist=False,
                    hist_kws={"alpha": 0.3},
                    norm_hist=True,
                    kde=True,
                    kde_kws={"shade": False, "alpha": 1.0, "linewidth": 0.6},
                    ax=ax,
                    color=cmap[n],
                )

            qc = np.percentile(q, pctl)
            q_kernel = stats.gaussian_kde(np.squeeze(q), bw_method=2 / q.std(ddof=1))
            yqc = q_kernel.evaluate(qc)
            ax.vlines(
                qc,
                np.zeros_like(qc),
                yqc,
                linewidth=0.4,
                linestyle="solid",
                color=cmap[n],
                alpha=1.0,
                transform=ax.transData,
            )

    sns.distplot(
        q_as19,
        bins=bins,
        hist=False,
        hist_kws={"alpha": 0.3},
        norm_hist=True,
        kde=True,
        kde_kws={"shade": False, "alpha": 1.0, "linewidth": 1.0},
        ax=ax,
        color=cmap[-1],
        label="AS19",
    )
    sns.distplot(
        p,
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

    pc = np.percentile(p, pctl)
    p_kernel = stats.gaussian_kde(np.squeeze(p), bw_method=2 / p.std(ddof=1))
    ypc = p_kernel.evaluate(pc)
    ax.vlines(
        pc, np.zeros_like(pc), ypc, linewidth=0.6, linestyle="solid", color="k", alpha=1.0, transform=ax.transData
    )

    qc_as19 = np.percentile(q_as19, pctl)
    q_as19_kernel = stats.gaussian_kde(np.squeeze(q_as19), bw_method=2 / q_as19.std(ddof=1))
    ypc_as19 = p_kernel.evaluate(qc_as19)
    ax.vlines(
        qc_as19,
        np.zeros_like(qc_as19),
        ypc_as19,
        linewidth=0.6,
        linestyle="solid",
        color=cmap[-1],
        alpha=1.0,
        transform=ax.transData,
    )

    ax.set_xlim(-10, 50)
    ax.set_ylim(0)
    ax.set_ylabel("Probability (1)")
    ax.set_xlabel("Sea-level contribution (cm sea-level equivalent)")
    sns.despine()
    legend = ax.legend(ncol=1)
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    set_size(3.2, 3.2 / golden_ratio)

    outfile = "gp_emulators.pdf"
    print("Saving figure to {}".format(outfile))
    fig.savefig(outfile, bbox_inches="tight")

    ax.set_xlim(25, 40)
    ax.set_ylim(0, 0.025)

    set_size(3.2, 3.2 / golden_ratio)

    outfile = "gp_emulators_95th.pdf"
    print("Saving figure to {}".format(outfile))
    fig.savefig(outfile, bbox_inches="tight")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    m_qc = []
    m_yqc = []
    for n, n_lhs in enumerate(range(500, 600, 100)):
        for k, method in enumerate(["exp", "expquad", "mat32", "mat52"]):
            df = les_df[les_df.n_lhs == n_lhs]
            q = df[df.method == method]["Y_mean"].values
            Q = np.histogram(q, bins=bins, density=True)[0]
            kldiv = kl_divergence(P, Q)
            sns.distplot(
                q,
                bins=bins,
                hist=False,
                hist_kws={"alpha": 0.3},
                norm_hist=True,
                kde=True,
                kde_kws={"shade": False, "alpha": 1.0, "linewidth": 0.6},
                ax=ax,
                label=f"{method} ({kldiv:.4f})",
                color=cmap2[k],
            )

        for k, method in enumerate(["exp-step", "expquad-step", "mat32-step", "mat52-step"]):
            df = les_df[les_df.n_lhs == n_lhs]
            q = df[df.method == method]["Y_mean"].values
            Q = np.histogram(q, bins=bins, density=True)[0]
            kldiv = kl_divergence(P, Q)

            sns.distplot(
                q,
                bins=bins,
                hist=False,
                hist_kws={"alpha": 0.3},
                norm_hist=True,
                kde=True,
                kde_kws={"shade": False, "alpha": 1.0, "linewidth": 0.6, "linestyle": "dashed",},
                ax=ax,
                label=f"{method} ({kldiv:.4f})",
                color=cmap2[k],
            )

            qc = np.percentile(q, pctl)
            q_kernel = stats.gaussian_kde(np.squeeze(q), bw_method=2 / q.std(ddof=1))
            yqc = q_kernel.evaluate(qc)
            # ax.vlines(
            #     qc,
            #     np.zeros_like(qc),
            #     yqc,
            #     linewidth=0.4,
            #     linestyle="solid",
            #     color=cmap2[k],
            #     alpha=1.0,
            #     transform=ax.transData,
            # )

    Q = np.histogram(q, bins=bins, density=True)[0]
    kldiv = kl_divergence(P, Q_as19)
    sns.distplot(
        q_as19,
        bins=bins,
        hist=False,
        hist_kws={"alpha": 0.3},
        norm_hist=True,
        kde=True,
        kde_kws={"shade": False, "alpha": 1.0, "linewidth": 1.0},
        ax=ax,
        color=cmap[-1],
        label=f"AS19 ({kldiv:.4})",
    )
    sns.distplot(
        p,
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

    pc = np.percentile(p, pctl)
    p_kernel = stats.gaussian_kde(np.squeeze(p), bw_method=2 / p.std(ddof=1))
    ypc = p_kernel.evaluate(pc)
    # ax.vlines(
    #     pc, np.zeros_like(pc), ypc, linewidth=0.6, linestyle="solid", color="k", alpha=1.0, transform=ax.transData
    # )

    qc_as19 = np.percentile(q_as19, pctl)
    q_as19_kernel = stats.gaussian_kde(np.squeeze(q_as19), bw_method=2 / q_as19.std(ddof=1))
    ypc_as19 = p_kernel.evaluate(qc_as19)
    # ax.vlines(
    #     qc_as19,
    #     np.zeros_like(qc_as19),
    #     ypc_as19,
    #     linewidth=0.6,
    #     linestyle="solid",
    #     color=cmap2[-1],
    #     alpha=1.0,
    #     transform=ax.transData,
    # )

    ax.set_xlim(-10, 50)
    ax.set_ylim(0)
    ax.set_ylabel("Probability (1)")
    ax.set_xlabel("Sea-level contribution (cm sea-level equivalent)")
    sns.despine()
    legend = ax.legend(ncol=1)
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    set_size(3.2, 3.2 / golden_ratio)

    outfile = "all_emulators.pdf"
    print("Saving figure to {}".format(outfile))
    fig.savefig(outfile, bbox_inches="tight")

    ax.set_xlim(25, 40)
    ax.set_ylim(0, 0.025)

    set_size(3.2, 3.2 / golden_ratio)

    outfile = "all_emulators_95th.pdf"
    print("Saving figure to {}".format(outfile))
    fig.savefig(outfile, bbox_inches="tight")

    # # for k, method in enumerate(["exp", "expquad", "mat32", "mat52"]):
    # for k, method in enumerate(["expquad"]):

    #     qc = m_qc[k]
    #     yqc = m_yqc[k]
    #     ax.plot(
    #         [qc, qc], [0, yqc], linewidth=1.25, linestyle="dotted", color=cmap2[k], alpha=1.0, transform=ax.transData
    #     )

    # # pc = np.percentile(p, pctl)
    # # p_kernel = stats.gaussian_kde(np.squeeze(p), bw_method=2 / p.std(ddof=1))
    # # ypc = p_kernel.evaluate(pc)
    # # ax.plot([pc, pc], [0, ypc], linewidth=1.25, linestyle="dotted", color="k", alpha=1.0, transform=ax.transData)

    # # qc_as19 = np.percentile(q_as19, pctl)
    # # q_as19_kernel = stats.gaussian_kde(np.squeeze(q_as19), bw_method=2 / q_as19.std(ddof=1))
    # # ypc_as19 = p_kernel.evaluate(qc_as19)
    # # ax.plot(
    # #     [qc_as19, qc_as19],
    # #     [0, ypc_as19],
    # #     linewidth=1.25,
    # #     linestyle="dotted",
    # #     color=cmap[-1],
    # #     alpha=1.0,
    # #     transform=ax.transData,
    # # )
    # ax.set_xlim(25, 55)
    # ax.set_ylim(0, 0.018)
    # set_size(3.2, 3.2 / golden_ratio)
    # legend = ax.legend(ncol=1)
    # legend.get_frame().set_linewidth(0.0)
    # legend.get_frame().set_alpha(0.0)

    # outfile = "gp_emulators_high.pdf"
    # print("Saving figure to {}".format(outfile))
    # fig.savefig(outfile, bbox_inches="tight")

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # sns.pointplot(
    #     x="method",
    #     y="distance",
    #     hue="n_lhs",
    #     order=["exp", "exp-step", "expquad", "expquad-step", "mat32", "mat32-step", "mat52", "mat52-step",],
    #     data=distance_df,
    #     ax=ax,
    #     join=False,
    #     palette=cmap,
    # )
    # legend = ax.legend(ncol=1, loc="upper right", bbox_to_anchor=(1.2, 1))
    # legend.get_frame().set_linewidth(0.0)
    # legend.get_frame().set_alpha(0.0)

    # # ax.set_ylim(1e-3, 1e1)
    # # ax.set_yscale("log")
    # ticklabels = ax.get_xticklabels()
    # for tick in ticklabels:
    #     tick.set_rotation(90)
    # set_size(4, 4 / golden_ratio)
    # fig.savefig("loo_distance.pdf", bbox_inches="tight")
