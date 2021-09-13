#!/usr/bin/env python

# Copyright (C) 2020 Andy Aschwanden

import matplotlib.lines as mlines
from netCDF4 import Dataset as NC
import numpy as np
import os
import pylab as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import statsmodels.api as sm

import matplotlib as mpl
import matplotlib.cm as cmx
import matplotlib.colors as colors


from pismemulator.utils import load_imbie
from scipy.interpolate import interp1d


def toDecimalYear(date):
    """
    Convert date to decimal year

    %In: toDecimalYear(datetime(2020, 10, 10))
    %Out: 2020.7732240437158

    """

    from datetime import datetime
    import time

    year = date.year
    startOfThisYear = datetime(year=year, month=1, day=1)
    startOfNextYear = datetime(year=year + 1, month=1, day=1)
    yearElapsed = (date - startOfThisYear).total_seconds()
    yearDuration = (startOfNextYear - startOfThisYear).total_seconds()
    fraction = yearElapsed / yearDuration

    return date.year + fraction


def set_size(w, h, ax=None):
    """w, h: width, height in inches"""

    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def plot_historical_with_calib(out_filename, df, df_ctrl, imbie):
    """
    Plot historical simulations and observations
    """

    xmin = 2008
    xmax = 2020
    ymin = -7500
    ymax = 500

    as19_ctrl_median = df_ctrl.groupby(by="Year")["Mass (Gt)"].quantile(0.50)

    fig = plt.figure(num="historical", clear=True)
    ax = fig.add_subplot(111)

    as19 = df[df["Ensemble"] == "AS19"]
    g = as19.groupby(by="Year")["Mass (Gt)"]
    as19_median = g.quantile(0.50)
    as19_low = g.quantile(0.05)
    as19_high = g.quantile(0.95)

    as19_ci = ax.fill_between(
        as19_median.index,
        as19_low,
        as19_high,
        color="0.8",
        alpha=1.0,
        linewidth=0.0,
        zorder=-11,
        label="AS19 90% c.i.",
    )
    as19_ci.set_zorder(-11)

    calib = df[df["Ensemble"] == "Calibrated"]
    g = calib.groupby(by="Year")["Mass (Gt)"]
    as19_calib_median = g.quantile(0.50)
    as19_calib_low = g.quantile(0.05)
    as19_calib_high = g.quantile(0.95)

    as19_ci_calib = ax.fill_between(
        as19_calib_median.index,
        as19_calib_low,
        as19_calib_high,
        color="0.6",
        alpha=1.0,
        linewidth=0.0,
        zorder=-10,
        label="Calibrated 90% c.i.",
    )

    resampled = df[df["Ensemble"] == "Resampled"]
    g = resampled.groupby(by="Year")["Mass (Gt)"]
    as19_resampled_median = g.quantile(0.50)
    as19_resampled_low = g.quantile(0.05)
    as19_resampled_high = g.quantile(0.95)

    as19_ci_resampled = ax.fill_between(
        as19_resampled_median.index,
        as19_resampled_low,
        as19_resampled_high,
        color="0.4",
        alpha=1.0,
        linewidth=0.0,
        zorder=-9,
        label="Resampled 90% c.i.",
    )

    imbie_ci = ax.fill_between(
        imbie["Year"],
        imbie["Mass (Gt)"] - 1 * imbie["Mass uncertainty (Gt)"],
        imbie["Mass (Gt)"] + 1 * imbie["Mass uncertainty (Gt)"],
        color=imbie_sigma_color,
        alpha=0.75,
        linewidth=0,
        label="Observational uncertainty",
    )
    imbie_ci.set_zorder(5)

    imbie_line = ax.plot(
        imbie["Year"],
        imbie["Mass (Gt)"],
        "-",
        color=imbie_signal_color,
        linewidth=signal_lw,
        label="Observed (IMBIE)",
        zorder=20,
    )

    l_es_median = ax.plot(
        as19_median.index,
        as19_median,
        color="#fd8d3c",
        linewidth=signal_lw,
        label="Median(AS19 Ensemble)",
    )
    l_es_calib_median = ax.plot(
        as19_calib_median.index,
        as19_calib_median,
        color="#6baed6",
        linewidth=signal_lw,
        label="Median(Calibrated Ensemble)",
    )
    l_es_resampled_median = ax.plot(
        as19_resampled_median.index,
        as19_resampled_median,
        color="k",
        linewidth=signal_lw,
        label="Median(Resampled Ensemble)",
    )
    l_ctrl_median = ax.plot(
        as19_ctrl_median.index,
        as19_ctrl_median,
        color="#fd8d3c",
        linestyle="dashed",
        linewidth=signal_lw,
        label="Median(AS19 CTRL)",
    )

    ax.axhline(0, color="k", linestyle="dotted", linewidth=0.6)

    legend = ax.legend(
        handles=[
            imbie_line[0],
            l_ctrl_median[0],
            l_es_median[0],
            l_es_calib_median[0],
            l_es_resampled_median[0],
            imbie_ci,
            as19_ci,
            as19_ci_calib,
            as19_ci_resampled,
        ],
        loc="lower left",
    )
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    ax.set_xlabel("Year")
    ax.set_ylabel(f"Cumulative mass change\nsince {proj_start} (Gt)")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax_sle = ax.twinx()
    ax_sle.set_ylabel(f"Contribution to sea-level \nsince {proj_start} (cm SLE)")
    ax_sle.set_ylim(-ymin * gt2cmSLE, -ymax * gt2cmSLE)

    set_size(5, 2)
    fig.savefig(out_filename, bbox_inches="tight")


def plot_partitioning(out_filename, df, df_ctrl, imbie):

    fig, axs = plt.subplots(2, 1, sharex="col", figsize=[4.75, 3.5])
    fig.subplots_adjust(hspace=0.1, wspace=0.25)

    for k, v in enumerate(["SMB", "D"]):
        as19 = df[df["Ensemble"] == "AS19"]
        g = as19.groupby(by="Year")[f"{v} (Gt/yr)"]
        as19_median = g.quantile(0.50)
        as19_std = g.std()
        as19_low = g.quantile(0.05)
        as19_high = g.quantile(0.95)
        as19_ctrl_median = df_ctrl.groupby(by="Year")[f"{v} (Gt/yr)"].quantile(0.50)

        as19_ci = axs[k].fill_between(
            as19_median.index,
            as19_low,
            as19_high,
            color="0.8",
            alpha=01.0,
            linewidth=0.0,
            zorder=-11,
            label="AS19 90% c.i.",
        )

        calib = df[df["Ensemble"] == "Calibrated"]
        g = calib.groupby(by="Year")[f"{v} (Gt/yr)"]
        as19_calib_median = g.quantile(0.50)
        as19_calib_std = g.std()
        as19_calib_low = g.quantile(0.05)
        as19_calib_high = g.quantile(0.95)

        as19_calib_ci = axs[k].fill_between(
            as19_calib_median.index,
            as19_calib_low,
            as19_calib_high,
            color="0.6",
            alpha=1.0,
            linewidth=0.0,
            zorder=-10,
            label="Calibrated 90% c.i.",
        )

        resampled = df[df["Ensemble"] == "Resampled"]
        g = resampled.groupby(by="Year")[f"{v} (Gt/yr)"]
        as19_resampled_median = g.quantile(0.50)
        as19_resampled_std = g.std()
        as19_resampled_low = g.quantile(0.05)
        as19_resampled_high = g.quantile(0.95)

        as19_resampled_ci = axs[k].fill_between(
            as19_resampled_median.index,
            as19_resampled_low,
            as19_resampled_high,
            color="0.4",
            alpha=1.0,
            linewidth=0.0,
            zorder=-10,
            label="Resampled 90% c.i.",
        )

        imbie_ci = axs[k].fill_between(
            imbie["Year"],
            imbie[f"{v} (Gt/yr)"] - 1 * imbie[f"{v} uncertainty (Gt/yr)"],
            imbie[f"{v} (Gt/yr)"] + 1 * imbie[f"{v} uncertainty (Gt/yr)"],
            color=imbie_sigma_color,
            alpha=0.5,
            linewidth=0,
            label="Observational uncertainty",
        )

        axs[k].plot(
            imbie["Year"],
            imbie[f"{v} (Gt/yr)"],
            color=imbie_signal_color,
            linewidth=signal_lw,
            linestyle="solid",
        )

        l_es_median = axs[k].plot(
            as19_median.index,
            as19_median,
            color="#fd8d3c",
            linewidth=signal_lw,
            label="Median(AS19 Ensemble)",
        )
        l_ctrl_median = axs[k].plot(
            as19_ctrl_median.index,
            as19_ctrl_median,
            color="#fd8d3c",
            linewidth=signal_lw,
            linestyle="dashed",
            label="Median(AS19 CTRL)",
        )
        l_es_calib_median = axs[k].plot(
            as19_calib_median.index,
            as19_calib_median,
            color="#6baed6",
            linewidth=signal_lw,
            label="Median(Calibrated Ensemble)",
        )
        l_es_resampled_median = axs[k].plot(
            as19_resampled_median.index,
            as19_resampled_median,
            color="k",
            linewidth=signal_lw,
            label="Median(Calibrated Ensemble)",
        )
        axs[k].set_ylabel(f"{v} (Gt/yr)")

    imbie_line = mlines.Line2D([], [], color=imbie_signal_color, linewidth=signal_lw, label="Observed (IMBIE)")

    legend = axs[0].legend(
        handles=[
            imbie_line,
            l_ctrl_median[0],
            l_es_median[0],
            l_es_calib_median[0],
            l_es_resampled_median[0],
            imbie_ci,
            as19_ci,
            as19_calib_ci,
            as19_resampled_ci,
        ],
        loc="lower left",
        ncol=3,
    )
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    axs[k].set_xlim(2010, 2020)
    axs[0].set_ylim(-500, 1000)
    axs[1].set_ylim(-1500, 0)

    set_size(5, 3)
    fig.savefig(out_filename, bbox_inches="tight")


def plot_sle_pdfs(out_filename, df, year=2100):

    df = df[df["Year"] == 2100]
    # Draw a nested violinplot and split the violins for easier comparison
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.violinplot(
        data=df,
        y="RCP",
        x="SLE (cm)",
        order=rcps,
        hue="Ensemble",
        hue_order=["Resampled", "AS19"],
        split=True,
        cut=0.0,
        inner="quart",
        palette=["0.4", "0.8"],
        linewidth=1,
        orient="h",
        ax=ax,
    )
    plt.title("SLE PDF at 2100")
    set_size(5, 2.5)
    fig.savefig(out_filename, bbox_inches="tight")


def load_df(respone_file, samples_file):

    response = pd.read_csv(respone_file)
    response["SLE (cm)"] = -response["Mass (Gt)"] / 362.5 / 10
    response = response.astype({"RCP": int})
    samples = pd.read_csv(samples_file).rename(columns={"id": "Experiment"})
    return pd.merge(response, samples, on="Experiment")


def resample_ensemble_by_data(imbie_calib_period, as19_calib_period, rcps, fudge_factor=3.0, verbose=False):
    imbie_interp_mean = interp1d(imbie_calib_period["Year"], imbie_calib_period["Mass (Gt)"])
    imbie_interp_std = interp1d(imbie_calib_period["Year"], imbie_calib_period["Mass uncertainty (Gt)"])
    resampled_list = []
    for rcp in rcps:
        log_likes = []
        experiments = np.unique(as19_calib_period["Experiment"])
        evals = []
        for i in experiments:
            exp_ = as19_calib_period[(as19_calib_period["Experiment"] == i) & (as19_calib_period["RCP"] == rcp)]
            log_like = 0.0
            for year, exp_mass in zip(exp_["Year"], exp_["Mass (Gt)"]):
                try:
                    imbie_mass = imbie_interp_mean(year)
                    imbie_std = imbie_interp_std(year) * fudge_factor
                    log_like -= 0.5 * ((exp_mass - imbie_mass) / imbie_std) ** 2 + 0.5 * np.log(
                        2 * np.pi * imbie_std ** 2
                    )
                except ValueError:
                    pass
            if log_like != 0:
                evals.append(i)
                log_likes.append(log_like)
                if verbose:
                    print(f"{rcp_dict[rcp]}, Experiment {i:.0f}: {log_like:.2f}")
        experiments = np.array(evals)
        w = np.array(log_likes)
        w -= w.mean()
        weights = np.exp(w)
        weights /= weights.sum()
        resampled_experiments = np.random.choice(experiments, 500, p=weights)
        new_frame = []
        for i in resampled_experiments:
            new_frame.append(as19_calib[(as19_calib["Experiment"] == i) & (as19_calib["RCP"] == rcp)])
        as19_resampled = pd.concat(new_frame)
        resampled_list.append(as19_resampled)

    as19_resampled_26, as19_resampled_45, as19_resample_85 = resampled_list
    as19_resampled = pd.concat(resampled_list)
    return as19_resampled


signal_lw = 0.75
imbie_signal_color = "#238b45"
imbie_sigma_color = "#a1d99b"

secpera = 3.15569259747e7
gt2cmSLE = 1.0 / 362.5 / 10.0

rcps = [26, 45, 85]
rcp_col_dict = {85: "#990002", 45: "#5492CD", 26: "#003466"}
rcp_shade_col_dict = {85: "#F4A582", 45: "#92C5DE", 26: "#4393C3"}
rcp_dict = {26: "RCP 2.6", 45: "RCP 4.5", 85: "RCP 8.5"}
palette_dict = {"AS19": "0.75", "Calibrated": "0.5", "Resampled": "0.25"}

# cm = sns.color_palette("ch:s=-.2,r=.6", n_colors=6, as_cmap=False).as_hex()
# cm = sns.color_palette("flare", n_colors=3, as_cmap=False).as_hex()
# palette_dict = {"AS19": cm[0], "Calibrated": cm[1], "Resampled": cm[2]}

calibration_start = 2010
calibration_end = 2020
proj_start = 2008

fontsize = 6
lw = 0.65
aspect_ratio = 0.35
markersize = 2

params = {
    "backend": "ps",
    "axes.linewidth": 0.25,
    "lines.linewidth": lw,
    "axes.labelsize": fontsize,
    "font.size": fontsize,
    "xtick.direction": "in",
    "xtick.labelsize": fontsize,
    "xtick.major.size": 2.5,
    "xtick.major.width": 0.25,
    "ytick.direction": "in",
    "ytick.labelsize": fontsize,
    "ytick.major.size": 2.5,
    "ytick.major.width": 0.25,
    "legend.fontsize": fontsize,
    "lines.markersize": markersize,
    "font.size": fontsize,
}

plt.rcParams.update(params)


if __name__ == "__main__":

    # Load AS19 (original LES)
    as19 = load_df("../data/as19/aschwanden_et_al_2019_les_2008_norm.csv.gz", "../data/samples/lhs_samples_500.csv")
    # Load AS19 (original CTRL)
    as19_ctrl = load_df("../data/as19/aschwanden_et_al_2019_ctrl.csv.gz", "../data/samples/lhs_control.csv")
    # Load AS19 (with calibrated ice dynamics)
    as19_calib = load_df(
        "../data/as19/aschwanden_et_al_2019_mc_2008_norm.csv.gz", "../data/samples/lhs_plus_mc_samples.csv"
    )

    as19_time = (as19["Year"] >= calibration_start) & (as19["Year"] <= calibration_end)
    as19_period = as19[as19_time]
    as19_calib_time = (as19_calib["Year"] >= calibration_start) & (as19_calib["Year"] <= calibration_end)
    as19_calib_period = as19_calib[as19_calib_time]

    imbie = load_imbie()
    imbie_calib_time = (imbie["Year"] >= calibration_start) & (imbie["Year"] <= calibration_end)
    imbie_calib_period = imbie[imbie_calib_time]

    as19_resampled = resample_ensemble_by_data(imbie_calib_period, as19_calib_period, rcps)

    as19["Ensemble"] = "AS19"
    as19_calib["Ensemble"] = "Calibrated"
    as19_resampled["Ensemble"] = "Resampled"
    all_df = (
        pd.concat([as19, as19_calib, as19_resampled])
        .drop_duplicates(subset=None, keep="first", inplace=False)
        .reset_index()
    )

    year = 2100
    plot_sle_pdfs(f"sle_pdf_resampled_{year}.pdf", all_df, year=year)

    plot_partitioning("historical_partitioning.pdf", all_df, as19_ctrl, imbie)
    plot_historical_with_calib("historical.pdf", all_df, as19_ctrl, imbie)

    all_2100_df = all_df[(all_df["Year"] == year)]

    fig, axs = plt.subplots(4, 4, figsize=[6.2, 6.2])
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    sns.histplot(
        data=all_2100_df,
        x="GCM",
        hue="Ensemble",
        common_norm=False,
        palette=palette_dict.values(),
        bins=[-0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25],
        stat="density",
        multiple="dodge",
        linewidth=0.8,
        ax=axs[0, 0],
    )

    sns.kdeplot(
        data=all_2100_df,
        x="PRS",
        hue="Ensemble",
        common_grid=False,
        common_norm=False,
        palette=palette_dict.values(),
        linewidth=0.8,
        ax=axs[1, 0],
    )
    sns.kdeplot(
        data=all_2100_df,
        x="FICE",
        hue="Ensemble",
        common_grid=False,
        common_norm=False,
        palette=palette_dict.values(),
        linewidth=0.8,
        ax=axs[0, 1],
    )
    sns.kdeplot(
        data=all_2100_df,
        x="FSNOW",
        hue="Ensemble",
        common_grid=False,
        common_norm=False,
        palette=palette_dict.values(),
        linewidth=0.8,
        ax=axs[1, 1],
    )
    sns.kdeplot(
        data=all_2100_df,
        x="RFR",
        hue="Ensemble",
        common_grid=False,
        common_norm=False,
        palette=palette_dict.values(),
        linewidth=0.8,
        ax=axs[2, 1],
    )
    sns.histplot(
        data=all_2100_df,
        x="OCM",
        hue="Ensemble",
        common_norm=False,
        palette=palette_dict.values(),
        bins=[-1.25, -0.75, -0.25, 0.25, 0.75, 1.25],
        stat="density",
        multiple="dodge",
        linewidth=0.8,
        ax=axs[0, 2],
    )
    sns.histplot(
        data=all_2100_df,
        x="OCS",
        hue="Ensemble",
        common_norm=False,
        palette=palette_dict.values(),
        bins=[-1.25, -0.75, -0.25, 0.25, 0.75, 1.25],
        stat="density",
        multiple="dodge",
        linewidth=0.8,
        ax=axs[1, 2],
    )
    sns.histplot(
        data=all_2100_df,
        x="TCT",
        hue="Ensemble",
        common_norm=False,
        palette=palette_dict.values(),
        bins=[-1.25, -0.75, -0.25, 0.25, 0.75, 1.25],
        stat="density",
        multiple="dodge",
        linewidth=0.8,
        ax=axs[2, 2],
    )
    sns.kdeplot(
        data=all_2100_df,
        x="VCM",
        hue="Ensemble",
        common_grid=False,
        common_norm=False,
        palette=palette_dict.values(),
        linewidth=0.8,
        ax=axs[3, 2],
    )
    sns.kdeplot(
        data=all_2100_df,
        x="SIAE",
        hue="Ensemble",
        common_grid=False,
        common_norm=False,
        palette=palette_dict.values(),
        linewidth=0.8,
        ax=axs[0, 3],
    )
    sns.kdeplot(
        data=all_2100_df,
        x="PPQ",
        hue="Ensemble",
        common_grid=False,
        common_norm=False,
        palette=palette_dict.values(),
        linewidth=0.8,
        ax=axs[1, 3],
    )

    for ax, col in zip(axs[0], ["Climate", "Surface", "Ocean", "Dynamics"]):
        ax.set_title(col)

    axs[2, 0].set_axis_off()
    axs[3, 0].set_axis_off()
    axs[3, 1].set_axis_off()
    axs[2, 3].set_axis_off()
    axs[3, 3].set_axis_off()

    for ax in axs.reshape(-1):
        ax.legend([], [], frameon=False)

    l_as19 = mlines.Line2D([], [], color=palette_dict["AS19"], linewidth=0.8, label="AS19")
    l_calib = mlines.Line2D([], [], color=palette_dict["Calibrated"], linewidth=0.8, label="Calibrated")
    l_resampled = mlines.Line2D([], [], color=palette_dict["Resampled"], linewidth=0.8, label="Resampled")
    legend = axs[2, 0].legend(handles=[l_as19, l_calib, l_resampled], loc="center left", title="Ensemble")
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    fig.tight_layout()
    fig.savefig("calibrated_kde.pdf")
