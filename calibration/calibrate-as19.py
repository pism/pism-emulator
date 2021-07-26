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


def plot_historical(out_filename, df, df_ctrl, imbie):
    """
    Plot historical simulations and observations
    """

    def plot_signal(g):
        m_df = g[-1]
        x = m_df["Year"]
        y = m_df["Mass (Gt)"]

        return ax.plot(x, y, color=simulated_signal_color, linewidth=simulated_signal_lw)

    xmin = 2008
    xmax = 2020
    ymin = -10000
    ymax = 1000

    g = df.groupby(by="Year")["Mass (Gt)"]
    as19_median = g.quantile(0.50)
    as19_std = g.std()
    as19_low = g.quantile(0.05)
    as19_high = g.quantile(0.95)

    as19_ctrl_median = df_ctrl.groupby(by="Year")["Mass (Gt)"].quantile(0.50)

    fig = plt.figure(num="historical", clear=True)
    ax = fig.add_subplot(111)

    as19_ci = ax.fill_between(
        as19_median.index,
        as19_low,
        as19_high,
        color="0.6",
        alpha=1.0,
        linewidth=0.0,
        zorder=-11,
        label="AS19 90% c.i.",
    )

    as19_ci = ax.fill_between(
        as19_median.index,
        as19_low,
        as19_high,
        color="0.4",
        alpha=1.0,
        linewidth=0.0,
        zorder=-10,
        label="AS19 90% c.i.",
    )

    imbie_fill = ax.fill_between(
        imbie["Year"],
        imbie["Mass (Gt)"] - 1 * imbie["Mass uncertainty (Gt)"],
        imbie["Mass (Gt)"] + 1 * imbie["Mass uncertainty (Gt)"],
        color=imbie_sigma_color,
        alpha=0.5,
        linewidth=0,
    )
    imbie_fill.set_zorder(5)
    imbie_line = ax.plot(
        imbie["Year"],
        imbie["Mass (Gt)"],
        "-",
        color=imbie_signal_color,
        linewidth=imbie_signal_lw,
        label="Observed (IMBIE)",
    )

    l_es_median = ax.plot(
        as19_median.index,
        as19_median,
        color="k",
        linewidth=0.6,
        label="Median(Ensemble)",
    )
    l_ctrl_median = ax.plot(
        as19_ctrl_median.index,
        as19_ctrl_median,
        color="k",
        linewidth=0.6,
        linestyle="dashed",
        label="Median(CTRL)",
    )

    ax.axhline(0, color="k", linestyle="dotted", linewidth=0.6)

    legend = ax.legend(handles=[imbie_line[0], l_es_median[0], l_ctrl_median[0], as19_ci], loc="lower left")
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


def plot_historical_with_calib(out_filename, df, df_calib, df_ctrl, imbie):
    """
    Plot historical simulations and observations
    """

    xmin = 2008
    xmax = 2022
    ymin = -10000
    ymax = 1000

    as19_ctrl_median = df_ctrl.groupby(by="Year")["Mass (Gt)"].quantile(0.50)

    fig = plt.figure(num="historical", clear=True)
    ax = fig.add_subplot(111)

    g = df.groupby(by="Year")["Mass (Gt)"]
    as19_median = g.quantile(0.50)
    as19_low = g.quantile(0.05)
    as19_high = g.quantile(0.95)

    as19_ci = ax.fill_between(
        as19_median.index,
        as19_low,
        as19_high,
        color="0.6",
        alpha=0.5,
        linewidth=0.0,
        zorder=10,
        label="AS19 90% c.i.",
    )
    as19_ci.set_zorder(-11)

    g = df_calib.groupby(by="Year")["Mass (Gt)"]
    as19_calib_median = g.quantile(0.50)
    as19_calib_low = g.quantile(0.05)
    as19_calib_high = g.quantile(0.95)

    as19_ci_calib = ax.fill_between(
        as19_calib_median.index,
        as19_calib_low,
        as19_calib_high,
        color="0.4",
        alpha=0.5,
        linewidth=0.0,
        zorder=10,
        label="Calibrated 90% c.i.",
    )
    as19_ci_calib.set_zorder(-10)

    imbie_fill = ax.fill_between(
        imbie["Year"],
        imbie["Mass (Gt)"] - 1 * imbie["Mass uncertainty (Gt)"],
        imbie["Mass (Gt)"] + 1 * imbie["Mass uncertainty (Gt)"],
        color=imbie_sigma_color,
        alpha=0.5,
        linewidth=0,
    )
    imbie_fill.set_zorder(5)

    imbie_line = ax.plot(
        imbie["Year"],
        imbie["Mass (Gt)"],
        "-",
        color=imbie_signal_color,
        linewidth=imbie_signal_lw,
        label="Observed (IMBIE)",
    )

    l_es_median = ax.plot(
        as19_median.index,
        as19_median,
        color="k",
        linewidth=0.6,
        label="Median(AS19 Ensemble)",
    )
    l_es_calib_median = ax.plot(
        as19_calib_median.index,
        as19_calib_median,
        color="k",
        linewidth=0.8,
        label="Median(Calibrated Ensemble)",
    )
    l_ctrl_median = ax.plot(
        as19_ctrl_median.index,
        as19_ctrl_median,
        color="k",
        linewidth=0.6,
        linestyle="dashed",
        label="Median(AS19 CTRL)",
    )

    ax.axhline(0, color="k", linestyle="dotted", linewidth=0.6)

    legend = ax.legend(
        handles=[imbie_line[0], l_es_median[0], l_es_calib_median[0], l_ctrl_median[0], as19_ci, as19_ci_calib],
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


def plot_partioning(out_filename, df, df_calib, df_ctrl, imbie):

    fig, axs = plt.subplots(2, 1, sharex="col", figsize=[4.75, 3.5])
    fig.subplots_adjust(hspace=0.1, wspace=0.25)

    for k, v in enumerate(["SMB", "D"]):
        g = df.groupby(by="Year")[f"{v} (Gt/yr)"]
        as19_median = g.quantile(0.50)
        as19_std = g.std()
        as19_low = g.quantile(0.05)
        as19_high = g.quantile(0.95)
        as19_ctrl_median = df_ctrl.groupby(by="Year")[f"{v} (Gt/yr)"].quantile(0.50)

        as19_ci = axs[k].fill_between(
            as19_median.index,
            as19_low,
            as19_high,
            color="0.6",
            alpha=0.5,
            linewidth=0.0,
            zorder=-11,
            label="AS19 90% c.i.",
        )

        g = df_calib.groupby(by="Year")[f"{v} (Gt/yr)"]
        as19_median = g.quantile(0.50)
        as19_std = g.std()
        as19_low = g.quantile(0.05)
        as19_high = g.quantile(0.95)

        as19_calib_ci = axs[k].fill_between(
            as19_median.index,
            as19_low,
            as19_high,
            color="0.4",
            alpha=0.5,
            linewidth=0.0,
            zorder=-10,
            label="Calibrated 90% c.i.",
        )

        axs[k].fill_between(
            imbie["Year"],
            imbie[f"{v} (Gt/yr)"] - 1 * imbie[f"{v} uncertainty (Gt/yr)"],
            imbie[f"{v} (Gt/yr)"] + 1 * imbie[f"{v} uncertainty (Gt/yr)"],
            color=imbie_sigma_color,
            alpha=0.5,
            linewidth=0,
        )

        axs[k].plot(
            imbie["Year"],
            imbie[f"{v} (Gt/yr)"],
            color=imbie_signal_color,
            linewidth=imbie_signal_lw,
            linestyle="solid",
        )

        l_es_median = axs[k].plot(
            as19_median.index,
            as19_median,
            color="k",
            linewidth=0.6,
            label="Median(Ensemble)",
        )
        l_ctrl_median = axs[k].plot(
            as19_ctrl_median.index,
            as19_ctrl_median,
            color="k",
            linewidth=0.6,
            linestyle="dashed",
            label="Median(CTRL)",
        )
        axs[k].set_ylabel(f"{v} (Gt/yr)")

    imbie_line = mlines.Line2D([], [], color=imbie_signal_color, linewidth=imbie_signal_lw, label="IMBIE")

    legend = axs[1].legend(
        handles=[imbie_line, l_es_median[0], l_ctrl_median[0], as19_ci, as19_calib_ci], loc="lower left", ncol=2
    )
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    axs[k].set_xlim(2010, 2020)
    axs[0].set_ylim(-250, 750)
    axs[1].set_ylim(-1500, 0)

    set_size(5, 3)
    fig.savefig(out_filename, bbox_inches="tight")


def plot_sle_pdfs(out_filename, df):

    # Draw a nested violinplot and split the violins for easier comparison
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.violinplot(
        data=df,
        y="RCP",
        x="SLE (cm)",
        order=rcps,
        hue="Ensemble",
        hue_order=["Calibrated", "AS19"],
        split=True,
        cut=0.0,
        inner="quart",
        palette=["0.6", "0.8"],
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


secpera = 3.15569259747e7

simulated_signal_lw = 0.3
simulated_signal_color = "#bdbdbd"
imbie_signal_lw = 0.75
imbie_signal_color = "#005a32"
imbie_sigma_color = "#a1d99b"

gt2cmSLE = 1.0 / 362.5 / 10.0

rcps = [26, 45, 85]
rcp_col_dict = {85: "#990002", 45: "#5492CD", 26: "#003466"}
rcp_shade_col_dict = {85: "#F4A582", 45: "#92C5DE", 26: "#4393C3"}
rcp_dict = {26: "RCP 2.6", 45: "RCP 4.5", 85: "RCP 8.5"}

calibration_start = 2010
calibration_end = 2020
proj_start = 2008

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

    plot_partioning("historical_partioning_as19.pdf", as19, as19_calib, as19_ctrl, imbie)
    plot_historical("historical_as19.pdf", as19, as19_ctrl, imbie)
    plot_historical_with_calib("historical_calib.pdf", as19, as19_calib, as19_ctrl, imbie)

    as19_2100 = as19[as19["Year"] == 2100]
    numeric_cols = as19_2100.select_dtypes(exclude="number")
    as19_2100.drop(numeric_cols, axis=1, inplace=True)

    as19_calib_2100 = as19_calib[as19_calib["Year"] == 2100]
    numeric_cols = as19_calib_2100.select_dtypes(exclude="number")
    as19_calib_2100.drop(numeric_cols, axis=1, inplace=True)

    as19_2100["Ensemble"] = "AS19"
    as19_calib_2100["Ensemble"] = "Calibrated"
    as19_all_2100 = pd.concat([as19_2100, as19_calib_2100]).astype({"Ensemble": str})
    as19_all_2100["ID"] = [
        f"rcp_dict{k}, {l} " for k, l in zip(as19_all_2100["RCP"].values, as19_all_2100["Ensemble"].values)
    ]

    plot_sle_pdfs("sle_pdf_2100.pdf", as19_all_2100)

    from scipy.stats import norm

    imbie_mean = imbie_calib_period.mean()
    imbie_std = imbie_calib_period.std()

    as19_period_mean = as19_period.groupby(by=["RCP", "Experiment"]).mean().reset_index()
    as19_period_std = as19_period.groupby(by=["RCP", "Experiment"]).std().reset_index()
    as19_calib_period_mean = as19_calib_period.groupby(by=["RCP", "Experiment"]).mean().reset_index()
    as19_calib_period_std = as19_calib_period.groupby(by=["RCP", "Experiment"]).std().reset_index()

    fig, axs = plt.subplots(3, 3, sharex="col", sharey="row", figsize=[6.2, 5])
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for k, v in enumerate(["Mass (Gt)", "SMB (Gt/yr)", "D (Gt/yr)"]):
        imbie_gauss_dist = norm(imbie_mean[v], imbie_std[v])

        for l, rcp in enumerate(rcps):

            as19_gauss_dist = norm(imbie_mean[v], imbie_std[v])

            v_as19_period_mean = as19_period_mean[as19_period_mean["RCP"] == rcp][v]
            v_as19_period_std = as19_period_std[as19_period_std["RCP"] == rcp][v]

            v_as19_calib_period_mean = as19_calib_period_mean[as19_calib_period_mean["RCP"] == rcp][v]
            v_as19_calib_period_std = as19_calib_period_std[as19_calib_period_std["RCP"] == rcp][v]

            weights_as19 = imbie_gauss_dist.pdf(v_as19_period_mean) / imbie_gauss_dist.pdf(v_as19_period_mean).sum()
            weights_calibrated = (
                imbie_gauss_dist.pdf(v_as19_calib_period_mean) / imbie_gauss_dist.pdf(v_as19_calib_period_mean).sum()
            )
            l_we = sns.kdeplot(
                data=as19_calib_2100[as19_calib_2100["RCP"] == rcp],
                x="SLE (cm)",
                weights=weights_calibrated,
                color="k",
                linestyle="solid",
                linewidth=0.5,
                ax=axs[k, l],
                label="Weighted",
            )
            l_ca = sns.kdeplot(
                data=as19_calib_2100[as19_calib_2100["RCP"] == rcp],
                x="SLE (cm)",
                color="k",
                linestyle="dotted",
                linewidth=0.5,
                ax=axs[k, l],
                label="Calibrated",
            )
            l_as19 = sns.kdeplot(
                data=as19_2100[as19_2100["RCP"] == rcp],
                x="SLE (cm)",
                ax=axs[k, l],
                color="k",
                linestyle="dashed",
                linewidth=0.5,
                label="AS19",
            )
            # sns.kdeplot(
            #     data=as19_2100[as19_2100["RCP"] == rcp],
            #     x="SLE (cm)",
            #     weights=weights_as19,
            #     color=rcp_col_dict[rcp],
            #     linestyle="dashed",
            #     ax=axs[k, l],
            #     label="AS19 Weighted Ensemble",
            # )
            axs[-1, l].set(xlabel=None)
        axs[k, 0].set(ylabel=None)
    [axs[0, l].set_title(rcp_dict[rcp], color=rcp_col_dict[rcp]) for l, rcp in enumerate(rcps)]
    legend = axs[2, 1].legend()
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    fig.supxlabel("Contribution to sea-level since 2008 (cm SLE)")
    fig.supylabel("Density")
    fig.savefig("calibrated.pdf", bbox_inches="tight")

    # Initialize the FacetGrid object

    g = sns.FacetGrid(as19_all_2100, row="ID", hue="RCP", aspect=10, height=1, palette=rcp_shade_col_dict)

    # Draw the densities in a few steps
    g.map(sns.violinplot, "SLE (cm)", bw_adjust=10, clip_on=False, inner="quartile", fill=True, alpha=0.5)
    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, 0.2, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)

    g.map(label, "ID")

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=0)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
