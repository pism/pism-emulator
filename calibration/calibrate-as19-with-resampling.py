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


def plot_historical(
    out_filename,
    simulated=None,
    observed=None,
    ensembles=["AS19", "Resampled"],
    quantiles=[0.05, 0.95],
    sigma=1,
    simulated_ctrl=None,
):
    """
    Plot historical simulations and observations
    """

    xmin = 2008
    xmax = 2020
    ymin = -10000
    ymax = 500

    credibility_interval = int(np.round((quantiles[-1] - quantiles[0]) * 100))

    fig = plt.figure(num="historical", clear=True)
    ax = fig.add_subplot(111)

    legend_handles = []
    if simulated is not None:
        for ens in ensembles:
            sim = simulated[simulated["Ensemble"] == ens]
            g = sim.groupby(by="Year")["Mass (Gt)"]
            sim_median = g.quantile(0.50)
            sim_low = g.quantile(quantiles[0])
            sim_high = g.quantile(quantiles[-1])

            l_es_median = ax.plot(
                sim_median.index,
                sim_median,
                color=ts_median_palette_dict[ens],
                linewidth=signal_lw,
                label=f"Median({ens} Ensemble)",
            )
            legend_handles.append(l_es_median[0])
            ci = ax.fill_between(
                sim_median.index,
                sim_low,
                sim_high,
                color=ts_fill_palette_dict[ens],
                alpha=1.0,
                linewidth=0.0,
                zorder=-11,
                label=f"{ens} {credibility_interval}% c.i.",
            )
            legend_handles.append(ci)

    if observed is not None:
        obs_line = ax.plot(
            observed["Year"],
            observed["Mass (Gt)"],
            "-",
            color=obs_signal_color,
            linewidth=signal_lw,
            label="Observed (IMBIE)",
            zorder=20,
        )
        legend_handles.append(obs_line[0])
        obs_ci = ax.fill_between(
            observed["Year"],
            observed["Mass (Gt)"] - sigma * observed["Mass uncertainty (Gt)"],
            observed["Mass (Gt)"] + sigma * observed["Mass uncertainty (Gt)"],
            color=obs_sigma_color,
            alpha=0.75,
            linewidth=0,
            zorder=5,
            label=f"Observational uncertainty ({sigma}-sigma)",
        )
        legend_handles.append(obs_ci)

    ax.axhline(0, color="k", linestyle="dotted", linewidth=0.6)

    legend = ax.legend(
        handles=legend_handles,
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


def plot_projection(out_filename, simulated=None, ensemble="Resampled", quantiles=[0.05, 0.95], violins=False):
    """
    Plot historical simulations and observations
    """

    xmin = 2008
    xmax = 2100
    ymax = 40
    ymin = 0

    if violins:
        fig, axs = plt.subplots(
            1,
            2,
            sharey="col",
            figsize=[6, 3],
            gridspec_kw=dict(width_ratios=[8, 1]),
        )
        fig.subplots_adjust(hspace=0.1, wspace=0.05)
        ax = axs[0]
    else:
        fig = plt.figure(num="historical", clear=True)
        ax = fig.add_subplot(111)

    legend_handles = []
    if simulated is not None:
        for rcp in rcps:
            sim = simulated[(simulated["Ensemble"] == ensemble) & (simulated["RCP"] == rcp)]
            g = sim.groupby(by="Year")["SLE (cm)"]
            sim_median = g.quantile(0.50)

            l_es_median = ax.plot(
                sim_median.index,
                sim_median,
                color=rcp_col_dict[rcp],
                linewidth=signal_lw,
                label=f"{rcp_dict[rcp]}",
            )
            legend_handles.append(l_es_median[0])

            credibility_interval = int(np.round((quantiles[-1] - quantiles[0]) * 100))
            sim_low = g.quantile(quantiles[0])
            sim_high = g.quantile(quantiles[-1])
            ci = ax.fill_between(
                sim_median.index,
                sim_low,
                sim_high,
                color=rcp_shade_col_dict[rcp],
                alpha=0.4,
                linewidth=0.5,
                zorder=-11,
                label=f"{rcp_dict[rcp]} {credibility_interval}% c.i.",
            )
            legend_handles.append(ci)
            if len(quantiles) == 4:
                credibility_interval = int(np.round((quantiles[-2] - quantiles[1]) * 100))
                sim_low = g.quantile(quantiles[1])
                sim_high = g.quantile(quantiles[-2])
                ci = ax.fill_between(
                    sim_median.index,
                    sim_low,
                    sim_high,
                    color=rcp_shade_col_dict[rcp],
                    alpha=1.0,
                    linewidth=0.5,
                    zorder=-11,
                    label=f"{rcp_dict[rcp]} {credibility_interval}% c.i.",
                )
                legend_handles.append(ci)

    legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
    )
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    ax.set_xlabel("Year")
    ax.set_ylabel(f"Contribution to sea-level\nsince {proj_start} (cm SLE)")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if violins:
        sns.violinplot(
            data=simulated[(simulated["Ensemble"] == ensemble) & (simulated["Year"] == 2100)],
            x="RCP",
            y="SLE (cm)",
            hue="RCP",
            palette=rcp_col_dict.values(),
            ax=axs[1],
        )
        sns.despine(ax=axs[1], left=True, bottom=True)
        axs[1].set_ylabel(None)
        axs[1].axes.xaxis.set_visible(False)
        axs[1].axes.yaxis.set_visible(False)

    set_size(5, 2)
    fig.savefig(out_filename, bbox_inches="tight")


def plot_partitioning(
    out_filename,
    simulated=None,
    observed=None,
    ensembles=["AS19", "Resampled"],
    quantiles=[0.05, 0.95],
    sigma=1,
    simulated_ctrl=None,
):

    ncol = 0
    if simulated is not None:
        ncol += len(ensembles)
    if observed is not None:
        ncol += 1

    credibility_interval = int(np.round((quantiles[-1] - quantiles[0]) * 100))

    fig, axs = plt.subplots(2, 1, sharex="col", figsize=[5, 3])
    fig.subplots_adjust(hspace=0.1, wspace=0.25)

    legend_handles = []
    if simulated is not None:
        for ens in ensembles:
            sim = simulated[simulated["Ensemble"] == ens]
            for k, v in enumerate(["SMB", "D"]):
                g = sim.groupby(by="Year")[f"{v} (Gt/yr)"]
                sim_median = g.quantile(0.50)
                sim_low = g.quantile(quantiles[0])
                sim_high = g.quantile(quantiles[-1])

                l_es_median = axs[k].plot(
                    sim_median.index,
                    sim_median,
                    color=ts_median_palette_dict[ens],
                    linewidth=signal_lw,
                    label=f"Median({ens} Ensemble)",
                )
                ci = axs[k].fill_between(
                    sim_median.index,
                    sim_low,
                    sim_high,
                    color=ts_fill_palette_dict[ens],
                    alpha=1.0,
                    linewidth=0.0,
                    zorder=-11,
                    label=f"{ens} {credibility_interval}% c.i.",
                )
                if k == 0:
                    legend_handles.append(l_es_median[0])
                    legend_handles.append(ci)

    if observed is not None:
        for k, v in enumerate(["SMB", "D"]):
            obs_line = axs[k].plot(
                observed["Year"],
                observed[f"{v} (Gt/yr)"],
                "-",
                color=obs_signal_color,
                linewidth=signal_lw,
                label="Observed (IMBIE)",
                zorder=20,
            )
            obs_ci = axs[k].fill_between(
                observed["Year"],
                observed[f"{v} (Gt/yr)"] - sigma * observed[f"{v} uncertainty (Gt/yr)"],
                observed[f"{v} (Gt/yr)"] + sigma * observed[f"{v} uncertainty (Gt/yr)"],
                color=obs_sigma_color,
                alpha=0.75,
                linewidth=0,
                zorder=5,
                label=f"Observational uncertainty ({sigma}-sigma)",
            )
            if k == 0:
                legend_handles.append(obs_line[0])
                legend_handles.append(obs_ci)

    for k, v in enumerate(["SMB", "D"]):
        axs[k].set_ylabel(f"{v} (Gt/yr)")
    legend = axs[0].legend(handles=legend_handles, loc="upper right", ncol=ncol)
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    axs[k].set_xlim(2010, 2020)
    axs[0].set_ylim(-250, 1250)
    axs[1].set_ylim(-1500, 0)

    set_size(5, 3)
    fig.savefig(out_filename, bbox_inches="tight")


def plot_sle_pdfs(
    out_filename,
    df,
    year=2100,
    ensembles=["AS19", "Resampled"],
):

    df = df[df["Year"] == 2100]
    # Draw a nested violinplot and split the violins for easier comparison
    fig = plt.figure()
    ax = fig.add_subplot(111)
    v = sns.violinplot(
        data=df,
        y="RCP",
        x="SLE (cm)",
        order=rcps,
        hue="Ensemble",
        hue_order=ensembles[::-1],
        split=True,
        cut=0.0,
        inner="quart",
        palette=[ts_fill_palette_dict[ens] for ens in ensembles[::-1]],
        linewidth=0.5,
        orient="h",
        ax=ax,
    )
    plt.title("SLE PDF at 2100")
    set_size(3.2, 1.6)
    fig.savefig(out_filename, bbox_inches="tight")


def plot_histograms(out_filename, df):
    fig, axs = plt.subplots(4, 4, figsize=[6.2, 4.0])
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    sns.histplot(
        data=df,
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
        data=df,
        x="PRS",
        hue="Ensemble",
        common_grid=False,
        common_norm=False,
        palette=palette_dict.values(),
        linewidth=0.8,
        ax=axs[1, 0],
    )
    sns.kdeplot(
        data=df,
        x="FICE",
        hue="Ensemble",
        common_grid=False,
        common_norm=False,
        palette=palette_dict.values(),
        linewidth=0.8,
        ax=axs[0, 1],
    )
    sns.kdeplot(
        data=df,
        x="FSNOW",
        hue="Ensemble",
        common_grid=False,
        common_norm=False,
        palette=palette_dict.values(),
        linewidth=0.8,
        ax=axs[1, 1],
    )
    sns.kdeplot(
        data=df,
        x="RFR",
        hue="Ensemble",
        common_grid=False,
        common_norm=False,
        palette=palette_dict.values(),
        linewidth=0.8,
        ax=axs[2, 1],
    )
    sns.histplot(
        data=df,
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
        data=df,
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
        data=df,
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
        data=df,
        x="VCM",
        hue="Ensemble",
        common_grid=False,
        common_norm=False,
        palette=palette_dict.values(),
        linewidth=0.8,
        ax=axs[3, 2],
    )
    sns.kdeplot(
        data=df,
        x="SIAE",
        hue="Ensemble",
        common_grid=False,
        common_norm=False,
        palette=palette_dict.values(),
        linewidth=0.8,
        ax=axs[0, 3],
    )
    sns.kdeplot(
        data=df,
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
    fig.savefig(out_filename)


def load_df(respone_file, samples_file):

    response = pd.read_csv(respone_file)
    response["SLE (cm)"] = -response["Mass (Gt)"] / 362.5 / 10
    response = response.astype({"RCP": int})
    samples = pd.read_csv(samples_file).rename(columns={"id": "Experiment"})
    return pd.merge(response, samples, on="Experiment")


def resample_ensemble_by_data(observed_calib_period, as19_calib_period, rcps, fudge_factor=3.0, verbose=False):
    """
    Resampling algorithm by Douglas C. Brinkerhoff
    """
    observed_interp_mean = interp1d(observed_calib_period["Year"], observed_calib_period["Mass (Gt)"])
    observed_interp_std = interp1d(observed_calib_period["Year"], observed_calib_period["Mass uncertainty (Gt)"])
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
                    observed_mass = observed_interp_mean(year)
                    observed_std = observed_interp_std(year) * fudge_factor
                    log_like -= 0.5 * ((exp_mass - observed_mass) / observed_std) ** 2 + 0.5 * np.log(
                        2 * np.pi * observed_std ** 2
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


signal_lw = 1.0
obs_signal_color = "#238b45"
obs_sigma_color = "#a1d99b"

secpera = 3.15569259747e7
gt2cmSLE = 1.0 / 362.5 / 10.0

rcps = [26, 45, 85]
rcp_col_dict = {26: "#003466", 45: "#5492CD", 85: "#990002"}
rcp_shade_col_dict = {26: "#4393C3", 45: "#92C5DE", 85: "#F4A582"}
rcp_dict = {26: "RCP 2.6", 45: "RCP 4.5", 85: "RCP 8.5"}
palette_dict = {"AS19": "0.8", "Calibrated": "0.6", "Resampled": "0.4"}
ts_fill_palette_dict = {"AS19": "0.8", "Calibrated": "0.6", "Resampled": "0.4"}
ts_median_palette_dict = {"AS19": "0.4", "Calibrated": "0.2", "Resampled": "0.0"}

# cm = sns.color_palette("ch:s=-.2,r=.6", n_colors=6, as_cmap=False).as_hex()
# ts_median_palette_dict = {"AS19": cm[0], "Calibrated": cm[1], "Resampled": cm[2]}

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

    observed = load_imbie()
    observed_calib_time = (observed["Year"] >= calibration_start) & (observed["Year"] <= calibration_end)
    observed_calib_period = observed[observed_calib_time]

    as19_resampled = resample_ensemble_by_data(observed_calib_period, as19_calib_period, rcps)

    as19["Ensemble"] = "AS19"
    as19_calib["Ensemble"] = "Calibrated"
    as19_resampled["Ensemble"] = "Resampled"
    all_df = (
        pd.concat([as19, as19_calib, as19_resampled])
        .drop_duplicates(subset=None, keep="first", inplace=False)
        .reset_index()
    )

    year = 2100
    all_2100_df = all_df[(all_df["Year"] == year)]

    plot_partitioning("historical_partitioning.pdf", simulated=all_df, observed=observed)
    plot_historical("historical.pdf", simulated=all_df, observed=observed)
    plot_projection("projection.pdf", simulated=all_df, quantiles=[0.05, 0.16, 0.84, 0.95])
    plot_sle_pdfs(f"sle_pdf_resampled_{year}.pdf", all_df, year=year)
    plot_histograms(f"histograms_{year}.pdf", all_2100_df)
