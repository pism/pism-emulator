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
    """ w, h: width, height in inches """

    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def trend_f(df, x_var, y_var):
    m_df = df[(df[x_var] >= calibration_start) & (df[x_var] <= calibration_end)]
    x = m_df[x_var]
    y = m_df[y_var]
    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit()
    p = ols.params
    bias = p[0]
    trend = p[1]
    trend_sigma = ols.bse[-1]

    return pd.Series([trend, trend_sigma])


def calculate_trend(df, x_var, y_var, y_units):

    x = x_var
    y = f"{y_var} ({y_units})"
    y_var_trend = f"{y_var} Trend ({y_units}/yr)"
    y_var_sigma = f"{y_var} Trend Error ({y_units}/yr)"
    r_df = df.groupby(by=["RCP", "Experiment"]).apply(trend_f, x, y)
    r_df = r_df.reset_index().rename({0: y_var_trend, 1: y_var_sigma}, axis=1)

    return r_df


def calculate_mean(df, x_var, y_var, y_units):
    m_df = df[(df[x_var] >= calibration_start) & (df[x_var] <= calibration_end)]
    r_df = m_df.groupby(by=["RCP", "Experiment"]).mean()
    r_df = (
        r_df[f"{y_var} ({y_units})"]
        .reset_index()
        .rename({f"{y_var} ({y_units})": f"{y_var} Mean ({y_units})"}, axis=1)
    )
    return r_df


secpera = 3.15569259747e7

grace_signal_lw = 0.6
mouginot_signal_lw = 0.6
simulated_signal_lw = 0.3
grace_signal_color = "#084594"
grace_sigma_color = "#9ecae1"
mouginot_signal_color = "#a63603"
mouginot_sigma_color = "#fdbe85"
simulated_signal_color = "#bdbdbd"

gt2cmSLE = 1.0 / 362.5 / 10.0


calibration_start = 2010
calibration_end = 2020

# Greenland only though this could easily be extended to Antarctica
domain = {"GIS": "../data/validation/greenland_mass_200204_202008.txt"}

for d, data in domain.items():
    print(f"Analyzing {d}")

    
    # Load the GRACE data
    grace = pd.read_csv(
        data, header=30, delim_whitespace=True, skipinitialspace=True, names=["Year", "Mass (Gt)", "Error (Gt)"]
    )
    # Normalize GRACE signal to the starting date of the projection
    grace["Mass (Gt)"] -= np.interp(calibration_start, grace["Year"], grace["Mass (Gt)"])

    # Get the GRACE trend
    grace_time = (grace["Year"] >= calibration_start) & (grace["Year"] <= calibration_end)
    grace_hist_df = grace[grace_time]
    x = grace_hist_df["Year"]
    y = grace_hist_df["Mass (Gt)"]
    s = grace_hist_df["Error (Gt)"]
    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit()
    p = ols.params
    grace_bias = p[0]
    grace_trend = p[1]
    grace_trend_stderr = ols.bse[1]

    # Load the Mankoff ice discharge data
    man_d = pd.read_csv("../data/validation/GIS_D.csv", parse_dates=[0])
    man_d["Year"] = [toDecimalYear(d) for d in man_d["Date"]]
    man_d = man_d.astype({"Discharge [Gt yr-1]": float})
    man_d["Discharge [Gt yr-1]"] = -man_d["Discharge [Gt yr-1]"]
    man_err = pd.read_csv("../data/validation/GIS_err.csv", parse_dates=[0])
    man_err["Year"] = [toDecimalYear(d) for d in man_err["Date"]]
    man_err = man_err.astype({"Discharge Error [Gt yr-1]": float})
    man = pd.merge(man_d, man_err, on="Year").drop(columns=["Date_x", "Date_y"])

    # Load AS19
    as19 = pd.read_csv("../data/validation/aschwanden_et_al_2019_les.gz")
    as19["SLE (cm)"] = -as19["Mass (Gt)"] / 362.5 / 10
    as19 = as19.astype({"RCP": int, "Experiment": int})

    samples_file = "../data/samples/lhs_samples_500.csv"
    samples = pd.read_csv(samples_file).rename(columns={"id": "Experiment"})

    as19 = pd.merge(as19, samples, on="Experiment")
    params = samples.columns[1::]

    as19_2100 = as19[as19["Year"] == 2100]

    mass_trend = calculate_trend(as19, "Year", "Mass", "Gt")
    mass_trend["interval"] = pd.arrays.IntervalArray.from_arrays(
        mass_trend["Mass Trend (Gt/yr)"] - mass_trend["Mass Trend Error (Gt/yr)"],
        mass_trend["Mass Trend (Gt/yr)"] + mass_trend["Mass Trend Error (Gt/yr)"],
    )
    discharge_mean = calculate_mean(as19, "Year", "D", "Gt/yr")
    for beta in [1, 2, 3]:
        # This should NOT be the standard error of the OLS regression:
        mass_trend[f"Mass Trend {beta}-sigma (Gt/yr)"] = mass_trend["interval"].array.overlaps(
            pd.Interval(grace_trend - beta * grace_trend_stderr, grace_trend + beta * grace_trend_stderr)
        )
        discharge_mean[f"D Mean {beta}-sigma (Gt/yr)"] = np.abs(discharge_mean["D Mean (Gt/yr)"] + 500) < beta * 50
        

    as19_2100 = pd.merge(as19_2100, mass_trend, on=["RCP", "Experiment"])
    as19_2100 = pd.merge(as19_2100, discharge_mean, on=["RCP", "Experiment"])

    rcp_list = ["26", "45", "85"]
    rcp_col_dict = {"CTRL": "k", 85: "#990002", 45: "#5492CD", 26: "#003466"}
    rcp_shade_col_dict = {"CTRL": "k", "85": "#F4A582", "45": "#92C5DE", "26": "#4393C3"}
    rcp_dict = {26: "RCP 2.6", 45: "RCP 4.5", 85: "RCP 8.5"}
    param_bins_dict = {
        "GCM": np.arange(-0.5, 4.5, 1),
        "FICE": np.arange(4, 13, 0.25),
        "FSNOW": np.arange(2, 8, 0.25),
        "PRS": np.arange(5, 8, 1),
        "RFR": np.arange(0.2, 0.8, 0.1),
        "OCM": np.arange(-1.5, 2.5, 1),
        "OCS": np.arange(-1.5, 2.5, 1),
        "TCT": np.arange(-1.5, 2.5, 1),
        "VCM": np.arange(0.6, 1.3, 0.1),
        "PPQ": np.arange(0.2, 0.9, 0.1),
        "SIAE": np.arange(1, 10, 0.25),
    }

    fig_k, ax_k = plt.subplots(len(params), 3, sharey="row", figsize=[20, 20])
    fig_k.subplots_adjust(hspace=0.25, wspace=0.25)

    fig_p, ax_p = plt.subplots(len(params), 3, sharey="row", figsize=[20, 20])
    fig_p.subplots_adjust(hspace=0.25, wspace=0.25)

    fig_r, ax_r = plt.subplots(len(params), 1, sharey="row", figsize=[6, 20])
    fig_r.subplots_adjust(hspace=0.25, wspace=0.25)

    cmap = sns.color_palette("mako_r", n_colors=4)

    for q, rcp in enumerate([26, 45, 85]):
        for p, param in enumerate(params):
            sns.histplot(
                data=as19_2100[as19_2100["RCP"] == rcp],
                y=param,
                bins=param_bins_dict[param],
                stat="probability",
                element="step",
                fill=False,
                color=cmap[-1],
                ax=ax_p[p, q],
            )
            sns.kdeplot(
                data=as19_2100[as19_2100["RCP"] == rcp],
                y=param,
                color=cmap[-1],
                ax=ax_k[p, q],
            )
            sns.kdeplot(
                data=as19_2100,
                y=param,
                color=cmap[-1],
                ax=ax_r[p],
            )

    sle_min = np.floor(as19_2100["SLE (cm)"].min())
    sle_max = np.ceil(as19_2100["SLE (cm)"].max())
    bin_width = 1
    sle_bins = np.arange(sle_min, sle_max, bin_width)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.histplot(
        data=as19_2100,
        x="SLE (cm)",
        hue="RCP",
        palette=["#003466", "#5492CD", "#990002"],
        bins=sle_bins,
        stat="probability",
        ax=ax,
    )
    fig.savefig("calibrated_histogram_2100.pdf")
    fig, ax = plt.subplots(1, 3, sharey="row", figsize=[6, 3])
    fig.subplots_adjust(hspace=0.05, wspace=0.30)
    for q, rcp in enumerate([26, 45, 85]):
        df = as19_2100[as19_2100["RCP"] == rcp]
        sns.kdeplot(data=df, x="SLE (cm)", ax=ax[q], color=cmap[-1])
        ax[q].set_title(f"RCP {rcp}")
        for (beta, c) in zip([1, 2, 3], cmap[::-1]):
            df_calib = df[df[f"Mass Trend {beta}-sigma (Gt/yr)"] == True]
            sns.kdeplot(data=df_calib, x="SLE (cm)", ax=ax[q], color=c, linewidth=0.75)
            df_calib = df[df[f"D Mean {beta}-sigma (Gt/yr)"] == True]
            sns.kdeplot(data=df_calib, x="SLE (cm)", ax=ax[q], color=c, linewidth=0.75, linestyle="dashed")
    l_s = []
    for m, label in enumerate(["AS19", "3-sigma", "2-sigma", "1-sigma"]):
        l_ = mlines.Line2D(
            [],
            [],
            color=cmap[::-1][m],
            linewidth=1.0,
            linestyle="solid",
            label=label,
        )
        l_s.append(l_)

    legend = ax[-1].legend(
        handles=l_s,
        loc="upper right",
    )
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)
    fig.savefig("sle_pdf_rcps_2100.pdf", bbox_inches="tight")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.kdeplot(
        data=as19_2100,
        x="SLE (cm)",
        hue="RCP",
        palette=["#003466", "#5492CD", "#990002"],
        ax=ax,
    )
    k = 0
    for beta, lw in zip([1, 2, 3], [0.25, 0.5, 0.75]):
        as19_2100_calib = as19_2100[as19_2100[f"Mass Trend {beta}-sigma (Gt/yr)"] == True]
        for rcp in [26, 45, 85]:
            no = len(as19_2100_calib[as19_2100_calib["RCP"] == rcp])
            ratio = no / len(as19_2100[as19_2100["RCP"] == rcp]) * 100
            print(f"RCP {rcp} {beta}-sigma: {no} ({ratio:.1f} %)")
        sns.kdeplot(
            data=as19_2100_calib,
            x="SLE (cm)",
            hue="RCP",
            linewidth=lw,
            linestyle="dashed",
            ax=ax,
        )
        for q, rcp in enumerate([26, 45, 85]):
            for p, param in enumerate(params):
                sns.histplot(
                    data=as19_2100_calib[as19_2100_calib["RCP"] == rcp],
                    y=param,
                    bins=param_bins_dict[param],
                    stat="probability",
                    element="step",
                    fill=False,
                    color=cmap[k],
                    ax=ax_p[p, q],
                )
                sns.kdeplot(
                    data=as19_2100_calib[as19_2100_calib["RCP"] == rcp],
                    y=param,
                    color=cmap[k],
                    ax=ax_k[p, q],
                )
                sns.kdeplot(
                    data=as19_2100_calib,
                    y=param,
                    color=cmap[k],
                    ax=ax_r[p],
                )

        k += 1
    ax.set_xlim(sle_min, sle_max)
    fig.savefig("sle_pdf_2100.pdf")

    l_s = []
    for m, label in enumerate(["AS19", "3-sigma", "2-sigma", "1-sigma"]):
        l_ = mlines.Line2D(
            [],
            [],
            color=cmap[::-1][m],
            linewidth=1.0,
            linestyle="solid",
            label=label,
        )
        l_s.append(l_)

    legend = ax_r[-1].legend(
        handles=l_s,
        loc="upper right",
    )
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    fig_p.savefig("marginal_distributions_hist.pdf")
    fig_k.savefig("marginal_distributions_kde_rcp.pdf")
    fig_r.savefig("marginal_distributions_kde.pdf")
    # ax.text(-340, 2.2, "Observed (GRACE)", rotation=90, fontsize=12)

    # param="GCM"
    # vals = as19_2100[param].values
    # p = np.histogram(vals,bins=param_bins_dict[param], density=True)
    # plt.plot([0,1,2,3],p[0], ".", color=cmap[-1])
    # for beta in [1, 2, 3]:
    #     vals = as19_2100[as19_2100[f"{beta}-sigma"] == True][param].values
    #     p = np.histogram(vals,bins=param_bins_dict[param], density=True)
    #     plt.plot([0,1,2,3],p[0], ".", color=cmap[beta])

    def plot_d(g):
        return ax.plot(g[-1]["Year"], g[-1]["D (Gt/yr)"], linewidth=0.5, color="0.5", zorder=-1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    df = as19
    x_var = "Year"
    y_var = "D (Gt/yr)"
    df = df[(df[x_var] >= calibration_start) & (df[x_var] <= calibration_end)]
    [plot_d(g) for g in df.groupby(by=["RCP", "Experiment"])]
    ax.fill_between(
        man["Year"],
        man["Discharge [Gt yr-1]"] - man["Discharge Error [Gt yr-1]"],
        man["Discharge [Gt yr-1]"] + man["Discharge Error [Gt yr-1]"],
        linewidth=0,
        color=cmap[0],
        alpha=0.5,
        zorder=2,
    )
    ax.plot(man["Year"], man["Discharge [Gt yr-1]"], linewidth=2, color=cmap[0],label="Mankoff")
    legend = ax.legend()
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    ax.set_xlim(calibration_start, calibration_end)
    ax.set_xlabel("Year")
    ax.set_ylabel("Discharge (Gt/yr)")
    fig.savefig("discharge_calibrated.pdf")
