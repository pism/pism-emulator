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
    m_df = df[(df[x_var] >= trend_start) & (df[x_var] <= trend_end)]
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
    y_var_sigma = f"{y_var} Trend Sigma ({y_units}/yr)"
    r_df = df.groupby(by=["RCP", "Experiment"]).apply(trend_f, x, y)
    r_df = r_df.reset_index().rename({0: y_var_trend, 1: y_var_sigma}, axis=1)

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


trend_start = 2008
trend_end = 2020

# Greenland only though this could easily be extended to Antarctica
domain = {"GIS": "../data/validation/greenland_mass_200204_202008.txt"}

for d, data in domain.items():
    print(f"Analyzing {d}")

    # Load the GRACE data
    grace = pd.read_csv(
        data, header=30, delim_whitespace=True, skipinitialspace=True, names=["Year", "Mass (Gt)", "Sigma (Gt)"]
    )
    # Normalize GRACE signal to the starting date of the projection
    grace["Mass (Gt)"] -= np.interp(trend_start, grace["Year"], grace["Mass (Gt)"])

    # Get the GRACE trend
    grace_time = (grace["Year"] >= trend_start) & (grace["Year"] <= trend_end)
    grace_hist_df = grace[grace_time]
    x = grace_hist_df["Year"]
    y = grace_hist_df["Mass (Gt)"]
    s = grace_hist_df["Sigma (Gt)"]
    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit()
    p = ols.params
    grace_bias = p[0]
    grace_trend = p[1]
    grace_trend_stderr = ols.bse[1]

    as19 = pd.read_csv("../data/validation/aschwanden_et_al_2019_les.gz")
    as19["SLE (cm)"] = -as19["Mass (Gt)"] / 362.5 / 10
    as19 = as19.astype({"RCP": int, "Experiment": int})

    samples_file = "../data/samples/lhs_samples_500.csv"
    samples = pd.read_csv(samples_file).rename(columns={"id": "Experiment"})

    as19 = pd.merge(as19, samples, on="Experiment")
    params = samples.columns[1::]

    trend = calculate_trend(as19, "Year", "Mass", "Gt")
    as19_2100 = as19[as19["Year"] == 2100]
    trend["interval"] = pd.arrays.IntervalArray.from_arrays(
        trend["Mass Trend (Gt/yr)"] - trend["Mass Trend Sigma (Gt/yr)"],
        trend["Mass Trend (Gt/yr)"] + trend["Mass Trend Sigma (Gt/yr)"],
    )
    for beta in [1, 2, 3]:

        trend[f"{beta}-sigma"] = trend["interval"].array.overlaps(
            pd.Interval(grace_trend - beta * grace_trend_stderr, grace_trend + beta * grace_trend_stderr)
        )

    as19_2100 = pd.merge(as19_2100, trend, on=["RCP", "Experiment"])

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

    # bins = np.arange(np.floor(trend["Mass Trend (Gt/yr)"].min()), np.ceil(trend["Mass Trend (Gt/yr)"].max()), 10)
    # sns.histplot(data=trend, x="Mass Trend (Gt/yr)", bins=bins, color="0.5")
    # ax = plt.gca()

    # # Plot dashed line for GRACE
    # # ax.fill_between(
    # #     [grace_trend - grace_trend_stderr, grace_trend + grace_trend_stderr],
    # #     [ymin, ymin],
    # #     [ymax, ymax],
    # #     color=grace_sigma_color,
    # # )
    # ax.axvline(grace_trend, linestyle="solid", color=grace_signal_color, linewidth=1)
    # ax.axvline(grace_trend - grace_trend_stderr, linestyle="dotted", color=grace_signal_color, linewidth=1)
    # ax.axvline(grace_trend + grace_trend_stderr, linestyle="dotted", color=grace_signal_color, linewidth=1)

    # fig = plt.gcf()
    # fig.savefig("trend_histogram.pdf")

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
    ax.set_xlim(sle_min, sle_max)

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
        as19_2100_calib = as19_2100[as19_2100[f"{beta}-sigma"] == True]
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

    # rcps = []
    # exps = []
    # trends = []
    # sigmas = []
    # for g in as19.groupby(by=["RCP", "Experiment"]):
    #     m_df = g[-1][(g[-1]["Year"] >= trend_start) & (g[-1]["Year"] <= trend_end)]
    #     x = m_df["Year"]
    #     y = m_df["Mass (Gt)"]
    #     X = sm.add_constant(x)
    #     ols = sm.OLS(y, X).fit()
    #     p = ols.params
    #     model_bias = p[0]
    #     model_trend = p[1]
    #     model_trend_sigma = ols.bse[-1]
    #     rcps.append(g[0][0])
    #     exps.append(g[0][2])
    #     trends.append(model_trend)
    #     sigmas.append(model_trend_sigma)
