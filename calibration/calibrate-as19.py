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


def load_df(respone_file, samples_file):

    response = pd.read_csv(respone_file)
    response["SLE (cm)"] = -response["Mass (Gt)"] / 362.5 / 10
    response = response.astype({"RCP": int, "Experiment": int})
    samples = pd.read_csv(samples_file).rename(columns={"id": "Experiment"})

    return pd.merge(response, samples, on="Experiment")


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

rcp_list = ["26", "45", "85"]
rcp_col_dict = {"CTRL": "k", 85: "#990002", 45: "#5492CD", 26: "#003466"}
rcp_shade_col_dict = {"CTRL": "k", "85": "#F4A582", "45": "#92C5DE", "26": "#4393C3"}
rcp_dict = {26: "RCP 2.6", 45: "RCP 4.5", 85: "RCP 8.5"}

calibration_start = 2010
calibration_end = 2020

if __name__ == "__main__":

    # Load AS19
    as19 = load_df("../data/as19/aschwanden_et_al_2019_les_2008_norm.csv.gz", "../data/samples/lhs_samples_500.csv")
    as19_mc = load_df(
        "../data/as19/aschwanden_et_al_2019_mc_2008_norm.csv.gz", "../data/samples/lhs_plus_mc_samples.csv"
    )

    # Load AS19 with calibrated ice dynamics
    as19_2100 = as19[as19["Year"] == 2100]
    as19_mc_2100 = as19_mc[as19_mc["Year"] == 2100]

    imbie = load_imbie()

    sle_min = np.floor(as19_2100["SLE (cm)"].min())
    sle_max = np.ceil(as19_2100["SLE (cm)"].max())
    bin_width = 1
    sle_bins = np.arange(sle_min, sle_max, bin_width)

    numeric_cols = as19_2100.select_dtypes(exclude="number")
    as19_2100.drop(numeric_cols, axis=1, inplace=True)
    numeric_cols = as19_mc_2100.select_dtypes(exclude="number")
    as19_mc_2100.drop(numeric_cols, axis=1, inplace=True)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # sns.kdeplot(data=as19_mc_2100, x="SLE (cm)", hue="RCP", palette=["#003466"], ax=ax)
    # sns.kdeplot(
    #     data=as19_2100,
    #     x="SLE (cm)",
    #     hue="RCP",
    #     palette=["#003466", "#5492CD", "#990002"],
    #     ax=ax,
    #     fill=True,
    # )
    # for rcp in [26, 45, 85]:
    #     m_df = as19_2100[as19_2100["RCP"] == rcp]
    #     median = m_df.groupby(by=["Year"]).quantile(0.50)["SLE (cm)"].values[0]
    #     plt.axvline(median, color=rcp_col_dict[rcp], linestyle="--")

    as19_2100["ENS"] = "LES"
    as19_mc_2100["ENS"] = "MC"
    as19_all = pd.concat([as19_2100, as19_mc_2100])
    # Draw a nested violinplot and split the violins for easier comparison
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.violinplot(
        data=as19_all, y="RCP", x="SLE (cm)", hue="ENS", split=True, inner="quart", linewidth=1, orient="h", ax=ax
    )
    set_size(5, 2.5)
    fig.savefig("sle_pdf.pdf", bbox_inches="tight")
