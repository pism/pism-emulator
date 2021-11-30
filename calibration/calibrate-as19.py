#!/usr/bin/env python

# Copyright (C) 2020-21 Andy Aschwanden, Douglas J. Brinkerhoff

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import colors
import numpy as np
import os
import pylab as plt
import pandas as pd
import seaborn as sns
from functools import reduce
from itertools import cycle

from pismemulator.utils import load_imbie
from scipy.interpolate import interp1d


def add_inner_title(ax, title, loc="upper left", size=7, **kwargs):
    """
    Adds an inner title to a given axis, with location loc.

    from http://matplotlib.sourceforge.net/examples/axes_grid/demo_axes_grid2.html
    """
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke

    prop = dict(size=size, weight="bold")
    at = AnchoredText(title, loc=loc, prop=prop, pad=0.0, borderpad=0.5, frameon=False, **kwargs)
    ax.add_artist(at)
    return at


def color_tint(m_color, alpha):
    m_color = list(colors.to_rgba(m_color))
    m_color[-1] = alpha
    m_color = np.array(m_color) * 255
    return rgba2rgb(m_color) / 255


def rgba2rgb(rgba, background=(255, 255, 255)):

    rgb = np.zeros((3), dtype="float32")
    r, g, b, a = rgba[0], rgba[1], rgba[2], rgba[3]

    a = np.asarray(a, dtype="float32") / 255.0

    R, G, B = background

    rgb[0] = r * a + (1.0 - a) * R
    rgb[1] = g * a + (1.0 - a) * G
    rgb[2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype="uint8")


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
    ensembles=["AS19", "Flow+Mass Calib."],
    quantiles=[0.05, 0.95],
    sigma=2,
    simulated_ctrl=None,
):
    """
    Plot historical simulations and observations
    """

    xmin = 2008
    xmax = 2020
    ymin = -10000
    ymax = 500

    fig = plt.figure(num="historical", clear=True, figsize=[4.6, 1.6])
    ax = fig.add_subplot(111)

    if simulated is not None:
        for r, ens in enumerate(ensembles):
            legend_handles = []
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
                label="Median",
            )
            legend_handles.append(l_es_median[0])
            ci = ax.fill_between(
                sim_median.index,
                sim_low,
                sim_high,
                color=ts_fill_palette_dict[ens],
                alpha=0.75,
                linewidth=0.0,
                zorder=-11,
                label=f"{quantiles[0]*100:.0f}-{quantiles[-1]*100:.0f}%",
            )
            legend_handles.append(ci)

            legend = ax.legend(
                handles=legend_handles, loc="lower left", ncol=1, title=ens, bbox_to_anchor=(r * 0.2, 0.01)
            )
            legend.get_frame().set_linewidth(0.0)
            legend.get_frame().set_alpha(0.0)
            ax.add_artist(legend)

    if observed is not None:
        legend_handles = []
        obs_line = ax.plot(
            observed["Year"],
            observed["Mass (Gt)"],
            "-",
            color=obs_signal_color,
            linewidth=signal_lw,
            label="Mean",
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
            label=f"{sigma}-$\sigma$",
        )
        legend_handles.append(obs_ci)

        legend = ax.legend(
            handles=legend_handles,
            loc="lower left",
            ncol=1,
            title="Observed (IMBIE)",
            bbox_to_anchor=((r + 1.0) * 0.2, 0.01),
        )
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)
        ax.add_artist(legend)

    ax.axhline(0, color="k", linestyle="dotted", linewidth=0.6)

    ax.set_xlabel("Year")
    ax.set_ylabel(f"Cumulative mass change\nsince {proj_start} (Gt)")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax_sle = ax.twinx()
    ax_sle.set_ylabel(f"Contribution to sea-level \nsince {proj_start} (cm SLE)")
    ax_sle.set_ylim(-ymin * gt2cmSLE, -ymax * gt2cmSLE)

    fig.savefig(out_filename, bbox_inches="tight")


def plot_projection(
    out_filename,
    simulated=None,
    ensemble="Flow+Mass Calib. S2",
    quantiles=[0.05, 0.95],
    bars=None,
    quantile_df=None,
):
    """
    Plot historical simulations and observations
    """

    xmin = 2008
    xmax = 2100
    ymax = 45
    ymin = -0.5

    if bars:
        fig, axs = plt.subplots(
            1,
            4,
            sharey="row",
            figsize=[6.0, 2.0],
            gridspec_kw=dict(width_ratios=[20, 1, 1, 1]),
        )
        fig.subplots_adjust(hspace=0.1, wspace=0.05)
        ax = axs[0]
    else:
        fig = plt.figure(figsize=[5.2, 2.2])
        ax = fig.add_subplot(111)

    if simulated is not None:
        for r, rcp in enumerate(rcps):
            legend_handles = []
            sim = simulated[(simulated["Ensemble"] == ensemble) & (simulated["RCP"] == rcp)]
            g = sim.groupby(by="Year")["SLE (cm)"]
            sim_median = g.quantile(0.50)

            l_es_median = ax.plot(
                sim_median.index,
                sim_median,
                color=rcp_col_dict[rcp],
                linewidth=signal_lw,
                label="Median",
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
                alpha=0.5,
                linewidth=0.5,
                zorder=-11,
                label=f"{quantiles[0]*100:.0f}-{quantiles[-1]*100:.0f}%",
            )
            legend_handles.append(ci)
            if len(quantiles) == 4:
                sim_low = g.quantile(quantiles[1])
                sim_high = g.quantile(quantiles[-2])
                ci = ax.fill_between(
                    sim_median.index,
                    sim_low,
                    sim_high,
                    color=rcp_shade_col_dict[rcp],
                    alpha=0.85,
                    linewidth=0.5,
                    zorder=-11,
                    label=f"{quantiles[1]*100:.0f}-{quantiles[-2]*100:.0f}%",
                )
                legend_handles.append(ci)

            legend = ax.legend(
                handles=legend_handles, title=rcp_dict[rcp], loc="upper left", bbox_to_anchor=(r * 0.2, 0.99)
            )
            legend.get_frame().set_linewidth(0.0)
            legend.get_frame().set_alpha(0.0)
            ax.add_artist(legend)

    ax.set_xlabel("Year")
    ax.set_ylabel(f"Contribution to sea-level\nsince {proj_start} (cm SLE)")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if bars is not None:
        width = 1.0
        legend_elements = []
        hatch_pattern_dict = {
            "Flow+Mass Calib.": "\\\\\\",
            "Flow+Mass Calib. S1": "\\\\\\",
            "Flow+Mass Calib. S2": "\\\\\\",
            "Flow+Mass Calib. S3": "\\\\\\",
            "Flow+Mass Calib.": "\\\\\\",
            "Flow Calib.": "......",
            "AS19": "",
        }
        hatch_patterns = [hatch_pattern_dict[ensemble] for ensemble in bars]
        hatches = cycle(hatch_patterns)
        q_df = make_quantile_df(simulated[simulated["Year"] == 2100], quantiles=[0.05, 0.16, 0.5, 0.84, 0.95])
        for k, rcp in enumerate(rcps):
            df = q_df[q_df["RCP"] == rcp]
            for e, ens in enumerate(bars):
                hatch = next(hatches)
                s_df = df[df["Ensemble"] == ens]
                rect1 = plt.Rectangle(
                    (e + 0.4, s_df[[0.05]].values[0][0]),
                    0.2,
                    s_df[[0.95]].values[0][0] - s_df[[0.05]].values[0][0],
                    color=rcp_shade_col_dict[rcp],
                    alpha=1.0,
                    lw=0,
                )
                rect2 = plt.Rectangle(
                    (e + 0.2, s_df[[0.16]].values[0][0]),
                    0.6,
                    s_df[[0.84]].values[0][0] - s_df[[0.16]].values[0][0],
                    color=rcp_shade_col_dict[rcp],
                    alpha=1.0,
                    lw=0,
                )
                rect3 = plt.Rectangle(
                    (e + 0.4, s_df[[0.05]].values[0][0]),
                    0.2,
                    s_df[[0.95]].values[0][0] - s_df[[0.05]].values[0][0],
                    color="k",
                    alpha=1.0,
                    fill=False,
                    lw=0.25,
                    # hatch=hatch,
                    label=ens,
                )
                rect4 = plt.Rectangle(
                    (e + 0.2, s_df[[0.16]].values[0][0]),
                    0.6,
                    s_df[[0.84]].values[0][0] - s_df[[0.16]].values[0][0],
                    color="k",
                    alpha=1.0,
                    fill=False,
                    lw=0.25,
                    # hatch=hatch,
                )
                axs[k + 1].add_patch(rect3)
                axs[k + 1].add_patch(rect4)
                axs[k + 1].add_patch(rect1)
                axs[k + 1].add_patch(rect2)
                axs[k + 1].plot(
                    [e, e + width],
                    [s_df[[0.50]].values[0][0], s_df[[0.50]].values[0][0]],
                    color=rcp_col_dict[rcp],
                    lw=signal_lw,
                )
                if k == 0:
                    legend_elements.append(
                        Patch(facecolor=None, edgecolor="k", fill=None, lw=0.25, hatch=hatch, label=ens),
                    )
        legend_2 = ax.legend(handles=legend_elements, loc="upper right")
        legend_2.get_frame().set_linewidth(0.0)
        legend_2.get_frame().set_alpha(0.0)

        sns.despine(ax=axs[1], left=True, bottom=True)
        axs[1].set_ylabel(None)
        axs[1].axes.xaxis.set_visible(False)
        axs[1].axes.yaxis.set_visible(False)

        sns.despine(ax=axs[2], left=True, bottom=True)
        axs[2].set_ylabel(None)
        axs[2].axes.xaxis.set_visible(False)
        axs[2].axes.yaxis.set_visible(False)

        sns.despine(ax=axs[3], left=True, bottom=True)
        axs[3].set_ylabel(None)
        axs[3].axes.xaxis.set_visible(False)
        axs[3].axes.yaxis.set_visible(False)

    fig.savefig(out_filename, bbox_inches="tight")


def plot_partitioning(
    out_filename,
    simulated=None,
    observed=None,
    ensembles=["AS19", "Flow Calib.", "Flow+Mass Calib."],
    quantiles=[0.05, 0.95],
    sigma=2,
    simulated_ctrl=None,
):

    ncol = 0
    if simulated is not None:
        ncol += len(ensembles)
    if observed is not None:
        ncol += 1

    fig, axs = plt.subplots(
        3,
        1,
        sharex="col",
        figsize=[3.2, 4.2],
        gridspec_kw=dict(height_ratios=[1.5, 1, 1]),
    )
    fig.subplots_adjust(hspace=0.1, wspace=0.25)

    if simulated is not None:
        for r, ens in enumerate(ensembles):
            legend_handles = []
            sim = simulated[simulated["Ensemble"] == ens]
            for k, (v, u) in enumerate(zip(["Mass", "SMB", "D"], ["Gt", "Gt/yr", "Gt/yr"])):
                g = sim.groupby(by="Year")[f"{v} ({u})"]
                sim_median = g.quantile(0.50)
                sim_low = g.quantile(quantiles[0])
                sim_high = g.quantile(quantiles[-1])

                l_es_median = axs[k].plot(
                    sim_median.index,
                    sim_median,
                    color=ts_median_palette_dict[ens],
                    linewidth=signal_lw,
                    zorder=r,
                    label=f"Median",
                )
                ci = axs[k].fill_between(
                    sim_median.index,
                    sim_low,
                    sim_high,
                    color=ts_fill_palette_dict[ens],
                    alpha=0.75,
                    linewidth=0.0,
                    zorder=-11,
                    label=f"{quantiles[0]*100:.0f}-{quantiles[-1]*100:.0f}%",
                )
                if k == 0:
                    legend_handles.append(l_es_median[0])
                    legend_handles.append(ci)

            legend = axs[0].legend(
                handles=legend_handles, loc="lower left", ncol=1, title=ens, bbox_to_anchor=(r * 0.16, 0.005)
            )
            legend.get_frame().set_linewidth(0.0)
            legend.get_frame().set_alpha(0.0)
            axs[0].add_artist(legend)

    if observed is not None:
        legend_handles = []
        for k, (v, u) in enumerate(zip(["Mass", "SMB", "D"], ["Gt", "Gt/yr", "Gt/yr"])):
            obs_line = axs[k].plot(
                observed["Year"],
                observed[f"{v} ({u})"],
                "-",
                color=obs_signal_color,
                linewidth=signal_lw,
                label="Mean",
                zorder=20,
            )
            obs_ci = axs[k].fill_between(
                observed["Year"],
                observed[f"{v} ({u})"] - sigma * observed[f"{v} uncertainty ({u})"],
                observed[f"{v} ({u})"] + sigma * observed[f"{v} uncertainty ({u})"],
                color=obs_sigma_color,
                alpha=0.75,
                linewidth=0,
                zorder=5,
                label=f"{sigma}-$\sigma$",
            )
            if k == 0:
                legend_handles.append(obs_line[0])
                legend_handles.append(obs_ci)

        legend = axs[0].legend(
            handles=legend_handles,
            loc="lower left",
            ncol=1,
            title="Observed (IMBIE)",
            bbox_to_anchor=((r + 1.5) * 0.16, 0.005),
        )
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)
        axs[0].add_artist(legend)

    for k, (v, u) in enumerate(
        zip([f"Cumulative mass change \nsince {proj_start}", "SMB", "D"], ["Gt", "Gt/yr", "Gt/yr"])
    ):
        axs[k].set_ylabel(f"{v} ({u})")

    for k, (v, u) in enumerate(zip(["d", "e", "f"], ["", "", ""])):
        add_inner_title(axs[k], f"{v}) {u}")

    axs[-1].set_xlim(2008, 2020)
    axs[-1].set_xlabel("Year")
    axs[0].set_ylim(-12500, 2500)
    axs[1].set_ylim(-500, 1000)
    axs[2].set_ylim(-1500, 0)

    fig.savefig(out_filename, bbox_inches="tight")


def plot_posterior_sle_pdfs(
    out_filename,
    df,
    year=2100,
):

    df = df[df["Year"] == year]
    q_df = make_quantile_df(df, quantiles=[0.05, 0.16, 0.5, 0.84, 0.95])
    ensembles = ["AS19", "Flow Calib.", "Flow+Mass Calib."]
    alphas = [0.8, 0.9, 1.0]

    fig, axs = plt.subplots(
        6,
        1,
        sharex="col",
        figsize=[3.2, 4.2],
        gridspec_kw=dict(height_ratios=[0.75, 4] * 3),
    )
    fig.subplots_adjust(hspace=0.0, wspace=-0.0)
    for k, rcp in enumerate(rcps):
        m_df = df[df["RCP"] == rcp]
        p_df = q_df[q_df["RCP"] == rcp]

        sns.kdeplot(
            data=m_df,
            x="SLE (cm)",
            hue="Ensemble",
            hue_order=ensembles,
            common_norm=False,
            common_grid=True,
            multiple="layer",
            fill=True,
            lw=0,
            palette=[color_tint(rcp_col_dict[rcp], alpha) for alpha in alphas],
            ax=axs[k * 2 + 1],
        )

        sns.kdeplot(
            data=m_df,
            x="SLE (cm)",
            hue="Ensemble",
            hue_order=ensembles,
            common_norm=False,
            common_grid=True,
            multiple="layer",
            fill=False,
            lw=0.25,
            palette=["k"] * len(ensembles),
            ax=axs[k * 2 + 1],
        )

        for e, ens in enumerate(ensembles):
            s_df = p_df[p_df["Ensemble"] == ens]
            mk_df = df[df["Ensemble"] == ens]

            alpha = alphas[e] - 0.2 * (len(ensembles) - e)
            m_color = color_tint(rcp_col_dict[rcp], alpha)
            lw = 0.25

            #             sns.kdeplot(
            #                 data=mk_df,
            #                 x="SLE (cm)",
            #                 common_norm=False,
            #                 common_grid=True,
            #                 multiple="layer",
            #                 fill=True,
            #                 lw=0.0,
            #                 color=m_color,
            #                 ax=axs[k * 2 + 1],
            #         )
            #             sns.kdeplot(
            #                 data=mk_df,
            #                 x="SLE (cm)",
            #                 common_norm=False,
            #                 common_grid=True,
            #                 multiple="layer",
            #                 fill=False,
            #                 lw=0.5,
            #                 color="k",
            #                 ax=axs[k * 2 + 1],
            #         )

            rect0 = plt.Rectangle(
                (s_df[[0.5]].values[0][0] - 0.1, e + 0.1),
                0.2,
                0.8,
                color="k",
                lw=0,
            )

            rect1 = plt.Rectangle(
                (s_df[[0.05]].values[0][0], e + 0.4),
                s_df[[0.95]].values[0][0] - s_df[[0.05]].values[0][0],
                0.2,
                color=m_color,
                alpha=1,
                lw=0,
            )
            rect2 = plt.Rectangle(
                (s_df[[0.16]].values[0][0], e + 0.2),
                s_df[[0.84]].values[0][0] - s_df[[0.16]].values[0][0],
                0.6,
                color=m_color,
                alpha=1,
                lw=0,
            )
            rect3 = plt.Rectangle(
                (s_df[[0.05]].values[0][0], e + 0.4),
                s_df[[0.95]].values[0][0] - s_df[[0.05]].values[0][0],
                0.2,
                color="k",
                alpha=1,
                fill=False,
                lw=lw,
            )
            rect4 = plt.Rectangle(
                (s_df[[0.16]].values[0][0], e + 0.2),
                s_df[[0.84]].values[0][0] - s_df[[0.16]].values[0][0],
                0.6,
                color="k",
                alpha=1,
                fill=False,
                lw=lw,
            )

            axs[(k * 2)].add_patch(rect1)
            axs[(k * 2)].add_patch(rect3)
            axs[(k * 2)].add_patch(rect2)
            axs[(k * 2)].add_patch(rect4)
            axs[(k * 2)].add_patch(rect0)

            axs[(k * 2)].set_ylabel(None)
            axs[(k * 2)].axes.xaxis.set_visible(False)
            axs[(k * 2)].axes.yaxis.set_visible(False)
            sns.despine(ax=axs[(k * 2)], left=True, bottom=True)
            sns.despine(ax=axs[(k * 2) + 1], top=True)

            axs[(k * 2)].set_ylim(0, len(ensembles))

        axs[k].legend().remove()
        axs[k * 2 + 1].legend().remove()

    for k, rcp in enumerate(rcps):
        add_inner_title(axs[k * 2 + 1], rcp_dict[rcp])

    l_as19 = Patch(
        facecolor=str(alphas[2] - 0.2 * (len(ensembles) - 2)), edgecolor="0.0", linewidth=0.25, label="Prior (AS19)"
    )
    l_mass = Patch(
        facecolor=str(alphas[1] - 0.2 * (len(ensembles) - 1)),
        edgecolor="0.0",
        linewidth=0.25,
        label="Posterior (Flow Calib.)",
    )
    l_calib = Patch(
        facecolor=str(alphas[0] - 0.2 * (len(ensembles) - 0)),
        edgecolor="0.0",
        linewidth=0.25,
        label="Posterior (Flow+Mass Calib.)",
    )

    legend_1 = axs[1].legend(
        handles=[l_as19, l_mass, l_calib],
        loc="upper left",
        bbox_to_anchor=(0.4, 1.1, 0, 0),
    )
    legend_1.get_frame().set_linewidth(0.0)
    legend_1.get_frame().set_alpha(0.0)
    axs[1].add_artist(legend_1)

    # l_ = []
    # for rcp in rcps:
    #     l_.append(Line2D([], [], color=rcp_col_dict[rcp], label=rcp_dict[rcp]))
    # legend_2 = axs[3].legend(
    #     handles=l_,
    #     loc="upper left",
    #     bbox_to_anchor=(0.4, 1.1, 0, 0),
    # )
    # legend_2.get_frame().set_linewidth(0.0)
    # legend_2.get_frame().set_alpha(0.0)
    # axs[3].add_artist(legend_2)

    fig.tight_layout()
    fig.savefig(out_filename)


def compare_sle_pdfs(
    out_filename,
    df,
    year=2100,
):

    df = df[df["Year"] == year]

    q_df = make_quantile_df(df, quantiles=[0.05, 0.16, 0.5, 0.84, 0.95])

    mass_ensembles = [
        "Mass Calib. S3",
        "Mass Calib. S2",
        "Mass Calib. S1",
    ]
    flowmass_ensembles = [
        "Flow+Mass Calib. S3",
        "Flow+Mass Calib. S2",
        "Flow+Mass Calib. S1",
    ]
    all_ensembles = ["AS19", "Flow Calib."] + mass_ensembles + flowmass_ensembles
    mass_ensembles_palette = ["#756bb1", "#9e9ac8", "#cbc9e2"]
    flowmass_ensembles_palette = ["#3182bd", "#6baed6", "#bdd7e7"]
    all_ensembles_palette = ["#636363", "#31a354"] + mass_ensembles_palette + flowmass_ensembles_palette
    fig, axs = plt.subplots(
        6,
        1,
        sharex="col",
        figsize=[3.2, 4.2],
        gridspec_kw=dict(height_ratios=[1.75, 4] * 3),
    )
    fig.subplots_adjust(hspace=0.0, wspace=-0.0)
    for k, rcp in enumerate(rcps):
        m_df = df[df["RCP"] == rcp]
        p_df = q_df[q_df["RCP"] == rcp]

        sns.kdeplot(
            data=m_df,
            x="SLE (cm)",
            hue="Ensemble",
            hue_order=all_ensembles,
            fill=False,
            lw=0.5,
            palette=all_ensembles_palette,
            ax=axs[k * 2 + 1],
        )

        for e, ens in enumerate(all_ensembles):
            s_df = p_df[p_df["Ensemble"] == ens]

            lw = 0.1

            rect0 = plt.Rectangle(
                (s_df[[0.5]].values[0][0] - 0.1, e + 0.1),
                0.2,
                0.8,
                color="k",
                lw=0,
            )

            rect1 = plt.Rectangle(
                (s_df[[0.05]].values[0][0], e + 0.4),
                s_df[[0.95]].values[0][0] - s_df[[0.05]].values[0][0],
                0.2,
                color=all_ensembles_palette[e],
                alpha=1,
                lw=0,
            )
            rect2 = plt.Rectangle(
                (s_df[[0.16]].values[0][0], e + 0.2),
                s_df[[0.84]].values[0][0] - s_df[[0.16]].values[0][0],
                0.6,
                color=all_ensembles_palette[e],
                alpha=1,
                lw=0,
            )
            rect3 = plt.Rectangle(
                (s_df[[0.05]].values[0][0], e + 0.4),
                s_df[[0.95]].values[0][0] - s_df[[0.05]].values[0][0],
                0.2,
                color="k",
                alpha=1,
                fill=False,
                lw=lw,
            )
            rect4 = plt.Rectangle(
                (s_df[[0.16]].values[0][0], e + 0.2),
                s_df[[0.84]].values[0][0] - s_df[[0.16]].values[0][0],
                0.6,
                color="k",
                alpha=1,
                fill=False,
                lw=lw,
            )

            axs[(k * 2)].add_patch(rect1)
            axs[(k * 2)].add_patch(rect3)
            axs[(k * 2)].add_patch(rect2)
            axs[(k * 2)].add_patch(rect4)
            axs[(k * 2)].add_patch(rect0)

            axs[(k * 2)].set_ylabel(None)
            axs[(k * 2)].axes.xaxis.set_visible(False)
            axs[(k * 2)].axes.yaxis.set_visible(False)
            sns.despine(ax=axs[(k * 2)], left=True, bottom=True)
            sns.despine(ax=axs[(k * 2) + 1], top=True)

            axs[(k * 2)].set_ylim(0, len(all_ensembles))
            axs[k * 2 + 1].legend().remove()

    for k, rcp in enumerate(rcps):
        add_inner_title(axs[k * 2], rcp_dict[rcp])

    l_as19 = Patch(facecolor="0.8", edgecolor="0.0", linewidth=0.75, label="AS19")
    l_flow = Patch(facecolor="0.8", edgecolor="0.0", linewidth=0.5, label="Flow Calib.")
    l_calib_s3 = Patch(facecolor="0.6", edgecolor="0.0", linewidth=0.25, label="Flow+Mass Calib. $\\beta$=3")
    l_calib_s2 = Patch(facecolor="0.4", edgecolor="0.0", linewidth=0.25, label="Flow+Mass Calib. $\\beta$=2")

    l_calib_s1 = Patch(facecolor="0.4", edgecolor="0.0", linewidth=0.25, label="Flow+Mass Calib. $\\beta$=1")

    # legend_1 = axs[1].legend(
    #     handles=[l_as19, l_flow, l_calib_s3, l_calib_s2, l_calib_s1],
    #     loc="upper left",
    #     bbox_to_anchor=(0.48, 1.1, 0, 0),
    # )
    # legend_1.get_frame().set_linewidth(0.0)
    # legend_1.get_frame().set_alpha(0.0)
    # axs[1].add_artist(legend_1)

    l_ = []
    for e, ens in enumerate(all_ensembles):
        l_.append(Line2D([], [], color=all_ensembles_palette[e], label=ens))
    legend = axs[1].legend(
        handles=l_,
        loc="upper left",
        bbox_to_anchor=(0.48, 1.2, 0, 0),
    )
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)
    axs[1].add_artist(legend)

    fig.tight_layout()
    fig.savefig(out_filename)


def compare_sle_pdfs_1(
    out_filename,
    df,
    year=2100,
):

    df = df[df["Year"] == year]

    q_df = make_quantile_df(df, quantiles=[0.05, 0.16, 0.5, 0.84, 0.95])

    mass_ensembles = [
        "Mass Calib. S3",
        "Mass Calib. S2",
        "Mass Calib. S1",
    ]
    flowmass_ensembles = [
        "Flow+Mass Calib. S3",
        "Flow+Mass Calib. S2",
        "Flow+Mass Calib. S1",
    ]
    all_ensembles = ["AS19", "Flow Calib."] + mass_ensembles + flowmass_ensembles
    mass_ensembles_palette = ["#756bb1", "#9e9ac8", "#cbc9e2"]
    flowmass_ensembles_palette = ["#3182bd", "#6baed6", "#bdd7e7"]
    all_ensembles_palette = ["#636363", "#31a354"] + mass_ensembles_palette + flowmass_ensembles_palette
    fig, axs = plt.subplots(
        2,
        1,
        sharex="col",
        figsize=[3.2, 1.8],
        gridspec_kw=dict(height_ratios=[1.5, 4]),
    )
    fig.subplots_adjust(hspace=0.0, wspace=-0.0)
    m_df = df
    p_df = q_df

    sns.kdeplot(
        data=m_df,
        x="SLE (cm)",
        hue="Ensemble",
        hue_order=all_ensembles,
        fill=False,
        lw=0.5,
        palette=all_ensembles_palette,
        ax=axs[1],
    )

    for e, ens in enumerate(all_ensembles):
        s_df = p_df[p_df["Ensemble"] == ens]
        k_df = df[df["Ensemble"] == ens]

        lw = 0.1

        rect0 = plt.Rectangle(
            (s_df[[0.5]].values[0][0] - 0.1, e + 0.1),
            0.2,
            0.8,
            color="k",
            lw=0,
        )

        rect1 = plt.Rectangle(
            (s_df[[0.05]].values[0][0], e + 0.4),
            s_df[[0.95]].values[0][0] - s_df[[0.05]].values[0][0],
            0.2,
            color=all_ensembles_palette[e],
            alpha=1,
            lw=0,
        )
        rect2 = plt.Rectangle(
            (s_df[[0.16]].values[0][0], e + 0.2),
            s_df[[0.84]].values[0][0] - s_df[[0.16]].values[0][0],
            0.6,
            color=all_ensembles_palette[e],
            alpha=1,
            lw=0,
        )
        rect3 = plt.Rectangle(
            (s_df[[0.05]].values[0][0], e + 0.4),
            s_df[[0.95]].values[0][0] - s_df[[0.05]].values[0][0],
            0.2,
            color="k",
            alpha=1,
            fill=False,
            lw=lw,
        )
        rect4 = plt.Rectangle(
            (s_df[[0.16]].values[0][0], e + 0.2),
            s_df[[0.84]].values[0][0] - s_df[[0.16]].values[0][0],
            0.6,
            color="k",
            alpha=1,
            fill=False,
            lw=lw,
        )

        axs[0].add_patch(rect1)
        axs[0].add_patch(rect3)
        axs[0].add_patch(rect2)
        axs[0].add_patch(rect4)
        axs[0].add_patch(rect0)

        axs[0].set_ylabel(None)
        axs[0].axes.xaxis.set_visible(False)
        axs[0].axes.yaxis.set_visible(False)
        sns.despine(ax=axs[0], left=True, bottom=True)
        sns.despine(ax=axs[1], top=True)

        axs[0].set_ylim(0, len(all_ensembles))
        axs[1].legend().remove()

    l_as19 = Patch(facecolor="0.8", edgecolor="0.0", linewidth=0.75, label="AS19")
    l_flow = Patch(facecolor="0.8", edgecolor="0.0", linewidth=0.5, label="Flow Calib.")
    l_calib_s3 = Patch(facecolor="0.6", edgecolor="0.0", linewidth=0.25, label="Flow+Mass Calib. $\\beta$=3")
    l_calib_s2 = Patch(facecolor="0.4", edgecolor="0.0", linewidth=0.25, label="Flow+Mass Calib. $\\beta$=2")

    l_calib_s1 = Patch(facecolor="0.4", edgecolor="0.0", linewidth=0.25, label="Flow+Mass Calib. $\\beta$=1")

    # legend_1 = axs[1].legend(
    #     handles=[l_as19, l_flow, l_calib_s3, l_calib_s2, l_calib_s1],
    #     loc="upper left",
    #     bbox_to_anchor=(0.48, 1.1, 0, 0),
    # )
    # legend_1.get_frame().set_linewidth(0.0)
    # legend_1.get_frame().set_alpha(0.0)
    # axs[1].add_artist(legend_1)

    l_ = []
    for e, ens in enumerate(all_ensembles):
        l_.append(Line2D([], [], color=all_ensembles_palette[e], label=ens))
    legend = axs[1].legend(
        handles=l_,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.2, 0, 0),
    )
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)
    axs[1].add_artist(legend)

    fig.tight_layout()
    fig.savefig(out_filename)


def plot_histograms(out_filename, df):

    fig, axs = plt.subplots(4, 4, figsize=[6.2, 4.4])
    fig.subplots_adjust(hspace=0.5, wspace=0.4)

    f = sns.histplot(
        data=df,
        x="GCM",
        hue="Ensemble",
        common_norm=False,
        palette=palette_dict.values(),
        bins=[-0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25],
        stat="density",
        multiple="dodge",
        linewidth=0.5,
        ax=axs[0, 0],
    )
    f.get_legend().set_bbox_to_anchor([1, -2])
    f = sns.kdeplot(
        data=df,
        x="PRS",
        hue="Ensemble",
        common_grid=False,
        common_norm=False,
        palette=ts_median_palette_dict.values(),
        linewidth=0.75,
        ax=axs[1, 0],
    )
    f.get_legend().set_bbox_to_anchor([1, -1.75])
    sns.kdeplot(
        data=df,
        x="FICE",
        hue="Ensemble",
        common_grid=False,
        common_norm=False,
        palette=ts_median_palette_dict.values(),
        linewidth=0.75,
        ax=axs[0, 1],
        legend=False,
    )
    sns.kdeplot(
        data=df,
        x="FSNOW",
        hue="Ensemble",
        common_grid=False,
        common_norm=False,
        palette=ts_median_palette_dict.values(),
        linewidth=0.75,
        ax=axs[1, 1],
        legend=False,
    )
    sns.kdeplot(
        data=df,
        x="RFR",
        hue="Ensemble",
        common_grid=False,
        common_norm=False,
        palette=ts_median_palette_dict.values(),
        linewidth=0.5,
        ax=axs[2, 1],
        legend=False,
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
        linewidth=0.5,
        ax=axs[0, 2],
        legend=False,
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
        linewidth=0.5,
        ax=axs[1, 2],
        legend=False,
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
        linewidth=0.5,
        ax=axs[2, 2],
        legend=False,
    )
    sns.kdeplot(
        data=df,
        x="VCM",
        hue="Ensemble",
        common_grid=False,
        common_norm=False,
        palette=ts_median_palette_dict.values(),
        linewidth=0.75,
        ax=axs[3, 2],
        legend=False,
    )
    sns.kdeplot(
        data=df,
        x="SIAE",
        hue="Ensemble",
        common_grid=False,
        common_norm=False,
        palette=ts_median_palette_dict.values(),
        linewidth=0.5,
        ax=axs[0, 3],
        legend=False,
    )
    sns.kdeplot(
        data=df,
        x="PPQ",
        hue="Ensemble",
        common_grid=False,
        common_norm=False,
        palette=ts_median_palette_dict.values(),
        linewidth=0.75,
        ax=axs[1, 3],
        legend=False,
    )

    for ax, col in zip(axs[0], ["Climate", "Surface", "Ocean", "Ice Dynamics"]):
        ax.set_title(col)

    axs[2, 0].set_axis_off()
    axs[3, 0].set_axis_off()
    axs[3, 1].set_axis_off()
    axs[2, 3].set_axis_off()
    axs[3, 3].set_axis_off()

    fig.savefig(out_filename)


def load_df(respone_file, samples_file):

    response = pd.read_csv(respone_file)
    response["SLE (cm)"] = -response["Mass (Gt)"] / 362.5 / 10
    response = response.astype({"RCP": int})
    samples = pd.read_csv(samples_file).rename(columns={"id": "Experiment"})
    return pd.merge(response, samples, on="Experiment")


def resample_ensemble_by_data(
    observed,
    simulated,
    rcps,
    calibration_start=2008,
    calibration_end=2020,
    fudge_factor=2.0,
    n_samples=500,
    verbose=False,
):
    """
    Resampling algorithm by Douglas C. Brinkerhoff


    Parameters
    ----------
    observed : pandas.DataFrame
        A dataframe with observations
    simulated : pandas.DataFrame
        A dataframe with simulations
    calibration_start : float
        Start year for calibration
    calibration_end : float
        End year for calibration
    fudge_factor : float
        Tolerance for simulations. Calculated as fudge_factor * standard deviation of observed
    n_samples : int
        Number of samples to draw.

    """

    observed_calib_time = (observed["Year"] >= calibration_start) & (observed["Year"] <= calibration_end)
    observed_calib_period = observed[observed_calib_time]
    # Should we interpolate the simulations at observed time?
    observed_interp_mean = interp1d(observed_calib_period["Year"], observed_calib_period["Mass (Gt)"])
    observed_interp_std = interp1d(observed_calib_period["Year"], observed_calib_period["Mass uncertainty (Gt)"])

    simulated_calib_time = (simulated["Year"] >= calibration_start) & (simulated["Year"] <= calibration_end)
    simulated_calib_period = simulated[simulated_calib_time]

    resampled_list = []
    for rcp in rcps:
        log_likes = []
        experiments = np.unique(simulated_calib_period["Experiment"])
        evals = []
        for i in experiments:
            exp_ = simulated_calib_period[
                (simulated_calib_period["Experiment"] == i) & (simulated_calib_period["RCP"] == rcp)
            ]
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
        resampled_experiments = np.random.choice(experiments, n_samples, p=weights)
        new_frame = []
        for i in resampled_experiments:
            new_frame.append(simulated[(simulated["Experiment"] == i) & (simulated["RCP"] == rcp)])
        simulated_resampled = pd.concat(new_frame)
        resampled_list.append(simulated_resampled)

    simulated_resampled = pd.concat(resampled_list)
    return simulated_resampled


def make_quantile_table(q_df):
    ensembles = ["AS19", "Flow Calib.", "Mass Calib.", "Flow+Mass Calib. S3"]
    table_header = """
    \\begin{table}
    \\fontsize{6}{7.2}\selectfont
    \centering
    \caption{This is a table with scientific results.}
    \medskip
    \\begin{tabular}{p{.06\\textwidth}p{.105\\textwidth}p{.08\\textwidth}p{.105\\textwidth}p{.08\\textwidth}p{.105\\textwidth}p{.08\\textwidth}p{.105\\textwidth}p{.08\\textwidth}}
    \hline
    """

    ls = []

    f = "".join([f"& \multicolumn{{2}}{{l}}{{{ens}}}" for ens in ensembles])
    ls.append(f"{f} \\\\ \n")
    ls.append("\cline{2-9} \\\\ \n")
    f = "& {:.0f}th [{:.0f}th, {:.0f}th] & [{:.0f}th, {:.0f}th] ".format(*(np.array(quantiles) * 100))
    g = f * len(ensembles)
    ls.append(f" {g} \\\\ \n")
    q_str = "& percentiles " * len(ensembles) * 2
    ls.append(f"{q_str} \\\\ \n")
    sle_str = "& (cm SLE) " * len(ensembles * 2)
    ls.append(f"{sle_str} \\\\ \n")
    ls.append("\hline \n")

    for rcp in rcps:
        a = q_df[q_df["RCP"] == rcp]
        f = "& ".join(
            [
                "{:.0f} [{:.0f}, {:.0f}] & [{:.0f}, {:.0f} ]".format(*a[a["Ensemble"] == ens].values[0][2::])
                for ens in ensembles
            ]
        )
        ls.append(f"{rcp_dict[rcp]} & {f} \\\\ \n")

    table_footer = """
    \hline
    \end{tabular}
    \caption{tab:sle}
    \end{table}
    """

    print("".join([table_header, *ls, table_footer]))


def make_quantile_df(df, quantiles):
    q_dfs = [
        df.groupby(by=["RCP", "Ensemble"])["SLE (cm)"].quantile(q).reset_index().rename(columns={"SLE (cm)": q})
        for q in quantiles
    ]
    q_df = reduce(lambda df1, df2: pd.merge(df1, df2, on=["RCP", "Ensemble"]), q_dfs)
    a_dfs = [
        df.groupby(by=["Ensemble"])["SLE (cm)"].quantile(q).reset_index().rename(columns={"SLE (cm)": q})
        for q in quantiles
    ]
    a_df = reduce(lambda df1, df2: pd.merge(df1, df2, on=["Ensemble"]), a_dfs)
    a_df["RCP"] = "Union"
    return pd.concat([q_df, a_df])


signal_lw = 1.0
obs_signal_color = "#6a51a3"
obs_sigma_color = "#cbc9e2"

secpera = 3.15569259747e7
gt2cmSLE = 1.0 / 362.5 / 10.0

rcps = [26, 45, 85]
rcpss = [26, 45, 85, "Union"]
rcp_col_dict = {26: "#003466", 45: "#5492CD", 85: "#990002"}
rcp_shade_col_dict = {26: "#4393C3", 45: "#92C5DE", 85: "#F4A582"}
rcp_dict = {26: "RCP 2.6", 45: "RCP 4.5", 85: "RCP 8.5"}
palette_dict = {
    "AS19": "0.70",
    "Flow Calib.": "0.8",
    "Mass Calib.": "#fee6ce",
    "Flow+Mass Calib.": "0.60",
}
ts_fill_palette_dict = {
    "AS19": "0.80",
    "Flow Calib.": "0.70",
    "Mass Calib.": "#fee6ce",
    "Flow+Mass Calib.": "0.60",
}
ts_median_palette_dict = {
    "AS19": "0.50",
    "Flow Calib.": "0.25",
    "Mass Calib.": "#e6550d",
    "Flow+Mass Calib.": "0.0",
}

# cm = sns.color_palette("ch:s=-.2,r=.6", n_colors=6, as_cmap=False).as_hex()
# ts_median_palette_dict = {"AS19": cm[0], "Flow Calib.": cm[1], "Resampled": cm[2]}

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
    "hatch.linewidth": 0.25,
}

plt.rcParams.update(params)


if __name__ == "__main__":

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Two-step Bayesian calibration for Aschwanden et al (2019) ."
    parser.add_argument(
        "--as19_results_file",
        nargs=1,
        help="Comma-separated file with AS19 results",
        default="../data/as19/aschwanden_et_al_2019_les_2008_norm.csv.gz",
    )
    parser.add_argument(
        "--as19_samples_file",
        nargs=1,
        help="Comma-separated file with AS19 samples",
        default="../data/samples/lhs_samples_500.csv",
    )
    parser.add_argument(
        "--calibrated_results_file",
        nargs=1,
        help="Comma-separated file with calibrated results",
        default="../data/as19/aschwanden_et_al_2019_mc_2008_norm.csv.gz",
    )
    parser.add_argument(
        "--calibrated_samples_file",
        nargs=1,
        help="Comma-separated file with calibrated samples",
        default="../data/samples/lhs_plus_mc_samples.csv",
    )
    options = parser.parse_args()

    # Load Observations
    observed = load_imbie()

    # Load AS19 (original LES)
    as19 = load_df(options.as19_results_file, options.as19_samples_file)
    # Load AS19 (original CTRL)
    as19_ctrl = load_df("../data/as19/aschwanden_et_al_2019_ctrl.csv.gz", "../data/samples/lhs_control.csv")
    # Load AS19 (with calibrated ice dynamics)
    calib = load_df(options.calibrated_results_file, options.calibrated_samples_file)

    # Bayesian calibration: resampling
    as19_resampled = resample_ensemble_by_data(observed, as19, rcps, fudge_factor=2)
    as19_resampled_3 = resample_ensemble_by_data(observed, as19, rcps, fudge_factor=3)
    as19_resampled_2 = resample_ensemble_by_data(observed, as19, rcps, fudge_factor=2)
    as19_resampled_1 = resample_ensemble_by_data(observed, as19, rcps, fudge_factor=1)
    as19_calib_resampled_3 = resample_ensemble_by_data(observed, calib, rcps, fudge_factor=3)
    as19_calib_resampled_2 = resample_ensemble_by_data(observed, calib, rcps, fudge_factor=2)
    as19_calib_resampled_1 = resample_ensemble_by_data(observed, calib, rcps, fudge_factor=1)
    as19_calib_resampled = resample_ensemble_by_data(observed, calib, rcps, fudge_factor=2)

    as19["Ensemble"] = "AS19"
    calib["Ensemble"] = "Flow Calib."
    as19_resampled["Ensemble"] = "Mass Calib."
    as19_resampled_3["Ensemble"] = "Mass Calib. S3"
    as19_resampled_2["Ensemble"] = "Mass Calib. S2"
    as19_resampled_1["Ensemble"] = "Mass Calib. S1"
    as19_calib_resampled_3["Ensemble"] = "Flow+Mass Calib. S3"
    as19_calib_resampled_2["Ensemble"] = "Flow+Mass Calib. S2"
    as19_calib_resampled_1["Ensemble"] = "Flow+Mass Calib. S1"
    as19_calib_resampled["Ensemble"] = "Flow+Mass Calib."
    all_df = (
        pd.concat(
            [
                as19,
                calib,
                as19_resampled,
                as19_resampled_3,
                as19_resampled_2,
                as19_resampled_1,
                as19_calib_resampled_3,
                as19_calib_resampled_2,
                as19_calib_resampled_1,
                as19_calib_resampled,
            ]
        )
        .drop_duplicates(subset=None, keep="first", inplace=False)
        .reset_index()
    )

    year = 2100
    all_2100_df = all_df[(all_df["Year"] == year)]
    quantiles = [0.5, 0.05, 0.95, 0.16, 0.84]
    q_df = make_quantile_df(all_2100_df, quantiles)

    plot_partitioning("historical_partitioning.pdf", simulated=all_df, observed=observed)
    plot_historical("historical.pdf", simulated=all_df, observed=observed)
    plot_projection("projection_as19.pdf", simulated=all_df, ensemble="AS19", quantiles=[0.05, 0.16, 0.84, 0.95])
    plot_projection(
        "projection_flow.pdf", simulated=all_df, ensemble="Flow Calib.", quantiles=[0.05, 0.16, 0.84, 0.95]
    )
    plot_projection(
        "projection_flowmass.pdf", simulated=all_df, ensemble="Flow+Mass Calib.", quantiles=[0.05, 0.16, 0.84, 0.95]
    )
    plot_projection(
        "projection_as19_bars.pdf",
        simulated=all_df,
        ensemble="AS19",
        quantiles=[0.05, 0.16, 0.84, 0.95],
        bars=["AS19"],
    )
    plot_projection(
        "projection_flow_bars.pdf",
        simulated=all_df,
        ensemble="Flow Calib.",
        quantiles=[0.05, 0.16, 0.84, 0.95],
        bars=["AS19", "Flow Calib."],
    )
    plot_projection(
        "projection_calibrated_bars.pdf",
        simulated=all_df,
        ensemble="Flow+Mass Calib. S2",
        quantiles=[0.05, 0.16, 0.84, 0.95],
        bars=["Flow+Mass Calib. S2"],
    )
    plot_projection(
        "projection_calibrated_bars_s1.pdf",
        simulated=all_df,
        ensemble="Flow+Mass Calib. S1",
        quantiles=[0.05, 0.16, 0.84, 0.95],
        bars=["AS19", "Flow Calib.", "Flow+Mass Calib. S1"],
    )
    plot_projection(
        "projection_calibrated_bars_s2.pdf",
        simulated=all_df,
        ensemble="Flow+Mass Calib. S2",
        quantiles=[0.05, 0.16, 0.84, 0.95],
        bars=["AS19", "Flow Calib.", "Flow+Mass Calib. S2"],
    )
    plot_projection(
        "projection_calibrated_bars_s3.pdf",
        simulated=all_df,
        ensemble="Flow+Mass Calib. S3",
        quantiles=[0.05, 0.16, 0.84, 0.95],
        bars=["AS19", "Flow Calib.", "Flow+Mass Calib. S3"],
    )
    plot_posterior_sle_pdfs(f"sle_pdf_{year}.pdf", all_df, year=year)
    compare_sle_pdfs_1(f"sle_pdf_betas_{year}.pdf", all_df, year=year)
    plot_histograms(f"histograms_{year}.pdf", all_2100_df)

    make_quantile_table(q_df)

    q_df["90%"] = q_df[0.95] - q_df[0.05]
    q_df["68%"] = q_df[0.84] - q_df[0.16]
    q_df.astype({"90%": np.float32, "68%": np.float32})

    q_abs = q_df[q_df["Ensemble"] == "Flow+Mass Calib."][["90%", "68%", 0.5]].reset_index(drop=True) - q_df[
        q_df["Ensemble"] == "AS19"
    ][["90%", "68%", 0.5]].reset_index(drop=True)

    q_rel = q_abs / q_df[q_df["Ensemble"] == "AS19"][["90%", "68%", 0.5]].reset_index(drop=True) * 100

    q_abs["RCP"] = rcpss
    q_rel["RCP"] = rcpss
    print(q_rel)
