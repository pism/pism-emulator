#!/usr/bin/env python

# Copyright (C) 2020-22 Andy Aschwanden, Douglas J. Brinkerhoff

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from functools import reduce
from itertools import cycle

import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import interp1d
from scipy.stats import beta
from scipy.stats.distributions import randint, truncnorm, uniform

from pismemulator.utils import load_imbie, load_imbie_csv
from pismemulator.utils import param_keys_dict as keys_dict


def add_inner_title(ax, title, loc="upper left", size=7, **kwargs):
    """
    Adds an inner title to a given axis, with location loc.

    from http://matplotlib.sourceforge.net/examples/axes_grid/demo_axes_grid2.html
    """
    from matplotlib.offsetbox import AnchoredText

    prop = dict(size=size, weight="bold")
    at = AnchoredText(
        title, loc=loc, prop=prop, pad=0.0, borderpad=0.5, frameon=False, **kwargs
    )
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
    left = ax.figure.subplotpars.left
    right = ax.figure.subplotpars.right
    top = ax.figure.subplotpars.top
    bottom = ax.figure.subplotpars.bottom
    figw = float(w) / (right - left)
    figh = float(h) / (top - bottom)
    ax.figure.set_size_inches(figw, figh)


def plot_historical(
    out_filename,
    simulated=None,
    observed=None,
    ensembles=["AS19", "Flow+Mass Calib."],
    quantiles=[0.05, 0.95],
    sigma=2,
    simulated_ctrl=None,
    xlims=[2008, 2021],
    ylims=[-10000, 500],
):
    """
    Plot historical simulations and observations
    """

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
                handles=legend_handles,
                loc="lower left",
                ncol=1,
                title=ens,
                bbox_to_anchor=(r * 0.2, 0.01),
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

        if simulated is None:
            r = 0
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

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax_sle = ax.twinx()
    ax_sle.set_ylabel(f"Contribution to sea-level \nsince {proj_start} (cm SLE)")
    ax_sle.set_ylim(-np.array(ylims) * gt2cmSLE)

    fig.savefig(out_filename, bbox_inches="tight")
    plt.close(fig)


def plot_projection(
    out_filename,
    simulated=None,
    ensemble="Flow+Mass Calib.",
    quantiles=[0.05, 0.95],
    bars=None,
    quantile_df=None,
    xlims=[2008, 2100],
    ylims=[-0.5, 45],
):
    """
    Plot historical simulations and observations
    """

    if bars:
        fig, axs = plt.subplots(
            1,
            4,
            sharey="row",
            figsize=[6.0, 2.2],
            gridspec_kw=dict(width_ratios=[60, len(bars), len(bars), len(bars)]),
        )
        fig.subplots_adjust(hspace=0.1, wspace=0.05)
        ax = axs[0]
    else:
        fig = plt.figure(figsize=[5.2, 2.2])
        ax = fig.add_subplot(111)

    if simulated is not None:
        for r, rcp in enumerate(rcps):
            legend_handles = []
            sim = simulated[
                (simulated["Ensemble"] == ensemble) & (simulated["RCP"] == rcp)
            ]
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
                handles=legend_handles,
                title=rcp_dict[rcp],
                loc="upper left",
                bbox_to_anchor=(r * 0.2, 0.99),
            )
            legend.get_frame().set_linewidth(0.0)
            legend.get_frame().set_alpha(0.0)
            ax.add_artist(legend)

    ax.set_xlabel("Year")
    ax.set_ylabel(f"Contribution to sea-level\nsince {proj_start} (cm SLE)")

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

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
        q_df = make_quantile_df(
            simulated[simulated["Year"] == 2100],
            quantiles=[0.05, 0.16, 0.5, 0.84, 0.95],
        )
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
                        Patch(
                            facecolor=None,
                            edgecolor="k",
                            fill=None,
                            lw=0.25,
                            hatch=hatch,
                            label=ens,
                        ),
                    )
        # legend_2 = ax.legend(handles=legend_elements, loc="upper right")
        # legend_2.get_frame().set_linewidth(0.0)
        # legend_2.get_frame().set_alpha(0.0)

        for a in [1, 2, 3]:
            sns.despine(ax=axs[a], left=True, bottom=True)
            axs[a].set_ylabel(None)
            axs[a].axes.xaxis.set_visible(False)
            axs[a].axes.yaxis.set_visible(False)

    fig.savefig(out_filename, bbox_inches="tight")
    plt.close(fig)


def plot_partitioning(
    out_filename,
    simulated=None,
    observed=None,
    ensembles=["AS19", "Flow Calib.", "Flow+Mass Calib."],
    quantiles=[0.05, 0.95],
    sigma=2,
    simulated_ctrl=None,
    xlims=[2010, 2020],
):
    ncol = 0
    if simulated is not None:
        ncol += len(ensembles)
    if observed is not None:
        ncol += 1

    fig, axs = plt.subplots(
        2,
        1,
        sharex="col",
        figsize=[3.2, 2.6],
        gridspec_kw=dict(height_ratios=[1, 1]),
    )
    fig.subplots_adjust(hspace=0.1, wspace=0.25)

    if simulated is not None:
        for r, ens in enumerate(ensembles):
            legend_handles = []
            sim = simulated[simulated["Ensemble"] == ens]
            for k, (v, u) in enumerate(zip(["D", "SMB"], ["Gt/yr", "Gt/yr"])):
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
                    label="Median",
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

            legend = axs[1].legend(
                bbox_to_anchor=(1.04, 0.4 + r * 0.4),
                borderaxespad=0,
                handles=legend_handles,
                loc="lower left",
                ncol=1,
                title=ens,
            )
            legend._legend_box.align = "left"
            legend.get_frame().set_linewidth(0.0)
            legend.get_frame().set_alpha(0.0)
            axs[1].add_artist(legend)

    if observed is not None:
        legend_handles = []
        for k, (v, u) in enumerate(zip(["D", "SMB"], ["Gt/yr", "Gt/yr"])):
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

        legend = axs[1].legend(
            handles=legend_handles,
            loc="lower left",
            ncol=1,
            title="Observed (IMBIE)",
            bbox_to_anchor=(1.04, -0.1),
        )
        legend._legend_box.align = "left"
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)
        axs[1].add_artist(legend)

    for k, (v, u) in enumerate(zip(["D", "SMB"], ["Gt/yr", "Gt/yr"])):
        axs[k].set_ylabel(f"{v} ({u})")

    for k, (v, u) in enumerate(zip(["a", "b"], ["", ""])):
        add_inner_title(axs[k], f"{v}) {u}")

    axs[-1].set_xlim(xlims)
    axs[-1].set_xlabel("Year")
    axs[1].set_ylim(-750, 750)
    axs[0].set_ylim(-1500, 0)

    fig.savefig(out_filename, bbox_inches="tight")
    plt.close(fig)


def plot_posterior_sle_pdfs(
    out_filename,
    df,
    observed=None,
    rcps=[26, 45, 85],
    ensembles=["AS19", "Flow Calib.", "Flow+Mass Calib."],
    years=[2020, 2100],
    ylim=None,
):
    n_rcps = len(rcps)
    legend_rcp = 85
    alphas = [0.4, 0.7, 1.0]
    m_alphas = alphas[: len(ensembles)]

    fig, axs = plt.subplots(
        n_rcps * 2,
        2,
        sharex="col",
        figsize=[5.8, 4.2],
        gridspec_kw=dict(height_ratios=[0.30 * len(ensembles), 4] * n_rcps),
    )
    fig.subplots_adjust(hspace=0.0, wspace=0)
    for k, rcp in enumerate(rcps):
        for y, year in enumerate(years):
            y_df = df[df["Year"] == year]
            q_df = make_quantile_df(y_df, quantiles=[0.05, 0.16, 0.5, 0.84, 0.95])

            m_df = y_df[y_df["RCP"] == rcp]
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
                palette=[color_tint(rcp_col_dict[rcp], alpha) for alpha in m_alphas],
                ax=axs[k * 2 + 1, y],
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
                lw=0.8,
                palette=[color_tint(rcp_col_dict[rcp], alpha) for alpha in m_alphas],
                ax=axs[k * 2 + 1, y],
            )

            for e, ens in enumerate(ensembles):
                s_df = p_df[p_df["Ensemble"] == ens]
                mk_df = y_df[y_df["Ensemble"] == ens]

                alpha = alphas[e]
                m_color = color_tint(rcp_col_dict[rcp], alpha)
                lw = 0.25

                axs[(k * 2), y].vlines(
                    s_df[[0.5]].values[0][0], e, e + 1, colors="k", lw=1
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

                axs[(k * 2), y].add_patch(rect1)
                axs[(k * 2), y].add_patch(rect3)
                axs[(k * 2), y].add_patch(rect2)
                axs[(k * 2), y].add_patch(rect4)

                axs[(k * 2), y].set_ylabel(None)
                axs[(k * 2), y].axes.xaxis.set_visible(False)
                axs[(k * 2), y].axes.yaxis.set_visible(False)
                sns.despine(ax=axs[(k * 2), y], left=True, bottom=True)
                sns.despine(ax=axs[(k * 2) + 1, y], top=True)

                axs[(k * 2), y].set_ylim(0, len(ensembles))

                if y > 0:
                    axs[k * 2 + 1, y].set_ylabel(None)

                axs[k, y].legend().remove()
                axs[k * 2 + 1, y].legend().remove()

                axs[0, y].set_title(f"Year {year}")
                if ylim is not None:
                    axs[(k * 2) + 1, y].set_ylim(ylim)

                if (k == 0) and (e == 0) and (y == 0):
                    for pctl in [0.05, 0.16, 0.5, 0.84, 0.95]:
                        axs[0, 0].text(
                            s_df[[pctl]].values[0][0],
                            -1.5,
                            int(pctl * 100),
                            ha="center",
                            fontsize=5,
                        )

        if observed is not None:
            obs = observed[
                (observed["Year"] >= years[0]) & (observed["Year"] < years[0] + 1)
            ]
            obs_mean = obs["SLE (cm)"].mean()
            obs_std = obs["SLE uncertainty (cm)"].mean()
            axs[(k * 2) + 1, 0].axvline(obs_mean, c="k", lw=0.5)
            axs[(k * 2) + 1, 0].axvline(
                obs_mean - 2 * obs_std, c="k", lw=0.5, ls="dotted"
            )
            axs[(k * 2) + 1, 0].axvline(
                obs_mean + 2 * obs_std, c="k", lw=0.5, ls="dotted"
            )

    for k, rcp in enumerate(rcps):
        axs[k * 2, 0].text(
            -0.125,
            0.2,
            rcp_dict[rcp],
            transform=axs[k * 2, 0].transAxes,
            fontsize=7,
            fontweight="bold",
            horizontalalignment="left",
        )

    l_as19 = Patch(
        facecolor=color_tint(rcp_col_dict[legend_rcp], alphas[0]),
        edgecolor="0.0",
        linewidth=0.25,
        label="Prior (AS19)",
    )
    l_ismip6 = Patch(
        facecolor=color_tint(rcp_col_dict[legend_rcp], alphas[0]),
        edgecolor="0.0",
        linewidth=0.25,
        label="Prior (ISMIP6)",
    )
    l_flow = Patch(
        facecolor=color_tint(rcp_col_dict[legend_rcp], alphas[1]),
        edgecolor="0.0",
        linewidth=0.25,
        label="Posterior (Flow Calib.)",
    )
    l_mass = Patch(
        facecolor=color_tint(rcp_col_dict[legend_rcp], alphas[1]),
        edgecolor="0.0",
        linewidth=0.25,
        label="Posterior (Mass Calib.)",
    )
    l_calib = Patch(
        facecolor=color_tint(rcp_col_dict[legend_rcp], alphas[2]),
        edgecolor="0.0",
        linewidth=0.25,
        label="Posterior (Flow+Mass Calib.)",
    )
    l_ismip6_calib = Patch(
        facecolor=color_tint(rcp_col_dict[legend_rcp], alphas[2]),
        edgecolor="0.0",
        linewidth=0.25,
        label="Posterior (ISMIP6 Calib.)",
    )

    ens_label_dict = {
        "AS19": l_as19,
        "Flow Calib.": l_flow,
        "Mass Calib.": l_mass,
        "Flow+Mass Calib.": l_calib,
        "ISMIP6": l_ismip6,
        "ISMIP6 Calib.": l_ismip6_calib,
    }

    legend_1 = axs[-1, 0].legend(
        handles=[ens_label_dict[e] for e in ensembles],
        loc="lower left",
        bbox_to_anchor=(0.4, 0.45, 0, 0),
    )
    legend_1.get_frame().set_linewidth(0.0)
    legend_1.get_frame().set_alpha(0.0)
    axs[-1, 0].add_artist(legend_1)

    if observed is not None:
        l_obs_mean = Line2D(
            [], [], c="k", lw=0.5, ls="solid", label="Observed (IMBIE) mean"
        )
        l_obs_std = Line2D(
            [], [], c="k", lw=0.5, ls="dotted", label="Observed (IMBIE) $\pm2-\sigma$"
        )
        legend_2 = axs[-3, 0].legend(
            handles=[l_obs_mean, l_obs_std],
            loc="lower left",
            bbox_to_anchor=(0.4, 0.45, 0, 0),
        )
        legend_2.get_frame().set_linewidth(0.0)
        legend_2.get_frame().set_alpha(0.0)

    fig.tight_layout()
    fig.savefig(out_filename)
    plt.close(fig)


def plot_posterior_sle_pdf(
    out_filename,
    df,
    observed=None,
    year=2100,
    ensembles=["AS19", "Flow Calib.", "Flow+Mass Calib."],
    ylim=None,
):
    legend_rcp = 85
    alphas = [0.4, 0.7, 1.0]
    m_alphas = alphas[: len(ensembles)]
    fig, axs = plt.subplots(
        6,
        1,
        sharex="col",
        figsize=[3.2, 4.2],
        gridspec_kw=dict(height_ratios=[0.30 * len(ensembles), 4] * 3),
    )
    fig.subplots_adjust(hspace=0.0, wspace=0)
    for k, rcp in enumerate(rcps):
        y_df = df[df["Year"] == year]
        q_df = make_quantile_df(y_df, quantiles=[0.05, 0.16, 0.5, 0.84, 0.95])

        m_df = y_df[y_df["RCP"] == rcp]
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
            palette=[color_tint(rcp_col_dict[rcp], alpha) for alpha in m_alphas],
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
            lw=0.8,
            palette=[color_tint(rcp_col_dict[rcp], alpha) for alpha in m_alphas],
            ax=axs[k * 2 + 1],
        )

        for e, ens in enumerate(ensembles):
            s_df = p_df[p_df["Ensemble"] == ens]
            mk_df = y_df[y_df["Ensemble"] == ens]

            alpha = alphas[e]
            m_color = color_tint(rcp_col_dict[rcp], alpha)
            lw = 0.25

            axs[(k * 2)].vlines(s_df[[0.5]].values[0][0], e, e + 1, colors="k", lw=1)

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

            if (k == 0) and (e == 0):
                for pctl in [0.05, 0.16, 0.5, 0.84, 0.95]:
                    axs[0].text(
                        s_df[[pctl]].values[0][0], -1.5, int(pctl * 100), ha="center"
                    )

            axs[(k * 2)].set_ylabel(None)
            axs[(k * 2)].axes.xaxis.set_visible(False)
            axs[(k * 2)].axes.yaxis.set_visible(False)
            sns.despine(ax=axs[(k * 2)], left=True, bottom=True)
            sns.despine(ax=axs[(k * 2) + 1], top=True)

            axs[(k * 2)].set_ylim(0, len(ensembles))
            if ylim is not None:
                axs[(k * 2) + 1].set_ylim(ylim)

            axs[k].legend().remove()
            axs[k * 2 + 1].legend().remove()

        if observed is not None:
            obs = observed[(observed["Year"] >= year) & (observed["Year"] < year + 1)]
            obs_mean = obs["SLE (cm)"].mean()
            obs_std = obs["SLE uncertainty (cm)"].mean()
            axs[(k * 2) + 1].axvline(obs_mean, c="k", lw=0.5)
            axs[(k * 2) + 1].axvline(obs_mean - 2 * obs_std, c="k", lw=0.5, ls="dotted")
            axs[(k * 2) + 1].axvline(obs_mean + 2 * obs_std, c="k", lw=0.5, ls="dotted")

    axs[0].set_title(f"Year {year}")

    for k, rcp in enumerate(rcps):
        add_inner_title(axs[k * 2 + 1], rcp_dict[rcp])

    l_as19 = Patch(
        facecolor=color_tint(rcp_col_dict[legend_rcp], alphas[0]),
        edgecolor="0.0",
        linewidth=0.25,
        label="Prior (AS19)",
    )
    l_flow = Patch(
        facecolor=color_tint(rcp_col_dict[legend_rcp], alphas[1]),
        edgecolor="0.0",
        linewidth=0.25,
        label="Posterior (Flow Calib.)",
    )
    l_mass = Patch(
        facecolor=color_tint(rcp_col_dict[legend_rcp], alphas[1]),
        edgecolor="0.0",
        linewidth=0.25,
        label="Posterior (Mass Calib.)",
    )
    l_calib = Patch(
        facecolor=color_tint(rcp_col_dict[legend_rcp], alphas[2]),
        edgecolor="0.0",
        linewidth=0.25,
        label="Posterior (Flow+Mass Calib.)",
    )

    ens_label_dict = {
        "AS19": l_as19,
        "Flow Calib.": l_flow,
        "Mass Calib.": l_mass,
        "Flow+Mass Calib.": l_calib,
    }

    legend_1 = axs[-1].legend(
        handles=[ens_label_dict[e] for e in ensembles],
        loc="lower left",
        bbox_to_anchor=(0.4, 0.45, 0, 0),
    )
    legend_1.get_frame().set_linewidth(0.0)
    legend_1.get_frame().set_alpha(0.0)
    axs[-1].add_artist(legend_1)

    if observed is not None:
        l_obs_mean = Line2D(
            [], [], c="k", lw=0.5, ls="solid", label="Observed (IMBIE) mean"
        )
        l_obs_std = Line2D(
            [], [], c="k", lw=0.5, ls="dotted", label="Observed (IMBIE) $\pm2-\sigma$"
        )
        legend_2 = axs[-3].legend(
            handles=[l_obs_mean, l_obs_std],
            loc="lower left",
            bbox_to_anchor=(0.4, 0.45, 0, 0),
        )
        legend_2.get_frame().set_linewidth(0.0)
        legend_2.get_frame().set_alpha(0.0)

    fig.tight_layout()
    fig.savefig(out_filename)
    plt.close(fig)
    del fig


def plot_histograms(
    out_filename,
    df,
    X_prior=None,
    ensembles=["AS19", "Flow Calib.", "Flow+Mass Calib."],
    palette="binary",
):
    m_flow_keys = [
        "SIAE",
        "PPQ",
        "TEFO",
        "SSAN",
        "ZMIN",
        "ZMAX",
        "PHIMIN",
        "PHIMAX",
    ]
    m_star_keys = ["GCM", "RFR", "FICE", "FSNOW", "PRS", "OCM", "OCS", "TCT", "VCM"]
    m_keys = m_flow_keys + m_star_keys
    m_as19_df = df[df["Ensemble"] == "AS19"][m_keys]
    m_flow_df = df[df["Ensemble"] == "Flow Calib."][m_flow_keys]
    m_mass_df = df[df["Ensemble"] == "Flow+Mass Calib."][m_keys]

    p_dict = {
        "SIAE": {"axs": [0, 0], "bins": np.linspace(1, 4, 11)},
        "PPQ": {"axs": [0, 1], "bins": np.linspace(0.1, 0.9, 11)},
        "TEFO": {"axs": [0, 2], "bins": np.linspace(0.005, 0.035, 11)},
        "SSAN": {"axs": [0, 3], "bins": np.linspace(3.0, 3.5, 11)},
        "ZMIN": {"axs": [1, 0], "bins": np.linspace(-1000, 0, 11)},
        "ZMAX": {"axs": [1, 1], "bins": np.linspace(0, 1000, 11)},
        "PHIMIN": {"axs": [1, 2], "bins": np.linspace(5, 15, 11)},
        "PHIMAX": {"axs": [1, 3], "bins": np.linspace(40, 45, 11)},
        "GCM": {
            "axs": [4, 0],
            "bins": [-0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25],
            "dist": randint(0, 4),
        },
        "PRS": {
            "axs": [4, 1],
            "bins": np.linspace(5, 7, 11),
            "dist": uniform(loc=5, scale=2),
        },
        "FICE": {
            "axs": [3, 1],
            "bins": np.linspace(4, 12, 11),
            "dist": truncnorm(-4 / 4.0, 4.0 / 4, loc=8, scale=4),
        },
        "FSNOW": {
            "axs": [3, 0],
            "bins": np.linspace(2, 6, 11),
            "dist": truncnorm(-4.1 / 3, 4.1 / 3, loc=4.1, scale=1.5),
        },
        "RFR": {
            "axs": [3, 2],
            "bins": np.linspace(0.2, 0.8, 16),
            "dist": truncnorm(-0.4 / 0.3, 0.4 / 0.3, loc=0.5, scale=0.2),
        },
        "OCM": {
            "axs": [2, 0],
            "bins": [-1.25, -0.75, -0.25, 0.25, 0.75, 1.25],
            "dist": randint(-1, 2),
        },
        "OCS": {
            "axs": [2, 1],
            "bins": [-1.25, -0.75, -0.25, 0.25, 0.75, 1.25],
            "dist": randint(-1, 2),
        },
        "TCT": {
            "axs": [2, 2],
            "bins": [-1.25, -0.75, -0.25, 0.25, 0.75, 1.25],
            "dist": randint(-1, 2),
        },
        "VCM": {
            "axs": [2, 3],
            "bins": np.linspace(0.75, 1.25, 11),
            "dist": truncnorm(-0.35 / 0.2, 0.35 / 0.2, loc=1, scale=0.2),
        },
    }

    if X_prior is not None:
        X = X_prior[m_flow_keys]
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_keys = X_prior.keys()

        n_samples, n_parameters = X.shape

        X_min = (((X.min(axis=0) - X_mean) / X_std - 1e-3) * X_std + X_mean).values
        X_max = (((X.max(axis=0) - X_mean) / X_std + 1e-3) * X_std + X_mean).values

        alpha_b = 3.0
        beta_b = 3.0
        X_prior_b = (
            beta.rvs(alpha_b, beta_b, size=(100000, n_parameters)) * (X_max - X_min)
            + X_min
        )

    fig, axs = plt.subplots(
        5,
        4,
        figsize=[4.8, 5.2],
    )
    fig.subplots_adjust(hspace=1.25, wspace=0.0)

    cmap = sns.color_palette(palette, n_colors=3)

    for key in p_dict.keys():
        m_axs = p_dict[key]["axs"]
        m_bins = p_dict[key]["bins"]

        sns.histplot(
            data=df,
            x=key,
            hue="Ensemble",
            hue_order=ensembles,
            common_norm=False,
            bins=m_bins,
            palette=palette,
            stat="density",
            multiple="dodge",
            alpha=0.8,
            linewidth=0.2,
            ax=axs[m_axs[0], m_axs[1]],
            legend=False,
        )
        if key not in ["GCM", "OCM", "OCS", "TCT"]:
            sns.kdeplot(
                data=df,
                x=key,
                hue="Ensemble",
                hue_order=ensembles,
                clip=[m_bins[0], m_bins[-1]],
                common_norm=False,
                warn_singular=False,
                palette=palette,
                linewidth=lw * 1.25,
                ax=axs[m_axs[0], m_axs[1]],
                legend=False,
            )
        if (X_prior_b is not None) and (key in m_flow_keys):
            X_prior_m = pd.DataFrame(data=X_prior_b, columns=m_flow_keys)
            X_prior_hist, b = np.histogram(X_prior_m[key], m_bins, density=True)
            b = 0.5 * (b[1:] + b[:-1])

            axs[m_axs[0], m_axs[1]].plot(
                b,
                X_prior_hist,
                color="k",
                linewidth=lw,
                linestyle="dashed",
            )
        elif (X_prior_b is not None) and (key in m_star_keys):
            X_prior_b = X_prior[m_star_keys]
            X_prior_m = pd.DataFrame(data=X_prior_b, columns=m_star_keys)
            X_prior_hist, b = np.histogram(X_prior_m[key], m_bins, density=True)
            b = 0.5 * (b[1:] + b[:-1])
            if key not in ["GCM", "OCM", "OCS", "TCT"]:
                axs[m_axs[0], m_axs[1]].plot(
                    b,
                    X_prior_hist,
                    color="k",
                    linewidth=lw,
                    linestyle="dashed",
                )
            else:
                axs[m_axs[0], m_axs[1]].plot(
                    b[::2],
                    X_prior_hist[::2],
                    "s",
                    color="k",
                )
        else:
            pass
        # if (X_prior_b is not None) and (key in m_star_keys):
        #     axs[m_axs[0], m_axs[1]].plot(
        #         [0, 1],
        #         [0.5, 0.5],
        #         color="k",
        #         lw=lw,
        #         transform=axs[m_axs[0], m_axs[1]].transAxes,
        #     )

    handles = [
        Patch(
            facecolor=cmap[k],
            edgecolor="0.0",
            linewidth=0.25,
            label=ens,
        )
        for k, ens in enumerate(ensembles)
    ]
    if X_prior_b is not None:
        l_p = Line2D(
            [],
            [],
            c="k",
            lw=lw,
            ls="dashed",
            label="Prior",
        )

        handles.append(l_p)
    legend_1 = axs[4, 2].legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0, -0.2),
    )
    legend_1.get_frame().set_linewidth(0.0)
    legend_1.get_frame().set_alpha(0.0)

    axs[3, 3].set_axis_off()
    axs[4, 2].set_axis_off()
    axs[4, 3].set_axis_off()

    axs[0, 0].text(
        0,
        1.05,
        "$\mathbf{m}_{\mathrm{flow}}$",
        transform=axs[0, 0].transAxes,
        size=8,
    )
    axs[2, 0].text(0, 1.05, "$\mathbf{m}^{*}$", transform=axs[2, 0].transAxes, size=8)
    for ax in axs.flatten():
        ticklabels = ax.get_xticklabels()
        for tick in ticklabels:
            tick.set_rotation(30)

        ax.get_yaxis().set_visible(False)
        key = ax.get_xlabel()
        if key != "":
            ax.set_xlabel(keys_dict[key])
            # ax.text(0, 0.9, keys_dict[key],
            #         transform=ax.transAxes,
            #         )

    fig.tight_layout()
    fig.savefig(out_filename)
    plt.close(fig)


def plot_prior_histograms(out_filename, df):
    return None


def load_df(respone_file, samples_file, return_samples=False):
    response = pd.read_csv(respone_file)
    response["SLE (cm)"] = -response["Mass (Gt)"] / 362.5 / 10
    response = response.astype({"RCP": int})
    samples = pd.read_csv(samples_file).rename(columns={"id": "Experiment"})
    if return_samples:
        return pd.merge(response, samples, on="Experiment"), samples
    else:
        return pd.merge(response, samples, on="Experiment")


def resample_ensemble_by_data(
    observed,
    simulated,
    rcps=[26, 45, 85],
    calibration_start=2010,
    calibration_end=2020,
    fudge_factor=3,
    n_samples=500,
    verbose=False,
    m_var="Mass (Gt)",
    m_var_std="Mass uncertainty (Gt)",
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

    observed_calib_time = (observed["Year"] >= calibration_start) & (
        observed["Year"] <= calibration_end
    )
    observed_calib_period = observed[observed_calib_time]
    # print(observed_calib_period)
    # Should we interpolate the simulations at observed time?
    observed_interp_mean = interp1d(
        observed_calib_period["Year"], observed_calib_period[m_var]
    )
    observed_interp_std = interp1d(
        observed_calib_period["Year"], observed_calib_period[m_var_std]
    )

    simulated_calib_time = (simulated["Year"] >= calibration_start) & (
        simulated["Year"] <= calibration_end
    )
    simulated_calib_period = simulated[simulated_calib_time]

    resampled_list = []
    for rcp in rcps:
        log_likes = []
        experiments = np.unique(simulated_calib_period["Experiment"])
        evals = []
        for i in experiments:
            exp_ = simulated_calib_period[
                (simulated_calib_period["Experiment"] == i)
                & (simulated_calib_period["RCP"] == rcp)
            ]
            log_like = 0.0
            for year, exp_mass in zip(exp_["Year"], exp_[m_var]):
                try:
                    observed_mass = observed_interp_mean(year)
                    observed_std = observed_interp_std(year) * fudge_factor
                    log_like -= 0.5 * (
                        (exp_mass - observed_mass) / observed_std
                    ) ** 2 + 0.5 * np.log(2 * np.pi * observed_std**2)
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
        print(weights)
        resampled_experiments = np.random.choice(experiments, n_samples, p=weights)
        new_frame = []
        for i in resampled_experiments:
            new_frame.append(
                simulated[(simulated["Experiment"] == i) & (simulated["RCP"] == rcp)]
            )
        simulated_resampled = pd.concat(new_frame)
        resampled_list.append(simulated_resampled)

    simulated_resampled = pd.concat(resampled_list)

    return simulated_resampled


def make_quantile_table(q_df, quantiles):
    ensembles = ["AS19", "Flow Calib.", "Flow+Mass Calib."]
    table_header = """
    \\begin{table}
    \\fontsize{6}{7.2}\selectfont
    \centering
    \caption{This is a table with scientific results.}
    \medskip
    \\begin{tabular}{lccc}
    \hline
    """

    ls = []

    f = "".join([f"& {ens}" for ens in ensembles])
    ls.append(f"{f} \\\\ \n")
    ls.append("\cline{2-4} \\\\ \n")
    f = "& {:.0f}th [{:.0f}th, {:.0f}th]".format(*(np.array(quantiles) * 100))
    g = f * len(ensembles)
    ls.append(f" {g} \\\\ \n")
    q_str = "& percentiles " * len(ensembles)
    ls.append(f"{q_str} \\\\ \n")
    sle_str = "& (cm SLE) " * len(ensembles)
    ls.append(f"{sle_str} \\\\ \n")
    ls.append("\hline \n")

    for rcp in rcps:
        a = q_df[q_df["RCP"] == rcp]
        f = "& ".join(
            [
                "{:.0f} [{:.0f}, {:.0f}]".format(
                    *a[a["Ensemble"] == ens].values[0][2::]
                )
                for ens in ensembles
            ]
        )
        ls.append(f"{rcp_dict[rcp]} & {f} \\\\")

    table_footer = """
    \hline
    \end{tabular}
    \label{tab:sle}
    \end{table}
    """

    print("".join([table_header, *ls, table_footer]))


def make_quantile_df(df, quantiles):
    q_dfs = [
        df.groupby(by=["RCP", "Ensemble"])["SLE (cm)"]
        .quantile(q)
        .reset_index()
        .rename(columns={"SLE (cm)": q})
        for q in quantiles
    ]
    q_df = reduce(lambda df1, df2: pd.merge(df1, df2, on=["RCP", "Ensemble"]), q_dfs)
    a_dfs = [
        df.groupby(by=["Ensemble"])["SLE (cm)"]
        .quantile(q)
        .reset_index()
        .rename(columns={"SLE (cm)": q})
        for q in quantiles
    ]
    a_df = reduce(lambda df1, df2: pd.merge(df1, df2, on=["Ensemble"]), a_dfs)
    a_df["RCP"] = "Union"
    return pd.concat([q_df, a_df]).round(1)


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
    "AS19": "#c51b8a",
    "Flow Calib.": "#31a354",
    "Mass Calib.": "#2c7fb8",
    "Flow+Mass Calib.": "0.0",
    "ISMIP6": "#c51b8a",
    "ISMIP6 Calib.": "0.0",
}
ts_fill_palette_dict = {
    "AS19": "0.80",
    "Flow Calib.": "0.70",
    "Mass Calib.": "#fee6ce",
    "Flow+Mass Calib.": "0.60",
    "ISMIP6": "0.80",
    "ISMIP6 Calib.": "0.60",
}
ts_median_palette_dict = {
    "AS19": "0.6",
    "Flow Calib.": "0.3",
    "Mass Calib.": "#e6550d",
    "Flow+Mass Calib.": "0.0",
    "ISMIP6": "0.6",
    "ISMIP6 Calib.": "0.0",
}

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
    observed_f = load_imbie()
    observed = load_imbie_csv()
    # observed = observed_f

    # Load AS19 (original LES)
    as19 = load_df(options.as19_results_file, options.as19_samples_file)
    # Load AS19 (with calibrated ice dynamics)
    calib, calib_samples = load_df(
        options.calibrated_results_file,
        options.calibrated_samples_file,
        return_samples=True,
    )

    #    ismip6 = pd.read_csv("ismip6_gis_ctrl.csv.gz")
    # Bayesian calibration: resampling
    as19_resampled = resample_ensemble_by_data(observed, as19)
    as19_calib_resampled = resample_ensemble_by_data(observed, calib)

    as19["Ensemble"] = "AS19"
    calib["Ensemble"] = "Flow Calib."
    as19_resampled["Ensemble"] = "Mass Calib."
    as19_calib_resampled["Ensemble"] = "Flow+Mass Calib."
    all_df = (
        pd.concat(
            [
                as19,
                calib,
                as19_resampled,
                as19_calib_resampled,
            ]
        )
        .drop_duplicates(subset=None, keep="first", inplace=False)
        .reset_index()
        .astype({"Ensemble": str})
    )

    year = 2100
    all_2100_df = all_df[(all_df["Year"] == year)]
    quantiles = [0.5, 0.05, 0.95, 0.16, 0.84]

    plot_projection(
        "sle_timeseries_calib_2008_2100.pdf",
        all_df,
        ensemble="Flow+Mass Calib.",
        bars=["AS19", "Flow Calib.", "Flow+Mass Calib."],
    )
    plot_projection(
        "sle_timeseries_flow_2008_2100.pdf",
        all_df,
        ensemble="Flow Calib.",
        bars=["AS19", "Flow Calib."],
    )
    plot_projection(
        "sle_timeseries_as19_2008_2100.pdf", all_df, ensemble="AS19", bars=["AS19"]
    )
    plot_histograms(
        "marginal_posteriors_all.pdf",
        all_2100_df,
        X_prior=calib_samples,
        ensembles=["Flow Calib.", "Flow+Mass Calib."],
    )

    plot_partitioning(
        "historical_partitioning_calibrated.pdf", simulated=all_df, observed=observed_f
    )

    year = 2020
    plot_posterior_sle_pdf(
        f"sle_pdf_w_obs_as19_{year}.pdf",
        all_df,
        observed=observed,
        year=year,
        ensembles=["AS19"],
    )
    plot_posterior_sle_pdf(
        f"sle_pdf_w_obs_as19flow_{year}.pdf",
        all_df,
        observed=observed,
        year=year,
        ensembles=["AS19", "Flow Calib."],
    )
    plot_posterior_sle_pdf(
        f"sle_pdf_w_obs_calibrated_{year}.pdf",
        all_df,
        observed=observed,
        year=year,
        ensembles=["AS19", "Flow Calib.", "Flow+Mass Calib."],
    )

    years = [2020, 2100]
    plot_posterior_sle_pdfs(
        f"sle_pdf_w_obs_{years[0]}_{years[1]}.pdf",
        all_df,
        observed=observed,
        years=years,
    )

    q_df = make_quantile_df(all_2100_df, quantiles)
    make_quantile_table(q_df, quantiles=quantiles)

    y_abs_dfs = []
    y_rel_dfs = []
    for year in [2020, 2100]:
        q_df = make_quantile_df(all_df[(all_df["Year"] == year)], quantiles)
        q_df["90%"] = q_df[0.95] - q_df[0.05]
        q_df["68%"] = q_df[0.84] - q_df[0.16]
        q_df.astype({"90%": np.float32, "68%": np.float32})

        q_abs_dfs = []
        q_rel_dfs = []
        for a, b in zip(
            ["Flow Calib.", "Flow+Mass Calib.", "Flow+Mass Calib."],
            ["AS19", "Flow Calib.", "AS19"],
        ):
            q_abs = q_df[q_df["Ensemble"] == a][["90%", "68%", 0.5]].reset_index(
                drop=True
            ) - q_df[q_df["Ensemble"] == b][["90%", "68%", 0.5]].reset_index(drop=True)

            q_rel = (
                q_abs
                / q_df[q_df["Ensemble"] == b][["90%", "68%", 0.5]].reset_index(
                    drop=True
                )
                * 100
            )

            q_abs["Difference"] = f"{a} - {b}"
            q_abs["RCP"] = rcpss
            q_rel["Difference"] = f"{a} - {b}"
            q_rel["RCP"] = rcpss
            q_abs_dfs.append(q_abs)
            q_rel_dfs.append(q_rel)

        q_abs_df = pd.concat(q_abs_dfs)
        q_rel_df = pd.concat(q_rel_dfs)
        q_abs_df["Year"] = year
        q_rel_df["Year"] = year

        y_abs_dfs.append(q_abs_df)
        y_rel_dfs.append(q_rel_df)

    quantiles_abs_df = pd.concat(y_abs_dfs).round(2)
    quantiles_rel_df = pd.concat(y_rel_dfs).round(2)
