#!/usr/bin/env python

# Copyright (C) 2020-22 Andy Aschwanden, Douglas J. Brinkerhoff

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Sequence
from datetime import datetime
from functools import reduce
from itertools import cycle
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple, TypeAlias, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import pylab as plt
import seaborn as sns
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Patch
from pismemulator.utils import load_imbie, load_imbie_csv
from pismemulator.utils import param_keys_dict as keys_dict
from scipy.interpolate import interp1d
from scipy.stats import beta
from scipy.stats.distributions import randint, truncnorm, uniform

NDArrayF: TypeAlias = npt.NDArray[np.floating]
NDArrayU8: TypeAlias = npt.NDArray[np.uint8]


def add_inner_title(
    ax: Axes,
    title: str,
    loc: str = "upper left",
    size: float | int = 7,
    **kwargs: Any,
) -> AnchoredText:
    """
    Add an inner title (anchored text) to a Matplotlib axis.

    This is a convenience wrapper around :class:`matplotlib.offsetbox.AnchoredText`
    that places a title-like label inside the axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to which the anchored title is added.
    title : str
        Text to display.
    loc : str, optional
        Location specifier passed to :class:`matplotlib.offsetbox.AnchoredText`
        (e.g., ``"upper left"``, ``"upper right"``, ``"lower left"``, etc.).
        Default is ``"upper left"``.
    size : float or int, optional
        Font size for the title text. Default is 7.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`matplotlib.offsetbox.AnchoredText`.

    Returns
    -------
    matplotlib.offsetbox.AnchoredText
        The created anchored text artist (already added to ``ax``).

    Notes
    -----
    Adapted from a Matplotlib example:
    http://matplotlib.sourceforge.net/examples/axes_grid/demo_axes_grid2.html
    """
    prop = {"size": size, "weight": "bold"}
    at = AnchoredText(
        title, loc=loc, prop=prop, pad=0.0, borderpad=0.5, frameon=False, **kwargs
    )
    ax.add_artist(at)
    return at


def color_tint(m_color: Any, alpha: float) -> NDArrayF:
    """
    Apply an alpha tint to a Matplotlib color and return an RGB triple.

    Parameters
    ----------
    m_color : object
        Any Matplotlib color spec accepted by :func:`matplotlib.colors.to_rgba`
        (e.g., a color name, hex string, or RGB(A) tuple).
    alpha : float
        Alpha value to apply. Typically in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        RGB color as float array of shape ``(3,)`` with values in ``[0, 1]``.

    Notes
    -----
    The function converts ``m_color`` to RGBA, overwrites the alpha channel with
    ``alpha``, composites against a white background (via :func:`rgba2rgb`), and
    returns the resulting RGB normalized to ``[0, 1]``.
    """
    rgba = list(colors.to_rgba(m_color))
    rgba[-1] = float(alpha)
    rgba255 = np.asarray(rgba, dtype=float) * 255.0
    return rgba2rgb(rgba255) / 255.0


def rgba2rgb(
    rgba: npt.ArrayLike,
    background: tuple[float, float, float] = (255.0, 255.0, 255.0),
) -> NDArrayU8:
    """
    Composite an RGBA color over a background and return an RGB uint8 triple.

    Parameters
    ----------
    rgba : array_like
        RGBA values as ``(R, G, B, A)`` in the 0–255 range.
        The alpha channel ``A`` is interpreted in 0–255 and normalized internally.
    background : tuple of float, optional
        Background RGB values in the 0–255 range. Default is white
        ``(255, 255, 255)``.

    Returns
    -------
    numpy.ndarray
        RGB values as ``uint8`` array of shape ``(3,)`` in the 0–255 range.

    Notes
    -----
    This performs standard "source over" compositing:

    ``rgb = src_rgb * a + bg_rgb * (1 - a)``

    where ``a`` is alpha normalized to ``[0, 1]``.
    """
    rgba_arr = np.asarray(rgba, dtype=float).reshape(-1)
    if rgba_arr.size != 4:
        raise ValueError("rgba must contain exactly 4 values (R, G, B, A)")

    r, g, b, a255 = rgba_arr[0], rgba_arr[1], rgba_arr[2], rgba_arr[3]
    a = np.asarray(a255, dtype=np.float32) / 255.0

    Rb, Gb, Bb = background

    rgb = np.empty((3,), dtype=np.float32)
    rgb[0] = r * a + (1.0 - a) * Rb
    rgb[1] = g * a + (1.0 - a) * Gb
    rgb[2] = b * a + (1.0 - a) * Bb

    return np.asarray(rgb, dtype=np.uint8)


def toDecimalYear(date: datetime) -> float:
    """
    Convert a :class:`datetime.datetime` to a decimal year.

    Parameters
    ----------
    date : datetime.datetime
        Input date/time.

    Returns
    -------
    float
        Decimal year representation, e.g. ``2020.7732...``.

    Notes
    -----
    The fractional part is computed as the proportion of elapsed seconds between
    the start of ``date.year`` and the start of the next year, using the given
    datetime's exact timestamp.

    Examples
    --------
    >>> from datetime import datetime
    >>> toDecimalYear(datetime(2020, 10, 10))
    2020.7...
    """
    year = date.year
    start_of_year = datetime(year=year, month=1, day=1)
    start_of_next_year = datetime(year=year + 1, month=1, day=1)

    year_elapsed = (date - start_of_year).total_seconds()
    year_duration = (start_of_next_year - start_of_year).total_seconds()
    fraction = year_elapsed / year_duration

    return float(year) + fraction


def set_size(w: float, h: float, ax: Axes | None = None) -> None:
    """
    Set the figure size so the axes area matches a target width/height.

    Parameters
    ----------
    w : float
        Target axes width in inches.
    h : float
        Target axes height in inches.
    ax : matplotlib.axes.Axes or None, optional
        Axes whose figure will be resized. If None, uses the current axes
        (``plt.gca()``).

    Returns
    -------
    None
        Resizes the figure in place.

    Notes
    -----
    Matplotlib's ``figsize`` refers to the full figure size, not the inner axes.
    This helper accounts for subplot margins (``subplotpars``) so that the
    *inner* axes area is approximately ``(w, h)``.
    """
    if ax is None:
        ax = plt.gca()

    left = ax.figure.subplotpars.left
    right = ax.figure.subplotpars.right
    top = ax.figure.subplotpars.top
    bottom = ax.figure.subplotpars.bottom

    figw = float(w) / (right - left)
    figh = float(h) / (top - bottom)
    ax.figure.set_size_inches(figw, figh)


def plot_historical(
    out_filename: str,
    simulated: pd.DataFrame | None = None,
    observed: pd.DataFrame | None = None,
    ensembles: Sequence[str] = ("AS19", "Flow+Mass Calib."),
    quantiles: Sequence[float] = (0.05, 0.95),
    sigma: float = 2.0,
    simulated_ctrl: pd.DataFrame | None = None,
    xlims: Sequence[float] = (2008.0, 2021.0),
    ylims: Sequence[float] = (-10000.0, 500.0),
) -> None:
    """
    Plot historical simulations and observations.

    Parameters
    ----------
    out_filename : str
        Output path for the saved figure.
    simulated : pandas.DataFrame or None, optional
        Simulated ensemble output. If provided, must contain at least the columns
        ``"Ensemble"``, ``"Year"``, and ``"Mass (Gt)"``.
    observed : pandas.DataFrame or None, optional
        Observational time series (e.g., IMBIE). If provided, must contain the
        columns ``"Year"``, ``"Mass (Gt)"``, and ``"Mass uncertainty (Gt)"``.
    ensembles : sequence of str, optional
        Ensemble names to plot from ``simulated["Ensemble"]``. Default is
        ``("AS19", "Flow+Mass Calib.")``.
    quantiles : sequence of float, optional
        Lower/upper quantiles for the simulated uncertainty band. Must contain
        at least two values. Default is ``(0.05, 0.95)``.
    sigma : float, optional
        Number of observational standard deviations to shade for observations.
        Default is 2.
    simulated_ctrl : pandas.DataFrame or None, optional
        Optional control simulation (currently unused in this function body).
        Included for API compatibility.
    xlims : sequence of float, optional
        X-axis limits in decimal years. Default is ``(2008, 2021)``.
    ylims : sequence of float, optional
        Y-axis limits for cumulative mass change (Gt). Default is ``(-10000, 500)``.

    Returns
    -------
    None
        Saves a figure to ``out_filename`` and closes it.

    Notes
    -----
    This function relies on the following external variables defined in the
    calling module:

    - ``ts_median_palette_dict``: mapping ensemble name -> line color
    - ``ts_fill_palette_dict``: mapping ensemble name -> fill color
    - ``signal_lw``: line width for signals
    - ``obs_signal_color``: color for observation mean line
    - ``obs_sigma_color``: color for observation uncertainty fill
    - ``proj_start``: reference start year for labeling
    - ``gt2cmSLE``: conversion factor from Gt to cm SLE (for right y-axis)

    If you want this function to be fully self-contained, pass these in
    explicitly or wrap them into a config object.
    """
    fig = plt.figure(num="historical", clear=True, figsize=[4.6, 1.6])
    ax = fig.add_subplot(111)

    if simulated is not None:
        for r, ens in enumerate(ensembles):
            legend_handles = []
            sim = simulated[simulated["Ensemble"] == ens]
            g = sim.groupby(by="Year")["Mass (Gt)"]
            sim_median = g.quantile(0.50)
            sim_low = g.quantile(float(quantiles[0]))
            sim_high = g.quantile(float(quantiles[-1]))

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
                label=f"{float(quantiles[0]) * 100:.0f}-{float(quantiles[-1]) * 100:.0f}%",
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
            observed["Mass (Gt)"] - float(sigma) * observed["Mass uncertainty (Gt)"],
            observed["Mass (Gt)"] + float(sigma) * observed["Mass uncertainty (Gt)"],
            color=obs_sigma_color,
            alpha=0.75,
            linewidth=0.0,
            zorder=5,
            label=rf"{sigma}-\sigma",
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

    ax.axhline(0.0, color="k", linestyle="dotted", linewidth=0.6)

    ax.set_xlabel("Year")
    ax.set_ylabel(f"Cumulative mass change\nsince {proj_start} (Gt)")
    ax.set_xlim(tuple(xlims))
    ax.set_ylim(tuple(ylims))

    ax_sle = ax.twinx()
    ax_sle.set_ylabel(f"Contribution to sea-level \nsince {proj_start} (cm SLE)")
    ax_sle.set_ylim(-np.array(list(ylims), dtype=float) * gt2cmSLE)

    fig.savefig(out_filename, bbox_inches="tight")
    plt.close(fig)


def plot_projection(
    out_filename: str,
    simulated: pd.DataFrame | None = None,
    ensemble: str = "Flow+Mass Calib.",
    quantiles: Sequence[float] = (0.05, 0.95),
    bars: Sequence[str] | None = None,
    quantile_df: pd.DataFrame | None = None,
    xlims: Sequence[float] = (2008.0, 2100.0),
    ylims: Sequence[float] = (-0.5, 45.0),
) -> None:
    """
    Plot sea-level contribution projections by RCP (and optional summary bars).

    Parameters
    ----------
    out_filename : str
        Output path for the saved figure.
    simulated : pandas.DataFrame or None, optional
        Simulation results. If provided, must include at least the columns
        ``"Ensemble"``, ``"RCP"``, ``"Year"``, and ``"SLE (cm)"``.
    ensemble : str, optional
        Ensemble name to select from ``simulated["Ensemble"]`` for the time-series
        panel. Default is ``"Flow+Mass Calib."``.
    quantiles : sequence of float, optional
        Quantiles used for uncertainty shading. Typically length 2 (e.g.
        ``(0.05, 0.95)``). If length is 4, an additional darker inner band is
        plotted using ``quantiles[1]`` and ``quantiles[-2]``.
        Default is ``(0.05, 0.95)``.
    bars : sequence of str or None, optional
        If provided, adds one bar-style panel per RCP showing summary quantiles
        at year 2100 for the specified ensembles. If None, only the time-series
        panel is produced.
    quantile_df : pandas.DataFrame or None, optional
        Optional precomputed quantile table. If provided, it should be compatible
        with what :func:`make_quantile_df` returns. Currently unused in this code
        path (the function recomputes quantiles internally when ``bars`` is not None).
    xlims : sequence of float, optional
        X-axis limits in years. Default is ``(2008, 2100)``.
    ylims : sequence of float, optional
        Y-axis limits in cm SLE. Default is ``(-0.5, 45)``.

    Returns
    -------
    None
        Saves a figure to ``out_filename`` and closes it.

    Notes
    -----
    This function relies on several globals defined elsewhere in the module:

    - ``rcps``: iterable of RCP identifiers
    - ``rcp_dict``: mapping RCP -> display label
    - ``rcp_col_dict``: mapping RCP -> line color
    - ``rcp_shade_col_dict``: mapping RCP -> fill color
    - ``signal_lw``: line width used for median curves
    - ``proj_start``: reference year used for labeling
    - ``make_quantile_df``: helper that computes quantiles for a subset of ``simulated``

    If you want this function to be fully self-contained, pass these in as
    arguments or bundle them into a configuration object.

    The ``quantile_df`` argument is retained for API compatibility, but the
    provided snippet does not use it.
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
        axs = None  # for type checkers

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

            sim_low = g.quantile(float(quantiles[0]))
            sim_high = g.quantile(float(quantiles[-1]))
            ci = ax.fill_between(
                sim_median.index,
                sim_low,
                sim_high,
                color=rcp_shade_col_dict[rcp],
                alpha=0.5,
                linewidth=0.5,
                zorder=-11,
                label=f"{float(quantiles[0]) * 100:.0f}-{float(quantiles[-1]) * 100:.0f}%",
            )
            legend_handles.append(ci)

            if len(quantiles) == 4:
                sim_low = g.quantile(float(quantiles[1]))
                sim_high = g.quantile(float(quantiles[-2]))
                ci = ax.fill_between(
                    sim_median.index,
                    sim_low,
                    sim_high,
                    color=rcp_shade_col_dict[rcp],
                    alpha=0.85,
                    linewidth=0.5,
                    zorder=-11,
                    label=f"{float(quantiles[1]) * 100:.0f}-{float(quantiles[-2]) * 100:.0f}%",
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
    ax.set_xlim(tuple(xlims))
    ax.set_ylim(tuple(ylims))

    if bars is not None and axs is not None:
        width = 1.0
        legend_elements = []

        hatch_pattern_dict = {
            "Flow+Mass Calib.": "\\\\\\",
            "Flow+Mass Calib. S1": "\\\\\\",
            "Flow+Mass Calib. S2": "\\\\\\",
            "Flow+Mass Calib. S3": "\\\\\\",
            "Flow Calib.": "......",
            "AS19": "",
        }
        hatch_patterns = [hatch_pattern_dict.get(ens, "") for ens in bars]
        hatches = cycle(hatch_patterns)

        # Use provided quantile_df if supplied; otherwise compute from simulated at year 2100
        if quantile_df is not None:
            q_df = quantile_df
        else:
            if simulated is None:
                raise ValueError("simulated must be provided when bars is not None")
            q_df = make_quantile_df(
                simulated[simulated["Year"] == 2100],
                quantiles=[0.05, 0.16, 0.5, 0.84, 0.95],
            )

        for k, rcp in enumerate(rcps):
            df = q_df[q_df["RCP"] == rcp]
            for e, ens in enumerate(bars):
                hatch = next(hatches)
                s_df = df[df["Ensemble"] == ens]

                q05 = float(s_df[[0.05]].values[0][0])
                q16 = float(s_df[[0.16]].values[0][0])
                q50 = float(s_df[[0.50]].values[0][0])
                q84 = float(s_df[[0.84]].values[0][0])
                q95 = float(s_df[[0.95]].values[0][0])

                rect1 = plt.Rectangle(
                    (e + 0.4, q05),
                    0.2,
                    q95 - q05,
                    color=rcp_shade_col_dict[rcp],
                    alpha=1.0,
                    lw=0.0,
                )
                rect2 = plt.Rectangle(
                    (e + 0.2, q16),
                    0.6,
                    q84 - q16,
                    color=rcp_shade_col_dict[rcp],
                    alpha=1.0,
                    lw=0.0,
                )
                rect3 = plt.Rectangle(
                    (e + 0.4, q05),
                    0.2,
                    q95 - q05,
                    color="k",
                    alpha=1.0,
                    fill=False,
                    lw=0.25,
                    # hatch=hatch,
                    label=ens,
                )
                rect4 = plt.Rectangle(
                    (e + 0.2, q16),
                    0.6,
                    q84 - q16,
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
                    [e, e + width], [q50, q50], color=rcp_col_dict[rcp], lw=signal_lw
                )

                if k == 0:
                    legend_elements.append(
                        Patch(
                            facecolor="none",
                            edgecolor="k",
                            fill=False,
                            lw=0.25,
                            hatch=hatch,
                            label=ens,
                        )
                    )

        for a in (1, 2, 3):
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

    handles: list[Any] = [
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


def plot_prior_histograms(out_filename: str, df: pd.DataFrame) -> None:
    """
    Plot histograms of prior distributions.

    Parameters
    ----------
    out_filename : str
        Output path for the saved figure.
    df : pandas.DataFrame
        DataFrame containing prior samples/parameters to be visualized.

    Returns
    -------
    None
        This function currently does nothing and always returns ``None``.

    Notes
    -----
    This is a stub. Implement plotting logic (e.g., via Matplotlib/Seaborn)
    or remove the function if it is not used.
    """
    return None


@overload
def load_df(
    response_file: str | Path,
    samples_file: str | Path,
    *,
    return_samples: Literal[True],
) -> tuple[pd.DataFrame, pd.DataFrame]: ...


@overload
def load_df(
    response_file: str | Path,
    samples_file: str | Path,
    *,
    return_samples: Literal[False] = False,
) -> pd.DataFrame: ...


def load_df(
    response_file: str | Path,
    samples_file: str | Path,
    *,
    return_samples: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load response and sample tables and merge them on experiment ID.

    Parameters
    ----------
    response_file : str or pathlib.Path
        CSV file containing model responses. Must include at least
        ``"Experiment"``, ``"Mass (Gt)"``, and ``"RCP"``.
    samples_file : str or pathlib.Path
        CSV file containing sampled parameters. Must include an ``"id"`` column
        which will be renamed to ``"Experiment"`` for merging.
    return_samples : bool, optional
        If True, return a tuple ``(merged, samples)``. Otherwise, return only
        the merged DataFrame. Default is False.

    Returns
    -------
    pandas.DataFrame or tuple[pandas.DataFrame, pandas.DataFrame]
        If ``return_samples`` is False: merged DataFrame containing response and
        sample columns.
        If True: ``(merged, samples)``.

    Notes
    -----
    Adds a derived column ``"SLE (cm)"`` computed from mass change:

    ``SLE (cm) = -Mass(Gt) / 362.5 / 10``

    and coerces the ``"RCP"`` column to integer.
    """
    response = pd.read_csv(response_file)
    response["SLE (cm)"] = -response["Mass (Gt)"] / 362.5 / 10.0
    response = response.astype({"RCP": int})

    samples = pd.read_csv(samples_file).rename(columns={"id": "Experiment"})

    merged = pd.merge(response, samples, on="Experiment")

    if return_samples:
        return merged, samples
    return merged


def resample_ensemble_by_data(
    observed: pd.DataFrame,
    simulated: pd.DataFrame,
    *,
    rcps: Sequence[int] = (26, 45, 85),
    calibration_start: float = 2010.0,
    calibration_end: float = 2020.0,
    fudge_factor: float = 3.0,
    n_samples: int = 500,
    verbose: bool = False,
    m_var: str = "Mass (Gt)",
    m_var_std: str = "Mass uncertainty (Gt)",
) -> pd.DataFrame:
    """
    Resample simulated ensemble members using a Gaussian likelihood vs observations.

    This implements a resampling/importance sampling scheme attributed to
    Douglas C. Brinkerhoff. For each RCP, experiments are weighted by their
    (approximate) log-likelihood over the calibration period, then resampled
    according to normalized weights.

    Parameters
    ----------
    observed : pandas.DataFrame
        Observational time series. Must include:
        - ``"Year"``
        - ``m_var`` (default ``"Mass (Gt)"``)
        - ``m_var_std`` (default ``"Mass uncertainty (Gt)"``)
    simulated : pandas.DataFrame
        Simulated ensemble time series. Must include:
        - ``"Year"``, ``"Experiment"``, ``"RCP"``
        - ``m_var`` (default ``"Mass (Gt)"``)
    rcps : sequence of int, optional
        RCP identifiers to process. Default is ``(26, 45, 85)``.
    calibration_start : float, optional
        Start year (inclusive) of the calibration period. Default is 2010.
    calibration_end : float, optional
        End year (inclusive) of the calibration period. Default is 2020.
    fudge_factor : float, optional
        Multiplier applied to observational uncertainty, effectively inflating
        the standard deviation used in the likelihood. Default is 3.
    n_samples : int, optional
        Number of experiments to resample per RCP. Default is 500.
    verbose : bool, optional
        If True, print per-experiment log-likelihoods. Default is False.
    m_var : str, optional
        Name of the mass variable column. Default is ``"Mass (Gt)"``.
    m_var_std : str, optional
        Name of the observational uncertainty column. Default is
        ``"Mass uncertainty (Gt)"``.

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame containing resampled experiments for all requested
        RCPs.

    Notes
    -----
    The likelihood used is Gaussian with time-varying observation mean and
    standard deviation interpolated from the observed record. The log-likelihood
    contribution at each year is:

    ``-0.5 * ((x - mu)/sigma)^2 - 0.5 * log(2*pi*sigma^2)``

    where ``sigma`` is the interpolated observational uncertainty multiplied by
    ``fudge_factor``.

    This function depends on a global ``rcp_dict`` for display labels when
    ``verbose=True``.
    """
    observed_calib_time = (observed["Year"] >= calibration_start) & (
        observed["Year"] <= calibration_end
    )
    observed_calib_period = observed[observed_calib_time]

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

    resampled_list: list[pd.DataFrame] = []

    for rcp in rcps:
        log_likes: list[float] = []
        evals: list[Any] = []

        experiments = np.unique(simulated_calib_period["Experiment"])

        for exp_id in experiments:
            exp_ = simulated_calib_period[
                (simulated_calib_period["Experiment"] == exp_id)
                & (simulated_calib_period["RCP"] == rcp)
            ]

            log_like = 0.0
            for year, exp_mass in zip(exp_["Year"], exp_[m_var]):
                try:
                    observed_mass = float(observed_interp_mean(year))
                    observed_std = float(observed_interp_std(year)) * float(
                        fudge_factor
                    )
                    log_like -= 0.5 * ((exp_mass - observed_mass) / observed_std) ** 2
                    log_like -= 0.5 * np.log(2.0 * np.pi * observed_std**2)
                except ValueError:
                    # interp1d can raise if outside bounds; skip those points
                    pass

            if log_like != 0.0:
                evals.append(exp_id)
                log_likes.append(log_like)
                if verbose:
                    print(f"{rcp_dict[rcp]}, Experiment {exp_id:.0f}: {log_like:.2f}")

        experiments_arr = np.asarray(evals)
        w = np.asarray(log_likes, dtype=float)
        w -= w.mean()

        weights = np.exp(w)
        weights /= weights.sum()

        resampled_experiments = np.random.choice(experiments_arr, n_samples, p=weights)

        new_frame: list[pd.DataFrame] = []
        for exp_id in resampled_experiments:
            new_frame.append(
                simulated[
                    (simulated["Experiment"] == exp_id) & (simulated["RCP"] == rcp)
                ]
            )

        simulated_resampled = pd.concat(new_frame)
        resampled_list.append(simulated_resampled)

    return pd.concat(resampled_list)


def make_quantile_table(q_df: pd.DataFrame, quantiles: Sequence[float]) -> None:
    """
    Print a LaTeX table of sea-level contribution quantiles by RCP and ensemble.

    Parameters
    ----------
    q_df : pandas.DataFrame
        Quantile table as produced by :func:`make_quantile_df`. Must contain
        columns ``"RCP"``, ``"Ensemble"``, and one column per requested quantile
        (numeric columns with float keys like ``0.05``).
    quantiles : sequence of float
        Quantiles to report (e.g., ``(0.16, 0.5, 0.84)``). The table prints a
        median and lower/upper bounds using the first, middle, and last elements
        of this sequence.

    Returns
    -------
    None
        Prints LaTeX to stdout.

    Notes
    -----
    This function depends on global variables:

    - ``rcps``: iterable of RCP identifiers
    - ``rcp_dict``: mapping RCP -> display label
    """
    ensembles = ["AS19", "Flow Calib.", "Flow+Mass Calib."]

    table_header = r"""
    \begin{table}
    \fontsize{6}{7.2}\selectfont
    \centering
    \caption{This is a table with scientific results.}
    \medskip
    \begin{tabular}{lccc}
    \hline
    """

    ls: list[str] = []

    head = "".join([f"& {ens}" for ens in ensembles])
    ls.append(f"{head} \\\\ \n")
    ls.append(r"\cline{2-4} \\ " + "\n")

    q_pct = np.asarray(list(quantiles), dtype=float) * 100.0
    fmt = "& {:.0f}th [{:.0f}th, {:.0f}th]".format(*q_pct)
    ls.append(f" {fmt * len(ensembles)} \\\\ \n")

    ls.append(f"{'& percentiles ' * len(ensembles)} \\\\ \n")
    ls.append(f"{'& (cm SLE) ' * len(ensembles)} \\\\ \n")
    ls.append(r"\hline " + "\n")

    for rcp in rcps:
        a = q_df[q_df["RCP"] == rcp]
        row = "& ".join(
            [
                "{:.0f} [{:.0f}, {:.0f}]".format(
                    *a[a["Ensemble"] == ens].values[0][2::]
                )
                for ens in ensembles
            ]
        )
        ls.append(f"{rcp_dict[rcp]} & {row} \\\\")
    table_footer = r"""
    \hline
    \end{tabular}
    \label{tab:sle}
    \end{table}
    """
    print("".join([table_header, *ls, table_footer]))


def make_quantile_df(df: pd.DataFrame, quantiles: Sequence[float]) -> pd.DataFrame:
    """
    Compute a tidy quantile table for sea-level contributions.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table with at least columns ``"RCP"``, ``"Ensemble"``, and
        ``"SLE (cm)"``.
    quantiles : sequence of float
        Quantiles to compute (e.g., ``(0.05, 0.5, 0.95)``).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``"RCP"``, ``"Ensemble"``, and one column per
        quantile (with the quantile value used as the column name). Includes an
        additional row group with ``RCP == "Union"`` giving quantiles across all
        RCPs per ensemble.

    Notes
    -----
    The output is rounded to one decimal place.
    """
    q_dfs = [
        df.groupby(by=["RCP", "Ensemble"])["SLE (cm)"]
        .quantile(float(q))
        .reset_index()
        .rename(columns={"SLE (cm)": float(q)})
        for q in quantiles
    ]
    q_df = reduce(lambda df1, df2: pd.merge(df1, df2, on=["RCP", "Ensemble"]), q_dfs)

    a_dfs = [
        df.groupby(by=["Ensemble"])["SLE (cm)"]
        .quantile(float(q))
        .reset_index()
        .rename(columns={"SLE (cm)": float(q)})
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
    plot_posterior_sle_pdfs(
        f"sle_pdf_prior_posterior_w_obs_{years[0]}_{years[1]}.pdf",
        all_df,
        observed=observed,
        years=years,
        ensembles=["AS19", "Flow+Mass Calib."],
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
