# Copyright (C) 2019-25 Andy Aschwanden
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
"""
Plotting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pylab as plt
from matplotlib.colors import LogNorm


def plot_compare(
    F_p: np.ndarray,
    F_v: np.ndarray,
    dataset: Any,
    X_val_unscaled: Sequence[float],
    return_fig: bool = False,
) -> plt.Figure | None:
    """
    Plot observed (PISM) vs. predicted (Emulator) surface speeds and their difference.

    The figure contains three panels stacked vertically:
    1) PISM target speeds,
    2) Emulator-predicted speeds,
    3) Difference (Emulator − PISM).

    A log color scale is used for the speed panels. The Pearson correlation
    between flattened arrays and the RMSE are annotated. The unscaled input
    parameters for the sample are listed under the panels.

    Parameters
    ----------
    F_p : numpy.ndarray
        Predicted (Emulator) speed field, shape ``(ny, nx)``.
    F_v : numpy.ndarray
        Target (PISM) speed field, shape ``(ny, nx)``.
    dataset : Any
        Dataset-like object providing an attribute ``X_keys`` (sequence of
        parameter names) used for the text annotation.
    X_val_unscaled : Sequence[float]
        Unscaled parameter values corresponding to ``dataset.X_keys`` for annotation.
    return_fig : bool, default=False
        If ``True``, return the created :class:`matplotlib.figure.Figure`.

    Returns
    -------
    matplotlib.figure.Figure or None
        The created figure if ``return_fig`` is ``True``, otherwise ``None``.

    Notes
    -----
    - Speeds are shown with ``LogNorm(vmin=1, vmax=3000)``.
    - Difference panel uses a symmetric linear colormap in ``[-50, 50]`` m/yr.
    - Two colorbars are added on the right: one for speed and one for difference.
    - The figure is saved as ``"test_comp.pdf"`` in the current working directory.
    """
    cmap = "viridis"
    fig, axs = plt.subplots(
        nrows=3, ncols=1, sharex="col", sharey="row", figsize=(2.5, 8)
    )

    rmse = np.sqrt(((F_p - F_v) ** 2).mean())
    corr = np.corrcoef(F_v.flatten(), F_p.flatten())[0, 1]

    c1 = axs[0].imshow(F_v, origin="lower", cmap=cmap, norm=LogNorm(vmin=1, vmax=3e3))
    axs[1].imshow(F_p, origin="lower", cmap=cmap, norm=LogNorm(vmin=1, vmax=3e3))
    c2 = axs[2].imshow(F_p - F_v, origin="lower", vmin=-50, vmax=50, cmap="coolwarm")

    axs[1].text(0.01, 0.00, f"r={corr:.3f}", c="k", size=7, transform=axs[1].transAxes)
    axs[-1].text(
        0.01,
        -0.51,
        "\n".join([f"{k}: {v:.3f}" for k, v in zip(dataset.X_keys, X_val_unscaled)]),
        c="k",
        size=7,
        transform=axs[-1].transAxes,
    )
    axs[2].text(
        0.01, 0.00, f"RMSE: {rmse:.0f} m/yr", c="k", size=7, transform=axs[2].transAxes
    )

    for ax in axs:
        ax.set_axis_off()

    axs[0].text(
        0.01, 0.98, "PISM", c="k", size=7, weight="bold", transform=axs[0].transAxes
    )
    axs[1].text(
        0.01, 0.98, "Emulator", c="k", size=7, weight="bold", transform=axs[1].transAxes
    )
    axs[2].text(
        0.01,
        0.98,
        "PISM-Emulator",
        c="k",
        size=7,
        weight="bold",
        transform=axs[2].transAxes,
    )

    cb_ax = fig.add_axes([0.88, 0.525, 0.025, 0.15])
    plt.colorbar(
        c1,
        cax=cb_ax,
        shrink=0.9,
        label="speed (m/yr)",
        orientation="vertical",
        extend="both",
    )
    cb_ax2 = fig.add_axes([0.88, 0.15, 0.025, 0.15])
    plt.colorbar(
        c2,
        cax=cb_ax2,
        shrink=0.9,
        label="diff. (m/yr)",
        orientation="vertical",
        extend="both",
    )
    cb_ax.tick_params(labelsize=7)
    cb_ax.set_yticklabels([1, 10, 100, 1000])
    cb_ax2.tick_params(labelsize=7)

    fig.subplots_adjust(wspace=0.05, hspace=0.15)

    fig.savefig("test_comp.pdf")

    if return_fig:
        return fig
    return None


def plot_eigenglaciers(
    dataset: Any,
    data_loader: Any,
    model_index: int,
    emulator_dir: str | Path,
    nrows: int = 2,
    ncols: int = 3,
    figsize: tuple[float, float] = (3.2, 3.6),
    q: int = 6,
) -> None:
    """
    Plot and save eigenglacier basis vectors as 2D fields.

    This function retrieves the first ``q`` eigenglaciers from ``data_loader``,
    maps each basis vector from a sparse 1D representation back to a 2D grid
    using indices and masking provided by ``dataset.target``, and saves a PDF
    figure containing the first ``nrows * ncols`` eigenglaciers. Each panel is
    annotated with the corresponding normalized eigenvalue contribution
    (percentage of total).

    Parameters
    ----------
    dataset : Any
        Dataset providing grid metadata through ``dataset.target``. The target is
        expected to define:

        - ``nx`` and ``ny``: grid dimensions (ints)
        - ``sparse_idx_1d``: 1D indices into a flattened ``(ny, nx)`` array
        - ``mask_2d``: boolean mask with shape ``(ny, nx)`` (True = masked)
    data_loader : Any
        Object providing eigenglaciers via ``get_eigenglaciers(q=...)`` returning
        ``(V_hat, _, _, lamda)`` where:

        - ``V_hat`` has shape ``(n_nodes, q)``
        - ``lamda`` has shape ``(q,)`` and contains eigenvalues
    model_index : int
        Identifier used in the output filename
        (``eigenglaciers_{model_index}.pdf``).
    emulator_dir : str or pathlib.Path
        Base directory where the figure will be written under the subdirectory
        ``eigenglaciers``.
    nrows : int, optional
        Number of subplot rows. Default is 2.
    ncols : int, optional
        Number of subplot columns. Default is 3.
    figsize : tuple of float, optional
        Figure size in inches ``(width, height)``. Default is ``(3.2, 3.6)``.
    q : int, optional
        Number of eigenglaciers to retrieve from the loader. Default is 6.

    Returns
    -------
    None
        The figure is saved to disk.

    Raises
    ------
    ValueError
        If ``q <= 0`` or if the loader returns arrays with incompatible shapes.
    ZeroDivisionError
        If the eigenvalue sum is zero (cannot compute percentages).
    OSError
        If the output directory cannot be created or the figure cannot be saved.

    Notes
    -----
    Only the first ``nrows * ncols`` eigenglaciers are plotted. If
    ``nrows * ncols > q`` this function will raise an ``IndexError`` unless you
    guard against it (e.g., by iterating to ``min(q, nrows*ncols)``).
    """
    V_hat, _, _, lamda = data_loader.get_eigenglaciers(q=q)

    lamda_scaled = lamda / lamda.sum() * 100
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, sharex="col", sharey="row", figsize=figsize
    )
    for k, ax in enumerate(axs.ravel()):
        V = V_hat[:, k]
        data = np.zeros((dataset.target.ny, dataset.target.nx))
        data.put(dataset.target.sparse_idx_1d, V)
        eigen_glacier = np.ma.array(data=data, mask=dataset.target.mask_2d)
        _ = ax.imshow(
            eigen_glacier, origin="lower", cmap="twilight_shifted", vmin=-0.3, vmax=0.3
        )

        ax.text(
            0.05,
            -0.025,
            rf"$\Lambda_{k}$={lamda_scaled[k]:.1f}%",
            transform=ax.transAxes,
        )
        ax.axis("off")
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.tight_layout()

    fig_dir = Path(emulator_dir) / "eigenglaciers"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(fig_dir / f"eigenglaciers_{model_index}.pdf")


def plot_legacy_eigenglaciers(
    dataset: Any,
    data_loader: Any,
    model_index: int,
    emulator_dir: str | Path,
    nrows: int = 2,
    ncols: int = 3,
    figsize: tuple[float, float] = (3.2, 3.6),
    q: int = 6,
) -> None:
    """
    Plot and save legacy eigenglacier basis vectors as 2D fields.

    This is the legacy variant of :func:`plot_eigenglaciers`. It expects grid
    metadata to be available directly on ``dataset`` (rather than
    ``dataset.target``). The function retrieves eigenglaciers from ``data_loader``,
    maps each basis vector from a sparse 1D representation back to a 2D grid,
    and saves a PDF containing the first ``nrows * ncols`` eigenglaciers. Each
    panel is annotated with the corresponding normalized eigenvalue contribution
    (percentage of total).

    Parameters
    ----------
    dataset : Any
        Dataset providing grid metadata directly. Expected attributes:

        - ``nx`` and ``ny``: grid dimensions (ints)
        - ``sparse_idx_1d``: 1D indices into a flattened ``(ny, nx)`` array
        - ``mask_2d``: boolean mask with shape ``(ny, nx)`` (True = masked)
    data_loader : Any
        Object providing eigenglaciers via
        ``get_eigenglaciers(eigenvalues=True, q=...)`` returning
        ``(V_hat, _, _, lamda)`` where:

        - ``V_hat`` has shape ``(n_nodes, q)``
        - ``lamda`` has shape ``(q,)`` and contains eigenvalues
    model_index : int
        Identifier used in the output filename
        (``eigenglaciers_{model_index}.pdf``).
    emulator_dir : str or pathlib.Path
        Base directory where the figure will be written under the subdirectory
        ``eigenglaciers``.
    nrows : int, optional
        Number of subplot rows. Default is 2.
    ncols : int, optional
        Number of subplot columns. Default is 3.
    figsize : tuple of float, optional
        Figure size in inches ``(width, height)``. Default is ``(3.2, 3.6)``.
    q : int, optional
        Number of eigenglaciers to retrieve from the loader. Default is 6.

    Returns
    -------
    None
        The figure is saved to disk.

    Raises
    ------
    ValueError
        If ``q <= 0`` or if the loader returns arrays with incompatible shapes.
    ZeroDivisionError
        If the eigenvalue sum is zero (cannot compute percentages).
    OSError
        If the output directory cannot be created or the figure cannot be saved.

    Notes
    -----
    Only the first ``nrows * ncols`` eigenglaciers are plotted. If
    ``nrows * ncols > q`` this function will raise an ``IndexError`` unless you
    guard against it (e.g., by iterating to ``min(q, nrows*ncols)``).
    """
    V_hat, _, _, lamda = data_loader.get_eigenglaciers(eigenvalues=True, q=q)

    lamda_scaled = lamda / lamda.sum() * 100
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, sharex="col", sharey="row", figsize=figsize
    )
    for k, ax in enumerate(axs.ravel()):
        V = V_hat[:, k]
        data = np.zeros((dataset.ny, dataset.nx))
        data.put(dataset.sparse_idx_1d, V)
        eigen_glacier = np.ma.array(data=data, mask=dataset.mask_2d)
        _ = ax.imshow(
            eigen_glacier, origin="lower", cmap="twilight_shifted", vmin=-0.3, vmax=0.3
        )

        ax.text(
            0.05,
            -0.025,
            rf"$\Lambda_{k}$={lamda_scaled[k]:.1f}%",
            transform=ax.transAxes,
        )
        ax.axis("off")
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.tight_layout()

    fig_dir = Path(emulator_dir) / "eigenglaciers"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(fig_dir / f"eigenglaciers_{model_index}.pdf")
