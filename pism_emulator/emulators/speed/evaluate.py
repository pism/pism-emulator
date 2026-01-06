#!/bin/env python3

# Copyright (C) 2021 Andy Aschwanden, Douglas C Brinkerhoff
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

# pylint: disable=redefined-builtin,too-many-branches,too-many-statements

"""
Evaluate emulators.
"""

import inspect
from argparse import ArgumentParser
from os.path import dirname, realpath
from pathlib import Path
from typing import Mapping

import lightning as pl
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm.auto import tqdm

from pism_emulator.datasets import PISMInterpolatedDataset as PISMDataset
from pism_emulator.datasets import inverse_y_transform_np
from pism_emulator.emulators.nnemulator import DNNEmulator, NNEmulator
from pism_emulator.utils import param_keys_dict as keys_dict

EMULATORS: Mapping[str, type[pl.LightningModule]] = {
    "NN": NNEmulator,
    "DNN": DNNEmulator,
}

rcparams = {
    "axes.linewidth": 0.15,
    "xtick.major.size": 2.0,
    "xtick.major.width": 0.15,
    "ytick.major.size": 2.0,
    "ytick.major.width": 0.15,
    "hatch.linewidth": 0.15,
    "font.size": 6,
}

mpl.rcParams.update(rcparams)


def add_final_score_footer(
    fig: Figure,
    mae: float,
    mbe: float,
    rmse: float,
    r: float,
    r2: float,
    y: float = 0.01,
    fontsize: int = 9,
) -> None:
    """
    Add a formatted summary of evaluation metrics to the bottom of a figure.

    The footer is added using :meth:`matplotlib.figure.Figure.supxlabel`, which
    generally cooperates well with ``constrained_layout`` and reserves space for
    the label.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure that will receive the footer.
    mae : float
        Mean Absolute Error in m/yr.
    mbe : float
        Mean Bias Error (signed) in m/yr.
    rmse : float
        Root Mean Squared Error in m/yr.
    r : float
        Pearson correlation coefficient.
    r2 : float
        Coefficient of determination (r-squared).
    y : float, optional
        Vertical position of the footer in figure coordinates. Default is 0.01.
    fontsize : int, optional
        Font size used for the footer text. Default is 9.

    Returns
    -------
    None
        The function modifies ``fig`` in place.

    Notes
    -----
    Values are formatted for display (e.g., MAE/MBE with two decimals, RMSE with
    zero decimals). If you need different units or precision, pass pre-scaled
    values or adjust the formatting in this function.
    """
    score = (
        f"MAE={mae:.2f} m/yr, "
        f"MBE={mbe:.2f} m/yr, "
        f"RMSE={rmse:.0f} m/yr, "
        f"Pearson r={r:.2f}, "
        f"r²={r2:.2f}"
    )
    fig.supxlabel(f"Mean Score: {score}", y=y, fontsize=fontsize)


def current_script_directory() -> str:
    """
    Return the absolute directory containing the calling script.

    This helper inspects the current call stack to find the file path of the
    frame at index 0 (the immediate call site within this function) and returns
    its directory as an absolute path.

    Returns
    -------
    str
        Absolute path to the directory containing the script file.

    Raises
    ------
    RuntimeError
        If the script path cannot be determined (e.g., in some interactive
        environments where frames may not have a filename).

    Notes
    -----
    In notebooks, REPLs, or frozen/packaged applications, stack-based filename
    inspection can be unreliable. If you need a more robust approach, consider
    passing a reference path explicitly or using ``__file__`` when available.
    """
    frame = inspect.stack(context=0)[0]
    filename = frame.filename
    if not filename:
        raise RuntimeError(
            "Unable to determine current script directory from call stack."
        )
    return realpath(dirname(filename))


script_directory = current_script_directory()


def main():
    """
    Main.
    """
    parser = ArgumentParser()
    parser.add_argument("--emulator", choices=["NN", "DNN"], default="DNN")
    tmp, _ = parser.parse_known_args()
    parser.add_argument("--emulator-dir", default="emulator_ensemble")
    parser.add_argument("--mode", choices=["train", "validation"], default="train")
    parser.add_argument(
        "--samples-file", default="../data/samples/velocity_calibration_samples_100.csv"
    )
    parser.add_argument(
        "--target-file",
        default="../data/observed_speeds/greenland_vel_mosaic250_v1_g9000m.nc",
    )
    parser.add_argument("--target-var", type=str, default="velsurf_mag")
    parser.add_argument("--target-error-var", type=str, default="velsurf_mag_error")
    parser.add_argument("--sample-size", type=int, default=80)
    parser.add_argument("--y-lim", nargs=2, type=float, default=[1, 10e3])
    parser.add_argument("--y-transform", default="log10")
    parser.add_argument(
        "--training-files", nargs="+", help="PISM netCDF files", default=None
    )
    parser.add_argument("EMULATOR_FILES", nargs="*", help="Emulator ckpt")

    cls = EMULATORS[tmp.emulator]
    cls.add_model_specific_args(parser)
    Emulator = cls  # type: type[pl.LightningModule]
    # let the chosen model extend the parser
    if tmp.emulator == "NN":
        Emulator = NNEmulator
    elif tmp.emulator == "DNN":
        Emulator = DNNEmulator

    args = parser.parse_args()

    emulator_dir = args.emulator_dir
    emulator_files = args.EMULATOR_FILES

    y_transform = args.y_transform
    samples_file = args.samples_file
    target_file = args.target_file
    target_var = args.target_var
    target_error_var = args.target_error_var
    sample_size = args.sample_size
    y_lim = args.y_lim
    mode = args.mode
    if mode == "train":
        validation = False
    else:
        validation = True
    training_files = args.training_files

    torch.manual_seed(0)
    rng = np.random.default_rng(2021)

    dataset = PISMDataset(
        training_files=training_files,
        samples_file=samples_file,
        target_file=target_file,
        target_var=target_var,
        target_error_var=target_error_var,
        y_transform=y_transform,
        y_lim=y_lim,
    )
    X = dataset.samples.X
    F = dataset.samples.Y
    y_params = dict(dataset.cfg.y_transform_kwargs or {})

    n_members = len(F)
    k = min(sample_size, n_members)

    glaciers = sorted(rng.choice(n_members, size=k, replace=False).tolist())
    print(f"Glaciers selected: {glaciers}")

    # Calculate the mean by looping over emulators
    rmses = []
    maes = []
    mbes = []
    pearson_rs = []
    r2s = []

    plot_glaciers = sorted(rng.choice(glaciers, size=4, replace=False))
    print(f"Plot gllaciers selected: {plot_glaciers}")
    cmap = "viridis"
    fig = plt.figure(figsize=(6.4, 4.8), layout="constrained")

    # Outer layout: top block + bottom block (tweak height_ratios as you like)
    outer = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 0.4])

    # Add a skinny column on the left for row titles
    gs_top = outer[0].subgridspec(
        3,
        6,  # was 5; add 1 column for colorbars
        width_ratios=[0.12, 1, 1, 1, 1, 0.06],  # last is cbar column
        wspace=0.025,
        hspace=0.025,
    )
    # Axes for the 4 plot columns
    axs_top = np.array(
        [[fig.add_subplot(gs_top[r, c]) for c in range(1, 5)] for r in range(3)]
    )
    cax_log = fig.add_subplot(gs_top[0:2, 5])  # spans rows 0-1
    cax_diff = fig.add_subplot(gs_top[2, 5])  # spans row 2 only

    # Title column axes (one per row)
    row_titles = ["True", "Predicted", "Error"]
    for r, title in enumerate(row_titles):
        ax_lab = fig.add_subplot(gs_top[r, 0])
        ax_lab.axis("off")
        ax_lab.text(
            1.0,
            0.5,
            title,
            transform=ax_lab.transAxes,
            ha="right",
            va="center",
            fontsize=8,
        )

    # --- BOTTOM: 4 × 2 (i.e., 2 rows × 4 cols) ---
    gs_bot = outer[1].subgridspec(
        1, 5, width_ratios=[0.12, 1, 1, 1, 1], wspace=0.025, hspace=0.025
    )
    # Axes for the 4 plot columns
    axs_bot = np.array(
        [[fig.add_subplot(gs_bot[r, c]) for c in range(1, 5)] for r in range(1)]
    )

    n_emulators = len(emulator_files)

    n_glaciers = len(glaciers)
    _ = tqdm(
        total=n_emulators, position=0, leave=True, desc="Emulators", dynamic_ncols=True
    )
    p_glaciers = tqdm(
        total=n_glaciers, position=1, leave=True, desc="Glaciers", dynamic_ncols=True
    )

    p_metrics = tqdm(
        total=1, position=2, leave=True, bar_format="{desc}", dynamic_ncols=True
    )
    p_metrics.set_description_str("MAE=…, MBE=…, RMSE=…, r=…, r²=…")
    p_metrics.refresh()

    k = 0
    l = 1
    im_log = None  # Will be set in the plotting loop
    im_diff = None  # Will be set in the plotting loop
    for m in glaciers:
        p_glaciers.update(1)
        F_val = np.zeros((n_emulators, F.shape[1]))
        F_pred = np.zeros((n_emulators, F.shape[1]))
        for emulator_index, emulator_file in enumerate(emulator_files):
            e = Emulator.load_from_checkpoint(
                emulator_file,
                map_location="cpu",
            )
            e.eval()

            X_val = X[m]
            if isinstance(X_val, np.ndarray):
                X_val = torch.as_tensor(X_val)
            if X_val.dim() == 1:
                X_val = X_val.unsqueeze(0)  # (1, n_parameters)

            with torch.no_grad():
                F_v = F[m].detach().cpu().numpy()  # (n_nodes,)
                F_p = e(X_val, add_mean=True).detach().cpu().numpy()  # (1, n_nodes)

            F_val[emulator_index, :] = inverse_y_transform_np(
                F_v, name=y_transform, params=y_params, y_lim=y_lim
            )
            F_pred[emulator_index, :] = inverse_y_transform_np(
                F_p.squeeze(0), name=y_transform, params=y_params, y_lim=y_lim
            )

        rmse = np.sqrt(((F_pred.mean(axis=0) - F_val.mean(axis=0)) ** 2).mean())
        mae = mean_absolute_error(F_pred.mean(axis=0), F_val.mean(axis=0))
        mbe = (F_pred.mean(axis=0) - F_val.mean(axis=0)).mean()
        r = pearsonr(F_pred.mean(axis=0), F_val.mean(axis=0))[0]
        r2 = r2_score(F_pred.mean(axis=0), F_val.mean(axis=0))
        rmses.append(rmse)
        maes.append(mae)
        mbes.append(mbe)
        pearson_rs.append(r)
        r2s.append(r2)
        p_metrics.set_description_str(
            f"MAE={mae:.2f} m/yr, MBE={mbe:.2f} m/yr, RMSE={rmse:.0f} m/yr, "
            f"Pearson r={r:.4f}, r²={r2:.4f}"
        )
        p_metrics.refresh()

        if m in plot_glaciers:
            X_val_unscaled = (
                X_val.squeeze() * dataset.samples.X_std + dataset.samples.X_mean
            )

            F_val_2d = np.zeros((dataset.target.ny, dataset.target.nx))
            F_val_2d.put(dataset.target.sparse_idx_1d, F_val)

            F_pred_2d = np.zeros((dataset.target.ny, dataset.target.nx))
            F_pred_2d.put(dataset.target.sparse_idx_1d, F_pred)

            mask = np.logical_or(F_val_2d < 0.01, F_pred_2d < 0.01)
            F_val_2d = np.ma.array(data=F_val_2d, mask=mask)
            F_pred_2d = np.ma.array(data=F_pred_2d, mask=mask)

            im_log = axs_top[0, k].imshow(
                F_val_2d, origin="lower", cmap=cmap, norm=LogNorm(vmin=1e0, vmax=1e3)
            )
            axs_top[1, k].imshow(
                F_pred_2d, origin="lower", cmap=cmap, norm=LogNorm(vmin=1e0, vmax=1e3)
            )
            im_diff = axs_top[2, k].imshow(
                (F_pred_2d - F_val_2d) / F_val_2d,
                origin="lower",
                vmin=-0.2,
                vmax=0.2,
                cmap="coolwarm",
            )
            axs_bot[0, k].text(
                0.2,
                0.75,
                "\n".join(
                    [
                        f"{keys_dict[i]}: {j:.3f}"
                        for i, j in zip(dataset.samples.X_keys, X_val_unscaled)
                    ]
                ),
                c="k",
                transform=axs_bot[0, k].transAxes,
            )

            axs_bot[0, k].text(
                0.2,
                0.2,
                f"MAE = {mae:.1f} m/yr\nMBE = {mbe:.1f} m/yr\nRMSE = {rmse:.0f} m/yr\nr = {r2:.3f}",
                c="k",
                transform=axs_bot[0, k].transAxes,
            )
            k += 1
        l += 1

    # im_log and im_diff are guaranteed to be set since plot_glaciers is non-empty
    assert im_log is not None, "im_log should be set by plotting loop"
    assert im_diff is not None, "im_diff should be set by plotting loop"

    cb1 = fig.colorbar(im_log, cax=cax_log, extend="both")
    cb1.set_label("m/yr (log scale)")

    cb2 = fig.colorbar(im_diff, cax=cax_diff, extend="both")
    cb2.set_label("Relative error")

    rmse_mean = np.array(rmses).mean()
    mae_mean = np.array(maes).mean()
    mbe_mean = np.array(mbes).mean()
    pearson_r_mean = np.array(pearson_rs).mean()
    r2_mean = np.array(r2s).mean()

    add_final_score_footer(
        fig,
        mae=mae_mean,
        mbe=mbe_mean,
        rmse=rmse_mean,
        r=pearson_r_mean,
        r2=r2_mean,
        y=0.01,  # nudge closer/further from bottom if needed
        fontsize=6,
    )
    print("\n\n")
    print("\n\nFinal Score:\n=======================================================")
    print(
        f"MAE={mae_mean:.2f}m/yr, MBE={mbe_mean:.2f} m/yr, RMSE={rmse_mean:.0f} m/yr, Pearson r={pearson_r_mean:.2f}, r2={r2_mean:.2f}"
    )
    print("\n")

    for ax in axs_top.ravel():
        ax.set_axis_off()
    for ax in axs_bot.ravel():
        ax.set_axis_off()

    if validation:
        mode = "val"
    else:
        mode = "train"

    fig_dir = Path(f"{emulator_dir}/{mode}")
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig_name = fig_dir / Path(f"speed_emulator_{mode}.pdf")
    print(f"Saving to {fig_name.resolve()}")
    fig.savefig(fig_name)


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
