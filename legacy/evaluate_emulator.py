#!/bin/env python3

# Copyright (C) 2021,25 Andy Aschwanden, Douglas C Brinkerhoff
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

# pylint: disable=bare-except,possibly-used-before-assignment,redefined-builtin

"""
Evaluate emulators.
"""

import inspect
from argparse import ArgumentParser
from os.path import abspath, dirname, join, realpath
from pathlib import Path

import numpy as np
import pylab as plt
import torch
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm.auto import tqdm

from pism_emulator.datasets import LegacyPISMDataset as PISMDataset
from pism_emulator.emulators.nnemulator import LegacyNNEmulator as NNEmulator
from pism_emulator.utils import param_keys_dict as keys_dict


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

if __name__ == "__main__":
    __spec__ = None  # type: ignore

    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="../tests/training_data")
    parser.add_argument("--emulator_dir", default="emulator_ensemble")
    parser.add_argument("--num_models", type=int, default=50)
    parser.add_argument("--mode", choices=["train", "validation"], default="validation")
    parser.add_argument(
        "--samples_file",
        default=abspath(
            join(
                script_directory, "../data/samples/velocity_calibration_samples_50.csv"
            )
        ),
    )
    parser.add_argument(
        "--target_file",
        default="../data/observed_speeds/greenland_vel_mosaic250_v1_g9000m.nc",
    )
    parser.add_argument("--target_var", type=str, default="velsurf_mag")
    parser.add_argument("--target_error_var", type=str, default="velsurf_mag_error")
    parser.add_argument("--sample_size", type=int, default=80)

    parser = NNEmulator.add_model_specific_args(parser)
    args = parser.parse_args()
    hparams = vars(args)

    data_dir = args.data_dir
    emulator_dir = args.emulator_dir
    num_models = args.num_models
    samples_file = args.samples_file
    target_file = args.target_file
    target_var = args.target_var
    target_error_var = args.target_error_var
    sample_size = args.sample_size
    mode = args.mode
    if mode == "train":
        validation = False
    else:
        validation = True

    torch.manual_seed(0)
    rng = np.random.default_rng(2021)

    dataset = PISMDataset(
        data_dir=data_dir,
        samples_file=samples_file,
        target_file=target_file,
        target_var=target_var,
        target_error_var=target_error_var,
        threshold=1e7,
    )
    X = dataset.X
    F = dataset.Y
    n_members = len(F)
    if sample_size <= n_members:
        glaciers = rng.choice(range(n_members), size=sample_size, replace=False)
    else:
        glaciers = range(n_members)
    print(f"Glaciers selected: {glaciers}")

    # Calculate the mean by looping over emulators
    rmses = []
    maes = []
    mbes = []
    pearson_rs = []
    r2s = []

    plot_glaciers = rng.choice(glaciers, size=4, replace=False)

    cmap = "viridis"
    fig, axs = plt.subplots(
        nrows=4, ncols=4, sharex="col", sharey="row", figsize=(6.4, 10)
    )

    n_models = num_models
    n_glaciers = len(glaciers)
    p_models = tqdm(
        total=n_models, position=0, leave=True, desc="Models", dynamic_ncols=True
    )
    p_glaciers = tqdm(
        total=n_glaciers, position=1, leave=True, desc="Glaciers", dynamic_ncols=True
    )

    # text-only "bar" that stays as a single line
    p_metrics = tqdm(
        total=1, position=2, leave=True, bar_format="{desc}", dynamic_ncols=True
    )
    p_metrics.set_description_str("MAE=…, MBE=…, RMSE=…, r=…, r²=…")
    p_metrics.refresh()

    k = 0
    l = 1
    for m in range(n_glaciers):
        p_glaciers.update(1)
        F_val = np.zeros((num_models, F.shape[1]))
        F_pred = np.zeros((num_models, F.shape[1]))
        for model_index in range(n_models):
            p_models.update(1)
            emulator_file = join(emulator_dir, f"emulator_{model_index}.h5")
            state_dict = torch.load(emulator_file)
            e = NNEmulator(
                state_dict["l_1.weight"].shape[1],
                state_dict["V_hat"].shape[1],
                state_dict["V_hat"],
                state_dict["F_mean"],
                state_dict["area"],
                hparams,
            )
            e.load_state_dict(state_dict)
            e.eval()

            X_val = X[m]
            if isinstance(X_val, np.ndarray):
                X_val = torch.as_tensor(X_val)
            if X_val.dim() == 1:
                X_val = X_val.unsqueeze(0)  # (1, n_parameters)

            with torch.no_grad():
                F_v = F[m].detach().cpu().numpy()  # (n_nodes,)
                F_p = e(X_val, add_mean=True).detach().cpu().numpy()  # (1, n_nodes)

            # store per-model predictions; we'll ensemble by mean later
            F_val[model_index, :] = F_v
            F_pred[model_index, :] = F_p.squeeze(0)

        rmse = np.sqrt(
            ((10 ** F_pred.mean(axis=0) - 10 ** F_val.mean(axis=0)) ** 2).mean()
        )
        mae = mean_absolute_error(10 ** F_pred.mean(axis=0), 10 ** F_val.mean(axis=0))
        mbe = (10 ** F_pred.mean(axis=0) - 10 ** F_val.mean(axis=0)).mean()
        r = pearsonr(F_pred.mean(axis=0), F_val.mean(axis=0))[0]
        r2 = r2_score(F_pred.mean(axis=0), F_val.mean(axis=0))
        rmses.append(rmse)
        maes.append(mae)
        mbes.append(mbe)
        pearson_rs.append(r)
        r2s.append(r2)
        # after computing metrics for this model (or overall), update the fixed line:
        p_metrics.set_description_str(
            f"MAE={mae:.2f} m/yr, MBE={mbe:.2f} m/yr, RMSE={rmse:.0f} m/yr, "
            f"Pearson r={r:.4f}, r²={r2:.4f}"
        )
        p_metrics.refresh()
        if m in plot_glaciers:
            X_val_unscaled = X_val.squeeze() * dataset.X_std + dataset.X_mean

            F_val_2d = np.zeros((dataset.ny, dataset.nx))
            F_val_2d.put(dataset.sparse_idx_1d, 10**F_val)

            F_pred_2d = np.zeros((dataset.ny, dataset.nx))
            F_pred_2d.put(dataset.sparse_idx_1d, 10**F_pred)

            mask = np.logical_or(F_val_2d < 0.01, F_pred_2d < 0.01)
            F_val_2d = np.ma.array(data=F_val_2d, mask=mask)
            F_pred_2d = np.ma.array(data=F_pred_2d, mask=mask)

            c1 = axs[0, k].imshow(
                F_val_2d, origin="lower", cmap=cmap, norm=LogNorm(vmin=1, vmax=1e3)
            )
            axs[1, k].imshow(
                F_pred_2d, origin="lower", cmap=cmap, norm=LogNorm(vmin=1, vmax=1e3)
            )
            c2 = axs[2, k].imshow(
                F_pred_2d - F_val_2d,
                origin="lower",
                vmin=-50,
                vmax=50,
                cmap="coolwarm",
            )
            axs[-1, k].text(
                0.01,
                0.0,
                "\n".join(
                    [
                        f"{keys_dict[i]}: {j:.3f}"
                        for i, j in zip(dataset.X_keys, X_val_unscaled)
                    ]
                ),
                c="k",
                size=7,
                transform=axs[-1, k].transAxes,
            )

            axs[-1, k].text(
                0.01,
                0.75,
                f"MAE = {mae:.1f} m/yr\nMBE = {mbe:.1f} m/yr\nRMSE = {rmse:.0f} m/yr\nr = {r2:.3f}",
                c="k",
                size=7,
                transform=axs[-1, k].transAxes,
            )

            axs[0, k].set_axis_off()
            axs[1, k].set_axis_off()
            axs[2, k].set_axis_off()
            axs[-1, k].set_axis_off()

            k += 1
        l += 1

    rmse_mean = np.array(rmses).mean()
    mae_mean = np.array(maes).mean()
    mbe_mean = np.array(mbes).mean()
    pearson_r_mean = np.array(pearson_rs).mean()
    r2_mean = np.array(r2s).mean()
    print("\n\nFinal Score:\n=======================================================")
    print(
        f"MAE={mae_mean:.2f}m/yr, MBE={mbe_mean:.2f} m/yr, RMSE={rmse_mean:.0f} m/yr, Pearson r={pearson_r_mean:.2f}, r2={r2_mean:.2f}"
    )
    print("\n")
    axs[0, 0].text(
        0.01,
        0.98,
        "PISM",
        c="k",
        size=7,
        weight="bold",
        transform=axs[0, 0].transAxes,
    )
    axs[1, 0].text(
        0.01,
        0.98,
        "Emulator",
        c="k",
        size=7,
        weight="bold",
        transform=axs[1, 0].transAxes,
    )
    axs[2, 0].text(
        0.01,
        0.98,
        "PISM-Emulator",
        c="k",
        size=7,
        weight="bold",
        transform=axs[2, 0].transAxes,
    )
    # cb_ax = fig.add_axes([0.88, 0.65, 0.025, 0.15])
    # plt.colorbar(
    #     c1,
    #     cax=cb_ax,
    #     shrink=0.9,
    #     label="speed (m/yr)",
    #     orientation="vertical",
    #     extend="both",
    # )
    # cb_ax2 = fig.add_axes([0.88, 0.3, 0.025, 0.15])
    # plt.colorbar(
    #     c2,
    #     cax=cb_ax2,
    #     shrink=0.9,
    #     label="diff. (m/yr)",
    #     orientation="vertical",
    #     extend="both",
    # )
    # cb_ax.tick_params(labelsize=7)
    # cb_ax2.tick_params(labelsize=7)
    fig.subplots_adjust(wspace=0.0, hspace=0.0)

    if validation:
        mode = "val"
    else:
        mode = "train"

    fig_dir = Path(f"{emulator_dir}/{mode}")
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig_name = join(fig_dir, f"speed_emulator_{mode}.pdf")
    print(f"Saving to {fig_name}")
    fig.savefig(fig_name)
