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

from argparse import ArgumentParser

import numpy as np
import os
from os.path import join
from scipy.stats import dirichlet
import torch
import pylab as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from pismemulator.nnemulator import (
    NNEmulator,
    PISMDataset,
    PISMDataModule,
)
from pismemulator.utils import plot_validation, kl_divergence


if __name__ == "__main__":
    __spec__ = None

    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="../tests/training_data")
    parser.add_argument("--emulator_dir", default="emulator_ensemble")
    parser.add_argument("--num_models", type=int, default=50)
    parser.add_argument("--mode", choices=["train", "validation"], default="validation")
    parser.add_argument(
        "--samples_file",
        default="../data/samples/velocity_calibration_samples_20_lhs.csv",
    )
    parser.add_argument(
        "--target_file",
        default="../data/observed_speeds/greenland_vel_mosaic250_v1_g1800m.nc",
    )
    parser.add_argument("--train_size", type=float, default=1.0)

    parser = NNEmulator.add_model_specific_args(parser)
    args = parser.parse_args()
    hparams = vars(args)

    data_dir = args.data_dir
    emulator_dir = args.emulator_dir
    num_models = args.num_models
    samples_file = args.samples_file
    target_file = args.target_file
    train_size = args.train_size
    mode = args.mode
    if mode == "train":
        validation = False
    else:
        validation = True

    torch.manual_seed(0)

    dataset = PISMDataset(
        data_dir=data_dir,
        samples_file=samples_file,
        target_file=target_file,
        thinning_factor=1,
        threshold=1e7,
    )
    X = dataset.X
    F = dataset.Y
    n_samples = dataset.n_samples

    rmses = []
    log_rmses = []
    r2s = []
    corrs = []

    # Calculate the mean by looping over emulators

    dfs = []
    num_ens = F.shape[0]
    num_pts = F.shape[1]
    F_vals = np.zeros((num_models * num_ens, num_pts))
    F_preds = np.zeros((num_models * num_ens, num_pts))

    for model_index in range(0, num_models):
        print(f"Loading emulator {model_index}")

        emulator_file = join(emulator_dir, "emulator", f"emulator_{model_index}.h5")
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

        # plot_validation(
        #     e,
        #     dataset,
        #     model_index,
        #     emulator_dir,
        #     validation=validation,
        # )

        F_val = F.detach().numpy()
        F_pred = e(X, add_mean=True).detach().numpy()

        F_vals[model_index * num_ens : (model_index + 1) * num_ens, :] = F_val
        F_preds[model_index * num_ens : (model_index + 1) * num_ens, :] = F_pred

        # Loop over ensemble member
        for m in range(len(F_val)):

            F_v = F_val[m]
            F_p = F_pred[m]

            rmse = np.sqrt(((10**F_p - 10**F_v) ** 2).mean())
            rmses.append(rmse)
            r2 = r2_score(F_p, F_v)
            r2s.append(r2)
            log_rmse = np.sqrt(((F_p - F_v) ** 2).mean())
            log_rmses.append(log_rmse)
            corr = np.corrcoef(F_v, F_p)[0, 1]
            corrs.append(corr)
            df = pd.DataFrame(
                data=np.array([rmse, log_rmse, corr, r2, m, model_index]).reshape(
                    1, -1
                ),
                columns=[
                    "RMSD (m/yr)",
                    "RMSD log (m/yr)",
                    "Pearson r (1)",
                    "R2 (1)",
                    "Ensemble member",
                    "Model",
                ],
            ).astype({"Ensemble member": int, "Model": int})
            dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv(f"{emulator_dir}/emulator_stats_{mode}.csv.gz", compression="infer")
