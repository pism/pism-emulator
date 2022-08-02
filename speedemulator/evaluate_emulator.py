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

    # Calculate the mean by looping over emulators

    F_val = np.zeros((num_models, F.shape[0], F.shape[1]))
    F_pred = np.zeros((num_models, F.shape[0], F.shape[1]))
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

        F_v = F.detach().numpy()
        F_p = e(X, add_mean=True).detach().numpy()
        F_val[model_index, ...] = F_v
        F_pred[model_index, ...] = F_p

        del F_v, F_p, e

    # Calculate the mean velocity field (average over the number of ensmble members) for each emulator
    # calculate the root meant square for each emulator, and then get the mean
    print(
        np.mean(
            np.sqrt((10 ** F_val.mean(axis=0) - 10 ** F_pred.mean(axis=0)) ** 2).mean(
                axis=1
            )
        )
    )
    # Calculate the mean velocity field (averaged over the num_models) for each ensemble memember,
    # calculate the root mean square difference for each ensemble memeber, and then get the mean
    print(
        np.mean(
            np.sqrt(
                ((10 ** F_val.mean(axis=1) - 10 ** F_pred.mean(axis=1)) ** 2).mean(
                    axis=1
                )
            )
        )
    )
