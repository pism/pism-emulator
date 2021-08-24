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

from sklearn.metrics import mean_squared_error

from pismemulator.nnemulator import NNEmulator, PISMDataset, PISMDataModule
from pismemulator.utils import plot_validation


if __name__ == "__main__":
    __spec__ = None

    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="../tests/training_data")
    parser.add_argument("--validation_data_dir", default="../tests/training_data")
    parser.add_argument("--emulator_dir", default="emulator_ensemble")
    parser.add_argument("--num_models", type=int, default=50)
    parser.add_argument("--samples_file", default="../data/samples/velocity_calibration_samples_50.csv")
    parser.add_argument("--validation_samples_file", default="../data/samples/velocity_calibration_samples_50.csv")
    parser.add_argument(
        "--target_file",
        default="../tests/test_data/greenland_vel_mosaic250_v1_g9000m.nc",
    )
    parser.add_argument(
        "--validation_target_file",
        default="../tests/test_data/greenland_vel_mosaic250_v1_g9000m.nc",
    )
    parser.add_argument("--train_size", type=float, default=1.0)
    parser.add_argument("--thinning_factor", type=int, default=1)

    parser = NNEmulator.add_model_specific_args(parser)
    args = parser.parse_args()
    hparams = vars(args)

    data_dir = args.data_dir
    validation_data_dir = args.validation_data_dir
    emulator_dir = args.emulator_dir
    num_models = args.num_models
    samples_file = args.samples_file
    validation_samples_file = args.validation_samples_file
    target_file = args.target_file
    validation_target_file = args.validation_target_file
    train_size = args.train_size
    thinning_factor = args.thinning_factor

    # data_loader = PISMDataModule(X, F, omegas, omegas_0)

    # data_loader.prepare_data()
    # data_loader.setup(stage="fit")
    # n_eigenglaciers = data_loader.n_eigenglaciers
    # V_hat = data_loader.V_hat
    # F_mean = data_loader.F_mean
    # F_train = data_loader.F_bar

    validation_dataset = PISMDataset(
        data_dir=validation_data_dir,
        samples_file=validation_samples_file,
        target_file=validation_target_file,
        thinning_factor=1,
        threshold=500e3,
    )
    X = validation_dataset.X
    F = validation_dataset.Y
    n_samples = validation_dataset.n_samples

    omegas = torch.Tensor(dirichlet.rvs(np.ones(n_samples))).T
    omegas = omegas.type_as(X)
    omegas_0 = torch.ones_like(omegas) / len(omegas)

    validation_data_loader = PISMDataModule(
        X,
        F,
        omegas,
        omegas_0,
    )

    validation_data_loader.prepare_data()
    validation_data_loader.setup(stage="fit")
    F_mean = validation_data_loader.F_mean

    for model_index in range(num_models):
        print(f"Loading emulator {model_index}")
        emulator_file = join(emulator_dir, f"emulator_{0:03d}.h5".format(model_index))
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
        for idx in range(len(validation_data_loader.all_data)):
            (
                X_val,
                F_val,
                _,
                _,
            ) = validation_data_loader.all_data[idx]
            # X_val_unscaled = X_val * validation_dataset.X_std + validation_dataset.X_mean
            F_val = (
                (F_val + state_dict["F_mean"]).detach().numpy().reshape(validation_dataset.ny, validation_dataset.nx)
            )
            F_pred = e(X_val, add_mean=True).detach().numpy().reshape(validation_dataset.ny, validation_dataset.nx)
            mask = 10 ** F_val <= 1
            F_p = np.ma.array(data=10 ** F_pred, mask=mask)
            F_v = np.ma.array(data=10 ** F_val, mask=mask)
            rmse = np.sqrt(mean_squared_error(F_p, F_v))
            corr = np.corrcoef(F_val.flatten(), F_pred.flatten())[0, 1]
            print(model_index, idx, rmse, corr)
