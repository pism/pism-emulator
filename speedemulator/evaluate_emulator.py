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
    parser.add_argument("--emulator_dir", default="emulator_ensemble")
    parser.add_argument("--num_models", type=int, default=1)
    parser.add_argument("--samples_file", default="../data/samples/velocity_calibration_samples_50.csv")
    parser.add_argument(
        "--target_file",
        default="../tests/test_data/greenland_vel_mosaic250_v1_g9000m.nc",
    )
    parser.add_argument("--train_size", type=float, default=1.0)
    parser.add_argument("--thinning_factor", type=int, default=1)

    parser = NNEmulator.add_model_specific_args(parser)
    args = parser.parse_args()
    hparams = vars(args)

    data_dir = args.data_dir
    emulator_dir = args.emulator_dir
    num_models = args.num_models
    samples_file = args.samples_file
    target_file = args.target_file
    train_size = args.train_size
    thinning_factor = args.thinning_factor
    tb_logs_dir = f"{emulator_dir}/tb_logs"

    dataset = PISMDataset(
        data_dir=data_dir,
        samples_file=samples_file,
        target_file=target_file,
        thinning_factor=thinning_factor,
    )

    X = dataset.X
    F = dataset.Y
    area = dataset.normed_area
    n_grid_points = dataset.n_grid_points
    n_parameters = dataset.n_parameters
    n_samples = dataset.n_samples

    if not os.path.isdir(emulator_dir):
        os.makedirs(emulator_dir)

    omegas = torch.Tensor(dirichlet.rvs(np.ones(n_samples))).T
    omegas = omegas.type_as(X)
    omegas_0 = torch.ones_like(omegas) / len(omegas)

    if train_size == 1.0:
        data_loader = PISMDataModule(X, F, omegas, omegas_0)
    else:
        data_loader = PISMDataModule(X, F, omegas, omegas_0)

    data_loader.prepare_data()
    data_loader.setup(stage="fit")
    n_eigenglaciers = data_loader.n_eigenglaciers
    V_hat = data_loader.V_hat
    F_mean = data_loader.F_mean
    F_train = data_loader.F_bar

    corrs = []
    rmses = []
    F_ps = []
    F_vs = []
    rel_diff = []
    for model_index in range(num_models):
        print(f"Loading emulator {model_index}")
        emulator_file = join(emulator_dir, f"emulator_{0:03d}.h5".format(model_index))
        state_dict = torch.load(emulator_file)
        e = NNEmulator(
            state_dict["l_1.weight"].shape[1],
            state_dict["V_hat"].shape[1],
            state_dict["V_hat"],
            state_dict["F_mean"],
            dataset.normed_area,
            hparams,
        )
        e.load_state_dict(state_dict)
        for idx in range(len(data_loader.val_data)):
            (
                X_val,
                F_val,
                _,
                _,
            ) = data_loader.val_data[idx]
            X_val_scaled = X_val * dataset.X_std + dataset.X_mean
            F_val = (F_val + F_mean).detach().numpy().reshape(dataset.ny, dataset.nx)
            F_pred = e(X_val, add_mean=True).detach().numpy().reshape(dataset.ny, dataset.nx)
            mask = 10 ** F_val <= 1
            F_p = np.ma.array(data=10 ** F_pred, mask=mask)
            F_v = np.ma.array(data=10 ** F_val, mask=mask)
            rmse = np.sqrt(mean_squared_error(F_p, F_v))
            corr = np.corrcoef(F_val.flatten(), F_pred.flatten())[0, 1]
            rmses.append(rmse)
            corrs.append(corr)

            # Flatten predicted and validation speeds
            r = (F_p - F_v) / F_v
            rel_diff.append(r)

    S_rd = np.array(rel_diff).flatten()
