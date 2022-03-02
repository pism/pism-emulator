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

from pismemulator.nnemulator import (
    NNEmulator,
    PISMDataset,
    PISMODataset,
    PISMDataModule,
)
from pismemulator.utils import plot_validation


if __name__ == "__main__":
    __spec__ = None

    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="../tests/training_data")
    parser.add_argument("--emulator_dir", default="emulator_ensemble")
    parser.add_argument("--num_models", type=int, default=50)
    parser.add_argument(
        "--samples_file",
        default="../data/samples/velocity_calibration_samples_20_lhs.csv",
    )
    parser.add_argument(
        "--target_file",
        default="../data/observed_speeds/greenland_vel_mosaic250_v1_g1800m.nc",
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

    torch.manual_seed(0)

    dataset = PISMODataset(
        data_dir=data_dir,
        samples_file=samples_file,
        target_file=target_file,
        thinning_factor=1,
        threshold=1e7,
    )
    X = dataset.X
    F = dataset.Y
    n_samples = dataset.n_samples

    for model_index in range(0, num_models):
        print(f"Loading emulator {model_index}")
        np.random.seed(model_index)

        omegas = torch.Tensor(dirichlet.rvs(np.ones(n_samples))).T
        omegas = omegas.type_as(X)
        omegas_0 = torch.ones_like(omegas) / len(omegas)

        data_loader = PISMDataModule(
            X,
            F,
            omegas,
            omegas_0,
        )

        data_loader.prepare_data(q=5)
        data_loader.setup(stage="fit")

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

        F_mean = state_dict["F_mean"]
        plot_validation(
            e,
            F_mean,
            dataset,
            data_loader,
            model_index,
            emulator_dir,
            validation=True,
        )

        idx = 0
        (
            X_val,
            F_val,
            _,
            _,
        ) = data_loader.all_data[idx]

        F_val = (F_val + F_mean).detach().numpy()
        F_pred = e(X_val, add_mean=True).detach().numpy()

        F_val_2d = np.zeros((dataset.ny, dataset.nx))
        F_val_2d.put(dataset.sparse_idx_1d, F_val)

        F_pred_2d = np.zeros((dataset.ny, dataset.nx))
        F_pred_2d.put(dataset.sparse_idx_1d, F_pred)

        F_p = np.ma.array(data=10**F_pred_2d, mask=dataset.mask_2d)
        F_v = np.ma.array(data=10**F_val_2d, mask=dataset.mask_2d)
        rmse = np.sqrt(mean_squared_error(F_p, F_v))
        corr = np.corrcoef(F_v.flatten(), F_p.flatten())[0, 1]
        print(model_index, idx, rmse, corr)

        # for idx in range(len(data_loader.all_data)):
        #     (
        #         X_val,
        #         F_val,
        #         _,
        #         _,
        #     ) = data_loader.all_data[idx]

        #     F_val = (F_val + state_dict["F_mean"]).detach().numpy()
        #     F_pred = e(X_val, add_mean=True).detach().numpy()

        #     F_val_2d = np.zeros((dataset.ny, dataset.nx))
        #     F_val_2d.put(dataset.sparse_idx_1d, F_val)

        #     F_pred_2d = np.zeros((dataset.ny, dataset.nx))
        #     F_pred_2d.put(dataset.sparse_idx_1d, F_pred)

        #     F_p = np.ma.array(data=10**F_pred_2d, mask=dataset.mask_2d)
        #     F_v = np.ma.array(data=10**F_val_2d, mask=dataset.mask_2d)
        #     rmse = np.sqrt(mean_squared_error(F_p, F_v))
        #     corr = np.corrcoef(F_v.flatten(), F_p.flatten())[0, 1]
        #     print(model_index, idx, rmse, corr)
