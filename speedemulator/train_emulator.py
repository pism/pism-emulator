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
from scipy.stats import dirichlet

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pismemulator.nnemulator import NNEmulator, PISMDataset, PISMDataModule
from pismemulator.utils import plot_validation


if __name__ == "__main__":
    __spec__ = None

    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="../tests/training_data")
    parser.add_argument("--emulator_dir", default="emulator_ensemble")
    parser.add_argument("--num_models", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--samples_file", default="../data/samples/velocity_calibration_samples_50.csv")
    parser.add_argument(
        "--target_file",
        default="../tests/test_data/greenland_vel_mosaic250_v1_g9000m.nc",
    )
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--thinning_factor", type=int, default=1)

    parser = NNEmulator.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    hparams = vars(args)

    batch_size = args.batch_size
    data_dir = args.data_dir
    emulator_dir = args.emulator_dir
    max_epochs = args.max_epochs
    num_models = args.num_models
    num_workers = args.num_workers
    samples_file = args.samples_file
    target_file = args.target_file
    test_size = args.test_size
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

    torch.manual_seed(0)
    pl.seed_everything(0)
    np.random.seed(0)

    if not os.path.isdir(emulator_dir):
        os.makedirs(emulator_dir)

    for model_index in range(num_models):
        print(f"Training model {model_index} of {num_models}")
        omegas = torch.Tensor(dirichlet.rvs(np.ones(n_samples))).T
        omegas = omegas.type_as(X)
        omegas_0 = torch.ones_like(omegas) / len(omegas)

        data_loader = PISMDataModule(X, F, omegas, omegas_0, test_size=test_size, num_workers=num_workers)
        data_loader.prepare_data()
        data_loader.setup(stage="fit")
        n_eigenglaciers = data_loader.n_eigenglaciers
        V_hat = data_loader.V_hat
        F_mean = data_loader.F_mean
        F_train = data_loader.F_bar

        early_stop_callback = EarlyStopping(monitor="test_loss", min_delta=0.00, patience=5, verbose=False, mode="max")
        checkpoint_callback = ModelCheckpoint(dirpath=emulator_dir, filename="emulator_{epoch}_{model_index}")
        logger = TensorBoardLogger(tb_logs_dir, name=f"Emulator {model_index}")
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        e = NNEmulator(
            n_parameters,
            n_eigenglaciers,
            V_hat,
            F_mean,
            area,
            hparams,
        )
        callbacks = [lr_monitor, checkpoint_callback]
        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=callbacks,
            logger=logger,
            deterministic=True,
            num_sanity_val_steps=0,
        )
        trainer.fit(e, data_loader.train_all_loader, data_loader.val_all_loader)
        trainer.save_checkpoint(f"{emulator_dir}/emulator_{model_index:03d}.ckpt")
        torch.save(e.state_dict(), f"{emulator_dir}/emulator_{model_index:03d}.h5")

        plot_validation(e, F_mean, dataset, data_loader, model_index, emulator_dir)
