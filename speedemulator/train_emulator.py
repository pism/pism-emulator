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

import os
from argparse import ArgumentParser
from os.path import abspath, dirname, join, realpath

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, Timer
from lightning.pytorch.loggers import TensorBoardLogger
from scipy.stats import dirichlet

from pismemulator.nnemulator import (DNNEmulator, NNEmulator, PISMDataModule,
                                     PISMDataset)
from pismemulator.utils import plot_eigenglaciers


def current_script_directory():
    import inspect

    filename = inspect.stack(0)[0][1]
    return realpath(dirname(filename))


script_directory = current_script_directory()

if __name__ == "__main__":
    __spec__ = None

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", default=False, action="store_true")
    parser.add_argument(
        "--data_dir", default=abspath(join(script_directory, "../tests/training_data"))
    )
    parser.add_argument(
        "--emulator", choices=["NNEmulator", "DNNEmulator"], default="NNEmulator"
    )
    parser.add_argument("--emulator_dir", default="emulator_ensemble")
    parser.add_argument("--model_index", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=5)
    parser.add_argument("--q", type=int, default=100)
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
        default=abspath(
            join(
                script_directory,
                "../data/observed_speeds/greenland_vel_mosaic250_v1_g9000m.nc",
            )
        ),
    )
    parser.add_argument("--train_size", type=float, default=1.0)
    parser.add_argument("--thinning_factor", type=int, default=1)

    parser = NNEmulator.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    hparams = vars(args)

    batch_size = args.batch_size
    checkpoint = args.checkpoint
    data_dir = args.data_dir
    emulator_dir = args.emulator_dir
    max_epochs = args.max_epochs
    model_index = args.model_index
    num_workers = args.num_workers
    q = args.q
    samples_file = args.samples_file
    target_file = args.target_file
    train_size = args.train_size
    thinning_factor = args.thinning_factor
    tb_logs_dir = f"{emulator_dir}/tb_logs"

    callbacks = []

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
    pl.lite.utilities.seed.seed_everything(0)
    np.random.seed(model_index)

    if not os.path.isdir(emulator_dir):
        os.makedirs(emulator_dir)
        os.makedirs(os.path.join(emulator_dir, "emulator"))

    print(f"Training model {model_index}")
    omegas = torch.Tensor(dirichlet.rvs(np.ones(n_samples))).T
    omegas = omegas.type_as(X)
    omegas_0 = torch.ones_like(omegas) / len(omegas)

    if train_size == 1.0:
        data_loader = PISMDataModule(X, F, omegas, omegas_0, num_workers=num_workers)
    else:
        data_loader = PISMDataModule(
            X, F, omegas, omegas_0, train_size=train_size, num_workers=num_workers
        )

    data_loader.prepare_data(q=q)
    data_loader.setup(stage="fit")
    n_eigenglaciers = data_loader.n_eigenglaciers
    V_hat = data_loader.V_hat
    F_mean = data_loader.F_mean
    F_train = data_loader.F_bar

    plot_eigenglaciers(dataset, data_loader, model_index, emulator_dir)

    if checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath=emulator_dir,
            filename="emulator_{epoch}_{model_index}",
            every_n_epochs=0,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

    logger = TensorBoardLogger(tb_logs_dir, name=f"Emulator {model_index}")

    timer = Timer()
    callbacks.append(timer)

    e = NNEmulator(
        n_parameters,
        n_eigenglaciers,
        V_hat,
        F_mean,
        area,
        hparams,
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    if train_size == 1.0:
        train_loader = data_loader.train_all_loader
        val_loader = data_loader.val_all_loader
    else:
        train_loader = data_loader.train_loader
        val_loader = data_loader.val_loader

    trainer.fit(e, train_loader, val_loader)
    print(f"Training took {timer.time_elapsed():.0f}s")
    torch.save(e.state_dict(), f"{emulator_dir}/emulator/emulator_{model_index}.h5")
