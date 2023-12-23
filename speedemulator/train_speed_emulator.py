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
import warnings
from argparse import ArgumentParser
from os.path import abspath, dirname, join, realpath

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, Timer
from lightning.pytorch.loggers import TensorBoardLogger
from scipy.stats import dirichlet

from pism_emulator.datamodules import PISMDataModule
from pism_emulator.datasets import PISMDataset
from pism_emulator.nnemulator import NNEmulator
from pism_emulator.utils import plot_eigenglaciers

warnings.filterwarnings("ignore", ".*does not have many workers.*")


def current_script_directory():
    import inspect

    filename = inspect.stack(0)[0][1]
    return realpath(dirname(filename))


script_directory = current_script_directory()

if __name__ == "__main__":
    __spec__ = None

    parser = ArgumentParser()
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--checkpoint", default=False, action="store_true")
    parser.add_argument(
        "--data_dir", default=abspath(join(script_directory, "../tests/training_data"))
    )
    parser.add_argument("--devices", default="auto")
    parser.add_argument(
        "--emulator", choices=["NNEmulator", "DNNEmulator"], default="NNEmulator"
    )
    parser.add_argument("--emulator_dir", default="emulator_ensemble")
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--model_index", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
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
    parser.add_argument("--target_var", type=str, default="velsurf_mag")
    parser.add_argument("--target_error_var", type=str, default="velsurf_mag_error")
    parser.add_argument("--train_size", type=float, default=1.0)
    parser.add_argument("--thinning_factor", type=int, default=1)

    parser = NNEmulator.add_model_specific_args(parser)
    args = parser.parse_args()
    hparams = vars(args)

    accelerator = args.accelerator
    batch_size = args.batch_size
    checkpoint = args.checkpoint
    data_dir = args.data_dir
    devices = args.devices
    emulator_dir = args.emulator_dir
    model_index = args.model_index
    num_workers = args.num_workers
    max_epochs = args.max_epochs
    q = args.q
    samples_file = args.samples_file
    target_file = args.target_file
    target_var = args.target_var
    target_error_var = args.target_error_var
    train_size = args.train_size
    thinning_factor = args.thinning_factor
    tb_logs_dir = f"{emulator_dir}/tb_logs"

    callbacks: list = []

    dataset = PISMDataset(
        data_dir=data_dir,
        samples_file=samples_file,
        target_file=target_file,
        target_var=target_var,
        target_error_var=target_error_var,
        thinning_factor=thinning_factor,
        verbose=True,
    )

    X = dataset.X
    F = dataset.Y
    area = dataset.normed_area
    n_grid_points = dataset.n_grid_points
    n_parameters = dataset.n_parameters
    n_samples = dataset.n_samples

    torch.manual_seed(0)
    np.random.seed(model_index)

    if not os.path.isdir(emulator_dir):
        os.makedirs(emulator_dir)
        os.makedirs(os.path.join(emulator_dir, "emulator"))

    print(f"Training model {model_index}")
    omegas = torch.Tensor(dirichlet.rvs(np.ones(n_samples))).T
    omegas = omegas.type_as(X)
    omegas_0 = torch.ones_like(omegas) / len(omegas)

    if train_size == 1.0:
        data_loader = PISMDataModule(
            X, F, omegas, omegas_0, num_workers=num_workers, batch_size=batch_size
        )
    else:
        data_loader = PISMDataModule(
            X,
            F,
            omegas,
            omegas_0,
            train_size=train_size,
            num_workers=num_workers,
            batch_size=batch_size,
        )

    data_loader.prepare_data(q=q)
    data_loader.setup(stage="fit")
    n_eigenglaciers = data_loader.n_eigenglaciers
    V_hat = data_loader.V_hat
    F_mean = data_loader.F_mean

    plot_eigenglaciers(dataset, data_loader, model_index, emulator_dir, q=q)

    if checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{emulator_dir}/emulator",
            filename="emulator_{model_index}",
            every_n_epochs=0,
            save_last=True,
        )
        checkpoint_callback.CHECKPOINT_NAME_LAST = f"emulator_{model_index}"
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
    print(accelerator)
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        accelerator=accelerator,
        devices=devices,
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