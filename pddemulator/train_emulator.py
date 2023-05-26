#!/bin/env python3

# Copyright (C) 2021-22 Andy Aschwanden, Douglas C Brinkerhoff
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
from os.path import join
from pathlib import Path

from typing import Union

import lightning as pl
from lightning import LightningModule

import numpy as np
import pandas as pd
import pylab as plt
import torch
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.loggers import TensorBoardLogger

# from lightning.pytorch.tuner import Tuner
from pyDOE import lhs
from scipy.stats import dirichlet
from scipy.stats.distributions import uniform
from sklearn.metrics import mean_squared_error

from tqdm import tqdm
from scipy.stats import beta

import time

from pismemulator.nnemulator import PDDEmulator, TorchPDDModel
from pismemulator.datamodules import PDDDataModule
from pismemulator.utils import load_hirham_climate_w_std_dev

import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")


def draw_samples(n_samples=250, random_seed=2):
    np.random.seed(random_seed)

    distributions = {
        "f_snow": uniform(loc=1.0, scale=5.0),  # uniform between 1 and 6
        "f_ice": uniform(loc=3.0, scale=12),  # uniform between 3 and 15
        "refreeze_snow": uniform(loc=0.0, scale=1.0),  # uniform between 0 and 1
        "refreeze_ice": uniform(loc=0.0, scale=1.0),  # uniform between 0 and 1
        "temp_snow": uniform(loc=-2, scale=4.0),  # uniform between 0 and 1
        "temp_rain": uniform(loc=0.0, scale=4.0),  # uniform between 0 and 1
    }
    # Names of all the variables
    keys = [x for x in distributions.keys()]

    # Describe the Problem
    problem = {"num_vars": len(keys), "names": keys, "bounds": [[0, 1]] * len(keys)}

    # Generate uniform samples (i.e. one unit hypercube)
    unif_sample = lhs(len(keys), n_samples)

    # To hold the transformed variables
    dist_sample = np.zeros_like(unif_sample)

    # Now transform the unit hypercube to the prescribed distributions
    # For each variable, transform with the inverse of the CDF (inv(CDF)=ppf)
    for i, key in enumerate(keys):
        dist_sample[:, i] = distributions[key].ppf(unif_sample[:, i])

    # Save to CSV file using Pandas DataFrame and to_csv method
    header = keys
    # Convert to Pandas dataframe, append column headers, output as csv
    df = pd.DataFrame(data=dist_sample, columns=header)

    return df


if __name__ == "__main__":
    __spec__ = None

    parser = ArgumentParser()
    parser.add_argument("--emulator_dir", default="emulator_ensemble")
    parser.add_argument("--n_interpolate", type=int, default=12)
    parser.add_argument("--model_index", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--thinning_factor", type=int, default=1)
    parser.add_argument(
        "--training_file", type=str, default="DMI-HIRHAM5_1980_2020_MMS.nc"
    )
    parser.add_argument("--use_obs_sd", default=False, action="store_true")

    parser = PDDEmulator.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    hparams = vars(args)

    batch_size = args.batch_size
    emulator_dir = args.emulator_dir
    n_interpolate = args.n_interpolate
    max_epochs = args.max_epochs
    model_index = args.model_index
    num_workers = args.num_workers
    train_size = args.train_size
    thinning_factor = args.thinning_factor
    training_file = args.training_file
    tb_logs_dir = f"{emulator_dir}/tb_logs"
    use_observed_std_dev = args.use_obs_sd

    if not os.path.isdir(emulator_dir):
        os.makedirs(emulator_dir)
        os.makedirs(os.path.join(emulator_dir, "emulator"))

    torch.manual_seed(0)
    pl.seed_everything(0)
    np.random.seed(model_index)

    (
        temp,
        precip,
        std_dev,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = load_hirham_climate_w_std_dev(training_file, thinning_factor=thinning_factor)

    if not use_observed_std_dev:
        std_dev = np.zeros_like(temp)

    prior_df = draw_samples(n_samples=250)

    nt = temp.shape[0]
    X_m = []
    Y_m = []
    for k, row in prior_df.iterrows():
        m_f_snow = row["f_snow"]
        m_f_ice = row["f_ice"]
        m_r_snow = row["refreeze_snow"]
        m_r_ice = row["refreeze_ice"]
        m_temp_snow = row["temp_snow"]
        m_temp_rain = row["temp_rain"]
        params = np.hstack(
            [np.tile(row[k], (temp.shape[1], 1)) for k in range(len(row))]
        )

        pdd = TorchPDDModel(
            pdd_factor_snow=m_f_snow,
            pdd_factor_ice=m_f_ice,
            refreeze_snow=m_r_snow,
            refreeze_ice=m_r_ice,
            temp_snow=m_temp_snow,
            temp_rain=m_temp_rain,
            n_interpolate=n_interpolate,
        )
        result = pdd(temp, precip, std_dev)

        A = result["accu"]
        M = result["snow_melt"]
        R = result["runoff"]
        F = result["refreeze"]
        B = result["smb"]

        m_Y = torch.vstack(
            (
                A,
                M,
                R,
                F,
                B,
            )
        ).T
        Y_m.append(m_Y)
        X_m.append(
            torch.from_numpy(
                np.hstack(
                    (
                        temp.T,
                        precip.T,
                        std_dev.T,
                        params,
                    )
                )
            )
        )

    X = torch.vstack(X_m).type(torch.FloatTensor)
    Y = torch.vstack(Y_m).type(torch.FloatTensor)
    n_samples, n_parameters = X.shape
    n_outputs = Y.shape[1]

    # Normalize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_norm = torch.nan_to_num((X - X_mean) / X_std, 0)

    callbacks = []
    timer = Timer()
    callbacks.append(timer)

    print(f"Training model {model_index}")
    omegas = torch.Tensor(dirichlet.rvs(np.ones(n_samples))).T
    omegas = omegas.type(torch.FloatTensor)
    omegas_0 = torch.ones_like(omegas) / len(omegas)
    area = torch.ones_like(omegas)

    # Load training data
    data_loader = PDDDataModule(X, Y, omegas, omegas_0, num_workers=num_workers)
    data_loader.setup()

    # Generate emulator
    e = PDDEmulator(
        n_parameters,
        n_outputs,
        hparams,
    )

    # Setup trainer
    logger = TensorBoardLogger(tb_logs_dir, name=f"Emulator {model_index}")
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        default_root_dir=emulator_dir,
        num_sanity_val_steps=0,
    )
    train_loader = data_loader.train_loader
    val_loader = data_loader.val_loader

    # tuner = Tuner(trainer)

    # Auto-scale batch size by growing it exponentially (default)
    # tuner.scale_batch_size(model, mode="power")

    # Train the emulator
    emulator_file = f"{emulator_dir}/emulator/emulator_{model_index}.h5"
    trainer.fit(e, train_loader, val_loader)
    print(f"Training took {timer.time_elapsed():.0f}s")
    torch.save(e.state_dict(), emulator_file)

    X_val = torch.vstack([d[0] for d in data_loader.val_data])
    Y_val = torch.vstack([d[1] for d in data_loader.val_data])

    e.eval()

    Y_pred = e(X_val).detach().cpu()
    rmse = [
        np.sqrt(
            mean_squared_error(
                Y_pred.detach().cpu().numpy()[:, i],
                Y_val.detach().cpu().numpy()[:, i],
            )
        )
        for i in range(Y_val.shape[1])
    ]
    print("RMSE")
    print(f"A={rmse[0]:.6f}, M={rmse[1]:.6f}", f"R={rmse[2]:.6f}, F={rmse[3]:.6f}")

    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(12, 4))
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    axs[0].plot(Y_val[:, 0], Y_pred[:, 0], ".", ms=0.25, label="Accumulation")
    axs[1].plot(Y_val[:, 1], Y_pred[:, 1], ".", ms=0.25, label="Melt")
    axs[2].plot(Y_val[:, 2], Y_pred[:, 2], ".", ms=0.25, label="Runoff")
    axs[3].plot(Y_val[:, 3], Y_pred[:, 3], ".", ms=0.25, label="Refreeze")
    axs[4].plot(Y_val[:, 4], Y_pred[:, 4], ".", ms=0.25, label="SMB")
    for k in range(5):
        m_max = np.ceil(np.maximum(Y_val[:, k].max(), Y_pred[:, k].max()))
        m_min = np.floor(np.minimum(Y_val[:, k].min(), Y_pred[:, k].min()))
        axs[k].set_xlim(m_min, m_max)
        axs[k].set_ylim(m_min, m_max)
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()
    axs[4].legend()
    fig.savefig(f"{emulator_dir}/validation_{model_index}.pdf")
