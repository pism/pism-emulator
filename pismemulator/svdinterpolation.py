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

from collections import OrderedDict
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import xarray as xr

import torch
import torch.nn as nn
from torch import Tensor
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset

from pismemulator.metrics import L2MeanSquaredError


class DEMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        training_files=None,
        target_file=None,
        target_var="surface_altitude",
        training_var="usurf",
        thinning_factor=1,
        normalize_x=True,
        epsilon=0,
        return_numpy=False,
    ):
        self.training_files = training_files
        self.target_file = target_file
        self.target_var = target_var
        self.thinning_factor = thinning_factor
        self.training_var = training_var
        self.epsilon = epsilon
        self.normalize_x = normalize_x
        self.return_numpy = return_numpy
        self.load_target()
        self.load_data()

    def __getitem__(self, i):
        return tuple(d[i] for d in [self.X, self.Y])

    def __len__(self):
        return min(len(d) for d in [self.X, self.Y])

    def load_target(self):
        epsilon = self.epsilon
        return_numpy = self.return_numpy
        thinning_factor = self.thinning_factor
        print("Loading observations data")
        print(f"       - {self.target_file}")
        with xr.open_dataset(self.target_file) as ds:
            obs = ds.variables[self.target_var]
            mask = obs.isnull()
            m_mask = np.ones_like(mask)
            m_mask[mask == True] = 0
            obs = obs[::thinning_factor, ::thinning_factor]
            m_mask = m_mask[::thinning_factor, ::thinning_factor]
            I = torch.from_numpy(m_mask.ravel())
            R = torch.from_numpy(np.nan_to_num(obs.values.ravel(), 0))
            n_row, n_col = obs.shape
            self.I = I
            self.R = R
            self.obs_ny = n_row
            self.obs_nx = n_col
            self.Obs = obs
            self.mask = m_mask
            self.obs_mask = mask
            self.x = ds.x.values
            self.y = ds.y.values

    def load_data(self):
        epsilon = self.epsilon
        return_numpy = self.return_numpy
        thinning_factor = self.thinning_factor

        print("Loading training data")
        all_data = []
        for idx, m_file in tqdm(enumerate(self.training_files)):
            print(f"       - Loading {m_file}")
            with xr.open_dataset(m_file) as ds:
                data = ds.variables[self.training_var]
                data = np.squeeze(
                    np.nan_to_num(
                        data.values[
                            ::thinning_factor, ::thinning_factor, ::thinning_factor
                        ],
                        nan=epsilon,
                    )
                )

                nt, ny, nx = data.shape
                all_data.append(data.reshape(nt, -1))
                ds.close()
        X = torch.from_numpy(np.concatenate(all_data, axis=0))

        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        self.X_mean = X_mean
        self.X_std = X_std

        if self.normalize_x:
            X -= X_mean

        self.X = X

        self.train_nt = nt
        self.train_nx = nx
        self.train_ny = ny


class DEMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        X,
        R,
        I,
        q: int = 30,
        batch_size: int = 128,
        train_size: float = 0.9,
        num_workers: int = 0,
    ):
        super().__init__()
        self.X = X
        self.R = R
        self.I = I
        self.q = q
        self.batch_size = batch_size
        self.train_size = train_size
        self.num_workers = num_workers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser.add_argument("-q", type=int, default=30)

        return parent_parser

    def setup(self, stage: str = None):

        self.get_eigenglaciers(q=self.q)
        self.V_i = self.V[self.I] * self.S
        self.R_i = self.R[self.I].reshape(-1, 1)

        all_data = TensorDataset(self.V_i, self.R_i)
        self.all_data = all_data

        training_data, val_data = train_test_split(
            all_data, train_size=self.train_size, random_state=0
        )

        self.training_data = training_data
        self.val_data = val_data

        train_loader = DataLoader(
            dataset=training_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        self.train_loader = train_loader

        val_loader = DataLoader(
            dataset=val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        self.val_loader = val_loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def get_eigenglaciers(self, **kwargs):
        defaultKwargs = {
            "q": 30,
        }
        kwargs = {**defaultKwargs, **kwargs}
        q = kwargs["q"]
        print(f"Generating {q} eigenglaciers")
        U, S, V = torch.svd_lowrank(self.X, q=q)
        self.U = U
        self.S = S
        self.V = V


class LinearRegression(pl.LightningModule):
    def __init__(
        self,
        inputSize,
        outputSize,
        hparams,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.linear = torch.nn.Linear(inputSize, outputSize)

        self.train_loss = L2MeanSquaredError()
        self.val_loss = L2MeanSquaredError()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LinearRegression")
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--learning_rate", type=float, default=0.1)
        parser.add_argument("--l2_regularization", type=float, default=1e6)

        return parent_parser

    def forward(self, x):
        out = self.linear(x)
        return out

    def training_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self.forward(x)
        weight = self.linear.weight
        K = self.hparams.l2_regularization
        loss = self.train_loss(y_hat, y, weight, K)
        return {"loss": loss, "x": x, "y": y, "y_hat": y_hat, "weight": weight, "K": K}

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        weight = self.linear.weight
        K = self.hparams.l2_regularization
        loss = self.val_loss(y_hat, y, weight, K)
        return {"loss": loss, "x": x, "y": y, "y_hat": y_hat, "weight": weight, "K": K}

    def training_epoch_end(self, outputs):

        self.log(
            "train_loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def validation_epoch_end(self, outputs):

        self.log(
            "val_loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
b            self.parameters(), self.hparams.learning_rate, weight_decay=0.0
        )
        # This is an approximation to Doug's version:
        scheduler = {
            "scheduler": ExponentialLR(optimizer, 0.9975, verbose=False),
        }

        return [optimizer], [scheduler]
