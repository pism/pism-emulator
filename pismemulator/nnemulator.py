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
from glob import glob
import numpy as np
import pandas as pd
from os.path import join
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import xarray as xr

import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics import Metric
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from pismemulator.metrics import AbsoluteError, absolute_error


class DNNEmulator(pl.LightningModule):
    def __init__(
        self,
        n_parameters: int,
        n_eigenglaciers: int,
        V_hat: Tensor,
        F_mean: Tensor,
        area: Tensor,
        hparams,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        n_layers = self.hparams.n_layers
        n_hidden = self.hparams.n_hidden

        assert isinstance(n_hidden, (list, int, np.ndarray)), f"{n_hidden} is not  (list, int, np.ndarray)"

        # Inputs to hidden layer linear transformation
        self.l_first = nn.Linear(n_parameters, n_hidden)
        self.norm = nn.LayerNorm(n_hidden)
        self.dropout = nn.Dropout(p=0.0)

        models = []
        for  n in range(n_layers - 2):
            models.append(nn.Sequential(
                OrderedDict(
                    [
                        ("Linear", nn.Linear(n_hidden[n], n_hidden[n+1])),
                        ("LayerNorm", nn.LayerNorm(n_hidden)),
                        ("Dropout", nn.Dropout(p=0.1)),
                    ]))))
        self.dnn = nn.ModuleList(models)
        self.l_last = nn.Linear(n_hidden, n_eigenglaciers)

        self.V_hat = torch.nn.Parameter(V_hat, requires_grad=False)
        self.F_mean = torch.nn.Parameter(F_mean, requires_grad=False)

        self.register_buffer("area", area)

        self.train_ae = AbsoluteError()
        self.test_ae = AbsoluteError()

    def forward(self, x, add_mean=False):
        # Pass the input tensor through each of our operations

        a = self.l_first(x)
        a = self.norm(a)
        a = self.dropout(a)
        z = torch.relu(a)

        for dnn in self.dnn:
            a = dnn(z)
            z = torch.relu(a) + z

        z_last = self.l_last(z)

        if add_mean:
            F_pred = z_last @ self.V_hat.T + self.F_mean
        else:
            F_pred = z_last @ self.V_hat.T

        return F_pred

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("NNEmulator")
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--n_hidden", default=128)
        parser.add_argument("--learning_rate", type=float, default=0.01)

        return parent_parser

    def criterion_ae(self, F_pred, F_obs, omegas, area):
        instance_misfit = torch.sum(torch.abs((F_pred - F_obs)) ** 2 * area, axis=1)
        return torch.sum(instance_misfit * omegas.squeeze())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), self.hparams.learning_rate, weight_decay=0.0
        )
        # This is an approximation to Doug's version:
        scheduler = {
            "scheduler": ExponentialLR(optimizer, 0.9975, verbose=True),
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, f, o, _ = batch
        f_pred = self.forward(x)
        loss = absolute_error(f_pred, f, o, self.area)

        return loss

    def validation_step(self, batch, batch_idx):
        x, f, o, o_0 = batch
        f_pred = self.forward(x)

        self.log("train_loss", self.train_ae(f_pred, f, o, self.area))
        self.log("test_loss", self.test_ae(f_pred, f, o_0, self.area))

        return {"x": x, "f": f, "f_pred": f_pred, "o": o, "o_0": o_0}

    def validation_epoch_end(self, outputs):

        self.log(
            "train_loss",
            self.train_ae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test_loss",
            self.test_ae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )


class NNEmulator(pl.LightningModule):
    def __init__(
        self,
        n_parameters,
        n_eigenglaciers,
        V_hat,
        F_mean,
        area,
        hparams,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        n_hidden_1 = self.hparams.n_hidden_1
        n_hidden_2 = self.hparams.n_hidden_2
        n_hidden_3 = self.hparams.n_hidden_3
        n_hidden_4 = self.hparams.n_hidden_4

        # Inputs to hidden layer linear transformation
        self.l_1 = nn.Linear(n_parameters, n_hidden_1)
        self.norm_1 = nn.LayerNorm(n_hidden_1)
        self.dropout_1 = nn.Dropout(p=0.0)
        self.l_2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.norm_2 = nn.LayerNorm(n_hidden_2)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.l_3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.norm_3 = nn.LayerNorm(n_hidden_3)
        self.dropout_3 = nn.Dropout(p=0.5)
        self.l_4 = nn.Linear(n_hidden_3, n_hidden_4)
        self.norm_4 = nn.LayerNorm(n_hidden_3)
        self.dropout_4 = nn.Dropout(p=0.5)
        self.l_5 = nn.Linear(n_hidden_4, n_eigenglaciers)

        self.V_hat = torch.nn.Parameter(V_hat, requires_grad=False)
        self.F_mean = torch.nn.Parameter(F_mean, requires_grad=False)

        self.register_buffer("area", area)

        self.train_ae = AbsoluteError()
        self.test_ae = AbsoluteError()

    def forward(self, x, add_mean=False):
        # Pass the input tensor through each of our operations

        a_1 = self.l_1(x)
        a_1 = self.norm_1(a_1)
        a_1 = self.dropout_1(a_1)
        z_1 = torch.relu(a_1)

        a_2 = self.l_2(z_1)
        a_2 = self.norm_2(a_2)
        a_2 = self.dropout_2(a_2)
        z_2 = torch.relu(a_2) + z_1

        a_3 = self.l_3(z_2)
        a_3 = self.norm_3(a_3)
        a_3 = self.dropout_3(a_3)
        z_3 = torch.relu(a_3) + z_2

        a_4 = self.l_4(z_3)
        a_4 = self.norm_3(a_4)
        a_4 = self.dropout_3(a_4)
        z_4 = torch.relu(a_4) + z_3

        z_5 = self.l_5(z_4)
        if add_mean:
            F_pred = z_5 @ self.V_hat.T + self.F_mean
        else:
            F_pred = z_5 @ self.V_hat.T

        return F_pred

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("NNEmulator")
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--n_hidden_1", type=int, default=128)
        parser.add_argument("--n_hidden_2", type=int, default=128)
        parser.add_argument("--n_hidden_3", type=int, default=128)
        parser.add_argument("--n_hidden_4", type=int, default=128)
        parser.add_argument("--learning_rate", type=float, default=0.01)

        return parent_parser

    def criterion_ae(self, F_pred, F_obs, omegas, area):
        instance_misfit = torch.sum(torch.abs((F_pred - F_obs)) ** 2 * area, axis=1)
        return torch.sum(instance_misfit * omegas.squeeze())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), self.hparams.learning_rate, weight_decay=0.0
        )
        # This is an approximation to Doug's version:
        scheduler = {
            "scheduler": ExponentialLR(optimizer, 0.9975, verbose=True),
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, f, o, _ = batch
        f_pred = self.forward(x)
        loss = absolute_error(f_pred, f, o, self.area)

        return loss

    def validation_step(self, batch, batch_idx):
        x, f, o, o_0 = batch
        f_pred = self.forward(x)

        self.log("train_loss", self.train_ae(f_pred, f, o, self.area))
        self.log("test_loss", self.test_ae(f_pred, f, o_0, self.area))

        return {"x": x, "f": f, "f_pred": f_pred, "o": o, "o_0": o_0}

    def validation_epoch_end(self, outputs):

        self.log(
            "train_loss",
            self.train_ae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test_loss",
            self.test_ae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )


class PISMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir="path/to/dir",
        samples_file="path/to/file",
        target_file=None,
        target_var="velsurf_mag",
        target_corr_threshold=25.0,
        target_corr_var="thickness",
        target_error_var="velsurf_mag_error",
        training_var="velsurf_mag",
        thinning_factor=1,
        normalize_x=True,
        log_y=True,
        threshold=100e3,
        epsilon=0,
        return_numpy=False,
    ):
        self.data_dir = data_dir
        self.samples_file = samples_file
        self.target_file = target_file
        self.target_var = target_var
        self.target_corr_threshold = target_corr_threshold
        self.target_corr_var = target_corr_var
        self.target_error_var = target_error_var
        self.thinning_factor = thinning_factor
        self.threshold = threshold
        self.training_var = training_var
        self.epsilon = epsilon
        self.log_y = log_y
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
        ds = xr.open_dataset(self.target_file)
        data = ds.variables[self.target_var].squeeze()
        mask = data.isnull()
        data = np.nan_to_num(
            data.values[::thinning_factor, ::thinning_factor],
            nan=epsilon,
        )
        self.target_has_error = False
        if self.target_error_var in ds.variables:
            data_error = ds.variables[self.target_error_var].squeeze()
            data_error = np.nan_to_num(
                data_error.values[::thinning_factor, ::thinning_factor],
                nan=epsilon,
            )
            self.target_has_error = True

        self.target_has_corr = False
        if self.target_corr_var in ds.variables:
            data_corr = ds.variables[self.target_corr_var].squeeze()
            data_corr = np.nan_to_num(
                data_corr.values[::thinning_factor, ::thinning_factor],
                nan=epsilon,
            )
            mask = mask.where(data_corr >= self.target_corr_threshold, True)
            self.target_has_corr = True
        mask = mask[::thinning_factor, ::thinning_factor].values
        grid_resolution = np.abs(np.diff(ds.variables["x"][0:2]))[0]
        self.grid_resolution = grid_resolution
        ds.close()

        idx = (mask == False).nonzero()

        data = data[idx]
        Y_target_2d = data
        Y_target = np.array(data.flatten(), dtype=np.float32)
        if not return_numpy:
            Y_target = torch.from_numpy(Y_target)
        self.Y_target = Y_target
        self.Y_target_2d = Y_target_2d
        if self.target_has_error:
            data_error = data_error[idx]
            Y_target_error_2d = data_error
            Y_target_error = np.array(data_error.flatten(), dtype=np.float32)
            if not return_numpy:
                Y_target_error = torch.from_numpy(Y_target_error)

            self.Y_target_error = Y_target_error
            self.Y_target_error_2d = Y_target_error_2d
        if self.target_has_corr:
            data_corr = data_corr[idx]
            Y_target_corr_2d = data_corr
            Y_target_corr = np.array(data_corr.flatten(), dtype=np.float32)
            if not return_numpy:
                Y_target_corr = torch.from_numpy(Y_target_corr)

            self.Y_target_corr = Y_target_corr
            self.Y_target_corr_2d = Y_target_corr_2d
        self.mask_2d = mask
        self.sparse_idx_2d = idx
        self.sparse_idx_1d = np.ravel_multi_index(idx, mask.shape)

    def load_data(self):
        epsilon = self.epsilon
        return_numpy = self.return_numpy
        thinning_factor = self.thinning_factor

        identifier_name = "id"
        training_var = self.training_var
        training_files = glob(join(self.data_dir, "*.nc"))
        # glob can return duplicates, which must be removed
        training_files = list(OrderedDict.fromkeys(training_files))
        ids = [int(re.search("id_(.+?)_", f).group(1)) for f in training_files]
        samples = pd.read_csv(
            self.samples_file, delimiter=",", skipinitialspace=True
        ).squeeze("columns").sort_values(by=identifier_name)
        samples.index = samples[identifier_name]
        samples.index.name = None

        ids_df = pd.DataFrame(data=ids, columns=["id"])
        ids_df.index = ids_df[identifier_name]
        ids_df.index.name = None

        # It is possible that not all ensemble simulations succeeded and returned a value
        # so we much search for missing response values
        missing_ids = list(set(samples["id"]).difference(ids_df["id"]))
        if missing_ids:
            print(f"The following simulations are missing:\n   {missing_ids}")
            print("  ... adjusting priors")
            # and remove the missing samples and responses
            samples_missing_removed = samples[~samples["id"].isin(missing_ids)]
            samples = samples_missing_removed

        samples = samples.drop(samples.columns[0], axis=1)
        m_samples, n_parameters = samples.shape
        self.X_keys = samples.keys()

        ds0 = xr.open_dataset(training_files[0])
        _, ny, nx = (
            ds0.variables["velsurf_mag"]
            .values[:, ::thinning_factor, ::thinning_factor]
            .shape
        )

        ds0.close()
        self.nx = nx
        self.ny = ny

        response = np.zeros((m_samples, len(self.sparse_idx_1d)))

        print("  Loading data sets...")
        training_files.sort(key=lambda x: int(re.search("id_(.+?)_", x).group(1)))
        for idx, m_file in tqdm(enumerate(training_files)):
            ds = xr.open_dataset(m_file)
            data = np.squeeze(
                np.nan_to_num(
                    ds.variables[training_var].values[
                        :, ::thinning_factor, ::thinning_factor
                    ],
                    nan=epsilon,
                )
            )
            response[idx, :] = data[self.sparse_idx_2d].flatten()
            ds.close()

        print(self.sparse_idx_2d)
        p = response.max(axis=1) < self.threshold
        if self.log_y:
            response = np.log10(response)
            response[np.isneginf(response)] = 0

        X = np.array(samples[p], dtype=np.float32)
        Y = np.array(response[p], dtype=np.float32)
        if not return_numpy:
            X = torch.from_numpy(X)
            Y = torch.from_numpy(Y)
        Y[Y < 0] = 0

        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        self.X_mean = X_mean
        self.X_std = X_std

        if self.normalize_x:
            X = (X - X_mean) / X_std

        self.X = X
        self.Y = Y

        n_parameters = X.shape[1]
        self.n_parameters = n_parameters
        n_samples, n_grid_points = Y.shape
        self.n_samples = n_samples
        self.n_grid_points = n_grid_points

        normed_area = np.ones(n_grid_points)
        if not return_numpy:
            normed_area = torch.tensor(normed_area)
        normed_area /= normed_area.sum()
        self.normed_area = normed_area

    def return_original(self):
        if self.normalize_x:
            return self.X * self.X_std + self.X_mean
        else:
            return self.X


class PISMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        X,
        F,
        omegas,
        omegas_0,
        batch_size: int = 128,
        train_size: float = 0.9,
        num_workers: int = 0,
    ):
        super().__init__()
        self.X = X
        self.F = F
        self.omegas = omegas
        self.omegas_0 = omegas_0
        self.batch_size = batch_size
        self.train_size = train_size
        self.num_workers = num_workers

    def setup(self, stage: str = None):

        all_data = TensorDataset(self.X, self.F_bar, self.omegas, self.omegas_0)
        self.all_data = all_data

        training_data, val_data = train_test_split(
            all_data, train_size=self.train_size, random_state=0
        )
        self.training_data = training_data
        self.test_data = training_data

        self.val_data = val_data
        train_all_loader = DataLoader(
            dataset=all_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.train_all_loader = train_all_loader
        val_all_loader = DataLoader(
            dataset=all_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.val_all_loader = val_all_loader
        train_loader = DataLoader(
            dataset=training_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.train_loader = train_loader
        self.test_loader = train_loader
        val_loader = DataLoader(
            dataset=val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        self.val_loader = val_loader

    def prepare_data(self, **kwargs):
        V_hat, F_bar, F_mean = self.get_eigenglaciers(**kwargs)
        n_eigenglaciers = V_hat.shape[1]
        self.V_hat = V_hat
        self.F_bar = F_bar
        self.F_mean = F_mean
        self.n_eigenglaciers = n_eigenglaciers

    def get_eigenglaciers(self, **kwargs):
        print("Generating eigenglaciers")
        defaultKwargs = {
            "cutoff": 1.0,
            "q": 100,
            "svd_lowrank": True,
            "eigenvalues": False,
        }
        kwargs = {**defaultKwargs, **kwargs}
        F = self.F
        omegas = self.omegas
        n_grid_points = F.shape[1]
        F_mean = (F * omegas).sum(axis=0)
        F_bar = F - F_mean  # Eq. 28
        if kwargs["svd_lowrank"]:
            Z = torch.diag(torch.sqrt(omegas.squeeze() * n_grid_points))
            U, S, V = torch.svd_lowrank(Z @ F_bar, q=kwargs["q"])
            lamda = S**2 / (n_grid_points)
        else:
            S = F_bar.T @ torch.diag(omegas.squeeze()) @ F_bar  # Eq. 27

            lamda, V = torch.eig(S, eigenvectors=True)  # Eq. 26
            lamda = lamda[:, 0].squeeze()

        cutoff_index = torch.sum(
            torch.cumsum(lamda / lamda.sum(), 0) < kwargs["cutoff"]
        )
        print(f"...using the first {cutoff_index} eigen values")
        lamda_truncated = lamda.detach()[:cutoff_index]
        V = V.detach()[:, :cutoff_index]
        V_hat = V @ torch.diag(torch.sqrt(lamda_truncated))

        if kwargs["eigenvalues"]:
            return V_hat, F_bar, F_mean, lamda
        else:
            return V_hat, F_bar, F_mean

    def train_dataloader(self):
        return self.train_loader

    def validation_dataloader(self):
        return self.val_loader
