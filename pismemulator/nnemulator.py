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
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset

from pismemulator.utils import plot_validation


class NNEmulator(pl.LightningModule):
    def __init__(self, n_parameters, n_eigenglaciers, area, V_hat, F_mean, hparams, *args, **kwargs):
        super().__init__()
        #print(hparams)
        #self.hparams = hparams
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
        self.area = area

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.hparams.learning_rate, weight_decay=0.0)
        scheduler = {
            "scheduler": ExponentialLR(optimizer, 0.9975, verbose=True),
        }
        return [optimizer], [scheduler]

    def criterion_ae(self, F_pred, F_obs, omegas, area):
        instance_misfit = torch.sum(torch.abs(F_pred - F_obs) ** 2 * area, axis=1)
        return torch.sum(instance_misfit * omegas.squeeze())

    def training_step(self, batch, batch_idx):
        x, f, o, o_0 = batch
        f_pred = self.forward(x)
        loss = self.criterion_ae(f_pred, f, o, self.area)

        return loss

    def validation_step(self, batch, batch_idx):
        x, f, o, o_0 = batch
        return {"x": x, "f": f, "omegas": o, "omegas_0": o_0}

    def validation_epoch_end(self, outputs):
        x = []
        f = []
        omegas_0 = []
        omegas = []
        for k, o in enumerate(outputs):
            x.append(o["x"])
            f.append(o["f"])
            omegas.append(o["omegas"])
            omegas_0.append(o["omegas_0"])
        x = torch.vstack(x)
        f = torch.vstack(f)
        omegas = torch.vstack(omegas)
        omegas_0 = torch.vstack(omegas_0)
        f_pred = self.forward(x)
        train_loss = self.criterion_ae(f_pred, f, omegas, self.area)
        test_loss = self.criterion_ae(f_pred, f, omegas_0, self.area)

        self.log("train_loss", train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)


class PISMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir="path/to/dir",
        samples_file="path/to/file",
        target_file=None,
        target_var="velsurf_mag",
        thinning_factor=1,
        normalize_x=True,
        log_y=True,
        epsilon=1e-10,
    ):
        self.data_dir = data_dir
        self.samples_file = samples_file
        self.target_file = target_file
        self.target_var = target_var
        self.thinning_factor = thinning_factor
        self.epsilon = epsilon
        self.log_y = log_y
        self.normalize_x = normalize_x
        self.load_data()
        if target_file is not None:
            self.load_target()

    def load_target(self):
        epsilon = self.epsilon
        thinning_factor = self.thinning_factor
        ds = xr.open_dataset(self.target_file)
        data = np.nan_to_num(ds.variables[self.target_var].values[::thinning_factor, ::thinning_factor], epsilon)
        grid_resolution = np.abs(np.diff(ds.variables["x"][0:2]))[0]
        ds.close()

        Y_target_2d = data
        Y_target = np.array(data.flatten(), dtype=np.float32)
        Y_target = torch.from_numpy(Y_target)
        self.Y_target = Y_target
        self.Y_target_2d = Y_target_2d
        self.grid_resolution = grid_resolution

    def load_data(self):
        epsilon = self.epsilon
        thinning_factor = self.thinning_factor
        identifier_name = "id"
        training_files = glob(join(self.data_dir, "*.nc"))
        ids = [int(re.search("id_(.+?)_", f).group(1)) for f in training_files]
        samples = pd.read_csv(self.samples_file, delimiter=",", squeeze=True, skipinitialspace=True).sort_values(
            by=identifier_name
        )
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
        _, ny, nx = ds0.variables["velsurf_mag"].values[:, ::thinning_factor, ::thinning_factor].shape
        ds0.close()
        self.nx = nx
        self.ny = ny

        response = np.zeros((m_samples, ny * nx))

        print("  Loading data sets...")
        training_files.sort(key=lambda x: int(re.search("id_(.+?)_", x).group(1)))
        for idx, m_file in tqdm(enumerate(training_files)):
            ds = xr.open_dataset(m_file)
            data = np.nan_to_num(
                ds.variables["velsurf_mag"].values[:, ::thinning_factor, ::thinning_factor].flatten(), epsilon
            )
            response[idx, :] = data
            ds.close()

        if self.log_y:
            response = np.log10(response)
            response[np.isneginf(response)] = 0

        X = np.array(samples, dtype=np.float32)
        Y = np.array(response, dtype=np.float32)

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

        normed_area = torch.tensor(np.ones(n_grid_points))
        normed_area /= normed_area.sum()
        self.normed_area = normed_area

    def return_original(self):
        if self.normalize_x:
            return self.X * self.X_std + self.X_mean
        else:
            return self.X


class PISMDataModule(pl.LightningDataModule):
    def __init__(self, X, F, omegas, omegas_0, batch_size: int = 128, test_size: float = 0.1, num_workers: int = 0):
        super().__init__()
        self.X = X
        self.F = F
        self.omegas = omegas
        self.omegas_0 = omegas_0
        self.batch_size = batch_size
        self.test_size = test_size
        self.num_workers = num_workers

    def setup(self, stage: str = None):

        all_data = TensorDataset(self.X, self.F_bar, self.omegas, self.omegas_0)
        training_data, validation_data = train_test_split(all_data, test_size=self.test_size)
        self.training_data = training_data
        self.test_data = training_data
        self.validation_data = validation_data
        self.all_data = all_data
        all_loader = DataLoader(
            dataset=all_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

        train_loader = DataLoader(
            dataset=training_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        self.all_loader = all_loader
        self.train_loader = train_loader
        self.test_loader = train_loader
        validation_loader = DataLoader(
            dataset=validation_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        self.validation_loader = validation_loader

    def prepare_data(self):
        V_hat, F_bar, F_mean = self.get_eigenglaciers()
        n_eigenglaciers = V_hat.shape[1]
        self.V_hat = V_hat
        self.F_bar = F_bar
        self.F_mean = F_mean
        self.n_eigenglaciers = n_eigenglaciers

    def get_eigenglaciers(self, cutoff=0.999):
        F = self.F
        omegas = self.omegas
        n_grid_points = F.shape[1]
        F_mean = (F * omegas).sum(axis=0)
        F_bar = F - F_mean  # Eq. 28
        Z = torch.diag(torch.sqrt(omegas.squeeze() * n_grid_points))
        U, S, V = torch.svd_lowrank(Z @ F_bar, q=100)
        lamda = S ** 2 / (n_grid_points)

        cutoff_index = torch.sum(torch.cumsum(lamda / lamda.sum(), 0) < cutoff)
        lamda_truncated = lamda.detach()[:cutoff_index]
        V = V.detach()[:, :cutoff_index]
        V_hat = V @ torch.diag(torch.sqrt(lamda_truncated))

        return V_hat, F_bar, F_mean

    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.test_loader

    def validation_dataloader(self):
        return self.validation_loader