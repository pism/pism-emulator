#!/bin/env python3

from glob import glob
import numpy as np
import pandas as pd
import os
from os.path import join
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.stats import dirichlet
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import xarray as xr
import re
from tqdm import tqdm


class PISMDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir="path/to/dir", samples_file="path/to/file", thin=1, epsilon=1e-10):
        self.data_dir = data_dir
        self.samples_file = samples_file

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

        ds0 = xr.open_dataset(training_files[0])
        _, my, mx = ds0.variables["velsurf_mag"].values[:, ::thin, ::thin].shape
        ds0.close()

        response = np.zeros((m_samples, my * mx))

        print("  Loading data sets...")
        for idx, m_file in tqdm(enumerate(training_files)):
            ds = xr.open_dataset(m_file)
            data = np.nan_to_num(ds.variables["velsurf_mag"].values[:, ::thin, ::thin].flatten(), epsilon)
            response[idx, :] = data
            ds.close()

        self.samples = samples
        self.response = response


class PISMDataModule(pl.LightningDataModule):
    def __init__(self, X, F, omegas, batch_size: int = 128):
        super().__init__()
        self.X = X
        self.F = F
        self.omegas = omegas
        self.batch_size = batch_size

    def setup(self, stage: str = None):

        if stage == "fit" or stage is None:
            training_data = TensorDataset(self.X, self.F_bar, self.omegas)
            train_loader = DataLoader(dataset=training_data, batch_size=self.batch_size, shuffle=True)
            self.train_loader = train_loader

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
        F_mean = (F * omegas).sum(axis=0)
        F_bar = F - F_mean  # Eq. 28
        Z = torch.diag(torch.sqrt(omegas.squeeze() * n_grid_points))
        U, S, V = torch.svd_lowrank(Z @ F_bar, q=100)
        lamda = S ** 2 / (n_samples)

        cutoff_index = torch.sum(torch.cumsum(lamda / lamda.sum(), 0) < cutoff)
        lamda_truncated = lamda.detach()[:cutoff_index]
        V = V.detach()[:, :cutoff_index]
        V_hat = V @ torch.diag(torch.sqrt(lamda_truncated))

        return V_hat, F_bar, F_mean

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


def get_eigenglaciers(omegas, F, cutoff=0.999):
    F_mean = (F * omegas).sum(axis=0)
    F_bar = F - F_mean  # Eq. 28
    Z = torch.diag(torch.sqrt(omegas.squeeze() * n_grid_points))
    U, S, V = torch.svd_lowrank(Z @ F_bar, q=100)
    lamda = S ** 2 / (n_samples)

    cutoff_index = torch.sum(torch.cumsum(lamda / lamda.sum(), 0) < cutoff)
    lamda_truncated = lamda.detach()[:cutoff_index]
    V = V.detach()[:, :cutoff_index]
    V_hat = V @ torch.diag(torch.sqrt(lamda_truncated))  # A slight departure from the paper: Vhat is the
    # eigenvectors scaled by the eigenvalue size.  This
    # has the effect of allowing the outputs of the neural
    # network to be O(1).  Otherwise, it doesn't make
    # any difference.
    return V_hat, F_bar, F_mean


class GlacierEmulator(pl.LightningModule):
    def __init__(
        self, n_parameters, n_eigenglaciers, area, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, V_hat, F_mean
    ):
        super().__init__()
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1, weight_decay=0.0)
        scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer),
            "reduce_on_plateau": True,
            "monitor": "loss",
        }
        return [optimizer], [scheduler]

    def criterion_ae(self, F_pred, F_obs, omegas, area):
        instance_misfit = torch.sum(torch.abs((F_pred - F_obs)) ** 2 * area, axis=1)
        return torch.sum(instance_misfit * omegas.squeeze())

    def training_step(self, batch, batch_idx):
        x, f, o = batch
        f_pred = self.forward(x)
        loss = self.criterion_ae(f_pred, f, o, self.area)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss


# response_file = "log_speeds_prune_1.csv.gz"
# samples_file = "../data/samples/velocity_calibration_samples_100.csv"
# samples, response = prepare_data(samples_file, response_file)

dataset = PISMDataset(
    data_dir="../data/speeds_v2/", samples_file="../data/samples/velocity_calibration_samples_100.csv", thin=5
)
response = dataset.response
samples = dataset.samples.values

response = np.log10(response)
response[np.isneginf(response)] = 0

X = np.array(samples, dtype=np.float32)
F = np.array(response, dtype=np.float32)

X = torch.from_numpy(X)
F = torch.from_numpy(F)
F[F < 0] = 0

X_m = X.mean(axis=0)
X_s = X.std(axis=0)

X_train = (X - X_m) / X_s


n_parameters = X.shape[1]
n_samples, n_grid_points = F.shape

normed_area = torch.tensor(np.ones(n_grid_points))
normed_area /= normed_area.sum()

torch.manual_seed(0)
np.random.seed(0)

n_hidden_1 = 128
n_hidden_2 = 128
n_hidden_3 = 128
n_hidden_4 = 128

emulator_dir = "emulator_ensemble"

if not os.path.isdir(emulator_dir):
    os.makedirs(emulator_dir)

n_models = 1
max_epochs = 1000
batch_size = 128


for model_index in range(n_models):
    omegas = torch.tensor(dirichlet.rvs(np.ones(n_samples)), dtype=torch.float).T
    omegas_0 = torch.ones_like(omegas) / len(omegas)

    train_loader = PISMDataModule(X_train, F, omegas)
    train_loader.prepare_data()
    train_loader.setup(stage="fit")
    n_eigenglaciers = train_loader.n_eigenglaciers
    V_hat = train_loader.V_hat
    F_mean = train_loader.F_mean
    F_train = train_loader.F_bar
    e = GlacierEmulator(
        n_parameters, n_eigenglaciers, normed_area, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, V_hat, F_mean
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stop_callback = EarlyStopping(
        monitor="loss", min_delta=0.01, patience=10, verbose=False, mode="min", strict=True
    )
    logger = TensorBoardLogger(f"tb_logs_{model_index}", name="PISM Speed Emulator")
    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[lr_monitor, early_stop_callback], logger=logger)
    trainer.fit(e, train_loader)

    F_train_pred = e(X_train)
    # Make a prediction based on the model
    loss_train = e.criterion_ae(F_train_pred, F_train, omegas, normed_area)
    # Make a prediction based on the model
    loss_test = e.criterion_ae(F_train_pred, F_train, omegas_0, normed_area)

    torch.save(e.state_dict(), f"{emulator_dir}/emulator_pl_lr_{0:03d}.h5".format(model_index))
