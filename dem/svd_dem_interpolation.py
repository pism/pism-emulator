from argparse import ArgumentParser

import xarray as xr
import torch
import numpy as np
import pylab as plt
from glob import glob
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torchmetrics import MeanSquaredError
import pandas as pd
import matplotlib

matplotlib.use("agg")


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
            self.x = ds.x.values
            self.y = ds.y.values

    def load_data(self):
        epsilon = self.epsilon
        return_numpy = self.return_numpy
        thinning_factor = self.thinning_factor

        print("Loading training data")
        all_data = []
        for idx, m_file in enumerate(self.training_files):
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

        self.train_loss = MeanSquaredError()
        self.val_loss = MeanSquaredError()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--learning_rate", type=float, default=0.1)

        return parent_parser

    def forward(self, x):
        out = self.linear(x)
        return out

    def training_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self.forward(x)
        loss = self.train_loss(y_hat, y)
        return {"loss": loss, "x": x, "y": y, "y_hat": y_hat}

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        loss = self.val_loss(y_hat, y)
        return {"loss": loss, "x": x, "y": y, "y_hat": y_hat}

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
            self.parameters(), self.hparams.learning_rate, weight_decay=0.0
        )
        # This is an approximation to Doug's version:
        scheduler = {
            "scheduler": ExponentialLR(optimizer, 0.9975, verbose=True),
        }

        return [optimizer], [scheduler]


def cart2pol(x, y):
    """
    cartesian to polar coordinates
    """
    theta = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)
    return (theta, rho)


def hillshade(dem, dx):
    """
    shaded relief using the ESRI algorithm
    """

    # lighting azimuth
    azimuth = params["azimuth"]
    azimuth = 360.0 - azimuth + 90  # convert to mathematic unit
    if (azimuth > 360) or (azimuth == 360):
        azimuth = azimuth - 360
    azimuth = azimuth * (np.pi / 180)  # convert to radians

    # lighting altitude
    altitude = params["altitude"]
    altitude = (90 - altitude) * (np.pi / 180)  # convert to zenith angle in radians

    # calc slope and aspect (radians)
    fx, fy = np.gradient(dem, dx)  # uses simple, unweighted gradient of immediate
    [asp, grad] = cart2pol(fy, fx)  # convert to carthesian coordinates

    zf = params["zf"]
    grad = np.arctan(zf * grad)  # steepest slope
    # convert asp
    asp[asp < np.pi] = asp[asp < np.pi] + (np.pi / 2)
    asp[asp < 0] = asp[asp < 0] + (2 * np.pi)

    ## hillshade calculation
    h = 255.0 * (
        (np.cos(altitude) * np.cos(grad))
        + (np.sin(altitude) * np.sin(grad) * np.cos(azimuth - asp))
    )
    h[h < 0] = 0  # set hillshade values to min of 0.

    return h


params = {
    "altitude": 45,
    "azimuth": 45,
    "fill_value": 0,
    "zf": 25,
}


if __name__ == "__main__":
    __spec__ = None

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", default=False, action="store_true")
    parser.add_argument("--training_files", nargs="*", default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("-q", type=int, default=30)
    parser.add_argument(
        "--target_file",
        default="aerodem_1978_1987_wgs84_g1800m.nc",
    )
    parser.add_argument("--train_size", type=float, default=0.9)
    parser.add_argument("--thinning_factor", type=int, default=1)
    parser.add_argument("--outfile", type=str, default="dem_reconstructed.nc")

    parser = LinearRegression.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    hparams = vars(args)

    batch_size = args.batch_size
    checkpoint = args.checkpoint
    max_epochs = args.max_epochs
    num_workers = args.num_workers
    outfile = args.outfile
    q = args.q
    target_file = args.target_file
    train_size = args.train_size
    training_files = args.training_files
    thinning_factor = args.thinning_factor

    dataset = DEMDataset(
        training_files=training_files,
        target_file=target_file,
        thinning_factor=thinning_factor,
    )

    data_loader = DEMDataModule(
        dataset.X,
        dataset.R,
        dataset.I,
        q=q,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    data_loader.setup()
    m = LinearRegression(q, 1, hparams)
    trainer = pl.Trainer.from_argparse_args(
        args,
    )

    V = data_loader.V
    S = data_loader.S
    nx = dataset.obs_nx
    ny = dataset.obs_ny
    lamda = S**2
    V_hat = V @ torch.diag(torch.sqrt(lamda))
    nrows = 2
    ncols = 3

    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, sharex="col", sharey="row", figsize=[12, 14]
    )
    for k, ax in enumerate(axs.ravel()):
        eigen_glacier = V_hat[:, k].reshape(ny, nx)
        mask = eigen_glacier == 0
        eigen_glacier = np.ma.array(data=eigen_glacier, mask=mask)
        ax.imshow(
            eigen_glacier,
            origin="lower",
            cmap="twilight_shifted",
        )
        ax.axis("off")
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    fig.savefig(f"eigen_glaciers.pdf")

    for k in range(3):
        eigen_glacier = V_hat[:, k].reshape(ny, nx)
        mask = eigen_glacier == 0
        eigen_glacier = np.ma.array(data=eigen_glacier, mask=mask)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(
            eigen_glacier,
            origin="lower",
            cmap="twilight_shifted",
        )
        ax.axis("off")
        fig.savefig(f"eigen_glacier_{k}.pdf")

    trainer.fit(m, data_loader.train_dataloader(), data_loader.val_dataloader())
    S_est = m.linear.weight

    R_filled = (torch.diag(S_est.ravel()) @ V.T).T.sum(axis=1).reshape(1, ny, nx)
    R_filled += dataset.X_mean.reshape(1, ny, nx)
    R_filled[R_filled < 0] = 0
    time = pd.date_range("1980-1-1", periods=1)

    mds = xr.Dataset(
        {"surface_altitude": (("time", "y", "x"), R_filled.detach().numpy())},
        coords={
            "x": ("x", dataset.x),
            "y": ("y", dataset.y),
            "time": ("time", time),
        },
    ).to_netcdf(outfile, unlimited_dims="time")

    R_series = pd.Series(R_filled.detach().numpy().ravel())
    k = 0
    for frac in np.linspace(0.8, 1, 100):
        Sub = R_series.sample(frac=frac)
        R_f = R_filled.detach().numpy()
        R_s = np.zeros_like(R_f.ravel())
        mask = R_f.reshape(ny, nx) == 0
        R_s[Sub.index] = Sub.values
        dem = R_s.reshape(ny, nx)
        dem_mask = dem == 0
        dem_hs = hillshade(dem, 1800)
        dem_hs = np.ma.array(data=dem_hs, mask=(mask | dem_mask))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(dem_hs, origin="lower", cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        fig.savefig(f"image_{k:03d}.png", dpi=300)
        k += 1
        plt.close(plt.gcf())
        del fig, ax
