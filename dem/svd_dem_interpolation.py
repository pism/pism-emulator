from argparse import ArgumentParser

import xarray as xr
import torch
from torch import Tensor, tensor
import numpy as np
import pylab as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ExponentialLR

from pismemulator.metrics import L2MeanSquaredError
from pismemulator.svdinterpolation import LinearRegression, DEMDataset, DEMDataModule
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib

# matplotlib.use("agg")


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
    parser.add_argument("--add_mean", default=False, action="store_true")
    parser.add_argument("--checkpoint", default=False, action="store_true")
    parser.add_argument("--training_files", nargs="*", default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("-q", type=int, default=30)
    parser.add_argument(
        "--target_file",
        default="aerodem_1978_1987_wgs84_g1800m.nc",
    )
    parser.add_argument("--train_size", type=float, default=0.9)
    parser.add_argument("--outfile", type=str, default="dem_reconstructed.nc")

    parser = LinearRegression.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    hparams = vars(args)

    add_mean = args.add_mean
    batch_size = args.batch_size
    checkpoint = args.checkpoint
    max_epochs = args.max_epochs
    num_workers = args.num_workers
    outfile = args.outfile
    q = args.q
    target_file = args.target_file
    train_size = args.train_size
    training_files = args.training_files

    dataset = DEMDataset(
        training_files=training_files,
        target_file=target_file,
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

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e2, patience=10, verbose=False
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[early_stop_callback],
        check_val_every_n_epoch=10,
    )

    V = data_loader.V
    S = data_loader.S
    lamda = S**2
    V_hat = V @ torch.diag(torch.sqrt(lamda))
    nx = dataset.obs_nx
    ny = dataset.obs_ny

    nrows = 2
    ncols = 3

    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, sharex="col", sharey="row", figsize=[12, 14]
    )
    for k, ax in enumerate(axs.ravel()):
        V_k = V_hat[:, k]
        data = np.zeros((ny, nx))
        data.put(dataset.sparse_idx_1d, V_k)
        eigen_glacier = np.ma.array(data=data, mask=dataset.mask_2d)
        ax.imshow(
            eigen_glacier,
            origin="lower",
            cmap="twilight_shifted",
        )
        ax.axis("off")
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    fig.tight_layout()
    fig.savefig(f"eigen_glaciers.pdf")

    for k in range(3):
        V_k = V_hat[:, k]
        data = np.zeros((ny, nx))
        data.put(dataset.sparse_idx_1d, V_k)
        eigen_glacier = np.ma.array(data=data, mask=dataset.mask_2d)

        fig = plt.figure(figsize=[5, 8])
        ax = fig.add_subplot(111)
        ax.imshow(
            eigen_glacier,
            origin="lower",
            cmap="twilight_shifted",
        )
        ax.axis("off")
        fig.tight_layout()

        fig.savefig(f"eigen_glacier_{k}.pdf")

    # Train the model
    trainer.fit(m, data_loader.train_dataloader(), data_loader.val_dataloader())
    # Get the linear weights
    w = m.linear.weight

    M = (V * S) @ w.reshape(-1)
    M = M.detach().numpy()
    R_filled = np.zeros((1, ny, nx))
    R_filled.put(dataset.sparse_idx_1d, M)
    if add_mean:
        Xm = np.zeros((1, ny, nx))
        Xm.put(dataset.sparse_idx_1d, dataset.X_mean)
        R_filled += Xm
    R_filled[R_filled < 0] = 0
    time = pd.date_range("1980-1-1", periods=1)

    print(f"Saving result to {outfile}")
    mds = xr.Dataset(
        {"surface_altitude": (("time", "y", "x"), R_filled)},
        coords={
            "x": ("x", dataset.x),
            "y": ("y", dataset.y),
            "time": ("time", time),
        },
    ).to_netcdf(outfile, unlimited_dims="time")

    # R_series = pd.Series(R_filled.ravel())

    # for k in range(0, 10):
    #     obs_dem = dataset.Obs
    #     obs_mask = dataset.obs_mask
    #     obs_hs = hillshade(obs_dem, 1800)
    #     obs_hs = np.ma.array(data=obs_hs, mask=obs_mask)
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.imshow(obs_hs, origin="lower", cmap="gray", vmin=0, vmax=255)
    #     plt.axis("off")
    #     fig.savefig(f"image_{k:03d}.png", dpi=300)
    #     k += 1
    #     plt.close(plt.gcf())
    #     del fig, ax

    # for frac in np.linspace(0.8, 1, 100):
    #     Sub = R_series.sample(frac=frac)
    #     R_f = R_filled
    #     R_s = np.zeros_like(R_f.ravel())
    #     mask = R_f.reshape(ny, nx) == 0
    #     R_s[Sub.index] = Sub.values
    #     dem = R_s.reshape(ny, nx)
    #     dem_mask = dem == 0
    #     dem_hs = hillshade(dem, 1800)
    #     dem_hs = np.ma.array(data=dem_hs, mask=(mask | dem_mask))
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.imshow(dem_hs, origin="lower", cmap="gray", vmin=0, vmax=255)
    #     plt.axis("off")
    #     fig.savefig(f"image_{k:03d}.png", dpi=300)
    #     k += 1
    #     plt.close(plt.gcf())
    #     del fig, ax
