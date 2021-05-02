#!/bin/env python3

from argparse import ArgumentParser

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
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
import xarray as xr
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split


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

        Y_target = np.array(data.flatten(), dtype=np.float32)
        Y_target = torch.from_numpy(Y_target)
        self.Y_target = Y_target
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

        data = TensorDataset(self.X, self.F_bar, self.omegas, self.omegas_0)
        training_data, test_data = train_test_split(data, test_size=self.test_size)
        self.training_data = training_data
        self.test_data = test_data

        train_loader = DataLoader(
            dataset=training_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        self.train_loader = train_loader
        test_loader = DataLoader(
            dataset=test_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        self.test_loader = test_loader
        self.validation_loader = test_loader

    def prepare_data(self):
        V_hat, F_bar, F_mean = self.get_eigenglaciers()
        n_eigenglaciers = V_hat.shape[1]
        self.V_hat = V_hat
        self.F_bar = F_bar
        self.F_mean = F_mean
        self.n_eigenglaciers = n_eigenglaciers

    def get_eigenglaciers(self, cutoff=0.999):
        F = self.F
        n_grid_points = F.shape[1]
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

    def test_dataloader(self):
        return self.test_loader

    def validation_dataloader(self):
        return self.validation_loader


def criterion_ae(F_pred, F_obs, omegas, area):
    instance_misfit = torch.sum(torch.abs((F_pred - F_obs)) ** 2 * area, axis=1)
    return torch.sum(instance_misfit * omegas.squeeze())


def train_surrogate(e, X_train, F_train, omegas, area, batch_size=128, epochs=3000, eta_0=0.01, k=1000.0):

    omegas_0 = torch.ones_like(omegas) / len(omegas)
    training_data = TensorDataset(X_train, F_train, omegas)

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(e.parameters(), lr=eta_0, weight_decay=0.0)

    # Loop over the data
    for epoch in range(epochs):
        # Loop over each subset of data
        for param_group in optimizer.param_groups:
            param_group["lr"] = eta_0 * (10 ** (-epoch / k))

        for x, f, o in train_loader:
            e.train()
            # Zero out the optimizer's gradient buffer
            optimizer.zero_grad()

            f_pred = e(x)

            # Compute the loss
            loss = criterion_ae(f_pred, f, o, area)

            # Use backpropagation to compute the derivative of the loss with respect to the parameters
            loss.backward()

            # Use the derivative information to update the parameters
            optimizer.step()

        e.eval()
        F_train_pred = e(X_train)
        # Make a prediction based on the model
        loss_train = criterion_ae(F_train_pred, F_train, omegas, area)
        # Make a prediction based on the model
        loss_test = criterion_ae(F_train_pred, F_train, omegas_0, area)

        # Print the epoch, the training loss, and the test set accuracy.
        if epoch % 10 == 0:
            print(epoch, loss_train.item(), loss_test.item())


class Emulator(nn.Module):
    def __init__(self, n_parameters, n_eigenglaciers, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, V_hat, F_mean):
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


if __name__ == "__main__":
    __spec__ = None

    parser = ArgumentParser()
    parser.add_argument("--emulator_dir", default="emulator_ensemble")
    parser.add_argument("--num_models", type=int, default=1)
    parser.add_argument("--thinning_factor", type=int, default=1)

    args = parser.parse_args()
    hparams = vars(args)

    batch_size = 128
    n_hidden_1 = 128
    n_hidden_2 = 128
    n_hidden_3 = 128
    n_hidden_4 = 128
    emulator_dir = args.emulator_dir
    num_models = args.num_models
    thinning_factor = args.thinning_factor

    device = "cpu"

    dataset = PISMDataset(
        data_dir="../data/speeds_v2/",
        samples_file="../data/samples/velocity_calibration_samples_100.csv",
        target_file="../data/validation/greenland_vel_mosaic250_v1_g1800m.nc",
        thinning_factor=thinning_factor,
    )

    X = dataset.X
    F = dataset.Y
    n_grid_points = dataset.n_grid_points
    n_parameters = dataset.n_parameters
    n_samples = dataset.n_samples
    normed_area = dataset.normed_area

    torch.manual_seed(0)
    np.random.seed(0)

    if not os.path.isdir(emulator_dir):
        os.makedirs(emulator_dir)

    for model_index in range(num_models):
        print(f"Training model {model_index} of {num_models}")
        omegas = torch.tensor(dirichlet.rvs(np.ones(n_samples)), dtype=torch.float).T
        omegas_0 = torch.ones_like(omegas) / len(omegas)

        data_loader = PISMDataModule(
            X,
            F,
            omegas,
            omegas_0,
        )
        data_loader.prepare_data()
        data_loader.setup(stage="fit")
        n_eigenglaciers = data_loader.n_eigenglaciers
        V_hat = data_loader.V_hat
        F_mean = data_loader.F_mean
        F_train = data_loader.F_bar
        e = Emulator(n_parameters, n_eigenglaciers, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, V_hat, F_mean)
        e.to(device)

        train_surrogate(e, X, F_train, omegas, normed_area, epochs=1000)
        torch.save(e.state_dict(), f"{emulator_dir}/emulator_lr_{model_index}.h5")

    alpha = 0.01
    from scipy.special import gamma

    nu = 1.0

    models = []
    for i in range(num_models):
        state_dict = torch.load(f"{emulator_dir}/emulator_pl_lr_{i}.h5")
        e = GlacierEmulator(
            state_dict["l_1.weight"].shape[1],
            state_dict["V_hat"].shape[1],
            normed_area,
            state_dict["V_hat"],
            state_dict["F_mean"],
            hparams,
        )
        e.load_state_dict(state_dict)
        e.to(device)
        e.eval()
        models.append(e)

    sigma2 = 10 ** 2

    rho = 1.0 / (1e4 ** 2)
    point_area = (dataset.grid_resolution * thinning_factor) ** 2
    K = point_area * rho

    Tau = K * 1.0 / sigma2 * K

    sigma_hat = np.sqrt(sigma2 / K ** 2)

    from scipy.stats import beta

    alpha_b = 3.0
    beta_b = 3.0

    X_min = X.cpu().numpy().min(axis=0) - 1e-3
    X_max = X.cpu().numpy().max(axis=0) + 1e-3

    X_prior = beta.rvs(alpha_b, beta_b, size=(10000, X.shape[1])) * (X_max - X_min) + X_min

    # This is required for
    # X_bar = (X - X_min) / (X_max - X_min)
    # to work

    X_min = torch.tensor(X_min, dtype=torch.float32, device=device)
    X_max = torch.tensor(X_max, dtype=torch.float32, device=device)

    torch.manual_seed(0)
    np.random.seed(0)

    # Needs
    # alpha_b, beta_b: float
    # alpha: float
    # nu: float
    # gamma
    # sigma_hat
    U_target = dataset.Y_target

    X_posteriors = []
    for j, model in enumerate(models):
        X_0 = torch.tensor(X_prior.mean(axis=0), requires_grad=True, dtype=torch.float, device=device)
        mala = MALASampler(model)
        X_map = mala.find_MAP(X_0, U_target)
        # To reproduce the paper, n_iters should be 10^5
        X_posterior = mala.MALA(X_map, U_target, n_iters=10000, model_index=j, save_interval=1000, print_interval=100)
        X_posteriors.append(X_posterior)

    import pylab as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    # X_posterior = X_posterior*X_s.cpu().numpy() + X_m.cpu().numpy()
    X_prior = X_prior * dataset.X_std.cpu().numpy() + dataset.X_mean.cpu().numpy()

    X_hat = X_prior

    fig, axs = plt.subplots(nrows=8, ncols=8, figsize=(12, 12))
    X_list = []

    for model_index in range(num_models):
        X_list.append(np.load(open("./posterior_samples/X_posterior_model_{0:03d}.npy".format(model_index), "rb")))

        X_posterior = np.vstack(X_list)
        # X_posterior = X_posterior * dataset.X_std.cpu().numpy() + dataset.X_mean.cpu().numpy()

        C_0 = np.corrcoef((X_posterior - X_posterior.mean(axis=0)).T)
        Cn_0 = (np.sign(C_0) * C_0 ** 2 + 1) / 2.0

        color_post_0 = "#00B25F"
        color_post_1 = "#132DD6"
        color_prior = "#D81727"
        color_ensemble = "#BA9B00"
        color_other = "#20484E0"

    for i in range(8):
        for j in range(8):
            if i > j:

                axs[i, j].scatter(
                    X_posterior[:, j], X_posterior[:, i], c="k", s=0.5, alpha=0.05, label="Posterior", rasterized=True
                )
                min_val = min(X_hat[:, i].min(), X_posterior[:, i].min())
                max_val = max(X_hat[:, i].max(), X_posterior[:, i].max())
                bins_y = np.linspace(min_val, max_val, 30)

                min_val = min(X_hat[:, j].min(), X_posterior[:, j].min())
                max_val = max(X_hat[:, j].max(), X_posterior[:, j].max())
                bins_x = np.linspace(min_val, max_val, 30)

                # v = st.gaussian_kde(X_posterior[:,[j,i]].T)
                # bx = 0.5*(bins_x[1:] + bins_x[:-1])
                # by = 0.5*(bins_y[1:] + bins_y[:-1])
                # Bx,By = np.meshgrid(bx,by)

                # axs[i,j].contour(10**Bx,10**By,v(np.vstack((Bx.ravel(),By.ravel()))).reshape(Bx.shape),7,alpha=0.7,colors='black')

                axs[i, j].set_xlim(X_hat[:, j].min(), X_hat[:, j].max())
                axs[i, j].set_ylim(X_hat[:, i].min(), X_hat[:, i].max())

                # axs[i,j].set_xscale('log')
                # axs[i,j].set_yscale('log')

            elif i < j:
                patch_upper = Polygon(
                    np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]), facecolor=plt.cm.seismic(Cn_0[i, j])
                )
                # patch_lower = Polygon(np.array([[0.,0.],[1.,0.],[1.,1.]]),facecolor=plt.cm.seismic(Cn_1[i,j]))
                axs[i, j].add_patch(patch_upper)
                # axs[i,j].add_patch(patch_lower)
                if C_0[i, j] > -0.5:
                    color = "black"
                else:
                    color = "white"
                axs[i, j].text(
                    0.5,
                    0.5,
                    "{0:.2f}".format(C_0[i, j]),
                    fontsize=12,
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axs[i, j].transAxes,
                    color=color,
                )
                # if C_1[i,j]>-0.5:
                #    color = 'black'
                # else:
                #    color = 'white'

                # axs[i,j].text(0.75,0.25,'{0:.2f}'.format(C_1[i,j]),fontsize=12,horizontalalignment='center',verticalalignment='center',transform=axs[i,j].transAxes,color=color)

            elif i == j:
                min_val = min(X_hat[:, i].min(), X_posterior[:, i].min())
                max_val = max(X_hat[:, i].max(), X_posterior[:, i].max())
                bins = np.linspace(min_val, max_val, 30)
                # X_hat_hist,b = np.histogram(X_hat[:,i],bins,density=True)
                X_prior_hist, b = np.histogram(X_prior[:, i], bins, density=True)
                X_posterior_hist = np.histogram(X_posterior[:, i], bins, density=True)[0]
                b = 0.5 * (b[1:] + b[:-1])
                lw = 3.0
                axs[i, j].plot(
                    b, X_prior_hist, color=color_prior, linewidth=0.5 * lw, label="Prior", linestyle="dashed"
                )

                axs[i, j].plot(
                    b, X_posterior_hist, color="black", linewidth=lw, linestyle="solid", label="Posterior", alpha=0.7
                )

                # for X_ind in X_stack:
                #    X_hist,_ = np.histogram(X_ind[:,i],bins,density=False)
                #    X_hist=X_hist/len(X_posterior)
                #    X_hist=X_hist/(bins[1]-bins[0])
                #    axs[i,j].plot(10**b,X_hist,'b-',alpha=0.2,lw=0.5)

                if i == 1:
                    axs[i, j].legend(fontsize=8)
                axs[i, j].set_xlim(min_val, max_val)
                # axs[i,j].set_xscale('log')

            else:
                axs[i, j].remove()

    keys = dataset.X_keys

    for i, ax in enumerate(axs[:, 0]):
        ax.set_ylabel(keys[i])

    for j, ax in enumerate(axs[-1, :]):
        ax.set_xlabel(keys[j])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        plt.setp(ax.xaxis.get_minorticklabels(), rotation=45)
        if j > 0:
            ax.tick_params(axis="y", which="both", length=0)
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())

    for ax in axs[:-1, 1:].ravel():
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="both", which="both", length=0)

    # fig.savefig('speed_emulator_posterior.pdf')
