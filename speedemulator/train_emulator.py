#!/bin/env python3

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn

from pismemulator.utils import prepare_data

from torch.utils.data import TensorDataset


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


response_file = "log_speeds.csv.gz"
samples_file = "../data/samples/velocity_calibration_samples_100.csv"
samples, response = prepare_data(samples_file, response_file)

F = response.values
X = samples.values

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = torch.from_numpy(X)
F = torch.from_numpy(F)
F[F < 0] = 0

X = X.to(torch.float32)
F = F.to(torch.float32)

X = X.to(device)
F = F.to(device)

X_m = X.mean(axis=0)
X_s = X.std(axis=0)

X = (X - X_m) / X_s


n_parameters = X.shape[1]
n_samples, n_grid_points = F.shape

normed_area = torch.tensor(np.ones(n_grid_points), device=device)
normed_area /= normed_area.sum()

torch.manual_seed(0)
np.random.seed(0)

n_hidden_1 = 128
n_hidden_2 = 128
n_hidden_3 = 128
n_hidden_4 = 128

emulator_dir = "emulator_ensemble"

if not os.path.isdir(emulator_dir):
    os.makedir(emulator_dir)

n_models = 5
n_epochs = 3000
from scipy.stats import dirichlet

for model_index in range(n_models):
    omegas = torch.tensor(dirichlet.rvs(np.ones(n_samples)), dtype=torch.float, device=device).T

    V_hat, F_bar, F_mean = get_eigenglaciers(omegas, F)
    n_eigenglaciers = V_hat.shape[1]

    e = Emulator(n_parameters, n_eigenglaciers, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, V_hat, F_mean)

    e.to(device)

    train_surrogate(e, X, F_bar, omegas, normed_area, epochs=n_epochs)

    torch.save(e.state_dict(), f"{emulator_dir}/emulator_{0:03d}.h5".format(model_index))
