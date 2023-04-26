# Copyright (C) 2021 Andy Aschwanden
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


import lightning as pl
import numpy as np
import torch
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from scipy.stats import dirichlet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from pismemulator.datasets import PISMDataset
from pismemulator.nnemulator import DNNEmulator, NNEmulator


def test_dataset():

    """"""

    dataset = PISMDataset(
        data_dir="tests/training_data",
        samples_file="data/samples/velocity_calibration_samples_50.csv",
        target_file="tests/test_data/test_vel_g9000m.nc",
        thinning_factor=1,
    )

    X = dataset.X.detach().numpy()
    Y = dataset.Y.detach().numpy()

    with np.load("tests/test_samples.npz") as data:
        X_true = data["arr_0"]
    with np.load("tests/test_responses.npz") as data:
        Y_true = data["arr_0"]
    with np.load("tests/test_areas.npz") as data:
        normed_area_true = data["arr_0"]
    n_grid_points = dataset.n_grid_points
    n_parameters = dataset.n_parameters
    n_samples = dataset.n_samples
    normed_area = dataset.normed_area

    assert_equal(n_grid_points, 26237)
    assert_equal(n_parameters, 8)
    assert_equal(n_samples, 482)
    assert_array_almost_equal(X, X_true, decimal=4)
    assert_array_almost_equal(Y, Y_true, decimal=4)
    assert_array_almost_equal(normed_area, normed_area_true, decimal=4)


def test_emulator_equivalence():
    """
    Compare NNEmulator and DNNEmulator
    """

    torch.manual_seed(0)

    n_parameters = 5
    n_eigenglaciers = 10
    n_grid_points = 10000
    n_samples = 1000
    V_hat = torch.rand(n_grid_points, n_eigenglaciers, dtype=torch.float32)
    F_mean = torch.rand(n_grid_points, dtype=torch.float32)
    area = torch.ones_like(F_mean, dtype=torch.float64) / n_grid_points
    hparams = {
        "max_epochs": 100,
        "batch_size": 128,
        "n_hidden": 128,
        "n_hidden_1": 128,
        "n_hidden_2": 128,
        "n_hidden_3": 128,
        "n_hidden_4": 128,
        "n_layers": 4,
        "learning_rate": 0.1,
    }

    e = NNEmulator(
        n_parameters,
        n_eigenglaciers,
        V_hat,
        F_mean,
        area,
        hparams,
    )

    de = DNNEmulator(
        n_parameters,
        n_eigenglaciers,
        V_hat,
        F_mean,
        area,
        hparams,
    )

    X = torch.rand(n_samples, n_parameters, dtype=torch.float32)
    Y = torch.rand(n_samples, n_grid_points, dtype=torch.float32)

    omegas = torch.Tensor(dirichlet.rvs(np.ones(n_samples))).T
    omegas = omegas.type_as(X)
    omegas_0 = torch.ones_like(omegas) / len(omegas)

    dataset = TensorDataset(X, Y, omegas, omegas_0)
    training_data, val_data = train_test_split(dataset, train_size=0.9, random_state=0)
    train_loader = DataLoader(
        dataset=training_data,
        batch_size=hparams["batch_size"],
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=hparams["batch_size"],
        shuffle=True,
        pin_memory=True,
    )

    max_epochs = 100
    trainer_e = pl.Trainer(
        deterministic=True, max_epochs=max_epochs, num_sanity_val_steps=0
    )
    trainer_de = pl.Trainer(
        deterministic=True, max_epochs=max_epochs, num_sanity_val_steps=0
    )
    trainer_e.fit(e, train_loader, val_loader)
    trainer_de.fit(de, train_loader, val_loader)

    e.eval()
    de.eval()
    Y_e = e(X, add_mean=True).detach().numpy()
    Y_de = de(X, add_mean=True).detach().numpy()

    assert_array_almost_equal(Y_e, Y_de, decimal=1)
