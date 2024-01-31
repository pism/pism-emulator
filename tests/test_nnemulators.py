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


import random

import lightning as pl
import numpy as np
import torch
from numpy.testing import assert_array_almost_equal
from scipy.stats import dirichlet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.regression import MeanSquaredError

from pism_emulator.nnemulator import DNNEmulator, NNEmulator


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


def nn_setup(Emulator):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    g = torch.Generator()
    g.manual_seed(0)

    n_parameters = 5
    n_eigenglaciers = 10
    n_grid_points = 10000
    n_samples = 1000
    V_hat = torch.rand(n_grid_points, n_eigenglaciers, dtype=torch.float32)
    F_mean = torch.rand(n_grid_points, dtype=torch.float32)
    area = torch.ones_like(F_mean, dtype=torch.float32) / n_grid_points
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

    e = Emulator(
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
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=hparams["batch_size"],
        worker_init_fn=seed_worker,
        generator=g,
    )

    max_epochs = 20

    trainer_e = pl.Trainer(
        deterministic=True,
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        accelerator="cpu",
    )
    trainer_e.fit(e, train_loader, val_loader)

    e.eval()
    Y_e = e(X, add_mean=True)
    return Y_e


def test_emulator_equivalence():
    """
    Compare NNEmulator and DNNEmulator
    """

    Y_e = nn_setup(NNEmulator)
    Y_de = nn_setup(DNNEmulator)

    mean_squared_error = MeanSquaredError()
    mse = mean_squared_error(Y_e, Y_de)

    assert mse <= 1e-1
