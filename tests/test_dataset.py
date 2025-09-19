# Copyright (C) 2021, 2023 Andy Aschwanden
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


import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal

from pism_emulator.datasets import PISMDataset, PISMConfigDataset


def test_dataset():
    """"""

    dataset = PISMDataset(
        training_files="tests/training_data/*.nc",
        samples_file="data/samples/velocity_calibration_samples_50.csv",
        target_file="tests/test_data/test_vel_g9000m.nc",
        thin=1,
    )

    X = dataset.samples.X.detach().numpy()
    Y = dataset.samples.Y.detach().numpy()

    with np.load("tests/test_samples.npz") as data:
        X_true = data["arr_0"]
    with np.load("tests/test_responses.npz") as data:
        Y_true = data["arr_0"]
    with np.load("tests/test_areas.npz") as data:
        normed_area_true = data["arr_0"]
    n_grid_points = dataset.samples.n_grid_points
    n_parameters = dataset.samples.n_parameters
    n_samples = dataset.samples.n_samples
    normed_area = dataset.samples.normed_area

    assert_equal(n_grid_points, 26237)
    assert_equal(n_parameters, 8)
    assert_equal(n_samples, 482)
    assert_array_almost_equal(X, X_true, decimal=4)
    assert_array_almost_equal(Y, Y_true, decimal=4)
    assert_array_almost_equal(normed_area, normed_area_true, decimal=4)


def test_config_dataset():
    """"""

    dataset = PISMConfigDataset(
        training_files="tests/training_data/*.nc",
        target_file="tests/test_data/test_vel_g9000m.nc",
        thin=1,
    )

    X = dataset.samples.X.detach().numpy()
    Y = dataset.samples.Y.detach().numpy()

    with np.load("tests/test_samples.npz") as data:
        X_true = data["arr_0"]
    with np.load("tests/test_responses.npz") as data:
        Y_true = data["arr_0"]
    with np.load("tests/test_areas.npz") as data:
        normed_area_true = data["arr_0"]
    n_grid_points = dataset.samples.n_grid_points
    n_parameters = dataset.samples.n_parameters
    n_samples = dataset.samples.n_samples
    normed_area = dataset.samples.normed_area

    assert_equal(n_grid_points, 26237)
    assert_equal(n_parameters, 8)
    assert_equal(n_samples, 482)
    assert_array_almost_equal(X, X_true, decimal=4)
    assert_array_almost_equal(Y, Y_true, decimal=4)
    assert_array_almost_equal(normed_area, normed_area_true, decimal=4)
