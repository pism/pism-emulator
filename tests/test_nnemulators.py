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


import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal

from pismemulator.nnemulator import PISMDataset


def test_dataset():

    """"""

    dataset = PISMDataset(
        data_dir="training_data",
        samples_file="../data/samples/velocity_calibration_samples_100.csv",
        target_file="test_data/greenland_vel_mosaic250_v1_g9000m.nc",
        thinning_factor=1,
    )

    X = dataset.X.detach().numpy()
    Y = dataset.Y.detach().numpy()

    with np.load("training_samples.npz") as data:
        X_true = data["arr_0"]
    with np.load("training_responses.npz") as data:
        Y_true = data["arr_0"]
    with np.load("training_areas.npz") as data:
        normed_area_true = data["arr_0"]
    n_grid_points = dataset.n_grid_points
    n_parameters = dataset.n_parameters
    n_samples = dataset.n_samples
    normed_area = dataset.normed_area

    assert_equal(n_grid_points, 53504)
    assert_equal(n_parameters, 8)
    assert_equal(n_samples, 499)
    assert_array_almost_equal(X, X_true, decimal=4)
    assert_array_almost_equal(Y, Y_true, decimal=4)
    assert_array_almost_equal(normed_area, normed_area_true, decimal=4)