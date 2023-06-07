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
import torch
from numpy.testing import assert_array_almost_equal

from pismemulator.models import TorchDEBMModel


def test_CalovGreveIntegrand():
    sigma = np.array([2.0, 0.0, 1.0])
    temperature = np.array([0.0, 2.0, -1.0])

    sigma = torch.from_numpy(sigma)
    temperature = torch.from_numpy(temperature)

    debm = TorchDEBMModel()
    cgi = debm.CalovGreveIntegrand(sigma, temperature)

    assert_array_almost_equal(
        np.array([0.7979, 2.0000, 0.0833]), cgi.numpy(), decimal=4
    )


def test_hour_angle():
    phi = np.array([0.0, np.pi / 4.0, np.pi / 2.0])
    latitude = np.array([-np.pi / 2.0, 0.0, np.pi / 4.0])
    declination = np.array([np.pi / 8.0, 0.0, 0.0])

    phi = torch.from_numpy(phi)
    latitude = torch.from_numpy(latitude)
    declination = torch.from_numpy(declination)

    debm = TorchDEBMModel()
    hour_angle = debm.hour_angle(phi, latitude, declination)
    assert_array_almost_equal(np.array([0.0000, 0.7854, 0.0000]), hour_angle, decimal=4)
