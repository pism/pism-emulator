# Copyright (C) 2023-24 Andy Aschwanden
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

from pism_emulator.models import ReferencePDDModel, TorchDEBMModel, TorchPDDModel


def make_fake_climate() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make fake climate to test surface models
    """

    temp = np.array(
        [
            [-3.12],
            [-2.41],
            [-0.62],
            [1.93],
            [4.41],
            [6.20],
            [6.91],
            [6.21],
            [4.40],
            [1.92],
            [-0.61],
            [-2.41],
        ],
    )
    precip = np.array(
        [
            [1.58],
            [1.47],
            [1.18],
            [0.79],
            [0.39],
            [0.11],
            [-0.01],
            [0.10],
            [0.39],
            [0.79],
            [1.18],
            [1.47],
        ],
        dtype=np.float64,
    )
    sd = np.array(
        [
            [0.0],
            [0.18],
            [0.70],
            [1.40],
            [2.11],
            [2.61],
            [2.81],
            [2.61],
            [2.10],
            [1.40],
            [0.72],
            [0.18],
        ],
    )
    return temp, precip, sd


def test_torch_model():
    """
    Test the TorchPDDModel by comparing it to the ReferencePDDModel
    """
    temp, precip, sd = make_fake_climate()

    pdd_ref = ReferencePDDModel(
        pdd_factor_snow=0.003,
        pdd_factor_ice=0.008,
        refreeze_snow=0.6,
        refreeze_ice=0.1,
        temp_snow=0.0,
        temp_rain=2.0,
        interpolate_rule="linear",
        interpolate_n=52,
    )
    result_ref = pdd_ref(temp, precip, sd)

    pdd_torch = TorchPDDModel(
        pdd_factor_snow=3.0,
        pdd_factor_ice=8.0,
        refreeze_snow=0.6,
        refreeze_ice=0.1,
        temp_snow=0.0,
        temp_rain=2.0,
        interpolate_rule="linear",
        interpolate_n=52,
    )
    result_torch = pdd_torch.forward(temp, precip, sd)

    for m_var in [
        "temp",
        "prec",
        "accumulation_rate",
        "inst_pdd",
        "snow_depth",
        "snow_melt_rate",
        "ice_melt_rate",
        "melt_rate",
        "smb",
    ]:
        print(f"Comparing Reference and Torch implementation for variable {m_var}")
        assert_array_almost_equal(result_ref[m_var], result_torch[m_var], decimal=3)


# def test_TorchPDDModel():
#     ds = make_fake_climate()
#     pdd = TorchPDDModel()


def test_CalovGreveIntegrand():
    """
    Test the CalovGreveIntegrand
    """
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
    """
    Test the calculation of the hour angle
    """

    phi = np.array([0.0, np.pi / 4.0, np.pi / 2.0])
    latitude = np.array([-np.pi / 2.0, 0.0, np.pi / 4.0])
    declination = np.array([np.pi / 8.0, 0.0, 0.0])

    phi = torch.from_numpy(phi)
    latitude = torch.from_numpy(latitude)
    declination = torch.from_numpy(declination)

    debm = TorchDEBMModel()
    hour_angle = debm.hour_angle(phi, latitude, declination)
    assert_array_almost_equal(np.array([0.0000, 0.7854, 0.0000]), hour_angle, decimal=4)


def test_solar_longitude():
    """
    Test solar longitude
    """
    year_fraction = np.array([0.0, 1.0 / 12.0, 1.0])
    eccentricity = np.array([0.9, 1.0, 1.0])
    perhelion_longitude = np.array([np.pi / 8.0, -np.pi, 0.0])

    year_fraction = torch.from_numpy(year_fraction)
    eccentricity = torch.from_numpy(eccentricity)
    perhelion_longitude = torch.from_numpy(perhelion_longitude)

    debm = TorchDEBMModel()
    solar_longitude = debm.solar_longitude(
        year_fraction, eccentricity, perhelion_longitude
    )
    assert_array_almost_equal(
        np.array([-2.4174, -0.1785, 3.6222]), solar_longitude, decimal=4
    )


def test_distance_factor_present_day():
    """
    Test distance factor present day
    """
    year_fraction = np.array([0.0, 1.0 / 12.0, 1.0])

    year_fraction = torch.from_numpy(year_fraction)

    debm = TorchDEBMModel()
    d = debm.distance_factor_present_day(year_fraction)
    assert_array_almost_equal(np.array([1.0351, 1.0308, 1.0351]), d, decimal=4)


def test_distance_factor_paleo():
    """
    Test distance factor paleo
    """

    eccentricity = 0.0167
    obliquity = 23.44
    perihelion_longitude = 102.94719
    debm = TorchDEBMModel()
    d = debm.distance_factor_paleo(eccentricity, perihelion_longitude, obliquity)
