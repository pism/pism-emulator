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
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from pism_emulator.models.debm import DEBMModel


def test_year_fraction():
    """
    Test the year_fraction code.
    """

    time = np.array([-2000.12, 1981.21, 2024.01])

    debm = DEBMModel()

    year_fraction = debm.year_fraction(time)

    assert_array_almost_equal(np.array([0.88, 0.21, 0.01]), year_fraction, decimal=4)


def test_CalovGreveIntegrand():
    """
    Test the CalovGreveIntegrand
    """

    sigma = np.array([2.0, 0.0, 1.0])
    temperature = np.array([0.0, 2.0, -1.0])

    debm = DEBMModel()

    cgi = debm.CalovGreveIntegrand(sigma, temperature)

    assert_array_almost_equal(np.array([0.7979, 2.0000, 0.0833]), cgi, decimal=4)


def test_hour_angle():
    """
    Test the calculation of the hour angle
    """

    phi = np.array([0.0, np.pi / 4.0, np.pi / 2.0])
    latitude = np.array([-np.pi / 2.0, 0.0, np.pi / 4.0])
    declination = np.array([np.pi / 8.0, 0.0, 0.0])

    debm = DEBMModel()
    hour_angle = debm.hour_angle(phi, latitude, declination)
    assert_array_almost_equal(np.array([0.0000, 0.7854, 0.0000]), hour_angle, decimal=4)


def test_solar_longitude():
    """
    Test solar longitude
    """
    year_fraction = np.array([0.0, 1.0 / 12.0, 1.0])
    eccentricity = np.array([0.9, 1.0, 1.0])
    perhelion_longitude = np.array([np.pi / 8.0, -np.pi, 0.0])

    debm = DEBMModel()
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

    debm = DEBMModel()
    d = debm.distance_factor_present_day(year_fraction)
    assert_array_almost_equal(np.array([1.0351, 1.0308, 1.0351]), d, decimal=4)


def test_distance_factor_paleo():
    """
    Test distance factor paleo
    """

    eccentricity = np.array([0.0167, 0.03])
    obliquity = np.deg2rad(np.array([23.14, 22.10]))
    perihelion_longitude = np.deg2rad(np.array([102.94719, -44.3]))
    debm = DEBMModel()
    d = debm.distance_factor_paleo(eccentricity, perihelion_longitude, obliquity)
