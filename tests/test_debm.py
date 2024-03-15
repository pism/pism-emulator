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

from pism_emulator.models.debm import DEBMModel, TorchDEBMModel


def test_year_fraction():
    """
    Test the year_fraction code.
    """

    time = np.array([-2000.12, 1981.21, 2024.01])

    debm = DEBMModel()
    year_fraction = debm.year_fraction(time)
    assert_array_almost_equal(np.array([0.88, 0.21, 0.01]), year_fraction, decimal=4)

    debm = TorchDEBMModel()
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
    assert_array_almost_equal(np.array([0.79788456, 2.0, 0.08331547]), cgi, decimal=4)

    debm = TorchDEBMModel()
    cgi = debm.CalovGreveIntegrand(
        torch.from_numpy(sigma), torch.from_numpy(temperature)
    )
    assert_array_almost_equal(np.array([0.79788456, 2.0, 0.08331547]), cgi, decimal=4)


def test_hour_angle():
    """
    Test the calculation of the hour angle
    """

    phi = np.array([0.0, np.pi / 4.0, np.pi / 2.0])
    latitude = np.array([-np.pi / 2.0, 0.0, np.pi / 4.0])
    declination = np.array([np.pi / 8.0, 0.0, 0.0])

    debm = DEBMModel()
    hour_angle = debm.hour_angle(phi, latitude, declination)
    assert_array_almost_equal(np.array([0.0, 0.78539816, 0.0]), hour_angle, decimal=4)

    debm = TorchDEBMModel()
    hour_angle = debm.hour_angle(
        torch.from_numpy(phi), torch.from_numpy(latitude), torch.from_numpy(declination)
    )
    assert_array_almost_equal(np.array([0.0, 0.78539816, 0.0]), hour_angle, decimal=4)


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

    debm = TorchDEBMModel()
    solar_longitude = debm.solar_longitude(
        year_fraction, eccentricity, perhelion_longitude
    )
    assert_array_almost_equal(
        np.array([-2.4174, -0.1785, 3.6222]), solar_longitude, decimal=4
    )


def test_solar_declination_present_day():
    """
    Test solar declination present day
    """
    year_fraction = np.array([0.0, 1.0 / 12.0, 1.0])

    debm = DEBMModel()
    solar_declination = debm.solar_declination_present_day(year_fraction)
    assert_array_almost_equal(
        np.array([-0.402449, -0.30673297, -0.402449]), solar_declination, decimal=4
    )

    debm = TorchDEBMModel()
    solar_declination = debm.solar_declination_present_day(year_fraction)
    assert_array_almost_equal(
        np.array([-0.402449, -0.30673297, -0.402449]), solar_declination, decimal=4
    )


def test_solar_declination_paleo():
    """
    Test solar declination paleo
    """

    obliquity = np.array([np.pi / 4])
    solar_longitude = np.array([np.pi * 3 / 4])

    debm = DEBMModel()
    solar_declination = debm.solar_declination_paleo(obliquity, solar_longitude)
    assert_array_almost_equal(np.array([0.55536037]), solar_declination, decimal=4)

    debm = TorchDEBMModel()
    solar_declination = debm.solar_declination_paleo(obliquity, solar_longitude)
    assert_array_almost_equal(np.array([0.55536037]), solar_declination, decimal=4)


def test_distance_factor_present_day():
    """
    Test distance factor present day
    """
    year_fraction = np.array([0.0, 1.0 / 12.0, 1.0])

    debm = DEBMModel()
    d = debm.distance_factor_present_day(year_fraction)
    assert_array_almost_equal(np.array([1.03505, 1.03081244, 1.03505]), d, decimal=4)

    debm = TorchDEBMModel()
    d = debm.distance_factor_present_day(year_fraction)
    assert_array_almost_equal(np.array([1.03505, 1.03081244, 1.03505]), d, decimal=4)


def test_distance_factor_paleo():
    """
    Test distance factor paleo
    """

    eccentricity = np.array([0.0167, 0.03])
    obliquity = np.deg2rad(np.array([23.14, 22.10]))
    perihelion_longitude = np.deg2rad(np.array([102.94719, -44.3]))

    debm = DEBMModel()
    d = debm.distance_factor_paleo(eccentricity, perihelion_longitude, obliquity)
    assert_array_almost_equal(np.array([1.00592089, 1.02418709]), d, decimal=4)

    debm = TorchDEBMModel()
    d = debm.distance_factor_paleo(eccentricity, perihelion_longitude, obliquity)
    assert_array_almost_equal(np.array([1.00592089, 1.02418709]), d, decimal=4)


def test_insolation():
    """
    Test insolation
    """

    solar_constant = np.array([1361.0])
    distance_factor = np.array([1.1])
    hour_angle = np.array([0.8])
    latitude = np.array([np.pi / 4])
    declination = np.array([np.pi / 8])

    debm = DEBMModel()
    insolation = debm.insolation(
        solar_constant, distance_factor, hour_angle, latitude, declination
    )
    assert_array_almost_equal(np.array([1282.10500694]), insolation, decimal=4)

    debm = TorchDEBMModel()
    insolation = debm.insolation(
        solar_constant, distance_factor, hour_angle, latitude, declination
    )
    assert_array_almost_equal(np.array([1282.10500694]), insolation, decimal=4)


def test_orbital_parameters():
    """
    Test orbital parameters
    """
    time = np.array([2022.25])

    debm = DEBMModel()
    orbital_parameters = debm.orbital_parameters(time)
    assert_array_almost_equal(
        np.array([0.083785]), orbital_parameters["declination"], decimal=4
    )
    assert_array_almost_equal(
        np.array([1.000671]), orbital_parameters["distance_factor"], decimal=4
    )

    debm = DEBMModel(paleo_enabled=True)
    orbital_parameters = debm.orbital_parameters(time)
    assert_array_almost_equal(
        np.array([0.07843061]), orbital_parameters["declination"], decimal=4
    )
    assert_array_almost_equal(
        np.array([0.99889585]), orbital_parameters["distance_factor"], decimal=4
    )

    debm = TorchDEBMModel()
    orbital_parameters = debm.orbital_parameters(time)
    assert_array_almost_equal(
        np.array([0.083785]), orbital_parameters["declination"], decimal=4
    )
    assert_array_almost_equal(
        np.array([1.000671]), orbital_parameters["distance_factor"], decimal=4
    )

    debm = TorchDEBMModel(paleo_enabled=True)
    orbital_parameters = debm.orbital_parameters(time)
    assert_array_almost_equal(
        np.array([0.07843061]), orbital_parameters["declination"], decimal=4
    )
    assert_array_almost_equal(
        np.array([0.99889585]), orbital_parameters["distance_factor"], decimal=4
    )


def test_albedo():
    """
    Test albedo
    """

    debm = DEBMModel()
    melt_rate = np.array([1.0]) / debm.seconds_per_year()
    albedo = debm.albedo(melt_rate)
    assert_array_almost_equal(np.array([0.79704371]), albedo, decimal=4)

    debm = TorchDEBMModel()
    melt_rate = np.array([1.0]) / debm.seconds_per_year()
    albedo = debm.albedo(torch.from_numpy(melt_rate))
    assert_array_almost_equal(np.array([0.79704371]), albedo, decimal=4)


def test_atmospshere_transmissivity():
    """
    Test atmosphere transmissivity
    """

    elevation = np.array([0.0, 1000.0, 2000.0])
    debm = DEBMModel()
    transmissivity = debm.atmosphere_transmissivity(elevation)

    assert_array_almost_equal(np.array([0.65, 0.682, 0.714]), transmissivity, decimal=4)


def test_melt():
    """
    Test melt function
    """

    year_fraction = 0
    dt = 1 / 12
    temp = np.array([323.0])
    temp_sd = np.array([12.0])
    elevation = np.array([1000.0])
    latitude = np.array([np.pi / 4 * 3])
    albedo = np.array([0.47])

    debm = DEBMModel()
    melt_info = debm.melt(temp, temp_sd, albedo, elevation, latitude, year_fraction, dt)

    assert_almost_equal(3.41058271637613e-08, melt_info["insolation_melt"], decimal=4)
    assert_almost_equal(
        1.3995243305283176e-07, melt_info["temperature_melt"], decimal=4
    )
    assert_almost_equal(-9.003261572844215e-09, melt_info["offset_melt"], decimal=4)
    assert_almost_equal(1.6505499864374886e-07, melt_info["total_melt"], decimal=4)

    debm = TorchDEBMModel()
    melt_info = debm.melt(
        torch.from_numpy(temp),
        torch.from_numpy(temp_sd),
        torch.from_numpy(albedo),
        torch.from_numpy(elevation),
        torch.from_numpy(latitude),
        year_fraction,
        dt,
    )

    assert_almost_equal(3.41058271637613e-08, melt_info["insolation_melt"], decimal=4)
    assert_almost_equal(
        1.3995243305283176e-07, melt_info["temperature_melt"], decimal=4
    )
    assert_almost_equal(-9.003261572844215e-09, melt_info["offset_melt"], decimal=4)
    assert_almost_equal(1.6505499864374886e-07, melt_info["total_melt"], decimal=4)


def test_snow_accumulation():
    """
    The snow accumulation function
    """

    T = np.array([-10, -5, 0, 1, 4, 8]) + 273.15
    P = np.array([10, 0.2, 1.0, 0.2, 0.1, 0.4])

    debm = DEBMModel()
    accumulation_rate = debm.snow_accumulation(T, P)
    assert_array_almost_equal(
        np.array([10.0, 0.2, 1.0, 0.1, 0.0, 0.0]), accumulation_rate
    )

    debm = TorchDEBMModel()
    accumulation_rate = debm.snow_accumulation(torch.from_numpy(T), torch.from_numpy(P))
    assert_array_almost_equal(
        np.array([10.0, 0.2, 1.0, 0.1, 0.0, 0.0]), accumulation_rate
    )


def test_step():
    """
    Test step
    """

    max_melt = np.array([2.0, 0.0, 12.0])
    old_snow_depth = np.array([1.0, 4.0, 0.1])
    accumulation = np.array([0.1, 0.8, 0.0])
    ice_thickness = np.array([100.0, 1000.0, 0.0])

    debm = DEBMModel()
    changes = debm.step(ice_thickness, max_melt, old_snow_depth, accumulation)
    print(changes)

    assert_almost_equal(np.array([-1.0, 0.8, -0.1]), changes["snow_depth"], decimal=4)
    assert_almost_equal(np.array([2.0, 0.0, 0.0]), changes["melt"], decimal=4)
    assert_almost_equal(np.array([1.34, 0.0, 0.0]), changes["runoff"], decimal=4)
    assert_almost_equal(np.array([-1.24, 0.8, 0.0]), changes["smb"], decimal=4)

    debm = TorchDEBMModel()
    changes = debm.step(
        torch.from_numpy(ice_thickness),
        torch.from_numpy(max_melt),
        torch.from_numpy(old_snow_depth),
        torch.from_numpy(accumulation),
    )

    assert_almost_equal(np.array([-1.0, 0.8, -0.1]), changes["snow_depth"], decimal=4)
    assert_almost_equal(np.array([2.0, 0.0, 0.0]), changes["melt"], decimal=4)
    assert_almost_equal(np.array([1.34, 0.0, 0.0]), changes["runoff"], decimal=4)
    assert_almost_equal(np.array([-1.24, 0.8, 0.0]), changes["smb"], decimal=4)


def test_debm():
    """
    The snow accumulation function
    """

    T = (
        np.array(
            [
                [[1.0]],
                [[-10.0]],
                [[-8.0]],
                [[-4]],
                [[-2.0]],
                [[0]],
                [[1.0]],
                [[2.0]],
                [[3.0]],
                [[1.0]],
                [[-3.0]],
                [[-8.0]],
            ]
        )
        + 273.15
    )

    sd = np.array(
        [
            [[1.0]],
            [[2.0]],
            [[2.0]],
            [[1]],
            [[4.0]],
            [[4]],
            [[3.0]],
            [[2.0]],
            [[3.0]],
            [[1.0]],
            [[3.0]],
            [[1.0]],
        ]
    )

    P = np.array(
        [
            [[1.0]],
            [[10.0]],
            [[4.0]],
            [[4]],
            [[2.0]],
            [[1]],
            [[1.0]],
            [[2.0]],
            [[3.0]],
            [[1.0]],
            [[3.0]],
            [[8.0]],
        ]
    )

    elevation = np.zeros_like(T) + 1000
    ice_thickness = np.zeros_like(T) + 1000
    latitude = np.zeros_like(T) + 70

    debm = DEBMModel()
    result = debm(T, sd, P, elevation, ice_thickness, latitude)
    print(result)
    assert_array_almost_equal(
        np.array([10.0, 0.2, 1.0, 0.1, 0.0, 0.0]), accumulation_rate
    )
