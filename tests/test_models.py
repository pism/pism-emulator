# Copyright (C) 2023 Andy Aschwanden
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

import hashlib

import numpy as np
import torch
import xarray as xr
from numpy.testing import assert_array_almost_equal

from pismemulator.models import PDDModel, TorchDEBMModel, TorchPDDModel

PARAMETERS = {
    "pdd_factor_snow": 0.003,
    "pdd_factor_ice": 0.008,
    "refreeze_snow": 0.0,
    "refreeze_ice": 0.0,
    "temp_snow": 0.0,
    "temp_rain": 2.0,
    "interpolate_rule": "linear",
    "interpolate_n": 52,
}


# Default variable attributes
# ---------------------------

ATTRIBUTES = {
    # coordinate variables
    "x": {
        "axis": "X",
        "long_name": "x-coordinate in Cartesian system",
        "standard_name": "projection_x_coordinate",
        "units": "m",
    },
    "y": {
        "axis": "Y",
        "long_name": "y-coordinate in Cartesian system",
        "standard_name": "projection_y_coordinate",
        "units": "m",
    },
    "time": {
        "axis": "T",
        "long_name": "time",
        "standard_name": "time",
        "bounds": "time_bounds",
        "units": "yr",
    },
    "time_bounds": {},
    # climatic variables
    "temp": {"long_name": "near-surface air temperature", "units": "degC"},
    "prec": {"long_name": "ice-equivalent precipitation rate", "units": "m yr-1"},
    "stdv": {
        "long_name": "standard deviation of near-surface air temperature",
        "units": "K",
    },
    # cumulative quantities
    "smb": {
        "standard_name": "land_ice_surface_specific_mass_balance",
        "long_name": "cumulative ice-equivalent surface mass balance",
        "units": "m yr-1",
    },
    "pdd": {
        "long_name": "cumulative number of positive degree days",
        "units": "degC day",
    },
    "accu": {
        "long_name": "cumulative ice-equivalent surface accumulation",
        "units": "m",
    },
    "snow_melt": {
        "long_name": "cumulative ice-equivalent surface melt of snow",
        "units": "m",
    },
    "ice_melt": {
        "long_name": "cumulative ice-equivalent surface melt of ice",
        "units": "m",
    },
    "melt": {"long_name": "cumulative ice-equivalent surface melt", "units": "m"},
    "runoff": {
        "long_name": "cumulative ice-equivalent surface meltwater runoff",
        "units": "m yr-1",
    },
    # instantaneous quantities
    "inst_pdd": {
        "long_name": "instantaneous positive degree days",
        "units": "degC day",
    },
    "accu_rate": {
        "long_name": "instantaneous ice-equivalent surface accumulation rate",
        "units": "m yr-1",
    },
    "snow_melt_rate": {
        "long_name": "instantaneous ice-equivalent surface melt rate of snow",
        "units": "m yr-1",
    },
    "ice_melt_rate": {
        "long_name": "instantaneous ice-equivalent surface melt rate of ice",
        "units": "m yr-1",
    },
    "melt_rate": {
        "long_name": "instantaneous ice-equivalent surface melt rate",
        "units": "m yr-1",
    },
    "runoff_rate": {
        "long_name": "instantaneous ice-equivalent surface runoff rate",
        "units": "m yr-1",
    },
    "inst_smb": {
        "long_name": "instantaneous ice-equivalent surface mass balance",
        "units": "m yr-1",
    },
    "snow_depth": {"long_name": "depth of snow cover", "units": "m"},
}


def make_fake_climate():
    """Create an artificial temperature and precipitation file.
    This function is used if pypdd.py is called as a script without an input
    file. The file produced contains an idealized, three-dimensional (t, x, y)
    distribution of near-surface air temperature, precipitation rate and
    standard deviation of near-surface air temperature to be read by
    `PDDModel`.
    """

    # FIXME code could be simplified a lot more but we need a better test not
    # relying on exact reproducibility of this toy climate data.

    # assign coordinate values
    lx = ly = 750000
    x = xr.DataArray(np.linspace(-lx, lx, 201, dtype="f4"), dims="x")
    y = xr.DataArray(np.linspace(-ly, ly, 201, dtype="f4"), dims="y")
    time = xr.DataArray((np.arange(12, dtype="f4") + 0.5) / 12, dims="time")
    tboundsvar = np.empty((12, 2), dtype="f4")
    tboundsvar[:, 0] = time[:] - 1.0 / 24
    tboundsvar[:, 1] = time[:] + 1.0 / 24

    # seasonality index from winter to summer
    season = xr.DataArray(-np.cos(np.arange(12) * 2 * np.pi / 12), dims="time")

    # order of operation is dictated by test md5sum and legacy f4 dtype
    temp = 5 * season - 10 * x / lx + 0 * y
    prec = y / ly * (season.astype("f4") + 0 * x + np.sign(y))
    stdv = (2 + y / ly - x / lx) * (1 + season)

    # this is also why transpose is needed here, and final type conversion
    temp = temp.transpose("time", "x", "y").astype("f4")
    prec = prec.transpose("time", "x", "y").astype("f4")
    stdv = stdv.transpose("time", "x", "y").astype("f4")

    # assign variable attributes
    temp.attrs.update(ATTRIBUTES["temp"])
    prec.attrs.update(ATTRIBUTES["prec"])
    stdv.attrs.update(ATTRIBUTES["stdv"])

    # make a dataset
    ds = xr.Dataset(
        data_vars={"temp": temp, "prec": prec, "stdv": stdv},
        coords={
            "time": time,
            "x": x,
            "y": y,
            "time_bounds": (["time", "nv"], tboundsvar[:]),
        },
    )

    return ds


def test_PDDModel():
    ds = make_fake_climate()
    pdd = PDDModel()
    smb = pdd(ds.temp, ds.prec, ds.stdv, return_xarray=True)
    # check md5 sums against v0.3.0
    hashes = {
        "temp": "8d207b63be2e35c29e8cd33286071f4f",
        "prec": "80c26f1919752773b181cf94d6ab2c0a",
        "stdv": "8d287b76520c577725360015278c0155",
        "inst_pdd": "4fab3205cce36f649e2a3a9687aafcef",
        "accu_rate": "ddacb551907ccf774e102f9d81d2a09d",
        "snow_melt_rate": "c7798b3f9fc96bef74dea4a96705b1e8",
        "ice_melt_rate": "167097328babae93642e04c4ddedca98",
        "melt_rate": "d6e069684f3e1e7985a312364161a7b1",
        "runoff_rate": "d6e069684f3e1e7985a312364161a7b1",
        "inst_smb": "e6d2fe834f7fd14ad986663273bcf06f",
        "snow_depth": "020f7898d6b11499b587f4a74c445c2a",
        "pdd": "23ad49e711adf4fd0eb9f5d10aee683a",
        "accu": "8e1510c29e5e5cc3103309185bb76342",
        "snow_melt": "8cbf52e571a96268fb91a733c602f4fa",
        "ice_melt": "60d0700627ccdf7a4cb9195c8b5fce50",
        "melt": "9c8789adfb6a62631b377299f5c85f8f",
        "runoff": "9c8789adfb6a62631b377299f5c85f8f",
        "smb": "e6b3c05c89d07f8058f31a7159641a3a",
    }
    for name, m_hash in hashes.items():
        var = smb[name].data.astype("f4")
        assert hashlib.md5(var).hexdigest() == m_hash


def test_TorchPDDModel():
    ds = make_fake_climate()
    pdd = TorchPDDModel()


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


def test_solar_longitude():
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


def test_distance_from_present_day():
    year_fraction = np.array([0.0, 1.0 / 12.0, 1.0])

    year_fraction = torch.from_numpy(year_fraction)

    debm = TorchDEBMModel()
    d = debm.distance_factor_present_day(year_fraction)
    assert_array_almost_equal(np.array([1.0351, 1.0308, 1.0351]), d, decimal=4)
