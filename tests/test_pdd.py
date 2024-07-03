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
import xarray as xr
from numpy.testing import assert_array_almost_equal

from pism_emulator.models.pdd import ReferencePDDModel, TorchPDDModel


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


def make_fake_climate_2d(filename=None):
    """Create an artificial temperature and precipitation file.

    This function is used if pypdd.py is called as a script without an input
    file. The file produced contains an idealized, three-dimensional (t, x, y)
    distribution of near-surface air temperature, precipitation rate and
    standard deviation of near-surface air temperature to be read by
    `PDDModel.nco`.

    filename: str, optional
        Name of output file.
    """

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

    # write dataset to file
    if filename is not None:
        ds.to_netcdf(filename)

    # return dataset
    return ds


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


def test_torch_model_2d():
    """
    Test the TorchPDDModel by comparing it to the ReferencePDDModel
    """
    ds = make_fake_climate_2d()

    temp = ds["temp"].to_numpy()
    precip = ds["prec"].to_numpy()
    sd = ds["stdv"].to_numpy()

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


def test_snow_accumulation():
    """
    The snow accumulation function
    """

    T = np.array([-10, -5, 0, 1, 4, 8])
    P = np.array([10, 0.2, 1.0, 0.2, 0.1, 0.4])
    pdd = ReferencePDDModel(
        pdd_factor_snow=0.003,
        pdd_factor_ice=0.008,
        refreeze_snow=0.6,
        refreeze_ice=0.1,
        temp_snow=0.0,
        temp_rain=2.0,
        interpolate_rule="linear",
        interpolate_n=52,
    )

    accumulation_rate = pdd.accumulation_rate(T, P)
    assert_array_almost_equal(
        np.array([10.0, 0.2, 1.0, 0.1, 0.0, 0.0]), accumulation_rate
    )
