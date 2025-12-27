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
"""
Test Torch PDD models.
"""

from typing import Literal

import torch
from numpy.testing import assert_array_almost_equal

from pism_emulator.models.pdd import (
    PDD,
    ReferencePDDModel,
    make_fake_climate,
    make_fake_climate_2d,
)

PDDCompareKey = Literal[
    "accumulation_rate",
    "inst_pdd",
    "snow_depth",
    "snow_melt_rate",
    "ice_melt_rate",
    "melt_rate",
    "smb",
]

PDD_COMPARE_VARS: tuple[PDDCompareKey, ...] = (
    "accumulation_rate",
    "inst_pdd",
    "snow_depth",
    "snow_melt_rate",
    "ice_melt_rate",
    "melt_rate",
    "smb",
)


def test_torch_model():
    """
    Test the Lightning PDD Model by comparing it to the ReferencePDDModel.
    """
    temp, precip, sd = make_fake_climate()

    pdd_factor_snow = 3.0
    pdd_factor_ice = 8.0
    refreeze_snow = 0.6
    refreeze_ice = 0.1
    temp_snow = 0.0
    temp_rain = 2.0

    pdd_ref = ReferencePDDModel(
        pdd_factor_snow=pdd_factor_snow / 1e3,
        pdd_factor_ice=pdd_factor_ice / 1e3,
        refreeze_snow=refreeze_snow,
        refreeze_ice=refreeze_ice,
        temp_snow=temp_snow,
        temp_rain=temp_rain,
        interpolate_rule="linear",
        interpolate_n=12,
    )
    result_ref = pdd_ref(temp, precip, sd)

    pdd_torch_pl = PDD(temp.reshape(1, -1), precip.reshape(1, -1), sd.reshape(1, -1))
    x = torch.tensor(
        [
            pdd_factor_snow,
            pdd_factor_ice,
            refreeze_snow,
            refreeze_ice,
            temp_snow,
            temp_rain,
        ]
    )
    result_pl = pdd_torch_pl.forward(x, as_dict=True)
    for m_var in PDD_COMPARE_VARS:
        print(f"Comparing Reference and Torch implementation for variable {m_var}")
        assert_array_almost_equal(
            result_ref[m_var].reshape(1, -1), result_pl[m_var], decimal=3
        )


def test_torch_model_2d():
    """
    Test the TorchPDDModel by comparing it to the ReferencePDDModel.
    """
    ds = make_fake_climate_2d()

    temp = ds["temp"].to_numpy()
    precip = ds["prec"].to_numpy()
    sd = ds["stdv"].to_numpy()

    ds_pl = make_fake_climate_2d(torch_order=True)

    temp_pl = ds_pl["temp"].to_numpy()
    precip_pl = ds_pl["prec"].to_numpy()
    sd_pl = ds_pl["stdv"].to_numpy()

    pdd_factor_snow = 3.0
    pdd_factor_ice = 8.0
    refreeze_snow = 0.6
    refreeze_ice = 0.1
    temp_snow = 0.0
    temp_rain = 2.0

    pdd_ref = ReferencePDDModel(
        pdd_factor_snow=pdd_factor_snow / 1e3,
        pdd_factor_ice=pdd_factor_ice / 1e3,
        refreeze_snow=refreeze_snow,
        refreeze_ice=refreeze_ice,
        temp_snow=temp_snow,
        temp_rain=temp_rain,
        interpolate_rule="linear",
        interpolate_n=12,
    )
    result_ref = pdd_ref(temp, precip, sd)

    pdd_torch_pl = PDD(temp_pl, precip_pl, sd_pl)
    x = torch.tensor(
        [
            pdd_factor_snow,
            pdd_factor_ice,
            refreeze_snow,
            refreeze_ice,
            temp_snow,
            temp_rain,
        ]
    )
    result_pl = pdd_torch_pl.forward(x, return_dict=True)
    for m_var in PDD_COMPARE_VARS:
        print(f"Comparing Reference and Torch implementation for variable {m_var}")
        assert_array_almost_equal(
            result_ref[m_var],
            result_pl[m_var].permute(-2, -1, 0, 1).squeeze(),
            decimal=3,
        )
