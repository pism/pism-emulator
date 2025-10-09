# Copyright (C) 2019-25 Andy Aschwanden
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
Utils.
"""

from __future__ import annotations

import sys
from math import sqrt
from os import mkdir
from os.path import isdir, join
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pylab as plt
import xarray as xr
from matplotlib.colors import LogNorm
from numpy.typing import NDArray
from pyDOE3 import lhs
from scipy.stats.distributions import gamma, randint, truncnorm, uniform
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error

np.random.seed(0)


param_keys_dict = {
    "GCM": "GCM (1)",
    "FICE": "$f_i$ (mm K$^{-1}$ day$^{-1}$)",
    "FSNOW": "$f_s$ (mm K$^{-1}$ day$^{-1}$)",
    "RFR": r"$\psi (1)$",
    "PRS": r"$\omega$ (% K$^{-1}$)",
    "OCM": "$m_{t}$ (1)",
    "OCS": "$m_{x}$ (1)",
    "TCT": r"$h_{\mathrm{min}}$ (1)",
    "VCM": r"$\sigma_{\mathrm{max}}$ (MPa)",
    "SIAE": r"$E_{\mathrm{SIA}}$ (1)",
    "SSAN": r"$n_{\mathrm{SSA}}$ (1)",
    "TEFO": r"$\delta$ (1)",
    "PPQ": "$q$ (1)",
    "PHIMIN": r"$\phi_{\mathrm{min}}$ ($^{\circ}$)",
    "PHIMAX": r"$\phi_{\mathrm{max}}$ ($^{\circ}$)",
    "ZMIN": r"$z_{\mathrm{min}}$ (m)",
    "ZMAX": r"$z_{\mathrm{max}}$ (m)",
    "a_glen": "A (Pa^{-n} s^{-1})",
    "sia_e": r"$E_{\mathrm{SIA}}$ (1)",
    "ssa_e": r"$E_{\mathrm{SSA}}$ (1)",
    "ssa_n": r"$n_{\mathrm{SSA}}$ (1)",
    "ppq": "$q$ (1)",
    "tefo": r"$\delta$ (1)",
    "till_effective_fraction_overburden": r"$\delta$ (1)",
    "pseudo_plastic_uthershold": r"u_{\mathrm{thr} (m yr^{-1})}",
    "phi_min": r"$\phi_{\mathrm{min}}$ ($^{\circ}$)",
    "z_min": r"$z_{\mathrm{min}}$ (m)",
    "z_max": r"$z_{\mathrm{max}}$ (m)",
    "pseudo_plastic_uthreshold": r"$u_{\mathrm{th}}$ (m yr$^{-1}$)",
    "SIAe": r"$E_{\mathrm{SIA}}$ (1)",
    "SSAe": r"$E_{\mathrm{SSA}}$ (1)",
    "topg_to_phi_base": r"$b_{\mathrm{base}}$ (m)",
    "topg_to_phi_range": r"$b_{\mathrm{range}}$ (m)",
    "calving.vonmises_calving.sigma_max": "$\sigma_{\mathrm{max}}$ (kPa)",
    "geometry.front_retreat.prescribed.file": "Retreat Method",
    "ocean.th.gamma_T": "$\gamma_{S}$ (10$^{-4}$ 1)",
    "surface.given.file": "Climate Forcing",
    "ocean.th.file": "Ocean Forcing",
    "frontal_melt.routing.parameter_a": r"$a$ (10$^{-4}$ m$^{-\alpha}$ day$^{\alpha-1}$ Celsius$^{-\beta}$)",
    "frontal_melt.routing.parameter_b": r"$b$ (day$^{\alpha-1)}$ Celsius$^{-\beta}$)",
    "frontal_melt.routing.power_alpha": r"$\alpha$ (1)",
    "frontal_melt.routing.power_beta": r"$\beta$ (1)",
    "stress_balance.sia.enhancement_factor": r"$E_{\mathrm{SIA}}$ (1)",
    "stress_balance.ssa.enhancement_factor": r"$E_{\mathrm{SSA}}$ (1)",
    "stress_balance.blatter.enhancement_factor": r"$E_{\mathrm{HO}}$ (1)",
    "stress_balance.sis.Glen_exponent": r"$n_{\mathrm{SIA}}$ (1)",
    "stress_balance.ssa.Glen_exponent": "r$n_{\mathrm{SSA}}$ (1)",
    "stress_balance.blatter.Glen_exponent": r"$n_{\mathrm{HO}}$ (1)",
    "basal_resistance.pseudo_plastic.q": r"$q$ (1)",
    "basal_yield_stress.mohr_coulomb.till_effective_fraction_overburden": r"$\delta$ (1)",
    "basal_yield_stress.mohr_coulomb.topg_to_phi.phi_min": r"$\phi_{\mathrm{min}} (^{\circ{}})$",
    "basal_yield_stress.mohr_coulomb.topg_to_phi.phi_max": r"$\phi_{\mathrm{max}} (^{\circ{}})$",
    "basal_yield_stress.mohr_coulomb.topg_to_phi.topg_min": "r$z_{\mathrm{min}}$ (m)",
    "basal_yield_stress.mohr_coulomb.topg_to_phi.topg_max": r"$z_{\mathrm{max}}$ (m)",
    "basal_resistance.pseudo_plastic.u_threshold": r"$u_{\mathrm{th}}$",
    "basal_yield_stress.mohr_coulomb.till_phi_default": r"$\phi$ ($^{\circ}$)",
}


def load_hirham_climate(
    file: str | Path = "DMI-HIRHAM5_1980_MM.nc",
    thinning_factor: int = 1,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """
    Load monthly HIRHAM5 climate fields and return thinned arrays.

    The dataset is stacked over spatial dims (``rlat``, ``rlon``) into a 1-D
    ``z`` dimension, NaNs are dropped along ``z`` (per time step), and simple
    unit conversions are applied. Outputs are optionally thinned along the last
    axis by ``thinning_factor``.

    Parameters
    ----------
    file : str or pathlib.Path, default="DMI-HIRHAM5_1980_MM.nc"
        Path to a HIRHAM5 monthly NetCDF file containing variables such as
        ``tas``, ``rainfall``, ``snfall``, ``gld``, ``snmel``, ``rogl``, ``rfrz``.
    thinning_factor : int, default=1
        Subsampling stride applied to the last (stacked spatial) axis of the
        returned arrays. Use ``1`` for no thinning.

    Returns
    -------
    temp : ndarray
        Air temperature in °C, shape ``(..., M)``.
    precip : ndarray
        Total precipitation (rain + snow) in m/yr, shape ``(..., M)``.
    snowfall_sum : ndarray
        Annual sum of snowfall in m/yr over the time axis, shape ``(M,)``.
    melt_sum : ndarray
        Annual sum of snowmelt in m/yr over the time axis, shape ``(M,)``.
    runoff_sum : ndarray
        Annual sum of runoff in m/yr over the time axis, shape ``(M,)``.
    refreeze_sum : ndarray
        Annual sum of refreeze in m/yr over the time axis, shape ``(M,)``.
    smb_sum : ndarray
        Annual sum of surface mass balance in m/yr over the time axis, shape ``(M,)``.

    Notes
    -----
    - Conversions:
      - Temperature: K → °C.
      - Fluxes accumulated monthly: scaled to m/yr using 365.242198781 days/yr
        and divided by 12 for monthly mean fluxes where applicable.
    - The last dimension ``M`` corresponds to valid (non-NaN) stacked spatial
      points after dropping missing values.
    """
    with xr.open_dataset(file) as Obs:
        stacked = Obs.stack(z=("rlat", "rlon"))
        ncl_stacked = Obs.stack(z=("ncl4", "ncl5"))

        temp = stacked.tas.dropna(dim="z").values - 273.15
        rainfall = stacked.rainfall.dropna(dim="z").values * 365.242198781 / 1000.0
        snowfall = stacked.snfall.dropna(dim="z").values * 365.242198781 / 1000.0
        smb = stacked.gld.dropna(dim="z").values * 365.242198781 / 1000.0 / 12.0
        refreeze = (
            ncl_stacked.rfrz.dropna(dim="z").values * 365.242198781 / 1000.0 / 12.0
        )
        melt = stacked.snmel.dropna(dim="z").values * 365.242198781 / 1000.0 / 12.0
        runoff = stacked.rogl.dropna(dim="z").values * 365.242198781 / 1000.0 / 12.0
        precip = rainfall + snowfall

    return (
        temp[..., ::thinning_factor],
        precip[..., ::thinning_factor],
        snowfall.sum(axis=0)[::thinning_factor],
        melt.sum(axis=0)[::thinning_factor],
        runoff.sum(axis=0)[::thinning_factor],
        refreeze.sum(axis=0)[::thinning_factor],
        smb.sum(axis=0)[::thinning_factor],
    )


def load_hirham_climate_w_std_dev(
    file: str | Path = "DMI-HIRHAM5_1980_2020_MMS.nc",
    thinning_factor: int = 1,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    dict[str, NDArray[np.floating]],
]:
    """
    Load multi-year HIRHAM5 climate fields grouped by calendar year with std-dev.

    The dataset is thinned on spatial dims, stacked into a 1-D spatial axis,
    grouped by ``time.year``, and concatenated (horizontally) across years.
    Simple unit conversions are applied. An ``obs`` dictionary returns annual
    aggregates of several components.

    Parameters
    ----------
    file : str or Path, default="DMI-HIRHAM5_1980_2020_MMS.nc"
        Path to a multi-year HIRHAM5 NetCDF file containing variables such as
        ``tas``, ``tas_std_dev``, ``rainfall``, ``snfall``, ``gld``, ``snmel``,
        ``rogl``, ``rfrz``, ``sn``.
    thinning_factor : int, default=1
        Subsampling stride applied to spatial dims (``rlat``, ``rlon``, ``ncl4``,
        ``ncl5``) before stacking.

    Returns
    -------
    temp : ndarray
        Air temperature in °C, concatenated by year along axis 1; shape ``(12, M)`` if
        all months are present per year.
    precip : ndarray
        Total precipitation (rain + snow) in m/yr, concatenated by year; shape ``(12, M)``.
    temp_std_dev : ndarray
        Standard deviation of air temperature (same layout as ``temp``); shape ``(12, M)``.
    obs : dict of {str: ndarray}
        Annual aggregates over time for each spatial point:
        - ``"snow_depth"``: snow depth anomaly relative to first entry (same units as input), shape ``(M,)``.
        - ``"accumulation"``: annual snowfall sum in m/yr, shape ``(M,)``.
        - ``"melt"``: annual snowmelt sum in m/yr, shape ``(M,)``.
        - ``"runoff"``: annual runoff sum in m/yr, shape ``(M,)``.
        - ``"refreeze"``: annual refreeze sum in m/yr, shape ``(M,)``.
        - ``"smb"``: annual surface mass balance sum in m/yr, shape ``(M,)``.

    Notes
    -----
    - Conversions mirror :func:`load_hirham_climate`.
    - Arrays are built by ``np.hstack`` over groups of ``time.year``, yielding a
      block of 12 months per year concatenated along axis 1.
    """
    with xr.open_dataset(file) as Obs:
        nlat = len(Obs["rlat"])
        nlon = len(Obs["rlon"])

        Obs = Obs.isel(
            rlat=slice(0, nlat, thinning_factor),
            rlon=slice(0, nlon, thinning_factor),
            ncl4=slice(0, nlat, thinning_factor),
            ncl5=slice(0, nlon, thinning_factor),
        )
        stacked = Obs.stack(z=("rlat", "rlon"))
        ncl_stacked = Obs.stack(z=("ncl4", "ncl5"))

        temp = (
            np.hstack(
                [d.dropna(dim="z").values for _, d in stacked.tas.groupby("time.year")]
            )
            - 273.15
        )
        temp_std_dev = np.hstack(
            [
                d.dropna(dim="z").values
                for _, d in stacked.tas_std_dev.groupby("time.year")
            ]
        )
        rainfall = (
            np.hstack(
                [
                    d.dropna(dim="z").values
                    for _, d in stacked.rainfall.groupby("time.year")
                ]
            )
            * 365.242198781
            / 1000.0
        )
        snowfall = (
            np.hstack(
                [
                    d.dropna(dim="z").values
                    for _, d in stacked.snfall.groupby("time.year")
                ]
            )
            * 365.242198781
            / 1000.0
        )
        smb = (
            np.hstack(
                [d.dropna(dim="z").values for _, d in stacked.gld.groupby("time.year")]
            )
            * 365.242198781
            / 1000.0
            / 12.0
        )
        refreeze = (
            np.hstack(
                [
                    d.dropna(dim="z").values
                    for _, d in ncl_stacked.rfrz.groupby("time.year")
                ]
            )
            * 365.242198781
            / 1000.0
            / 12.0
        )
        snowmelt = (
            np.hstack(
                [
                    d.dropna(dim="z").values
                    for _, d in stacked.snmel.groupby("time.year")
                ]
            )
            * 365.242198781
            / 1000.0
            / 12.0
        )
        snowdepth = np.hstack(
            [d.dropna(dim="z").values for _, d in stacked.sn.groupby("time.year")]
        )
        runoff = (
            np.hstack(
                [d.dropna(dim="z").values for _, d in stacked.rogl.groupby("time.year")]
            )
            * 365.242198781
            / 1000.0
            / 12.0
        )
        precip = rainfall + snowfall

        obs: dict[str, NDArray[np.floating]] = {
            "snow_depth": snowdepth - snowdepth[0],
            "accumulation": snowfall.sum(axis=0),
            "melt": snowmelt.sum(axis=0),
            "runoff": runoff.sum(axis=0),
            "refreeze": refreeze.sum(axis=0),
            "smb": smb.sum(axis=0),
        }

    return temp, precip, temp_std_dev, obs


def load_hirham_climate_simple(file="DMI-HIRHAM5_1980_MM.nc", thinning_factor=1):
    """
    Read and return Obs
    """

    with xr.open_dataset(file) as Obs:
        stacked = Obs.stack(z=("rlat", "rlon"))
        ncl_stacked = Obs.stack(z=("ncl4", "ncl5"))

        temp = stacked.tas.dropna(dim="z").values
        rainfall = stacked.rainfall.dropna(dim="z").values
        snowfall = stacked.snfall.dropna(dim="z").values
        smb = stacked.gld.dropna(dim="z").values
        refreeze = ncl_stacked.rfrz.dropna(dim="z").values
        melt = stacked.snmel.dropna(dim="z").values
        precip = rainfall + snowfall
        runoff = stacked.rogl.dropna(dim="z").values

    return (
        (temp[::thinning_factor] - 273.15).reshape(1, -1),
        precip[::thinning_factor].reshape(1, -1),
        snowfall[::thinning_factor].reshape(1, -1),
        melt[::thinning_factor].reshape(1, -1),
        runoff[::thinning_factor].reshape(1, -1),
        smb[::thinning_factor].reshape(1, -1),
        refreeze[::thinning_factor].reshape(1, -1),
    )


def load_imbie_csv(
    proj_start: int = 2008,
    file: str | Path = "imbie_greenland_2021_Gt.csv",
) -> pd.DataFrame:
    """
    Load IMBIE Greenland mass balance CSV and derive sea-level equivalents (SLE).

    The CSV is expected to contain IMBIE columns for mass balance and cumulative
    mass balance (with uncertainties). Columns are renamed to a consistent schema,
    cumulative mass is re-referenced to ``proj_start``, and SLE quantities are
    computed assuming ``1 cm SLE = 362.5 Gt``.

    Parameters
    ----------
    proj_start : int, default=2008
        Reference year. The cumulative mass time series ``"Mass (Gt)"`` is
        shifted so that its value in this year becomes zero.
    file : str or pathlib.Path, default="imbie_greenland_2021_Gt.csv"
        Path to the IMBIE CSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with at least the following columns (after renaming and
        augmentation):
        - ``"Year"``
        - ``"Mass change (Gt/yr)"``
        - ``"Mass change uncertainty (Gt/yr)"``
        - ``"Mass (Gt)"``  (re-referenced so value at ``proj_start`` is 0)
        - ``"Mass uncertainty (Gt)"``
        - ``"SLE (cm)"``  (computed as ``-Mass(Gt) / 362.5 / 10``)
        - ``"SLE uncertainty (cm)"``  (propagated linearly from mass uncertainty)
        - ``"SLE change uncertainty (cm/yr)"``  (from mass-change uncertainty)

    Notes
    -----
    - The conversion used is ``cmSLE = 1 / 362.5 / 10`` (i.e., 362.5 Gt per cm SLE).
    - SLE is defined with opposite sign to mass: increasing mass lowers sea level,
      hence ``SLE = -Mass * cmSLE``.
    - The function assumes the CSV includes IMBIE-standard columns:
      ``"Mass balance (Gt/yr)"``, ``"Mass balance uncertainty (Gt/yr)"``,
      ``"Cumulative mass balance (Gt)"``, and
      ``"Cumulative mass balance uncertainty (Gt)"``. These are renamed to
      ``"Mass change (Gt/yr)"``, ``"Mass change uncertainty (Gt/yr)"``,
      ``"Mass (Gt)"``, and ``"Mass uncertainty (Gt)"``, respectively.

    Raises
    ------
    KeyError
        If expected IMBIE columns are missing.
    ValueError
        If ``proj_start`` is not present in the ``"Year"`` column.
    """
    df = pd.read_csv(file)

    # Rename IMBIE columns to a consistent internal schema
    rename_map = {
        "Mass balance (Gt/yr)": "Mass change (Gt/yr)",
        "Mass balance uncertainty (Gt/yr)": "Mass change uncertainty (Gt/yr)",
        "Cumulative mass balance (Gt)": "Mass (Gt)",
        "Cumulative mass balance uncertainty (Gt)": "Mass uncertainty (Gt)",
    }
    missing = [k for k in rename_map if k not in df.columns]
    if missing:
        raise KeyError(f"Missing expected IMBIE columns: {missing}")
    df = df.rename(columns=rename_map)

    # Re-reference cumulative mass to the project start year
    if not (df["Year"] == proj_start).any():
        raise ValueError(
            f"'proj_start' year {proj_start} not found in CSV 'Year' column."
        )
    ref_val = df.loc[df["Year"] == proj_start, "Mass (Gt)"].values
    # subtract the scalar reference from the mass time series
    df["Mass (Gt)"] = df["Mass (Gt)"] - ref_val

    # Sea-level equivalent conversions (362.5 Gt per cm SLE)
    cmSLE = 1.0 / 362.5 / 10.0
    df["SLE (cm)"] = -df["Mass (Gt)"] * cmSLE
    df["SLE uncertainty (cm)"] = df["Mass uncertainty (Gt)"] * cmSLE
    df["SLE change uncertainty (cm/yr)"] = df["Mass change uncertainty (Gt/yr)"] * cmSLE

    return df


def distributions_as19():
    r"""

    Returns the distributions used by Aschwanden et al (2019):

    @article{Aschwanden2019,
    author = {Aschwanden, Andy and Fahnestock, Mark A. and Truffer, Martin and Brinkerhoff, Douglas J. and Hock, Regine and Khroulev, Constantine and Mottram, Ruth and Khan, S. Abbas},
    doi = {10.1126/sciadv.aav9396},
    issn = {2375-2548},
    journal = {Science Advances},
    month = {jun},
    number = {6},
    pages = {eaav9396},
    title = {{Contribution of the Greenland Ice Sheet to sea level over the next millennium}},
    url = {http://advances.sciencemag.org/lookup/doi/10.1126/sciadv.aav9396},
    volume = {5},
    year = {2019}
    }

    """

    return {
        "GCM": randint(0, 4),
        "FICE": truncnorm(-4 / 4.0, 4.0 / 4, loc=8, scale=4),
        "FSNOW": truncnorm(-4.1 / 3, 4.1 / 3, loc=4.1, scale=1.5),
        "PRS": uniform(loc=5, scale=2),
        "RFR": truncnorm(-0.4 / 0.3, 0.4 / 0.3, loc=0.5, scale=0.2),
        "OCM": randint(-1, 2),
        "OCS": randint(-1, 2),
        "TCT": randint(-1, 2),
        "VCM": truncnorm(-0.35 / 0.2, 0.35 / 0.2, loc=0.4, scale=0.2),
        "PPQ": truncnorm(-0.35 / 0.2, 0.35 / 0.2, loc=0.6, scale=0.2),
        "SIAE": gamma(1.5, scale=0.8, loc=1),
    }
