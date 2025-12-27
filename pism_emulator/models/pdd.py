# Copyright (C) 2023-25 Andy Aschwanden
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

# pylint: disable=too-many-lines,too-many-instance-attributes,too-many-statements

"""
PPD model implementations.
"""

import argparse
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeAlias, TypedDict, TypeVar, cast

import lightning as pl
import numpy as np
import numpy.typing as npt
import scipy.special as sp
import torch
import xarray as xr
from scipy.interpolate import interp1d
from torch import Tensor

ArrayLike: TypeAlias = npt.ArrayLike
NDArrayF: TypeAlias = npt.NDArray[np.floating]

_T = TypeVar("_T", bound=type)


C = TypeVar("C", bound=type[Any])
InitFn: TypeAlias = Callable[..., None]


def _as_float_array(x: ArrayLike) -> NDArrayF:
    """
    Convert input to a NumPy floating array.

    Parameters
    ----------
    x : array_like
        Input data.

    Returns
    -------
    numpy.ndarray
        ``x`` converted to a floating-point :class:`numpy.ndarray`. A copy is made
        only if required by the dtype conversion.
    """
    return np.asarray(x, dtype=float)


def freeze_it(cls: C) -> C:
    """
    Freeze a class after initialization to prevent adding new attributes.

    This class decorator modifies ``__setattr__`` so that, after ``__init__``
    completes, attempts to set *new* attributes (i.e., attributes that do not
    already exist on the instance) are rejected.

    Existing attributes may still be modified.

    Parameters
    ----------
    cls : type
        Class to decorate.

    Returns
    -------
    type
        The same class, modified in-place (``__setattr__`` and ``__init__`` are
        wrapped).

    Notes
    -----
    This is a lightweight alternative to ``__slots__`` that preserves normal
    attribute access during initialization.

    By default this implementation raises :class:`AttributeError` when a new
    attribute is set after initialization. If you prefer the legacy behavior
    (print a warning and ignore), replace the ``raise`` with a ``print`` and
    ``return``.
    """
    # Class-level default; each instance gets its own flag set in wrapped __init__.
    setattr(cls, "__frozen", False)

    def frozensetattr(self: Any, key: str, value: Any) -> None:
        if getattr(self, "__frozen", False) and not hasattr(self, key):
            raise AttributeError(f"Class {cls.__name__} is frozen. Cannot set {key}.")
        object.__setattr__(self, key, value)

    orig_init = cast(InitFn, getattr(cls, "__init__"))

    @wraps(orig_init)
    def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        object.__setattr__(self, "__frozen", True)

    setattr(cls, "__setattr__", frozensetattr)
    setattr(cls, "__init__", wrapped_init)

    return cls


def make_fake_climate() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a small 1D (monthly) synthetic climate time series for tests.

    Returns 12-point climatologies intended for unit/integration tests of surface
    mass-balance components (e.g., PDD-style models). Arrays are shaped
    ``(12, 1)`` to be trivially broadcastable to larger grids.

    Returns
    -------
    temp : numpy.ndarray
        Monthly near-surface air temperature with shape ``(12, 1)`` and units °C.
    precip : numpy.ndarray
        Monthly precipitation rate with shape ``(12, 1)`` and units m yr⁻¹.
        (Note: values are synthetic and may include small negative entries).
    sd : numpy.ndarray
        Monthly standard deviation of near-surface air temperature with shape
        ``(12, 1)`` and units K.

    Notes
    -----
    This function is intentionally deterministic and uses hard-coded values so
    tests remain stable.
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


def make_fake_climate_2d(
    filename: str | None = None, n_years: int = 1, torch_order: bool = False
) -> xr.Dataset:
    """
    Create an idealized 2D synthetic climate dataset for tests.

    This generates an artificial monthly (12-point) climatology on a Cartesian
    grid. The base 12-month climatology is repeated ``n_years`` times to create
    a multi-year dataset. The resulting dataset contains near-surface air
    temperature (``temp``), precipitation rate (``prec``), and temperature
    standard deviation (``stdv``), along with CF-style coordinate metadata.

    Parameters
    ----------
    filename : str, optional
        If provided, write the dataset to this NetCDF file via ``to_netcdf``.
        If None (default), no file is written.
    n_years : int, optional
        Number of years to generate by repeating the 12-month climatology.
        The output will contain ``12 * n_years`` monthly time steps.
        Default is 1.
    torch_order : bool, optional
        If False (default), return data in "xarray-friendly" order with a
        single ``time`` dimension of length ``12 * n_years`` and dimensions
        ``(time, x, y)`` for each variable.

        If True, the dataset is transposed to ``(x, y, time)``, the ``time``
        coordinate is converted to calendar dates, and then the ``time`` axis
        is converted into a MultiIndex (``year``, ``month``) and unstacked.
        This yields variables with dimensions ``(x, y, year, month)``.

    Returns
    -------
    xr.Dataset
        Dataset with data variables:

        - ``temp`` : near-surface air temperature (degC)
        - ``prec`` : ice-equivalent precipitation rate (m yr-1)
        - ``stdv`` : standard deviation of near-surface air temperature (K)

        If ``torch_order`` is False, each has shape ``(time, x, y)`` where
        ``time = 12 * n_years``.

        If ``torch_order`` is True, each has shape ``(x, y, year, month)``
        where ``year = n_years`` and ``month = 12``.

        Coordinates include:

        - ``x`` / ``y`` : Cartesian coordinates in meters
        - ``time`` : monthly timestamps (or an unstacked (year, month) index if
          ``torch_order`` is True)

    Notes
    -----
    The construction order, dtype casts, and transposes are intentionally kept
    stable to preserve legacy test behavior (e.g., reproducibility checksums).

    The multi-year dataset is created by repeating the same 12-month climatology;
    it is *not* a transient climate simulation.
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

    # code could be simplified a lot more but we need a better test not
    # relying on exact reproducibility of this toy climate data.

    # assign coordinate values
    lx = ly = 750000
    x = xr.DataArray(np.linspace(-lx, lx, 201, dtype="f4"), dims="x")
    y = xr.DataArray(np.linspace(-ly, ly, 201, dtype="f4"), dims="y")
    time = xr.DataArray((np.arange(12, dtype="f4") + 0.5) / 12, dims="time")

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
        },
    )

    # tile data along time (keeps order + dtype stable)
    ds = xr.concat([ds] * n_years, dim="time")

    time = xr.date_range(start="1980-01-01", periods=len(ds.time), freq="MS")
    ds["time"].attrs.update(ATTRIBUTES["time"])

    if torch_order:
        ds = ds.transpose("x", "y", "time")
        ds = ds.assign_coords(
            time=time,
        )
        ds = (
            ds.assign_coords(
                year=ds["time"].dt.year,
                month=ds["time"].dt.month,
            )
            .set_index(time=("year", "month"))
            .unstack("time")
        )

    # write dataset to file
    if filename is not None:
        ds.to_netcdf(filename)

    return ds


class PDDResult(TypedDict):
    """Return type for :meth:`ReferencePDDModel.__call__`."""

    temp: NDArrayF
    prec: NDArrayF
    stdv: NDArrayF
    inst_pdd: NDArrayF
    accumulation_rate: NDArrayF
    snow_melt_rate: NDArrayF
    ice_melt_rate: NDArrayF
    melt_rate: NDArrayF
    runoff_rate: NDArrayF
    inst_smb: NDArrayF
    snow_depth: NDArrayF
    pdd: NDArrayF
    accumulation: NDArrayF
    snow_melt: NDArrayF
    ice_melt: NDArrayF
    melt: NDArrayF
    runoff: NDArrayF
    smb: NDArrayF


@freeze_it
class ReferencePDDModel:
    """
    Reference Positive Degree-Day (PDD) surface mass-balance model.

    This is a reference implementation of a PDD model that computes
    instantaneous and annually integrated fields including positive degree days,
    accumulation, melt, runoff, and surface mass balance (SMB).

    Model parameters are stored as public attributes and can be set at
    initialization.

    Parameters
    ----------
    pdd_factor_snow : float, optional
        Positive degree-day factor for snow melt (units depend on forcing
        conventions; commonly m w.e. per (degC day) or equivalent). Default is 0.003.
    pdd_factor_ice : float, optional
        Positive degree-day factor for ice melt. Default is 0.008.
    refreeze_snow : float, optional
        Fraction of snow melt that refreezes (0–1). Default is 0.0.
    refreeze_ice : float, optional
        Fraction of ice melt that refreezes (0–1). Default is 0.0.
    temp_snow : float, optional
        Temperature threshold (°C) below which all precipitation is snow.
        Default is 0.0.
    temp_rain : float, optional
        Temperature threshold (°C) above which all precipitation is rain.
        Default is 2.0.
    interpolate_rule : str, optional
        Interpolation rule passed to :func:`scipy.interpolate.interp1d`.
        Common values include ``"linear"``, ``"nearest"``, ``"zero"``,
        ``"slinear"``, ``"quadratic"``, and ``"cubic"``. Default is ``"linear"``.
    interpolate_n : int, optional
        Number of points used to interpolate one annual cycle. Default is 12.

    Notes
    -----
    Inputs to :meth:`__call__` are broadcast/expanded so that the largest input
    array determines the working shape. The first dimension is interpreted as
    time and treated as periodic for interpolation.
    """

    def __init__(
        self,
        pdd_factor_snow: float = 0.003,
        pdd_factor_ice: float = 0.008,
        refreeze_snow: float = 0.0,
        refreeze_ice: float = 0.0,
        temp_snow: float = 0.0,
        temp_rain: float = 2.0,
        interpolate_rule: str = "linear",
        interpolate_n: int = 12,
    ) -> None:
        """
        Initialize a reference PDD model.

        Parameters
        ----------
        pdd_factor_snow : float, optional
            Positive degree-day factor for snow melt. Default is 0.003.
        pdd_factor_ice : float, optional
            Positive degree-day factor for ice melt. Default is 0.008.
        refreeze_snow : float, optional
            Fraction of snow melt that refreezes (0–1). Default is 0.0.
        refreeze_ice : float, optional
            Fraction of ice melt that refreezes (0–1). Default is 0.0.
        temp_snow : float, optional
            Temperature threshold (°C) below which precipitation is entirely snow.
            Default is 0.0.
        temp_rain : float, optional
            Temperature threshold (°C) above which precipitation is entirely rain.
            Default is 2.0.
        interpolate_rule : str, optional
            Interpolation rule for :func:`scipy.interpolate.interp1d`. Default is ``"linear"``.
        interpolate_n : int, optional
            Number of time points to interpolate to across one year. Default is 12.
        """
        self.pdd_factor_snow = pdd_factor_snow
        self.pdd_factor_ice = pdd_factor_ice
        self.refreeze_snow = refreeze_snow
        self.refreeze_ice = refreeze_ice
        self.temp_snow = temp_snow
        self.temp_rain = temp_rain
        self.interpolate_rule = interpolate_rule
        self.interpolate_n = interpolate_n

    def __call__(
        self, temp: ArrayLike, prec: ArrayLike, stdv: ArrayLike = 0.0
    ) -> PDDResult:
        """
        Run the PDD model for the provided climate forcing.

        Parameters
        ----------
        temp : array_like
            Near-surface air temperature in degrees Celsius. May be:
            * time-varying with shape ``(T, Y, X)`` (or generally ``(T, ...)``),
            * time-constant with shape ``(... )``,
            * scalar-like.
        prec : array_like
            Precipitation rate in meters per year. Must be broadcast-compatible
            with ``temp`` after expansion.
        stdv : array_like, optional
            Standard deviation of near-surface air temperature in Kelvin. Default is 0.0.

        Returns
        -------
        dict
            Dictionary containing instantaneous time series and annual integrals.
            Keys include:

            * ``"inst_pdd"``, ``"accumulation_rate"``, ``"snow_melt_rate"``,
              ``"ice_melt_rate"``, ``"melt_rate"``, ``"runoff_rate"``, ``"inst_smb"``,
              ``"snow_depth"`` (instantaneous, time axis leading)
            * ``"pdd"``, ``"accumulation"``, ``"snow_melt"``, ``"ice_melt"``,
              ``"melt"``, ``"runoff"``, ``"smb"`` (annual integrals)

        Raises
        ------
        ValueError
            If the inputs cannot be expanded to a common shape.

        Notes
        -----
        The first axis is interpreted as time and treated as periodic when
        interpolating to ``self.interpolate_n`` points.
        """
        temp_arr = np.asarray(temp, dtype=float)
        prec_arr = np.asarray(prec, dtype=float)
        stdv_arr = np.asarray(stdv, dtype=float)

        maxshape = max(temp_arr.shape, prec_arr.shape, stdv_arr.shape)
        temp_arr = self._expand(temp_arr, maxshape)
        prec_arr = self._expand(prec_arr, maxshape)
        stdv_arr = self._expand(stdv_arr, maxshape)

        temp_arr = self._interpolate(temp_arr)
        prec_arr = self._interpolate(prec_arr)
        stdv_arr = self._interpolate(stdv_arr)

        accumulation_rate = self.accumulation_rate(temp_arr, prec_arr)
        inst_pdd = self.inst_pdd(temp_arr, stdv_arr)

        snow_depth = np.zeros_like(temp_arr)
        snow_melt_rate = np.zeros_like(temp_arr)
        ice_melt_rate = np.zeros_like(temp_arr)

        for i in range(len(temp_arr)):
            if i > 0:
                snow_depth[i] = snow_depth[i - 1]
            snow_depth[i] += accumulation_rate[i]
            snow_melt_rate[i], ice_melt_rate[i] = self.melt_rates(
                snow_depth[i], inst_pdd[i]
            )
            snow_depth[i] -= snow_melt_rate[i]

        melt_rate = snow_melt_rate + ice_melt_rate
        runoff_rate = (
            melt_rate
            - self.refreeze_snow * snow_melt_rate
            - self.refreeze_ice * ice_melt_rate
        )
        inst_smb = accumulation_rate - runoff_rate

        result: PDDResult = {
            "temp": temp_arr,
            "prec": prec_arr,
            "stdv": stdv_arr,
            "inst_pdd": inst_pdd,
            "accumulation_rate": accumulation_rate,
            "snow_melt_rate": snow_melt_rate,
            "ice_melt_rate": ice_melt_rate,
            "melt_rate": melt_rate,
            "runoff_rate": runoff_rate,
            "inst_smb": inst_smb,
            "snow_depth": snow_depth,
            "pdd": self._integrate(inst_pdd),
            "accumulation": self._integrate(accumulation_rate),
            "snow_melt": self._integrate(snow_melt_rate),
            "ice_melt": self._integrate(ice_melt_rate),
            "melt": self._integrate(melt_rate),
            "runoff": self._integrate(runoff_rate),
            "smb": self._integrate(inst_smb),
        }
        return result

    def _expand(self, array: NDArrayF, shape: tuple[int, ...]) -> NDArrayF:
        """
        Expand an input array to a target shape using simple broadcasting rules.

        This implements the legacy expansion policy used by the reference model:

        * If ``array.shape == shape``: returned unchanged.
        * If ``array.shape == (1, shape[1], shape[2])``: repeated along time.
        * If ``array.shape == shape[1:]``: repeated along time.
        * If ``array.shape == ()``: expanded to full ``shape`` via multiplication.

        Parameters
        ----------
        array : numpy.ndarray
            Input array to expand.
        shape : tuple of int
            Target shape.

        Returns
        -------
        numpy.ndarray
            Expanded array with shape ``shape``.

        Raises
        ------
        ValueError
            If the input shape cannot be expanded to the requested shape.
        """
        if array.shape == shape:
            res = array
        elif len(shape) >= 3 and array.shape == (1, shape[1], shape[2]):
            res = np.asarray([array[0]] * shape[0], dtype=float)
        elif array.shape == shape[1:]:
            res = np.asarray([array] * shape[0], dtype=float)
        elif array.shape == ():
            res = array * np.ones(shape, dtype=float)
        else:
            raise ValueError(
                f"could not expand array of shape {array.shape} to {shape}"
            )
        return res

    def _integrate(self, array: ArrayLike) -> NDArrayF:
        """
        Integrate a time series over one year.

        The input is assumed to have a leading time dimension representing samples
        within a year (e.g., monthly values). The method returns the mean-integral
        over the year using the class interpolation setting.

        Parameters
        ----------
        array : array_like
            Time series array with a leading time axis. Shape ``(T, ...)`` where
            ``T`` is the number of time samples.

        Returns
        -------
        numpy.ndarray
            Time-integrated array with shape ``(...)``.

        Notes
        -----
        The normalization uses ``max(self.interpolate_n - 1, 1)`` to avoid division
        by zero if ``interpolate_n`` is 1.
        """
        arr = _as_float_array(array)
        denom = max(int(self.interpolate_n) - 1, 1)
        return np.sum(arr, axis=0) / denom

    def _interpolate(self, array: ArrayLike) -> NDArrayF:
        """
        Interpolate a periodic time series through one year.

        This interpolates the input along its leading time axis to
        ``self.interpolate_n`` points using ``scipy.interpolate.interp1d``.
        The series is treated as periodic by padding with the last and first
        sample before interpolation.

        Parameters
        ----------
        array : array_like
            Input time series with shape ``(T, ...)``.

        Returns
        -------
        numpy.ndarray
            Interpolated time series with shape ``(self.interpolate_n, ...)``.

        Raises
        ------
        ValueError
            If the input has fewer than 1 time sample.
        """

        rule: str = str(self.interpolate_rule)
        npts: int = int(self.interpolate_n)

        arr = _as_float_array(array)
        if arr.shape[0] < 1:
            raise ValueError("array must have at least one time sample along axis 0")

        # Treat as periodic by adding wrap-around endpoints
        oldx = (np.arange(len(arr) + 2) - 0.5) / len(arr)
        oldy = np.vstack(([arr[-1]], arr, [arr[0]]))

        # Centered sampling in each bin
        newx = (np.arange(npts) + 0.5) / npts  # use 0.0 for PISM-like behaviour
        newy = interp1d(oldx, oldy, kind=rule, axis=0)(newx)
        return _as_float_array(newy)

    def inst_pdd(self, temp: ArrayLike, stdv: ArrayLike) -> NDArrayF:
        r"""
        Compute instantaneous positive degree days (Calov & Greve, 2005).

        Uses near-surface air temperature and its standard deviation to compute an
        effective temperature for melt (in :math:`^\circ\mathrm{C}`) using an
        integral formulation that accounts for sub-grid temperature variability.

        Parameters
        ----------
        temp : array_like
            Near-surface air temperature in degrees Celsius. Shape ``(...)``.
        stdv : array_like
            Standard deviation of near-surface air temperature in Kelvin. Must be
            broadcast-compatible with ``temp``.

        Returns
        -------
        numpy.ndarray
            Instantaneous positive degree days in :math:`^\circ\mathrm{C}\,\mathrm{day}`.
            Shape is the broadcasted shape of ``temp`` and ``stdv``.

        Notes
        -----
        Where ``stdv == 0``, this reduces to the positive part of ``temp``.
        The result is multiplied by 365.242198781 to convert to degree-days.
        """

        t = _as_float_array(temp)
        s = _as_float_array(stdv)

        # positive part of temperature
        positivepart = (t > 0.0) * t

        # Calov & Greve integrand; ignore division warnings
        with np.errstate(divide="ignore", invalid="ignore"):
            normtemp = t / (np.sqrt(2.0) * s)

        calovgreve = s / np.sqrt(2.0 * np.pi) * np.exp(-(normtemp**2)) + (
            t / 2.0
        ) * sp.erfc(  # pylint: disable=no-member
            -normtemp
        )

        teff = np.where(s == 0.0, positivepart, calovgreve)
        return _as_float_array(teff) * 365.242198781

    def accumulation_rate(self, temp: ArrayLike, prec: ArrayLike) -> NDArrayF:
        """
        Compute snowfall accumulation rate from temperature and precipitation.

        The fraction of precipitation that falls as snow decreases linearly from 1
        to 0 between temperature thresholds ``self.temp_snow`` and ``self.temp_rain``.
        Values are clipped to ``[0, 1]``.

        Parameters
        ----------
        temp : array_like
            Near-surface air temperature in degrees Celsius. Shape ``(...)``.
        prec : array_like
            Precipitation rate in meters per year. Must be broadcast-compatible
            with ``temp``.

        Returns
        -------
        numpy.ndarray
            Snowfall accumulation rate in meters per year, shape is the broadcasted
            shape of ``temp`` and ``prec``.

        Notes
        -----
        The snow fraction is computed as::

            snowfrac = clip((temp_rain - temp) / (temp_rain - temp_snow), 0, 1)
        """
        t = _as_float_array(temp)
        p = _as_float_array(prec)

        reduced_temp = (float(self.temp_rain) - t) / (
            float(self.temp_rain) - float(self.temp_snow)
        )
        snowfrac = np.clip(reduced_temp, 0.0, 1.0)
        return _as_float_array(snowfrac * p)

    def melt_rates(self, snow: ArrayLike, pdd: ArrayLike) -> tuple[NDArrayF, NDArrayF]:
        """
        Compute snow and ice melt rates from snowfall and positive degree days.

        Snow melt is computed from positive degree days (``pdd``) and the degree-day
        factor for snow (``self.pdd_factor_snow``). If all snow is melted and excess
        energy remains, ice melt is computed using ``self.pdd_factor_ice``.

        Parameters
        ----------
        snow : array_like
            Snow precipitation / snow availability (same units as desired melt output).
            Shape ``(...)``.
        pdd : array_like
            Positive degree days (or effective melt energy proxy). Must be
            broadcast-compatible with ``snow``.

        Returns
        -------
        snow_melt : numpy.ndarray
            Snow melt rate, shape is the broadcasted shape of ``snow`` and ``pdd``.
        ice_melt : numpy.ndarray
            Ice melt rate, shape is the broadcasted shape of ``snow`` and ``pdd``.

        Notes
        -----
        The potential snow melt is::

            pot_snow_melt = pdd_factor_snow * pdd

        Snow melt is limited by available snow, and ice melt is proportional to
        the remaining energy using the ratio of degree-day factors.
        """
        s = _as_float_array(snow)
        d = _as_float_array(pdd)

        ddf_snow = float(self.pdd_factor_snow)
        ddf_ice = float(self.pdd_factor_ice)

        pot_snow_melt = ddf_snow * d
        snow_melt = np.minimum(s, pot_snow_melt)
        ice_melt = (pot_snow_melt - snow_melt) * ddf_ice / ddf_snow

        return _as_float_array(snow_melt), _as_float_array(ice_melt)


class PDD(pl.LightningModule):
    """
    Positive Degree-Day (PDD) surface mass-balance component for Lightning.

    This implementation assumes **time is the last dimension** and performs all
    cumulative and reduction operations over ``dim=-1`` (instead of ``dim=0``).

    The climate inputs ``temp``, ``precip``, and ``stdv`` are stored as non-trainable
    buffers. Model parameters (degree-day factors, refreezing fractions, and
    temperature thresholds for snow/rain partitioning) are provided at call time via
    :meth:`forward`.

    Parameters
    ----------
    temp : array_like
        Near-surface air temperature time series. Must be convertible to a torch
        tensor. Supported shapes include ``(..., T)``, ``(..., 1)``, ``(...)``, or
        scalar-like; broadcasting is used internally. Units: °C.
    precip : array_like
        Precipitation rate with shapes broadcastable to ``temp``. Units: m yr⁻¹.
    stdv : array_like
        Standard deviation of near-surface air temperature with shapes broadcastable
        to ``temp``. Units: K.
    n_interpolate : int, optional
        Number of time points used to represent a year (``>= 1``). Default is 12.
        This affects the annual integration scaling in :meth:`_integrate`.
    predictor_vars : list of str or None, optional
        If provided, :meth:`forward` will select only these diagnostics (in the given
        order) when constructing the returned tensor. Each name must be a key in the
        diagnostics dictionary (e.g., ``"pdd"``, ``"smb"``, ``"runoff"``). Default is
        None (use all diagnostics).

    Notes
    -----
    * Time is assumed to represent one year and integration is performed over the
      **last** axis (``dim=-1``).
    * Degree-day factors are expected in mm w.e. d⁻¹ °C⁻¹ and are converted to
      m w.e. d⁻¹ °C⁻¹ internally.
    """

    def __init__(
        self,
        temp,
        precip,
        stdv,
        *,
        n_interpolate: int = 12,
        predictor_vars: list[str] | None = None,
    ) -> None:
        """
        Initialize the PDD module.

        Parameters
        ----------
        temp : array_like
            Near-surface air temperature time series, shape ``(..., T)`` (time last).
            Units: °C.
        precip : array_like
            Precipitation rate, broadcast-compatible with ``temp``. Units: m yr⁻¹.
        stdv : array_like
            Standard deviation of temperature, broadcast-compatible with ``temp``.
            Units: K.
        n_interpolate : int, optional
            Number of time points per year (``>= 1``). Default is 12.
        predictor_vars : list of str or None, optional
            Diagnostics to include in the returned tensor from :meth:`forward`.
            Default is None (include all diagnostics).

        Raises
        ------
        ValueError
            If ``n_interpolate < 1``.
        """
        super().__init__()

        if n_interpolate < 1:
            raise ValueError("n_interpolate must be >= 1")

        self.save_hyperparameters(ignore=["temp", "precip", "stdv"])

        self.register_buffer("temp", torch.as_tensor(temp))
        self.register_buffer("precip", torch.as_tensor(precip))
        self.register_buffer("stdv", torch.as_tensor(stdv))

        self.predictor_vars = predictor_vars

    @staticmethod
    def add_model_specific_args(
        parent_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """
        Add PDD-specific CLI arguments to an existing parser.

        Parameters
        ----------
        parent_parser : argparse.ArgumentParser
            Parser to which PDD arguments will be added.

        Returns
        -------
        argparse.ArgumentParser
            The updated parser with PDD arguments added.

        Notes
        -----
        This helper currently adds the ``--n_interpolate`` argument.
        """
        parser = parent_parser.add_argument_group("PDD")
        parser.add_argument("--n_interpolate", type=int, default=12)
        return parent_parser

    def forward(self, *args: Any, **kwargs: Any) -> Tensor | dict[str, Tensor]:
        """
        Compute PDD diagnostics from parameter tensors.

        Parameters
        ----------
        *args
            Positional arguments. If provided, ``args[0]`` is interpreted as ``X``.
        **kwargs
            Keyword arguments. Must contain ``X`` if ``*args`` is empty.

            The following keyword is recognized:

            return_dict : bool, optional
                If True, return a dictionary of diagnostics. If False (default),
                return a tensor constructed by concatenating selected diagnostics
                along the last dimension.

        Returns
        -------
        torch.Tensor or dict[str, torch.Tensor]
            If ``return_dict=True``, a dictionary mapping diagnostic names to tensors.
            Instantaneous diagnostics have shape ``(..., T)`` and annual/cumulative
            diagnostics have shape ``(..., 1)`` (via ``unsqueeze(-1)``).

            If ``return_dict=False``, returns a tensor formed by concatenating the
            selected diagnostics along the last axis. The output shape is
            ``(..., M)``, where ``M`` is the sum of the last-dimension sizes of the
            selected tensors (e.g., concatenating one ``(..., T)`` variable and two
            ``(..., 1)`` variables yields ``(..., T+2)``).

        Raises
        ------
        TypeError
            If ``X`` is not provided via ``*args`` or ``**kwargs``.
        KeyError
            If ``predictor_vars`` contains names not present in the diagnostics.
        """
        if args:
            X = args[0]
        elif "X" in kwargs:
            X = kwargs["X"]
        else:
            raise TypeError("forward() missing required argument: X")

        return_dict = bool(kwargs.get("return_dict", False))

        (
            pdd_factor_snow,
            pdd_factor_ice,
            refreeze_snow,
            refreeze_ice,
            temp_snow,
            temp_rain,
        ) = X

        # All arrays expected to have time on the last axis: (..., T)
        inst_pdd = self.inst_pdd(self.temp, self.stdv)  # (..., T)
        accumulation_rate = self.accumulation_rate(
            self.temp, self.precip, temp_snow, temp_rain
        )  # (..., T)

        # Degree-day factors: mm w.e. / day / °C -> m w.e. / day / °C
        ddf_snow = pdd_factor_snow / 1000.0
        ddf_ice = pdd_factor_ice / 1000.0

        potential_snow_melt = ddf_snow * inst_pdd  # (..., T)

        # Snowpack evolution over time axis (-1)
        u = accumulation_rate - potential_snow_melt  # (..., T)
        S = torch.cumsum(u, dim=-1)  # (..., T)

        S0 = torch.zeros_like(S[..., :1])  # (..., 1)
        S_ext = torch.cat([S0, S], dim=-1)  # (..., T+1)

        running_min = torch.cummin(S_ext, dim=-1).values[..., 1:]  # (..., T)
        snow_depth = S - torch.minimum(
            running_min, torch.zeros_like(running_min)
        )  # (..., T)

        snow_depth_prev = torch.cat(
            [torch.zeros_like(snow_depth[..., :1]), snow_depth[..., :-1]],
            dim=-1,
        )  # (..., T)
        intermediate_snow = snow_depth_prev + accumulation_rate  # (..., T)

        snow_melt_rate = intermediate_snow - snow_depth  # (..., T)

        # Ice melt only if potential snow melt exceeds available snow
        ratio = ddf_ice / ddf_snow
        ice_melt_rate = (potential_snow_melt - snow_melt_rate) * ratio
        ice_melt_rate = torch.clamp(ice_melt_rate, min=0.0)

        melt_rate = snow_melt_rate + ice_melt_rate
        snow_refreeze_rate = refreeze_snow * snow_melt_rate
        ice_refreeze_rate = refreeze_ice * ice_melt_rate
        refreeze_rate = snow_refreeze_rate + ice_refreeze_rate
        runoff_rate = melt_rate - refreeze_rate
        inst_smb = accumulation_rate - runoff_rate

        pdd = self._integrate(inst_pdd).unsqueeze(-1)
        accumulation = self._integrate(accumulation_rate).unsqueeze(-1)
        snow_melt = self._integrate(snow_melt_rate).unsqueeze(-1)
        ice_melt = self._integrate(ice_melt_rate).unsqueeze(-1)
        melt = self._integrate(melt_rate).unsqueeze(-1)
        runoff = self._integrate(runoff_rate).unsqueeze(-1)
        refreeze = self._integrate(refreeze_rate).unsqueeze(-1)
        snow_refreeze = self._integrate(snow_refreeze_rate).unsqueeze(-1)
        ice_refreeze = self._integrate(ice_refreeze_rate).unsqueeze(-1)
        smb = self._integrate(inst_smb).unsqueeze(-1)

        result: dict[str, Tensor] = {
            "inst_pdd": inst_pdd,
            "accumulation_rate": accumulation_rate,
            "snow_melt_rate": snow_melt_rate,
            "ice_melt_rate": ice_melt_rate,
            "melt_rate": melt_rate,
            "snow_refreeze_rate": snow_refreeze_rate,
            "ice_refreeze_rate": ice_refreeze_rate,
            "refreeze_rate": refreeze_rate,
            "runoff_rate": runoff_rate,
            "smb_rate": inst_smb,
            "snow_depth": snow_depth,
            "pdd": pdd,
            "accumulation": accumulation,
            "snow_melt": snow_melt,
            "ice_melt": ice_melt,
            "melt": melt,
            "runoff": runoff,
            "refreeze": refreeze,
            "snow_refreeze": snow_refreeze,
            "ice_refreeze": ice_refreeze,
            "smb": smb,
        }

        if return_dict:
            return result

        if self.predictor_vars is not None:
            missing = [k for k in self.predictor_vars if k not in result]
            if missing:
                raise KeyError(
                    f"Unknown predictor_vars: {missing}. Available: {sorted(result)}"
                )
            cols = [result[k] for k in self.predictor_vars]
        else:
            cols = list(result.values())

        cols2 = list(cols)
        m = torch.cat(cols2, dim=-1)
        return m

    def inst_pdd(self, temp: Tensor, stdv: Tensor) -> Tensor:
        """
        Compute instantaneous positive degree-days (PDD).

        Uses the effective temperature formulation (Calov & Greve, 2005) to account
        for sub-time-step variability assuming a normal distribution with standard
        deviation ``stdv``.

        Parameters
        ----------
        temp : torch.Tensor
            Near-surface air temperature (°C), shape ``(..., T)`` (time last).
        stdv : torch.Tensor
            Standard deviation of temperature (K), broadcast-compatible with ``temp``.

        Returns
        -------
        torch.Tensor
            Instantaneous positive degree-days with shape ``(..., T)``.
        """
        positivepart = torch.clamp(temp, min=0.0)

        sqrt2 = temp.new_tensor(2.0).sqrt()
        normtemp = temp / (sqrt2 * stdv)
        calovgreve = stdv / torch.sqrt(
            torch.tensor(2.0 * torch.pi, device=temp.device)
        ) * torch.exp(-(normtemp**2)) + temp * 0.5 * (1.0 - torch.erf(-normtemp))
        teff = torch.where(stdv == 0.0, positivepart, calovgreve)
        return torch.clamp(teff, min=0.0) * 365.242198781

    def accumulation_rate(
        self,
        temp: Tensor,
        precip: Tensor,
        temp_snow: float | Tensor,
        temp_rain: float | Tensor,
    ) -> Tensor:
        """
        Compute snowfall accumulation rate from air temperature and precipitation.

        The snowfall fraction decreases linearly from 1 to 0 as temperature rises
        from ``temp_snow`` to ``temp_rain``; values are clamped to ``[0, 1]``.
        For ``temp <= temp_snow`` all precipitation is snow; for
        ``temp >= temp_rain`` none is snow.

        Parameters
        ----------
        temp : torch.Tensor
            Near-surface air temperature (°C), shape ``(..., T)`` (time last).
        precip : torch.Tensor
            Precipitation rate (m yr⁻¹), broadcast-compatible with ``temp``.
        temp_snow : float or torch.Tensor
            Temperature (°C) below which precipitation is entirely snow.
        temp_rain : float or torch.Tensor
            Temperature (°C) above which precipitation is entirely rain. Must satisfy
            ``temp_rain > temp_snow`` (conceptually) and be broadcast-compatible.

        Returns
        -------
        torch.Tensor
            Snowfall accumulation rate (m yr⁻¹), shape ``(..., T)``.
        """
        reduced_temp = (temp_rain - temp) / (temp_rain - temp_snow)
        snowfrac = torch.clamp(reduced_temp, 0.0, 1.0)
        return snowfrac * precip

    def _integrate(self, array: Tensor) -> Tensor:
        """
        Integrate a time series over one year (time on last axis).

        Parameters
        ----------
        array : torch.Tensor
            Time series with a **last** time axis, shape ``(..., T)``.

        Returns
        -------
        torch.Tensor
            Time-integrated field with shape ``(...)``.

        Notes
        -----
        Integration uses a scaled sum:

        ``sum(array, dim=-1) / max(n_interpolate - 1, 1)``

        which behaves like an average when ``array`` represents samples over a year.
        """
        return torch.sum(array, dim=-1) / max(self.hparams.n_interpolate - 1, 1)
