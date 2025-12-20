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

import argparse
from collections.abc import Callable
from functools import wraps
from types import SimpleNamespace
from typing import Any, Sequence, TypedDict, TypeVar, Union, cast

import lightning as pl
import numpy as np
import numpy.typing as npt
import scipy.special as sp
import torch
import xarray as xr
from scipy.interpolate import interp1d
from torch import Tensor

ArrayLike = npt.ArrayLike
NDArrayF = npt.NDArray[np.floating]

_T = TypeVar("_T", bound=type)


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


def freeze_it(cls: _T) -> _T:
    """
    Class decorator that prevents adding new attributes after initialization.

    After the decorated class finishes running its ``__init__``, the instance is
    marked as "frozen". Subsequent attempts to set a **new** attribute (one that
    does not already exist on the instance) will be rejected. Updating existing
    attributes remains allowed.

    Parameters
    ----------
    cls : type
        Class to decorate.

    Returns
    -------
    type
        The same class with a wrapped ``__init__`` and an overridden ``__setattr__``.

    Notes
    -----
    This decorator enforces a lightweight "no new attributes after init" policy.
    It prints a message instead of raising an exception when rejecting an
    attribute assignment.

    Examples
    --------
    >>> @freeze_it
    ... class A:
    ...     def __init__(self) -> None:
    ...         self.x = 1
    ...
    >>> a = A()
    >>> a.x = 2      # allowed (existing attribute)
    >>> a.y = 3      # rejected (new attribute), prints a message  # doctest: +SKIP
    """
    cls.__frozen = False  # type: ignore[attr-defined]

    def frozensetattr(self: Any, key: str, value: Any) -> None:
        if getattr(self, "__frozen", False) and not hasattr(self, key):
            print(f"Class {cls.__name__} is frozen. Cannot set {key} = {value}")
            return
        object.__setattr__(self, key, value)

    def init_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> None:
            func(self, *args, **kwargs)
            object.__setattr__(self, "__frozen", True)

        return wrapper

    cls.__setattr__ = frozensetattr  # type: ignore[assignment]
    cls.__init__ = init_decorator(cls.__init__)  # type: ignore[method-assign]

    return cls


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
        from scipy.interpolate import interp1d

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
        import scipy.special as sp

        t = _as_float_array(temp)
        s = _as_float_array(stdv)

        # positive part of temperature
        positivepart = (t > 0.0) * t

        # Calov & Greve integrand; ignore division warnings
        with np.errstate(divide="ignore", invalid="ignore"):
            normtemp = t / (np.sqrt(2.0) * s)

        calovgreve = s / np.sqrt(2.0 * np.pi) * np.exp(-(normtemp**2)) + (
            t / 2.0
        ) * sp.erfc(-normtemp)

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

    This module computes instantaneous positive degree-days (PDD), snowfall
    accumulation, melt, refreezing, runoff, and surface mass balance (SMB) from
    near-surface temperature and precipitation time series using a simple
    temperature-index method (e.g., Calov & Greve, 2005).

    Inputs ``temp``, ``precip``, and ``stdv`` are stored as non-trainable buffers.
    The model parameters (degree-day factors, refreezing fractions, and temperature
    thresholds for snow/rain partitioning) are provided at call time via ``forward``.

    Parameters
    ----------
    temp : array_like
        Near-surface air temperature time series. Must be convertible to a
        torch tensor. Supported shapes include ``(T, Y, X)``, ``(1, Y, X)``,
        ``(Y, X)``, or scalar-like; broadcasting is used internally. Units: °C.
    precip : array_like
        Precipitation rate with shapes broadcastable to ``temp``. Units: m yr⁻¹.
    stdv : array_like
        Standard deviation of near-surface air temperature. Shapes must be
        broadcastable to ``temp``. Units: K.
    n_interpolate : int, optional
        Number of time points used to represent a year (``>= 1``). Default is 12.
        This affects the annual integration scaling in :meth:`_integrate`.
    predictor_vars : list of str, optional
        If provided, :meth:`forward` returns a stacked tensor containing only
        the requested variables (in the given order) instead of the full
        diagnostics dictionary. Each name must be a key in the diagnostics
        dictionary produced by :meth:`forward` (e.g., ``"pdd"``, ``"smb"``).

    Notes
    -----
    * Time is assumed to represent one year and integration is performed with a
      simple mean over the time axis scaled by the number of interpolation points.
    * Temperature is treated in degrees Celsius (°C). The standard deviation
      ``stdv`` is treated as Kelvin (K) but used as a magnitude in the same formula.
    """

    def __init__(
        self,
        temp,
        precip,
        stdv,
        *,
        n_interpolate: int = 12,
        predictor_vars: list[str] | None = None,
    ):
        """
        Initialize the PDD module.

        Parameters
        ----------
        temp : array_like
            Near-surface air temperature time series. Must be convertible to a
            torch tensor. Supported shapes include ``(T, Y, X)``, ``(1, Y, X)``,
            ``(Y, X)``, or scalar-like; broadcasting is used internally. Units: °C.
        precip : array_like
            Precipitation rate with shapes broadcastable to ``temp``. Units: m yr⁻¹.
        stdv : array_like
            Standard deviation of near-surface air temperature. Shapes must be
            broadcastable to ``temp``. Units: K.
        n_interpolate : int, optional
            Number of time points used to represent a year (``>= 1``). Default is 12.
            This affects the annual integration scaling in :meth:`_integrate`.
        predictor_vars : list of str, optional
            If provided, :meth:`forward` returns a stacked tensor containing only
            the requested variables (in the given order) instead of the full
            diagnostics dictionary. Each name must be a key in the diagnostics
            dictionary produced by :meth:`forward` (e.g., ``"pdd"``, ``"smb"``).

        Raises
        ------
        ValueError
            If ``n_interpolate < 1``.
        """
        super().__init__()

        if n_interpolate < 1:
            raise ValueError("n_interpolate must be >= 1")

        # Store only the small scalar hyperparameters; ignore the big arrays.
        self.save_hyperparameters(ignore=["temp", "precip", "stdv"])

        # Tensors
        temp_t = torch.as_tensor(temp)
        precip_t = torch.as_tensor(precip)
        stdv_t = torch.as_tensor(stdv)

        self.register_buffer("temp", temp_t)
        self.register_buffer("precip", precip_t)
        self.register_buffer("stdv", stdv_t)

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
        This helper adds the ``--n_interpolate`` argument.
        """
        parser = parent_parser.add_argument_group("PDD")
        parser.add_argument("--n_interpolate", type=int, default=12)

        return parent_parser

    def forward(self, x: Sequence[Tensor], **kwargs):
        """
        Compute PDD diagnostics from parameter tensors.

        Parameters
        ----------
        x : Sequence[torch.Tensor]
            Ordered sequence of six parameter tensors:

            ``(pdd_factor_snow, pdd_factor_ice, refreeze_snow, refreeze_ice,
            temp_snow, temp_rain)``.

            Parameters must be broadcast-compatible with the internal ``temp`` and
            ``precip`` buffers. Scalars are allowed.

            Interpreted units/conventions:
            * ``pdd_factor_snow`` : snow degree-day factor, mm w.e. d⁻¹ °C⁻¹
            * ``pdd_factor_ice``  : ice degree-day factor, mm w.e. d⁻¹ °C⁻¹
            * ``refreeze_snow``   : fraction of snow melt that refreezes (0–1)
            * ``refreeze_ice``    : fraction of ice melt that refreezes (0–1)
            * ``temp_snow``       : °C, snow/rain transition lower bound
            * ``temp_rain``       : °C, snow/rain transition upper bound

        **kwargs
            Additional keyword arguments accepted for API compatibility.
            Currently unused.

        Returns
        -------
        dict[str, torch.Tensor] or torch.Tensor
            If ``predictor_vars`` is None, returns a diagnostics dictionary with
            keys including (non-exhaustive): ``"inst_pdd"``, ``"accumulation_rate"``,
            ``"snow_melt_rate"``, ``"ice_melt_rate"``, ``"runoff_rate"``, ``"inst_smb"``,
            and annual integrals such as ``"pdd"``, ``"accumulation"``, ``"melt"``,
            ``"runoff"``, ``"refreeze"``, ``"smb"``, etc.

            If ``predictor_vars`` is provided, returns a 2D tensor of shape
            ``(N, K)`` where ``K = len(predictor_vars)`` and ``N`` is the number of
            elements in the broadcasted field (flattened by the stacking logic).

        Notes
        -----
        * ``temp_snow`` and ``temp_rain`` define a linear transition for snowfall
          fraction in ``[temp_snow, temp_rain]`` and are clamped outside this range.
        * Gradients flow through the linear region; clamps have zero gradients outside.
        """
        (
            pdd_factor_snow,
            pdd_factor_ice,
            refreeze_snow,
            refreeze_ice,
            temp_snow,
            temp_rain,
        ) = x

        inst_pdd = self.inst_pdd(self.temp, self.stdv)
        accumulation_rate = self.accumulation_rate(
            self.temp, self.precip, temp_snow, temp_rain
        )

        snow_depth = torch.zeros_like(self.temp)
        snow_melt_rate = torch.zeros_like(self.temp)
        ice_melt_rate = torch.zeros_like(self.temp)
        snow_refreeze_rate = torch.zeros_like(self.temp)
        ice_refreeze_rate = torch.zeros_like(self.temp)

        ddf_snow = pdd_factor_snow / 1000
        ddf_ice = pdd_factor_ice / 1000

        for i in range(len(self.temp)):
            if i == 0:
                intermediate_snow_depth = accumulation_rate[i]
            else:
                intermediate_snow_depth = snow_depth[i - 1] + accumulation_rate[i]

            potential_snow_melt = ddf_snow * inst_pdd[i]

            snow_melt_rate[i] = torch.minimum(
                intermediate_snow_depth, potential_snow_melt
            )

            ice_melt_rate[i] = (
                (potential_snow_melt - snow_melt_rate[i]) * ddf_ice / ddf_snow
            )

            snow_depth[i] = intermediate_snow_depth - snow_melt_rate[i]

        melt_rate = snow_melt_rate + ice_melt_rate
        snow_refreeze_rate = refreeze_snow * snow_melt_rate
        ice_refreeze_rate = refreeze_ice * ice_melt_rate
        refreeze_rate = snow_refreeze_rate + ice_refreeze_rate
        runoff_rate = melt_rate - refreeze_rate
        inst_smb = accumulation_rate - runoff_rate

        result = {
            "inst_pdd": inst_pdd,
            "accumulation_rate": accumulation_rate,
            "snow_melt_rate": snow_melt_rate,
            "ice_melt_rate": ice_melt_rate,
            "melt_rate": melt_rate,
            "snow_refreeze_rate": snow_refreeze_rate,
            "ice_refreeze_rate": ice_refreeze_rate,
            "refreeze_rate": refreeze_rate,
            "runoff_rate": runoff_rate,
            "inst_smb": inst_smb,
            "snow_depth": snow_depth,
            "pdd": self._integrate(inst_pdd),
            "accumulation": self._integrate(accumulation_rate),
            "snow_melt": self._integrate(snow_melt_rate),
            "ice_melt": self._integrate(ice_melt_rate),
            "melt": self._integrate(melt_rate),
            "runoff": self._integrate(runoff_rate),
            "refreeze": self._integrate(refreeze_rate),
            "snow_refreeze": self._integrate(snow_refreeze_rate),
            "ice_refreeze": self._integrate(ice_refreeze_rate),
            "smb": self._integrate(inst_smb),
        }

        if self.predictor_vars is not None:
            obs_pred = [result[k] for k in self.predictor_vars]
            return torch.vstack(obs_pred).T

        return result

    def inst_pdd(self, temp: Tensor, stdv: Tensor) -> Tensor:
        """
        Compute instantaneous positive degree-days (PDD).

        Uses the effective temperature formulation described by Calov & Greve (2005)
        to account for sub-monthly variability using a normal distribution with
        standard deviation ``stdv``.

        Parameters
        ----------
        temp : torch.Tensor
            Near-surface air temperature (°C).
        stdv : torch.Tensor
            Standard deviation of near-surface air temperature (K). Must be
            broadcast-compatible with ``temp``.

        Returns
        -------
        torch.Tensor
            Instantaneous positive degree-days (°C·days) with the same broadcasted
            shape and device as ``temp``.
        """
        positivepart = torch.clamp(temp, min=0.0)
        normtemp = temp / (torch.sqrt(torch.tensor(2.0, device=temp.device)) * stdv)
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
            Near-surface air temperature (°C). Arbitrary shape ``(...)``.
        precip : torch.Tensor
            Precipitation rate (m yr⁻¹). Must be broadcast-compatible with ``temp``.
        temp_snow : float or torch.Tensor
            Temperature (°C) below which precipitation is entirely snow (fraction = 1).
            Can be a scalar or broadcast-compatible tensor.
        temp_rain : float or torch.Tensor
            Temperature (°C) above which precipitation is entirely rain (fraction = 0).
            Must satisfy ``temp_rain > temp_snow`` and be broadcast-compatible.

        Returns
        -------
        torch.Tensor
            Snowfall accumulation rate (m yr⁻¹), same broadcasted shape and device
            as the inputs.

        Notes
        -----
        The snow fraction is
        ``snowfrac = clamp((temp_rain - temp) / (temp_rain - temp_snow), 0, 1)``.
        Gradients flow through the linear segment; the clamp introduces zero
        gradients outside ``[temp_snow, temp_rain]``.
        """
        reduced_temp = (temp_rain - temp) / (temp_rain - temp_snow)
        snowfrac = torch.clamp(reduced_temp, 0.0, 1.0)
        return snowfrac * precip

    def _integrate(self, array: Tensor) -> Tensor:
        """
        Integrate a time series over one year.

        Parameters
        ----------
        array : torch.Tensor
            Time series with a leading time axis, i.e. shape ``(T, ...)``.

        Returns
        -------
        torch.Tensor
            Time-integrated field on the current device with shape ``(...)``.

        Notes
        -----
        Integration uses a simple scaled sum:

        ``sum(array, dim=0) / max(n_interpolate - 1, 1)``

        which behaves like an average when ``array`` represents samples over a year.
        """
        return torch.sum(array, dim=0) / max(self.hparams.n_interpolate - 1, 1)
