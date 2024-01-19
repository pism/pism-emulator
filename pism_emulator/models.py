# Copyright (C) 2023-24 Andy Aschwanden, Maria Zeitz
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

from functools import wraps

import numpy as np
import scipy.special as sp
import torch
import xarray as xr
from scipy.interpolate import interp1d


def freeze_it(cls):
    cls.__frozen = False

    def frozensetattr(self, key, value):
        if self.__frozen and not hasattr(self, key):
            print(
                "Class {} is frozen. Cannot set {} = {}".format(
                    cls.__name__, key, value
                )
            )
        else:
            object.__setattr__(self, key, value)

    def init_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            self.__frozen = True

        return wrapper

    cls.__setattr__ = frozensetattr
    cls.__init__ = init_decorator(cls.__init__)

    return cls


@freeze_it
class ReferencePDDModel:
    # Copyright (c) 2013--2018, Julien Seguinot <seguinot@vaw.baug.ethz.ch>
    # GNU General Public License v3.0+ (https://www.gnu.org/licenses/gpl-3.0.txt)

    """Return a callable Positive Degree Day (PDD) model instance.

    Reference implementation

    Model parameters are held as public attributes, and can be set using
    corresponding keyword arguments at initialization time:

    *pdd_factor_snow* : float
        Positive degree-day factor for snow.
    *pdd_factor_ice* : float
        Positive degree-day factor for ice.
    *refreeze_snow* : float
        Refreezing fraction of melted snow.
    *refreeze_ice* : float
        Refreezing fraction of melted ice.
    *temp_snow* : float
        Temperature at which all precipitation falls as snow.
    *temp_rain* : float
        Temperature at which all precipitation falls as rain.
    *interpolate_rule* : [ 'linear' | 'nearest' | 'zero' |
                           'slinear' | 'quadratic' | 'cubic' ]
        Interpolation rule passed to `scipy.interpolate.interp1d`.
    *interpolate_n*: int
        Number of points used in interpolations.
    """

    def __init__(
        self,
        pdd_factor_snow=0.003,
        pdd_factor_ice=0.008,
        refreeze_snow=0.0,
        refreeze_ice=0.0,
        temp_snow=0.0,
        temp_rain=2.0,
        interpolate_rule="linear",
        interpolate_n=12,
    ):
        # set pdd model parameters
        self.pdd_factor_snow = pdd_factor_snow
        self.pdd_factor_ice = pdd_factor_ice
        self.refreeze_snow = refreeze_snow
        self.refreeze_ice = refreeze_ice
        self.temp_snow = temp_snow
        self.temp_rain = temp_rain
        self.interpolate_rule = interpolate_rule
        self.interpolate_n = interpolate_n

    def __call__(self, temp, prec, stdv=0.0, return_xarray: bool = False):
        """Run the positive degree day model.

        Use temperature, precipitation, and standard deviation of temperature
        to compute the number of positive degree days, accumulation and melt
        surface mass fluxes, and the resulting surface mass balance.

        *temp*: array_like
            Input near-surface air temperature in degrees Celcius.
        *prec*: array_like
            Input precipitation rate in meter per year.
        *stdv*: array_like (default 0.0)
            Input standard deviation of near-surface air temperature in Kelvin.

        By default, inputs are N-dimensional arrays whose first dimension is
        interpreted as time and as periodic. Arrays of dimensions
        N-1 are interpreted as constant in time and expanded to N dimensions.
        Arrays of dimension 0 and numbers are interpreted as constant in time
        and space and will be expanded too. The largest input array determines
        the number of dimensions N.

        Return the number of positive degree days ('pdd'), surface mass balance
        ('smb'), and many other output variables in a dictionary.
        """

        # ensure numpy arrays
        # FIXME use data arrays instead
        temp = np.asarray(temp)
        prec = np.asarray(prec)
        stdv = np.asarray(stdv)

        # expand arrays to the largest shape
        # FIXME use xarray auto-broadcasting instead
        maxshape = max(temp.shape, prec.shape, stdv.shape)
        temp = self._expand(temp, maxshape)
        prec = self._expand(prec, maxshape)
        stdv = self._expand(stdv, maxshape)

        # interpolate time-series
        # FIXME propagate data arrays, coordinates
        temp = self._interpolate(temp)
        prec = self._interpolate(prec)
        stdv = self._interpolate(stdv)

        # compute accumulation and pdd
        accumulation_rate = self.accumulation_rate(temp, prec)
        inst_pdd = self.inst_pdd(temp, stdv)

        # initialize snow depth and melt rates
        snow_depth = np.zeros_like(temp)
        snow_melt_rate = np.zeros_like(temp)
        ice_melt_rate = np.zeros_like(temp)

        # compute snow depth and melt rates
        for i in range(len(temp)):
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

        if return_xarray:
            # make a dataset
            # FIXME add coordinate variables
            result = xr.Dataset(
                data_vars={
                    "temp": (["time", "x", "y"], temp),
                    "prec": (["time", "x", "y"], prec),
                    "stdv": (["time", "x", "y"], stdv),
                    "inst_pdd": (["time", "x", "y"], inst_pdd),
                    "accumulation_rate": (["time", "x", "y"], accumulation_rate),
                    "snow_melt_rate": (["time", "x", "y"], snow_melt_rate),
                    "ice_melt_rate": (["time", "x", "y"], ice_melt_rate),
                    "melt_rate": (["time", "x", "y"], melt_rate),
                    "runoff_rate": (["time", "x", "y"], runoff_rate),
                    "inst_smb": (["time", "x", "y"], inst_smb),
                    "snow_depth": (["time", "x", "y"], snow_depth),
                    "pdd": (["x", "y"], self._integrate(inst_pdd)),
                    "accumulation": (["x", "y"], self._integrate(accumulation_rate)),
                    "snow_melt": (["x", "y"], self._integrate(snow_melt_rate)),
                    "ice_melt": (["x", "y"], self._integrate(ice_melt_rate)),
                    "melt": (["x", "y"], self._integrate(melt_rate)),
                    "runoff": (["x", "y"], self._integrate(runoff_rate)),
                    "smb": (["x", "y"], self._integrate(inst_smb)),
                }
            )
        else:
            result = {
                "temp": temp,
                "prec": prec,
                "stdv": stdv,
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

    def _expand(self, array, shape):
        """Expand an array to the given shape"""
        if array.shape == shape:
            res = array
        elif array.shape == (1, shape[1], shape[2]):
            res = np.asarray([array[0]] * shape[0])
        elif array.shape == shape[1:]:
            res = np.asarray([array] * shape[0])
        elif array.shape == ():
            res = array * np.ones(shape)
        else:
            raise ValueError(
                "could not expand array of shape %s to %s" % (array.shape, shape)
            )
        return res

    def _integrate(self, array):
        """Integrate an array over one year"""
        return np.sum(array, axis=0) / (self.interpolate_n - 1)

    def _interpolate(self, array):
        """Interpolate an array through one year."""
        from scipy.interpolate import interp1d

        rule = self.interpolate_rule
        npts = self.interpolate_n
        oldx = (np.arange(len(array) + 2) - 0.5) / len(array)
        oldy = np.vstack(([array[-1]], array, [array[0]]))
        newx = (np.arange(npts) + 0.5) / npts  # use 0.0 for PISM-like behaviour
        newy = interp1d(oldx, oldy, kind=rule, axis=0)(newx)
        return newy

    def inst_pdd(self, temp, stdv):
        """Compute instantaneous positive degree days from temperature.

        Use near-surface air temperature and standard deviation to compute
        instantaneous positive degree days (effective temperature for melt,
        unit degrees C) using an integral formulation (Calov and Greve, 2005).

        *temp*: array_like
            Near-surface air temperature in degrees Celcius.
        *stdv*: array_like
            Standard deviation of near-surface air temperature in Kelvin.
        """
        import scipy.special as sp

        # compute positive part of temperature everywhere
        positivepart = np.greater(temp, 0) * temp

        # compute Calov and Greve (2005) integrand, ignoring division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            normtemp = temp / (np.sqrt(2) * stdv)
        calovgreve = stdv / np.sqrt(2 * np.pi) * np.exp(
            -(normtemp**2)
        ) + temp / 2 * sp.erfc(-normtemp)

        # use positive part where sigma is zero and Calov and Greve elsewhere
        teff = np.where(stdv == 0.0, positivepart, calovgreve)

        # convert to degree-days
        return teff * 365.242198781

    def accumulation_rate(self, temp, prec):
        """Compute accumulation rate from temperature and precipitation.

        The fraction of precipitation that falls as snow decreases linearly
        from one to zero between temperature thresholds defined by the
        `temp_snow` and `temp_rain` attributes.

        *temp*: array_like
            Near-surface air temperature in degrees Celcius.
        *prec*: array_like
            Precipitation rate in meter per year.
        """

        # compute snow fraction as a function of temperature
        reduced_temp = (self.temp_rain - temp) / (self.temp_rain - self.temp_snow)
        snowfrac = np.clip(reduced_temp, 0, 1)

        # return accumulation rate
        return snowfrac * prec

    def melt_rates(self, snow, pdd):
        """Compute melt rates from snow precipitation and pdd sum.

        Snow melt is computed from the number of positive degree days (*pdd*)
        and the `pdd_factor_snow` model attribute. If all snow is melted and
        some energy (PDD) remains, ice melt is computed using `pdd_factor_ice`.

        *snow*: array_like
            Snow precipitation rate.
        *pdd*: array_like
            Number of positive degree days.
        """

        # parse model parameters for readability
        ddf_snow = self.pdd_factor_snow
        ddf_ice = self.pdd_factor_ice

        # compute a potential snow melt
        pot_snow_melt = ddf_snow * pdd

        # effective snow melt can't exceed amount of snow
        snow_melt = np.minimum(snow, pot_snow_melt)

        # ice melt is proportional to excess snow melt
        ice_melt = (pot_snow_melt - snow_melt) * ddf_ice / ddf_snow

        # return melt rates
        return (snow_melt, ice_melt)


@freeze_it
class PDDModel:
    """

    # Copyright (c) 2013--2018, Julien Seguinot <seguinot@vaw.baug.ethz.ch>
    # GNU General Public License v3.0+ (https://www.gnu.org/licenses/gpl-3.0.txt)

    A positive degree day model for glacier surface mass balance

    Return a callable Positive Degree Day (PDD) model instance.

    Model parameters are held as public attributes, and can be set using
    corresponding keyword arguments at initialization time:

    *pdd_factor_snow* : float
        Positive degree-day factor for snow.
    *pdd_factor_ice* : float
        Positive degree-day factor for ice.
    *refreeze_snow* : float
        Refreezing fraction of melted snow.
    *refreeze_ice* : float
        Refreezing fraction of melted ice.
    *temp_snow* : float
        Temperature at which all precipitation falls as snow.
    *temp_rain* : float
        Temperature at which all precipitation falls as rain.
    *interpolate_rule* : [ 'linear' | 'nearest' | 'zero' |
                           'slinear' | 'quadratic' | 'cubic' ]
        Interpolation rule passed to `scipy.interpolate.interp1d`.
    *interpolate_n*: int
        Number of points used in interpolations.
    """

    def __init__(
        self,
        pdd_factor_snow: float = 3.0,
        pdd_factor_ice: float = 8.0,
        refreeze_snow: float = 0.0,
        refreeze_ice: float = 0.0,
        temp_snow: float = 0.0,
        temp_rain: float = 0.0,
        interpolate_rule: str = "linear",
        interpolate_n: int = 52,
        *args,
        **kwargs,
    ):
        super().__init__()

        # set pdd model parameters
        self.pdd_factor_snow = pdd_factor_snow
        self.pdd_factor_ice = pdd_factor_ice
        self.refreeze_snow = refreeze_snow
        self.refreeze_ice = refreeze_ice
        self.temp_snow = temp_snow
        self.temp_rain = temp_rain
        self.interpolate_rule = interpolate_rule
        self.interpolate_n = interpolate_n

    @property
    def pdd_factor_snow(self):
        return self._pdd_factor_snow

    @pdd_factor_snow.setter
    def pdd_factor_snow(self, value):
        self._pdd_factor_snow = value

    @property
    def pdd_factor_ice(self):
        return self._pdd_factor_ice

    @pdd_factor_ice.setter
    def pdd_factor_ice(self, value):
        self._pdd_factor_ice = value

    @property
    def temp_snow(self):
        return self._temp_snow

    @temp_snow.setter
    def temp_snow(self, value):
        self._temp_snow = value

    @property
    def temp_ice(self):
        return self._temp_ice

    @temp_ice.setter
    def temp_ice(self, value):
        self._temp_ice = value

    @property
    def refreeze_snow(self):
        return self._refreeze_snow

    @refreeze_snow.setter
    def refreeze_snow(self, value):
        self._refreeze_snow = value

    @property
    def refreeze_ice(self):
        return self._refreeze_ice

    @refreeze_ice.setter
    def refreeze_ice(self, value):
        self._refreeze_ice = value

    def __call__(self, temp, prec, stdv=0.0, return_xarray: bool = False):
        """Run the positive degree day model.
        Use temperature, precipitation, and standard deviation of temperature
        to compute the number of positive degree days, accumulation and melt
        surface mass fluxes, and the resulting surface mass balance.
        *temp*: array_like
            Input near-surface air temperature in degrees Celcius.
        *prec*: array_like
            Input precipitation rate in meter per year.
        *stdv*: array_like (default 0.0)
            Input standard deviation of near-surface air temperature in Kelvin.
        By default, inputs are N-dimensional arrays whose first dimension is
        interpreted as time and as periodic. Arrays of dimensions
        N-1 are interpreted as constant in time and expanded to N dimensions.
        Arrays of dimension 0 and numbers are interpreted as constant in time
        and space and will be expanded too. The largest input array determines
        the number of dimensions N.
        Return the number of positive degree days ('pdd'), surface mass balance
        ('smb'), and many other output variables in a dictionary.
        """

        # ensure numpy arrays
        # FIXME use data arrays instead
        temp = np.asarray(temp)
        prec = np.asarray(prec)
        stdv = np.asarray(stdv)

        # expand arrays to the largest shape
        # FIXME use xarray auto-broadcasting instead
        maxshape = max(temp.shape, prec.shape, stdv.shape)
        temp = self._expand(temp, maxshape)
        prec = self._expand(prec, maxshape)
        stdv = self._expand(stdv, maxshape)

        # interpolate time-series
        # FIXME propagate data arrays, coordinates
        if (self.interpolate_n > 1) and (self.interpolate_n != temp.shape[0]):
            temp = self._interpolate(temp)
            prec = self._interpolate(prec)
            stdv = self._interpolate(stdv)

        # compute accumulation and pdd
        accumulation_rate = self.accumulation_rate(temp, prec)
        inst_pdd = self.inst_pdd(temp, stdv)

        # initialize snow depth and melt rates
        snow_depth = np.zeros_like(temp)
        snow_melt_rate = np.zeros_like(temp)
        ice_melt_rate = np.zeros_like(temp)

        # compute snow depth and melt rates
        for i in range(len(temp)):
            if i > 0:
                snow_depth[i] = snow_depth[i - 1]
            snow_depth[i] += accumulation_rate[i]
            snow_melt_rate[i], ice_melt_rate[i] = self.melt_rates(
                snow_depth[i], inst_pdd[i]
            )
            snow_depth[i] -= snow_melt_rate[i]
        melt_rate = snow_melt_rate + ice_melt_rate
        snow_refreeze_rate = self.refreeze_snow * snow_melt_rate
        ice_refreeze_rate = self.refreeze_ice * ice_melt_rate
        refreeze_rate = snow_refreeze_rate + ice_refreeze_rate
        runoff_rate = melt_rate - refreeze_rate
        inst_smb = accumulation_rate - runoff_rate

        if return_xarray:
            # make a dataset
            # FIXME add coordinate variables
            result = xr.Dataset(
                data_vars={
                    "temp": (["time", "x", "y"], temp),
                    "prec": (["time", "x", "y"], prec),
                    "stdv": (["time", "x", "y"], stdv),
                    "inst_pdd": (["time", "x", "y"], inst_pdd),
                    "accumulation_rate": (["time", "x", "y"], accumulation_rate),
                    "snow_melt_rate": (["time", "x", "y"], snow_melt_rate),
                    "ice_melt_rate": (["time", "x", "y"], ice_melt_rate),
                    "melt_rate": (["time", "x", "y"], melt_rate),
                    "runoff_rate": (["time", "x", "y"], runoff_rate),
                    "inst_smb": (["time", "x", "y"], inst_smb),
                    "snow_depth": (["time", "x", "y"], snow_depth),
                    "pdd": (["x", "y"], self._integrate(inst_pdd)),
                    "accumulation": (["x", "y"], self._integrate(accumulation_rate)),
                    "snow_melt": (["x", "y"], self._integrate(snow_melt_rate)),
                    "ice_melt": (["x", "y"], self._integrate(ice_melt_rate)),
                    "melt": (["x", "y"], self._integrate(melt_rate)),
                    "runoff": (["x", "y"], self._integrate(runoff_rate)),
                    "refreeze": (["x", "y"], self._integrate(refreeze_rate)),
                    "snow_refreeze": (["x", "y"], self._integrate(snow_refreeze_rate)),
                    "ice_refreeze": (["x", "y"], self._integrate(ice_refreeze_rate)),
                    "smb": (["x", "y"], self._integrate(inst_smb)),
                }
            )
        else:
            result = {
                "temp": temp,
                "prec": prec,
                "stdv": stdv,
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

        return result

    def _expand(self, array, shape):
        """Expand an array to the given shape"""
        if array.shape == shape:
            res = array
        elif array.shape == (1, shape[1], shape[2]):
            res = np.asarray([array[0]] * shape[0])
        elif array.shape == shape[1:]:
            res = np.asarray([array] * shape[0])
        elif array.shape == ():
            res = array * np.ones(shape)
        else:
            raise ValueError(
                "could not expand array of shape %s to %s" % (array.shape, shape)
            )
        return res

    def _integrate(self, array):
        """Integrate an array over one year"""
        return np.sum(array, axis=0) / (self.interpolate_n - 1)

    def _interpolate(self, array):
        """Interpolate an array through one year."""

        rule = self.interpolate_rule
        npts = self.interpolate_n
        oldx = (np.arange(len(array) + 2) - 0.5) / len(array)
        oldy = np.vstack(([array[-1]], array, [array[0]]))
        newx = (np.arange(npts) + 0.5) / npts  # use 0.0 for PISM-like behaviour
        newy = interp1d(oldx, oldy, kind=rule, axis=0)(newx)
        return newy

    def inst_pdd(self, temp, stdv):
        """Compute instantaneous positive degree days from temperature.
        Use near-surface air temperature and standard deviation to compute
        instantaneous positive degree days (effective temperature for melt,
        unit degrees C) using an integral formulation (Calov and Greve, 2005).
        *temp*: array_like
            Near-surface air temperature in degrees Celcius.
        *stdv*: array_like
            Standard deviation of near-surface air temperature in Kelvin.
        """

        # compute positive part of temperature everywhere
        positivepart = np.greater(temp, 0) * temp

        # compute Calov and Greve (2005) integrand, ignoring division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            normtemp = temp / (np.sqrt(2) * stdv)
        calovgreve = stdv / np.sqrt(2 * np.pi) * np.exp(
            -(normtemp**2)
        ) + temp / 2 * sp.erfc(-normtemp)

        # use positive part where sigma is zero and Calov and Greve elsewhere
        teff = xr.where(stdv == 0.0, positivepart, calovgreve)

        # convert to degree-days
        return teff * 365.242198781

    def accumulation_rate(self, temp, prec):
        """Compute accumulation rate from temperature and precipitation.
        The fraction of precipitation that falls as snow decreases linearly
        from one to zero between temperature thresholds defined by the
        `temp_snow` and `temp_rain` attributes.
        *temp*: array_like
            Near-surface air temperature in degrees Celcius.
        *prec*: array_like
            Precipitation rate in meter per year.
        """

        # compute snow fraction as a function of temperature
        reduced_temp = (self.temp_rain - temp) / (self.temp_rain - self.temp_snow)
        snowfrac = np.clip(reduced_temp, 0, 1)

        # return accumulation rate
        return snowfrac * prec

    def melt_rates(self, snow, pdd):
        """Compute melt rates from snow precipitation and pdd sum.
        Snow melt is computed from the number of positive degree days (*pdd*)
        and the `pdd_factor_snow` model attribute. If all snow is melted and
        some energy (PDD) remains, ice melt is computed using `pdd_factor_ice`.
        *snow*: array_like
            Snow precipitation rate.
        *pdd*: array_like
            Number of positive degree days.
        """

        # parse model parameters for readability
        ddf_snow = self.pdd_factor_snow
        ddf_ice = self.pdd_factor_ice

        # compute a potential snow melt
        pot_snow_melt = ddf_snow * pdd

        # effective snow melt can't exceed amount of snow
        snow_melt = np.minimum(snow, pot_snow_melt)

        # ice melt is proportional to excess snow melt
        ice_melt = (pot_snow_melt - snow_melt) * ddf_ice / ddf_snow

        # return melt rates
        return (snow_melt, ice_melt)


@freeze_it
class TorchPDDModel(torch.nn.modules.Module):
    """

    # Copyright (c) 2013--2018, Julien Seguinot <seguinot@vaw.baug.ethz.ch>
    # GNU General Public License v3.0+ (https://www.gnu.org/licenses/gpl-3.0.txt)

    A positive degree day model for glacier surface mass balance

    Return a callable Positive Degree Day (PDD) model instance.

    Model parameters are held as public attributes, and can be set using
    corresponding keyword arguments at initialization time:

    *pdd_factor_snow* : float
        Positive degree-day factor for snow.
    *pdd_factor_ice* : float
        Positive degree-day factor for ice.
    *refreeze_snow* : float
        Refreezing fraction of melted snow.
    *refreeze_ice* : float
        Refreezing fraction of melted ice.
    *temp_snow* : float
        Temperature at which all precipitation falls as snow.
    *temp_rain* : float
        Temperature at which all precipitation falls as rain.
    *interpolate_rule* : [ 'linear' | 'nearest' | 'zero' |
                           'slinear' | 'quadratic' | 'cubic' ]
        Interpolation rule passed to `scipy.interpolate.interp1d`.
    *interpolate_n*: int
        Number of points used in interpolations.
    """

    def __init__(
        self,
        pdd_factor_snow: float = 3.0,
        pdd_factor_ice: float = 8.0,
        refreeze_snow: float = 0.0,
        refreeze_ice: float = 0.0,
        temp_snow: float = 0.0,
        temp_rain: float = 2.0,
        interpolate_rule: str = "linear",
        interpolate_n: int = 12,
        device="cpu",
    ):
        super().__init__()

        # set pdd model parameters
        self.pdd_factor_snow = pdd_factor_snow
        self.pdd_factor_ice = pdd_factor_ice
        self.refreeze_snow = refreeze_snow
        self.refreeze_ice = refreeze_ice
        self.temp_snow = temp_snow
        self.temp_rain = temp_rain
        self.interpolate_rule = interpolate_rule
        self.interpolate_n = interpolate_n
        self.device = device

    @property
    def pdd_factor_snow(self):
        return self._pdd_factor_snow

    @pdd_factor_snow.setter
    def pdd_factor_snow(self, value):
        self._pdd_factor_snow = value

    @property
    def pdd_factor_ice(self):
        return self._pdd_factor_ice

    @pdd_factor_ice.setter
    def pdd_factor_ice(self, value):
        self._pdd_factor_ice = value

    @property
    def temp_snow(self):
        return self._temp_snow

    @temp_snow.setter
    def temp_snow(self, value):
        self._temp_snow = value

    @property
    def temp_ice(self):
        return self._temp_ice

    @temp_ice.setter
    def temp_ice(self, value):
        self._temp_ice = value

    @property
    def refreeze_snow(self):
        return self._refreeze_snow

    @refreeze_snow.setter
    def refreeze_snow(self, value):
        self._refreeze_snow = value

    @property
    def refreeze_ice(self):
        return self._refreeze_ice

    @refreeze_ice.setter
    def refreeze_ice(self, value):
        self._refreeze_ice = value

    def forward(self, temp, prec, stdv=0.0):
        """Run the positive degree day model.

        Use temperature, precipitation, and standard deviation of temperature
        to compute the number of positive degree days, accumulation and melt
        surface mass fluxes, and the resulting surface mass balance.

        *temp*: array_like
            Input near-surface air temperature in degrees Celcius.
        *prec*: array_like
            Input precipitation rate in meter per year.
        *stdv*: array_like (default 0.0)
            Input standard deviation of near-surface air temperature in Kelvin.

        By default, inputs are N-dimensional arrays whose first dimension is
        interpreted as time and as periodic. Arrays of dimensions
        N-1 are interpreted as constant in time and expanded to N dimensions.
        Arrays of dimension 0 and numbers are interpreted as constant in time
        and space and will be expanded too. The largest input array determines
        the number of dimensions N.

        Return the number of positive degree days ('pdd'), surface mass balance
        ('smb'), and many other output variables in a dictionary.
        """

        temp = torch.asarray(temp, device=self.device)
        prec = torch.asarray(prec, device=self.device)
        stdv = torch.asarray(stdv, device=self.device)

        # expand arrays to the largest shape
        maxshape = max(temp.shape, prec.shape, stdv.shape)
        temp = self._expand(temp, maxshape)
        prec = self._expand(prec, maxshape)
        stdv = self._expand(stdv, maxshape)

        # interpolate time-series
        if self.interpolate_n >= 1:
            temp = self._interpolate(temp)
            prec = self._interpolate(prec)
            stdv = self._interpolate(stdv)

        # compute accumulationmulation and pdd
        accumulation_rate = self.accumulation_rate(temp, prec)
        inst_pdd = self.inst_pdd(temp, stdv)

        # initialize snow depth, melt and refreeze rates
        snow_depth = torch.zeros_like(temp)
        snow_melt_rate = torch.zeros_like(temp)
        ice_melt_rate = torch.zeros_like(temp)
        snow_refreeze_rate = torch.zeros_like(temp)
        ice_refreeze_rate = torch.zeros_like(temp)

        # parse model parameters for readability
        ddf_snow = self.pdd_factor_snow / 1000
        ddf_ice = self.pdd_factor_ice / 1000

        for i in range(len(temp)):
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
        snow_refreeze_rate = self.refreeze_snow * snow_melt_rate
        ice_refreeze_rate = self.refreeze_ice * ice_melt_rate
        refreeze_rate = snow_refreeze_rate + ice_refreeze_rate
        runoff_rate = melt_rate - refreeze_rate
        inst_smb = accumulation_rate - runoff_rate

        # output
        return {
            "temp": temp,
            "prec": prec,
            "stdv": stdv,
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

    def _expand(self, array, shape):
        """Expand an array to the given shape"""
        if array.shape == shape:
            res = array
        elif array.shape == (1, shape[1], shape[2]):
            res = [array[0]] * shape[0]
        elif array.shape == shape[1:]:
            res = [array] * shape[0]
        elif array.shape == ():
            res = array * torch.ones(shape)
        else:
            raise ValueError(
                "could not expand array of shape %s to %s" % (array.shape, shape)
            )
        return res

    def _integrate(self, array):
        """Integrate an array over one year"""
        dx = torch.sum(array, axis=0) / (self.interpolate_n - 1)
        return dx.to(self.device)

    def _interpolate(self, array):
        """Interpolate an array through one year."""

        rule = self.interpolate_rule
        npts = self.interpolate_n
        oldx = (torch.arange(len(array) + 2, device=self.device) - 0.5) / len(array)
        oldy = torch.vstack((array[-1], array, array[0]))
        newx = (torch.arange(npts) + 0.5) / npts  # use 0.0 for PISM-like behaviour
        newy = interp1d(oldx.cpu(), oldy.cpu(), kind=rule, axis=0)(newx)
        interp = torch.from_numpy(newy)

        return interp.to(self.device)

    def inst_pdd(self, temp, stdv):
        """Compute instantaneous positive degree days from temperature.

        Use near-surface air temperature and standard deviation to compute
        instantaneous positive degree days (effective temperature for melt,
        unit degrees C) using an integral formulation (Calov and Greve, 2005).

        *temp*: array_like
            Near-surface air temperature in degrees Celcius.
        *stdv*: array_like
            Standard deviation of near-surface air temperature in Kelvin.
        """

        # compute positive part of temperature everywhere
        positivepart = torch.greater(temp, 0) * temp

        # compute Calov and Greve (2005) integrand, ignoring division by zero
        normtemp = temp / (torch.sqrt(torch.tensor(2)) * stdv)
        calovgreve = stdv / torch.sqrt(torch.tensor(2) * torch.pi) * torch.exp(
            -(normtemp**2)
        ) + temp / 2 * (1.0 - torch.erf(-normtemp))

        # use positive part where sigma is zero and Calov and Greve elsewhere
        teff = torch.where(stdv == 0.0, positivepart, calovgreve)
        snowfrac = torch.clip(teff, 0)

        # convert to degree-days
        return snowfrac * 365.242198781

    def accumulation_rate(self, temp, prec):
        """Compute accumulationmulation rate from temperature and precipitation.

        The fraction of precipitation that falls as snow decreases linearly
        from one to zero between temperature thresholds defined by the
        `temp_snow` and `temp_rain` attributes.

        *temp*: array_like
            Near-surface air temperature in degrees Celcius.
        *prec*: array_like
            Precipitation rate in meter per year.
        """

        # compute snow fraction as a function of temperature
        reduced_temp = (self.temp_rain - temp) / (self.temp_rain - self.temp_snow)
        snowfrac = torch.clip(reduced_temp, 0, 1)

        # return accumulationmulation rate
        return snowfrac * prec


class TorchDEBMModel(torch.nn.modules.Module):
    """
    This class implements dEBM-simple, the simple diurnal energy balance model described in

    M. Zeitz, R. Reese, J. Beckmann, U. Krebs-Kanzow, and R. Winkelmann, “Impact of the
    melt–albedo feedback on the future evolution of the Greenland Ice Sheet with
    PISM-dEBM-simple,” The Cryosphere, vol. 15, Art. no. 12, Dec. 2021.

    See also

    U. Krebs-Kanzow, P. Gierz, and G. Lohmann, “Brief communication: An ice surface melt
    scheme including the diurnal cycle of solar radiation,” The Cryosphere, vol. 12, Art.
    no. 12, Dec. 2018.

    and chapter 2 of

    K. N. Liou, Introduction to Atmospheric Radiation. Elsevier Science & Technology Books, 2002.

    """

    # air_temp_all_precip_as_rain = 275.15;
    # air_temp_all_precip_as_rain_doc = "Threshold temperature above which all precipitation is rain; must exceed :config:`surface.debm_simple.air_temp_all_precip_as_snow`";
    # air_temp_all_precip_as_rain_type = "number";
    # air_temp_all_precip_as_rain_units = "Kelvin";

    # air_temp_all_precip_as_snow = 273.15;
    # air_temp_all_precip_as_snow_doc = "Threshold temperature below which all precipitation is snow";
    # air_temp_all_precip_as_snow_type = "number";
    # air_temp_all_precip_as_snow_units = "Kelvin";

    # albedo_ice = 0.47;
    # albedo_ice_doc = "Albedo value for bare ice (lowest possible value in albedo parametrization)";
    # albedo_ice_type = "number";
    # albedo_ice_units = "1";

    # albedo_input.file = "";
    # albedo_input.file_doc = "Name of the file containing the variable :var:`albedo` to use instead of parameterizing it";
    # albedo_input.file_type = "string";

    # albedo_input.periodic = "no";
    # albedo_input.periodic_doc = "If true, interpret forcing data as periodic in time";
    # albedo_input.periodic_type = "flag";

    # albedo_ocean = 0.1;
    # albedo_ocean_doc = "Albedo of ice-free ocean";
    # albedo_ocean_type = "number";
    # albedo_ocean_units = "";

    # albedo_slope = -790;
    # albedo_slope_doc = "Slope in albedo parametrization";
    # albedo_slope_type = "number";
    # albedo_slope_units = "m2 s kg-1";

    # albedo_snow = 0.82;
    # albedo_snow_doc = "Albedo value for fresh snow (albedo without melting in albedo parametrization)";
    # albedo_snow_type = "number";
    # albedo_snow_units = "1";

    # c1 = 29.0;
    # c1_doc = "Tuning parameter controlling temperature-driven melt";
    # c1_type = "number";
    # c1_units = "W m-2 K-1";

    # c2 = -93.0;
    # c2_doc = "Tuning parameter controlling background melt";
    # c2_type = "number";
    # c2_units = "W m-2";

    # interpret_precip_as_snow = "no";
    # interpret_precip_as_snow_doc = "If true, interpret *all* precipitation as snow";
    # interpret_precip_as_snow_type = "flag";

    # max_evals_per_year = 52;
    # max_evals_per_year_doc = "Maximum number of air temperature and precipitation samples per year used to build location-dependent time series for computing melt and snow accumulation; the default means use weekly samples of the annual cycle";
    # max_evals_per_year_type = "integer";
    # max_evals_per_year_units = "count";

    # melting_threshold_temp = 266.65;
    # melting_threshold_temp_doc = "Threshold temperature below which no melting occurs";
    # melting_threshold_temp_type = "number";
    # melting_threshold_temp_units = "Kelvin";

    # paleo.eccentricity =  0.0167;
    # paleo.eccentricity_doc = "Eccentricity of the Earth's orbit";
    # paleo.eccentricity_type = "number";
    # paleo.eccentricity_units = "1";

    # paleo.enabled = "false";
    # paleo.enabled_doc = "If true, use orbital parameters to compute top of the atmosphere insolation";
    # paleo.enabled_option = "debm_simple_paleo";
    # paleo.enabled_type = "flag";

    # paleo.file = "";
    # paleo.file_doc = "File containing orbital parameters (:var:`eccentricity`, :var:`obliquity`, :var:`perihelion_longitude`) for paleo-simulations";
    # paleo.file_option = "debm_simple_paleo_file";
    # paleo.file_type = "string";

    # paleo.obliquity =  23.44;
    # paleo.obliquity_doc = "Mean obliquity (axial tilt) of the Earth.";
    # paleo.obliquity_type = "number";
    # paleo.obliquity_units = "degree";

    # paleo.perihelion_longitude =  102.94719;
    # paleo.perihelion_longitude_doc = "Mean longitude of the perihelion relative to the vernal equinox";
    # paleo.perihelion_longitude_type = "number";
    # paleo.perihelion_longitude_units = "degree";

    # paleo.periodic = "no";
    # paleo.periodic_doc = "If true, interpret forcing data as periodic in time";
    # paleo.periodic_type = "flag";

    # phi =  17.5;
    # phi_doc = "Threshold solar elevation angle above which melt is possible";
    # phi_type = "number";
    # phi_units = "degree";

    # positive_threshold_temp = 273.15;
    # positive_threshold_temp_doc = "Temperature threshold used to define the \"positive\" temperature";
    # positive_threshold_temp_type = "number";
    # positive_threshold_temp_units = "Kelvin";

    # refreeze = 0.6;
    # refreeze_doc = "Refreeze fraction: this fraction of snow melt is assumed to re-freeze. See also :config:`surface.debm_simple.refreeze_ice_melt`.";
    # refreeze_type = "number";
    # refreeze_units = "1";

    # refreeze_ice_melt = "yes";
    # refreeze_ice_melt_doc = "If set to 'yes', refreeze :config:`surface.debm_simple.refreeze` fraction of melted ice, otherwise all of the melted ice runs off.";
    # refreeze_ice_melt_type = "flag";

    # solar_constant = 1367.0;
    # solar_constant_doc = "Mean solar electromagnetic radiation (total solar irradiance) per unit area";
    # solar_constant_type = "number";
    # solar_constant_units = "W m-2";

    # std_dev = 5.0;
    # std_dev_doc = "Standard deviation of daily near-surface air temperature variation";
    # std_dev_type = "number";
    # std_dev_units = "Kelvin";

    # std_dev.file = "";
    # std_dev.file_doc = "The file to read :var:`air_temp_sd` (standard deviation of air temperature) from";
    # std_dev.file_option = "debm_simple_sd_file";
    # std_dev.file_type = "string";

    # std_dev.param.a = -0.15;
    # std_dev.param.a_doc = "Parameter `a` in `\\sigma = a \\cdot T + b`, with `T` in degrees Celsius. Used only if :config:`surface.debm_simple.std_dev.param.enabled` is set to yes.";
    # std_dev.param.a_type = "number";
    # std_dev.param.a_units = "1";

    # std_dev.param.b = 0.66;
    # std_dev.param.b_doc = "Parameter `b` in `\\sigma = a \\cdot T + b`, with `T` in degrees Celsius. Used only if :config:`surface.debm_simple.std_dev.param.enabled` is set to yes.";
    # std_dev.param.b_type = "number";
    # std_dev.param.b_units = "Kelvin";

    # std_dev.param.enabled = "no";
    # std_dev.param.enabled_doc = "Parameterize standard deviation as a linear function of air temperature over ice-covered grid cells. The region of application is controlled by :config:`geometry.ice_free_thickness_standard`.";
    # std_dev.param.enabled_type = "flag";

    # std_dev.periodic = "no";
    # std_dev.periodic_doc = "If true, interpret forcing data as periodic in time";
    # std_dev.periodic_type = "flag";

    # tau_a_intercept = 0.65;
    # tau_a_intercept_doc = "Intercept in the parametrization of atmosphere transmissivity";
    # tau_a_intercept_type = "number";
    # tau_a_intercept_units = "";

    # tau_a_slope = 0.000032;
    # tau_a_slope_doc = "Slope in the parametrization of atmosphere transmissivity";
    # tau_a_slope_type = "number";
    # tau_a_slope_units = "m-1";

    def __init__(
        self,
        air_temp_all_precip_as_rain: float = 275.15,
        air_temp_all_precip_as_snow: float = 273.15,
        albedo_ice: float = 0.47,
        albedo_input_periodic: bool = False,
        albedo_ocean: float = 0.1,
        albedo_slope: float = -790,
        albedo_snow: float = 0.82,
        c1: float = 29.0,
        c2: float = -93.0,
        interpret_precip_as_snow: bool = False,
        latent_heat_of_fusion: float = 3.34e5,
        max_evals_per_year: int = 52,
        melting_threshold_temp: float = 266.65,
        paleo_eccentricity: float = 0.0167,
        paleo_enabled: bool = False,
        paleo_obliquity: float = 23.44,
        paleo_perihelion_longitude: float = 102.94719,
        paleo_periodic: bool = False,
        phi: float = 17.5,
        positive_threshold_temp: float = 273.15,
        refreeze: float = 0.6,
        solar_constant: float = 1367.0,
        std_dev: float = 5.0,
        std_dev_param_a: float = -0.15,
        std_dev_param_b: float = 0.66,
        std_dev_param_enabled: bool = False,
        std_dev_periodic: bool = False,
        tau_a_intercept: float = 0.65,
        tau_a_slope: float = 0.000032,
        device="cpu",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.air_temp_all_precip_as_rain = air_temp_all_precip_as_rain
        self.air_temp_all_precip_as_snow = air_temp_all_precip_as_snow
        self.albedo_ice = albedo_ice
        self.albedo_input_periodic = albedo_input_periodic
        self.albedo_ocean = albedo_ocean
        self.albedo_slope = albedo_slope
        self.albedo_snow = albedo_snow
        self.c1 = c1
        self.c2 = c2
        self.interpret_precip_as_snow = interpret_precip_as_snow
        self.latent_heat_of_fusion = latent_heat_of_fusion
        self.max_evals_per_year = max_evals_per_year
        self.melting_threshold_temp = melting_threshold_temp
        self.paleo_eccentricity = paleo_eccentricity
        self.paleo_enabled = paleo_enabled
        self.paleo_obliquity = paleo_obliquity
        self.paleo_perihelion_longitude = paleo_perihelion_longitude
        self.paleo_periodic = paleo_periodic
        self.phi = phi
        self.positive_threshold_temp = positive_threshold_temp
        self.refreeze = refreeze
        self.solar_constant = solar_constant
        self.std_dev = std_dev
        self.std_dev_param_a = std_dev_param_a
        self.std_dev_param_b = std_dev_param_b
        self.std_dev_param_enabled = std_dev_param_enabled
        self.std_dev_periodic = std_dev_periodic
        self.tau_a_intercept = tau_a_intercept
        self.tau_a_slope = tau_a_slope
        self.device = device

    @property
    def air_temp_all_precip_as_rain(self):
        return self._air_temp_all_precip_as_rain

    @air_temp_all_precip_as_rain.setter
    def air_temp_all_precip_as_rain(self, value):
        self._air_temp_all_precip_as_rain = value

    @property
    def air_temp_all_precip_as_snow(self):
        return self._air_temp_all_precip_as_snow

    @air_temp_all_precip_as_snow.setter
    def air_temp_all_precip_as_snow(self, value):
        self._air_temp_all_precip_as_snow = value

    @property
    def albedo_ice(self):
        return self._albedo_ice

    @albedo_ice.setter
    def albedo_ice(self, value):
        self._albedo_ice = value

    @property
    def albedo_input_periodic(self):
        return self._albedo_input_periodic

    @albedo_input_periodic.setter
    def albedo_input_periodic(self, value):
        self._albedo_input_periodic = value

    @property
    def albedo_ocean(self):
        return self._albedo_ocean

    @albedo_ocean.setter
    def albedo_ocean(self, value):
        self._albedo_ocean = value

    @property
    def albedo_slope(self):
        return self._albedo_slope

    @albedo_slope.setter
    def albedo_slope(self, value):
        self._albedo_slope = value

    @property
    def albedo_snow(self):
        return self._albedo_snow

    @albedo_snow.setter
    def albedo_snow(self, value):
        self._albedo_snow = value

    @property
    def c1(self):
        return self._c1

    @c1.setter
    def c1(self, value):
        self._c1 = value

    @property
    def c2(self):
        return self._c2

    @c2.setter
    def c2(self, value):
        self._c2 = value

    @property
    def interpret_precip_as_snow(self):
        return self._interpret_precip_as_snow

    @interpret_precip_as_snow.setter
    def interpret_precip_as_snow(self, value):
        self._interpret_precip_as_snow = value

    @property
    def latent_heat_of_fusion(self):
        return self._latent_heat_of_fusion

    @latent_heat_of_fusion.setter
    def latent_heat_of_fusion(self, value):
        self._latent_heat_of_fusion = value

    @property
    def max_evals_per_year(self):
        return self._max_evals_per_year

    @max_evals_per_year.setter
    def max_evals_per_year(self, value):
        self._max_evals_per_year = value

    @property
    def melting_threshold_temp(self):
        return self._melting_threshold_temp

    @melting_threshold_temp.setter
    def melting_threshold_temp(self, value):
        self._melting_threshold_temp = value

    @property
    def paleo_eccentricity(self):
        return self._paleo_eccentricity

    @paleo_eccentricity.setter
    def paleo_eccentricity(self, value):
        self._paleo_eccentricity = value

    @property
    def paleo_enabled(self):
        return self._paleo_enabled

    @paleo_enabled.setter
    def paleo_enabled(self, value):
        self._paleo_enabled = value

    @property
    def paleo_obliquity(self):
        return self._paleo_obliquity

    @paleo_obliquity.setter
    def paleo_obliquity(self, value):
        self._paleo_obliquity = value

    @property
    def paleo_perihelion_longitude(self):
        return self._paleo_perihelion_longitude

    @paleo_perihelion_longitude.setter
    def paleo_perihelion_longitude(self, value):
        self._paleo_perihelion_longitude = value

    @property
    def paleo_periodic(self):
        return self._paleo_periodic

    @paleo_periodic.setter
    def paleo_periodic(self, value):
        self._paleo_periodic = value

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, value):
        self._phi = value

    @property
    def positive_threshold_temp(self):
        return self._positive_threshold_temp

    @positive_threshold_temp.setter
    def positive_threshold_temp(self, value):
        self._positive_threshold_temp = value

    @property
    def refreeze(self):
        return self._refreeze

    @refreeze.setter
    def refreeze(self, value):
        self._refreeze = value

    @property
    def solar_constant(self):
        return self._solar_constant

    @solar_constant.setter
    def solar_constant(self, value):
        self._solar_constant = value

    @property
    def std_dev(self):
        return self._std_dev

    @std_dev.setter
    def std_dev(self, value):
        self._std_dev = value

    @property
    def std_dev_param_a(self):
        return self._std_dev_param_a

    @std_dev_param_a.setter
    def std_dev_param_a(self, value):
        self._std_dev_param_a = value

    @property
    def std_dev_param_b(self):
        return self._std_dev_param_b

    @std_dev_param_b.setter
    def std_dev_param_b(self, value):
        self._std_dev_param_b = value

    @property
    def std_dev_param_enabled(self):
        return self._std_dev_param_enabled

    @std_dev_param_enabled.setter
    def std_dev_param_enabled(self, value):
        self._std_dev_param_enabled = value

    @property
    def std_dev_periodic(self):
        return self._std_dev_periodic

    @std_dev_periodic.setter
    def std_dev_periodic(self, value):
        self._std_dev_periodic = value

    @property
    def tau_a_intercept(self):
        return self._tau_a_intercept

    @tau_a_intercept.setter
    def tau_a_intercept(self, value):
        self._tau_a_intercept = value

    @property
    def tau_a_slope(self):
        return self._tau_a_slope

    @tau_a_slope.setter
    def tau_a_slope(self, value):
        self._tau_a_slope = value

    def CalovGreveIntegrand(self, sigma, temperature):
        """
        * The integrand in equation 6 of
        *
        * R. Calov and R. Greve, “A semi-analytical solution for the positive degree-day model
        * with stochastic temperature variations,” Journal of Glaciology, vol. 51, Art. no. 172,
        * 2005.
        *
        * @param[in] sigma standard deviation of daily variation of near-surface air temperature (Kelvin)
        * @param[in] temperature near-surface air temperature in "degrees Kelvin above the melting point"
        */
        """

        a = torch.maximum(temperature, torch.zeros_like(temperature))
        Z = temperature / (np.sqrt(2.0) * sigma)
        b = (sigma / np.sqrt(2.0 * torch.pi)) * torch.exp(-Z * Z) + (
            temperature / 2.0
        ) * (1.0 - torch.erf(-Z))

        return torch.where(sigma == 0, a, b)

    def hour_angle(self, phi, latitude, declination):
        """
        * The hour angle (radians) at which the sun reaches the solar angle `phi`
        *
        * Implements equation 11 in Krebs-Kanzow et al solved for h_phi.
        *
        * Equation 2 in Zeitz et al should be equivalent but misses "acos(...)".
        *
        * The return value is in the range [0, pi].
        *
        * @param[in] phi angle (radians)
        * @param[in] latitude latitude (radians)
        * @param[in] declination solar declination angle (radians)
        */
        """

        cos_h_phi = (torch.sin(phi) - torch.sin(latitude) * torch.sin(declination)) / (
            torch.cos(latitude) * torch.cos(declination)
        )
        return torch.acos(torch.clip(cos_h_phi, -1.0, 1.0))

    def solar_longitude(self, year_fraction, eccentricity, perihelion_longitude):
        """
        * Solar longitude (radians) at current time in the year.
        *
        * @param[in] year_fraction year fraction (between 0 and 1)
        * @param[in] eccentricity eccentricity of the earth’s orbit (no units)
        * @param[in] perihelion_longitude perihelion longitude (radians)
        *
        * Implements equation A2 in Zeitz et al.
        """

        E = eccentricity
        E2 = E * E
        E3 = E * E * E
        L_p = perihelion_longitude

        # Note: lambda = 0 at March equinox (80th day of the year)
        equinox_day_number = 80.0
        delta_lambda = 2.0 * torch.pi * (year_fraction - equinox_day_number / 365.0)
        beta = torch.sqrt(1.0 - E2)

        lambda_m = (
            -2.0
            * (
                (E / 2.0 + E3 / 8.0) * (1.0 + beta) * torch.sin(-L_p)
                - E2 / 4.0 * (1.0 / 2.0 + beta) * torch.sin(-2.0 * L_p)
                + E3 / 8.0 * (1.0 / 3.0 + beta) * torch.sin(-3.0 * L_p)
            )
            + delta_lambda
        )

        return (
            lambda_m
            + (2.0 * E - E3 / 4.0) * torch.sin(lambda_m - L_p)
            + (5.0 / 4.0) * E2 * torch.sin(2.0 * (lambda_m - L_p))
            + (13.0 / 12.0) * E3 * torch.sin(3.0 * (lambda_m - L_p))
        )

    def distance_factor_present_day(self, year_fraction):
        """
        * The unit-less factor scaling top of atmosphere insolation according to the Earth's
        * distance from the Sun.
        *
        * The returned value is `(d_bar / d)^2`, where `d_bar` is the average distance from the
        * Earth to the Sun and `d` is the *current* distance at a given time.
        *
        * Implements equation 2.2.9 from Liou (2002).
        *
        * Liou states: "Note that the factor (a/r)^2 never departs from the unity by more than
        * 3.5%." (`a/r` in Liou is equivalent to `d_bar/d` here.)
        """

        # These coefficients come from Table 2.2 in Liou 2002
        a0 = 1.000110
        a1 = 0.034221
        a2 = 0.000719
        b0 = 0.0
        b1 = 0.001280
        b2 = 0.000077

        t = 2.0 * torch.pi * year_fraction

        return (
            a0
            + b0
            + a1 * torch.cos(t)
            + b1 * torch.sin(t)
            + a2 * torch.cos(2.0 * t)
            + b2 * torch.sin(torch.tensor(2.0) * t)
        )

    def distance_factor_paleo(self, eccentricity, perhelion_longitude, solar_longitude):
        """
        Calculate paleo distance factor
        """

        E = eccentricity

        assert E != 1.0, f"Division by zero, eccentricity is {E}"

        return torch.pow(
            1.0
            + E
            * torch.cos(torch.tensor(solar_longitude - perhelion_longitude))
            / (1.0 - E**2),
            2.0,
        )
