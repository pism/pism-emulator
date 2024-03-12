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
from typing import Literal, Union

import numpy as np
import scipy.special as sp
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


class DEBMModel:
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

    def __init__(
        self,
        air_temp_all_precip_as_rain: float = 275.15,
        air_temp_all_precip_as_snow: float = 273.15,
        albedo_input_periodic: bool = False,
        albedo_min: float = 0.47,
        albedo_max: float = 0.82,
        albedo_slope: float = -790,
        c1: float = 29.0,
        c2: float = -93.0,
        ice_density: float = 917.0,
        interpolate_n: int = 52,
        interpolate_rule: Literal[
            "linear", "nearest", "zero", "slinear", "quadratic", "cubic"
        ] = "linear",
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
        water_density: float = 1000.0,
        device="cpu",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.air_temp_all_precip_as_rain = air_temp_all_precip_as_rain
        self.air_temp_all_precip_as_snow = air_temp_all_precip_as_snow
        self.albedo_min = albedo_min
        self.albedo_input_periodic = albedo_input_periodic
        self.albedo_slope = albedo_slope
        self.albedo_max = albedo_max
        self.c1 = c1
        self.c2 = c2
        self.ice_density = ice_density
        self.interpret_precip_as_snow = interpret_precip_as_snow
        self.interpolate_rule = interpolate_rule
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
        self.water_density = water_density
        self.device = device
        self.interpolate_n = interpolate_n

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
    def albedo_min(self):
        return self._albedo_min

    @albedo_min.setter
    def albedo_min(self, value):
        self._albedo_min = value

    @property
    def albedo_input_periodic(self):
        return self._albedo_input_periodic

    @albedo_input_periodic.setter
    def albedo_input_periodic(self, value):
        self._albedo_input_periodic = value

    @property
    def albedo_slope(self):
        return self._albedo_slope

    @albedo_slope.setter
    def albedo_slope(self, value):
        self._albedo_slope = value

    @property
    def albedo_max(self):
        return self._albedo_max

    @albedo_max.setter
    def albedo_max(self, value):
        self._albedo_max = value

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
    def ice_density(self):
        return self._ice_density

    @ice_density.setter
    def ice_density(self, value):
        self._ice_density = value

    @property
    def interpolate_n(self):
        return self._interpolate_n

    @interpolate_n.setter
    def interpolate_n(self, value):
        self._interpolate_n = value

    @property
    def interpolate_rule(self):
        return self._interpolate_rule

    @interpolate_rule.setter
    def interpolate_rule(self, value):
        self._interpolate_rule = value

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

    @property
    def water_density(self):
        return self._water_density

    @water_density.setter
    def water_density(self, value):
        self._water_density = value

    def __call__(
        self,
        temperature: np.ndarray,
        temperature_std_deviation: np.ndarray,
        precipitation: np.ndarray,
        surface_elevation: Union[None, np.ndarray] = None,
        latitude: Union[None, np.ndarray] = None,
    ) -> dict:
        """Run the DEBM model.
        Use temperature, precipitation to compute accumulation and melt
        surface mass fluxes, and the resulting surface mass balance.
        *temperature*: array_like
            Input near-surface air temperature in degrees Celcius.
        *precipitation*: array_like
            Input precipitation rate in meter per year.
        By default, inputs are N-dimensional arrays whose first dimension is
        interpreted as time and as periodic. Arrays of dimensions
        N-1 are interpreted as constant in time and expanded to N dimensions.
        Arrays of dimension 0 and numbers are interpreted as constant in time
        and space and will be expanded too. The largest input array determines
        the number of dimensions N.
        Return surface mass balance
        ('smb'), and many other output variables in a dictionary.
        """

        # ensure numpy arrays
        temperature = np.asarray(temperature)
        precipitation = np.asarray(precipitation)

        # expand arrays to the largest shape
        maxshape = max(temperature.shape, precipitation.shape)
        temperature = self._expand(temperature, maxshape)
        temperature_std_deviation = self._expand(temperature_std_deviation, maxshape)
        precipitation = self._expand(precipitation, maxshape)
        surface_elevation = self._expand(surface_elevation, maxshape)
        latitude = self._expand(latitude, maxshape)

        # interpolate time-series
        if (self.interpolate_n > 1) and (self.interpolate_n != temperature.shape[0]):
            temperature = self._interpolate(temperature)
            temperature_std_deviation = self._interpolate(temperature_std_deviation)
            precipitation = self._interpolate(precipitation)
            surface_elevation = self._interpolate(surface_elevation)
            latitude = self._interpolate(latitude)

        # compute accumulation
        accumulation = self.snow_accumulation(temperature, precipitation)

        # initialize snow depth and melt rates
        snow_depth = np.zeros_like(temperature)
        snow_melted = np.zeros_like(temperature)
        ice_melted = np.zeros_like(temperature)
        insolation_melt = np.zeros_like(temperature)
        temperature_melt = np.zeros_like(temperature)
        offset_melt = np.zeros_like(temperature)
        total_melt = np.zeros_like(temperature)
        melt = np.zeros_like(temperature)
        runoff = np.zeros_like(temperature)
        smb = np.zeros_like(temperature)

        nt = temperature.shape[0]
        dt = 1.0 / nt
        albedo = self.albedo(total_melt[0])
        for i in range(nt):
            time = dt * i
            year_fraction = self.year_fraction(time)
            melt_info = self.melt(
                temperature[i],
                temperature_std_deviation[i],
                albedo,
                surface_elevation[i],
                latitude[i],
                year_fraction,
                dt,
            )
            insolation_melt[i] = melt_info["insolation_melt"]
            temperature_melt[i] = melt_info["temperature_melt"]
            offset_melt[i] = melt_info["offset_melt"]
            total_melt[i] = melt_info["total_melt"]
            if i == 0:
                changes = self.step(total_melt[i], snow_depth[i], accumulation[i])
                snow_depth[i] = changes["snow_depth"]
                melt[i] = changes["melt"]
                runoff[i] = changes["runoff"]
                smb[i] = changes["smb"]
            else:
                changes = self.step(total_melt[i], snow_depth[i - 1], accumulation[i])
                snow_depth[i] = snow_depth[i-1] + changes["snow_depth"]
                melt[i] = total_melt[i-1] + changes["melt"]
                runoff[i] = runoff[i-1] + changes["runoff"]
                smb[i] = smb[i-1] + changes["smb"]
            albedo += self.albedo(total_melt[i] / dt)  ## ????

        result = {
            "temperature": temperature,
            "precipitation": precipitation,
            "smb": self._integrate(smb),
            "snow_depth": snow_depth,
            "runoff": self._integrate(runoff),
            "melt": self._integrate(melt),
            "temperature_melt": self._integrate(temperature_melt),
            "offset_melt": self._integrate(offset_melt),
            "insolation_melt": self._integrate(insolation_melt),
            "melt_rate": melt,
            "accumulation": self._integrate(accumulation),
            "accumulation_rate": accumulation,
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

    def snow_accumulation(
        self, temperature: np.ndarray, precipitation: np.ndarray
    ) -> np.ndarray:
        """Compute accumulation rate from temperature and precipitation.
        The fraction of precipitation that falls as snow decreases linearly
        from one to zero between temperature thresholds defined by the
        `temp_snow` and `temp_rain` attributes.
        *temperature*: array_like
            Near-surface air temperature in degrees Celcius.
        *precipitation*: array_like
            Precipitation rate in meter per year.
        """

        # compute snow fraction as a function of temperature
        reduced_temp = (self.air_temp_all_precip_as_rain - temperature) / (
            self.air_temp_all_precip_as_rain - self.air_temp_all_precip_as_snow
        )
        snowfrac = np.clip(reduced_temp, 0, 1)

        # return accumulation rate
        return snowfrac * precipitation

    def melt(
        self,
        temperature: np.ndarray,
        temperature_std_deviaton: np.ndarray,
        albedo: np.ndarray,
        surface_elevation: np.ndarray,
        latitude: np.ndarray,
        year_fraction: float,
        dt: float,
    ) -> dict:
        """
        Calculate melt
        """
        latitude_rad = np.deg2rad(latitude)
        declination, distance_factor = self.orbital_parameters(year_fraction)

        transmissivity = self.atmosphere_transmissivity(surface_elevation)
        phi_rad = np.deg2rad(self.phi)
        h_phi = self.hour_angle(phi_rad, latitude_rad, declination)
        insolation = self.insolation(
            self.solar_constant, distance_factor, h_phi, latitude_rad, declination
        )
        T_eff = self.CalovGreveIntegrand(
            temperature_std_deviaton,
            temperature - self.positive_threshold_temp,
        )
        eps = 1.0e-4
        T_eff = np.where(T_eff < eps, 0.0, T_eff)

        #  Note that in the line below we replace "Delta_t_Phi / Delta_t" with "h_Phi / pi". See
        #  equations 1 and 2 in Zeitz et al.
        A = dt * (h_phi / np.pi / (self.water_density * self.latent_heat_of_fusion))

        insolation_melt = A * (transmissivity * (1.0 - albedo) * insolation)
        temperature_melt = A * self.c1 * T_eff
        offset_melt = A * self.c2

        total_melt = insolation_melt + temperature_melt + offset_melt
        total_melt = np.maximum(total_melt, 0)

        return {
            "insolation_melt": insolation_melt,
            "temperature_melt": temperature_melt,
            "offset_melt": offset_melt,
            "total_melt": total_melt,
        }

    def step(
        self, max_melt: np.ndarray, old_snow_depth: np.ndarray, accumulation: np.ndarray
    ) -> dict:
        snow_depth = old_snow_depth
        snow_depth += accumulation

        snow_melted = np.where(max_melt < 0, 0.0, max_melt)
        snow_melted = np.where(max_melt <= snow_depth, max_melt, snow_depth)
        ice_melted = np.minimum(max_melt - snow_melted, snow_depth)
        snow_depth = np.maximum(snow_depth - snow_melted, 0.0)
        snow_depth -= old_snow_depth
        ice_melted = max_melt - snow_melted
        total_melt = snow_melted + ice_melted
        ice_created_by_refreeze = self.refreeze * snow_melted
        runoff = total_melt - ice_created_by_refreeze
        smb = accumulation - runoff

        result = {
            "snow_depth": snow_depth,
            "melt": total_melt,
            "runoff": runoff,
            "smb": smb,
        }
        return result

    def year_fraction(self, time: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Fractional part of a year

        Parameters
        ----------
        time : numpy.ndarray
            Decimal time

        Returns
        ----------
        year_fraction : numpy.ndarray
            Year fraction

        >>>    import numpy as np
        >>>    from pism_emulator.models.debm import DEBMModel

        >>>    debm = DEBMModel()

        >>>    time = np.array([1980.5])
        >>>    year_fraction = debm.year_fraction(time)
        array([0.5])
        """
        return time - np.floor(time)

    def CalovGreveIntegrand(
        self,
        sigma: np.ndarray,
        temperature: np.ndarray,
    ) -> np.ndarray:
        """
        The integrand in equation 6 of
        R. Calov and R. Greve, “A semi-analytical solution for the positive degree-day model
        with stochastic temperature variations,” Journal of Glaciology, vol. 51, Art. no. 172,
        2005.

        Parameters
        ----------
        sigma : numpy.ndarray
            sigma standard deviation of daily variation of near-surface air temperature (Kelvin)
        temperature : numpy.ndarray
            temperature near-surface air temperature in "degrees Kelvin above the melting point"

        Returns
        ----------
        int: numpy.ndarray
            Integrand of Eq 6.

        Examples
        ----------

        >>>    import numpy as np
        >>>    from pism_emulator.models.debm import DEBMModel

        >>>    debm = DEBMModel()

        >>>    sigma = np.array([4.2, 2.0])
        >>>    temperature = np.array([1.0, 2.1])

        >>>    cgi = debm.CalovGreveIntegrand(sigma, temperature)
        >>>    cgi
        array([2.22282761, 2.25136026])
        """

        a = np.maximum(temperature, np.zeros_like(temperature))
        Z = temperature / (np.sqrt(2.0) * sigma)
        b = (sigma / np.sqrt(2.0 * np.pi)) * np.exp(-Z * Z) + (temperature / 2.0) * (
            1.0 - sp.erf(-Z)
        )

        return np.where(sigma == 0, a, b)

    def hour_angle(
        self, phi: np.ndarray, latitude: np.ndarray, declination: np.ndarray
    ) -> np.ndarray:
        """
        The hour angle (radians) at which the sun reaches the solar angle `phi`

        Implements equation 11 in Krebs-Kanzow et al solved for h_phi.

        Equation 2 in Zeitz et al should be equivalent but misses "acos(...)".

        The return value is in the range [0, pi].

        Parameters
        ----------
        phi : float, list of floats or numpy.ndarray
            angle (radians)
        latitude : float, list of floats or numpy.ndarray
            latitude (radians)
        declination : float, list of floats or numpy.ndarray
            solar declination angle (radians)

        Returns
        ----------
        angle: numpy.ndarray
            hour angle (radians)

        Examples
        ----------

        >>>    import numpy as np
        >>>    from pism_emulator.models.debm import DEBMModel

        >>>    debm = DEBMModel()

        >>>    phi = np.array([np.pi / 4])
        >>>    latitude = np.array([np.pi / 4])
        >>>    declination = np.array([np.pi / 8])
        >>>    hour_angle = debm.hour_angle(phi, latitude, declination)
        >>>    hour_angle
        array([0.839038303283583])
        """

        cos_h_phi = (np.sin(phi) - np.sin(latitude) * np.sin(declination)) / (
            np.cos(latitude) * np.cos(declination)
        )
        return np.arccos(np.clip(cos_h_phi, -1.0, 1.0))

    def solar_longitude(
        self,
        year_fraction: np.ndarray,
        eccentricity: np.ndarray,
        perihelion_longitude: np.ndarray,
    ) -> np.ndarray:
        """
        Solar longitude (radians) at current time in the year.

        Implements equation A2 in Zeitz et al.

        Parameters
        ----------
        year_fraction : numpy.ndarray
            fraction (1)
        eccentricity : numpy.ndarray
            eccentricity of the earth's orbit (1)
        perihelion_longigute : numpy.ndarray
            perihelion longitude (radians)

        Returns
        ----------
        solar_longitude: numpy.ndarray
            solar longitude at current time in the year (radians)

        Examples
        ----------

        >>>    import numpy as np
        >>>    from pism_emulator.models.debm import DEBMModel

        >>>    debm = DEBMModel()

        >>>    year_fraction = np.array([0.5])
        >>>    eccentricity = np.array([0.1])
        >>>    perihelion_longitude = np.array([np.pi/4])
        >>>    solar_longitude = debm.solar_longitude(year_fraction, eccentricity, perihelion_longitude)
        >>>    solar_longitude
        array([2.08753274])

        """

        E = eccentricity
        E2 = E * E
        E3 = E * E * E
        L_p = perihelion_longitude

        # Note: lambda = 0 at March equinox (80th day of the year)
        equinox_day_number = 80.0
        delta_lambda = 2.0 * np.pi * (year_fraction - equinox_day_number / 365.0)
        beta = np.sqrt(1.0 - E2)

        lambda_m = (
            -2.0
            * (
                (E / 2.0 + E3 / 8.0) * (1.0 + beta) * np.sin(-L_p)
                - E2 / 4.0 * (1.0 / 2.0 + beta) * np.sin(-2.0 * L_p)
                + E3 / 8.0 * (1.0 / 3.0 + beta) * np.sin(-3.0 * L_p)
            )
            + delta_lambda
        )

        return (
            lambda_m
            + (2.0 * E - E3 / 4.0) * np.sin(lambda_m - L_p)
            + (5.0 / 4.0) * E2 * np.sin(2.0 * (lambda_m - L_p))
            + (13.0 / 12.0) * E3 * np.sin(3.0 * (lambda_m - L_p))
        )

    def distance_factor_present_day(self, year_fraction: np.ndarray) -> np.ndarray:
        """
        The unit-less factor scaling top of atmosphere insolation according to the Earth's
        distance from the Sun.

        The returned value is `(d_bar / d)^2`, where `d_bar` is the average distance from the
        Earth to the Sun and `d` is the *current* distance at a given time.

        Implements equation 2.2.9 from Liou (2002).

        Liou states: "Note that the factor (a/r)^2 never departs from the unity by more than
        3.5%." (`a/r` in Liou is equivalent to `d_bar/d` here.)

        Parameters
        ----------
        year_fraction : numpy.ndarray
            fraction (1)

        Returns
        ----------
        distance_factor_present_day: numpy.ndarray
            distance factor present day ()

        Examples
        ----------

        >>>    import numpy as np
        >>>    from pism_emulator.models.debm import DEBMModel

        >>>    debm = DEBMModel()

        >>>    year_fraction = np.array([0.5])
        >>>    distance_factor_present_day = debm.distance_factor_present_day(year_fraction)
        >>>    distance_factor_present_day
        array([0.966608])
        """

        # These coefficients come from Table 2.2 in Liou 2002
        a0 = 1.000110
        a1 = 0.034221
        a2 = 0.000719
        b0 = 0.0
        b1 = 0.001280
        b2 = 0.000077

        t = 2.0 * np.pi * year_fraction

        return (
            a0
            + b0
            + a1 * np.cos(t)
            + b1 * np.sin(t)
            + a2 * np.cos(2.0 * t)
            + b2 * np.sin(2.0 * t)
        )

    def distance_factor_paleo(
        self,
        eccentricity: np.ndarray,
        perhelion_longitude: np.ndarray,
        solar_longitude: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate paleo distance factor

        Parameters
        ----------
        eccentricity : numpy.ndarray
            eccentricity of the earth’s orbit (no units)
        perhelion_longitude : numpy.ndarray
            perihelion longitude (radians)
        solar_longitude : numpy.ndarray
            solar longitude (radians)

        Returns
        ----------
        distance_factor_present_day: numpy.ndarray
            distance factor present day ()

        Examples
        ----------

        >>>    import numpy as np
        >>>    from pism_emulator.models.debm import DEBMModel

        >>>    debm = DEBMModel()

        >>>    year_fraction = np.array([0.5])
        >>>    eccentricity = np.array([0.1])
        >>>    perihelion_longitude = np.array([np.pi/4])
        >>>    distance_factor_paleo = debm.distance_factor_paleo(year_fraction, eccentricity, perihelion_longitude)
        >>>    distance_factor_paleo
        array([2.29859373])
        """

        E = eccentricity

        assert np.isin(E, 1).any() == False, f"Division by zero, eccentricity is {E}"

        return (
            1.0 + E * np.cos(solar_longitude - perhelion_longitude) / (1.0 - E**2)
        ) ** 2.0

    def solar_declination_present_day(self, year_fraction: np.ndarray) -> np.ndarray:
        """
        Solar declination (radians)

        Implements equation 2.2.10 from Liou (2002)

        Parameters
        ----------
        year_fraction : numpy.ndarray
            fraction (1)

        Returns
        ----------
        solar_declination: numpy.ndarray
            Solar declination present day ()

        Examples
        ----------

        >>>    import numpy as np
        >>>    from pism_emulator.models.debm import DEBMModel

        >>>    debm = DEBMModel()

        >>>    year_fraction = np.array([0.5])
        >>>    solar_declination = debm.solar_declination_present_day(year_fraction)
        array([0.402769])
        """

        # These coefficients come from Table 2.2 in Liou 2002
        a0 = 0.006918
        a1 = -0.399912
        a2 = -0.006758
        a3 = -0.002697
        b0 = 0.0
        b1 = 0.070257
        b2 = 0.000907
        b3 = 0.000148

        t = 2.0 * np.pi * year_fraction

        return (
            a0
            + b0
            + a1 * np.cos(t)
            + b1 * np.sin(t)
            + a2 * np.cos(2.0 * t)
            + b2 * np.sin(2.0 * t)
            + a3 * np.cos(3.0 * t)
            + b3 * np.sin(3.0 * t)
        )

    def solar_declination_paleo(
        self, obliquity: np.ndarray, solar_longitude: np.ndarray
    ) -> np.ndarray:
        """
        Solar declination (radians). This is the "paleo" version used when
        the trigonometric expansion (equation 2.2.10 in Liou 2002) is not valid.

        The return value is in the range [-pi/2, pi/2].

        Implements equation in the text just above equation A1 in Zeitz et al.

        See also equation 2.2.4 of Liou (2002).

        Parameters
        ----------
        obliquity : numpy.ndarray
            fraction (radians)
        solar_longitude : numpy.ndarray
            Solar longitude (radians)

        Returns
        ----------
        solar_declination: numpy.ndarray
            Solar declination paleo (radians)

        Examples
        ----------

        >>>    import numpy as np
        >>>    from pism_emulator.models.debm import DEBMModel

        >>>    debm = DEBMModel()

        >>>    obliquity = np.array([np.pi / 4])
        >>>    solar_longitude = np.array([np.pi * 3 / 4])
        >>>    solar_declination = debm.solar_declination_paleo(obliquity, solar_longitude)
        >>>   solar_declination
        array([0.55536037])
        """

        solar_declination = np.arcsin(np.sin(obliquity * np.sin(solar_longitude)))
        assert np.all(
            np.abs(solar_declination) <= np.pi / 4
        ), f"{solar_declination} not within [-pi/4, pi/4] bounds"
        return solar_declination

    def insolation(
        self,
        solar_constant: np.ndarray,
        distance_factor: np.ndarray,
        hour_angle: np.ndarray,
        latitude: np.ndarray,
        declination: np.ndarray,
    ) -> np.ndarray:
        """

        Average top of atmosphere insolation (rate) during the daily melt period, in W/m^2.

        This should be equation 5 in Zeitz et al or equation 12 in Krebs-Kanzow et al, but both
        of these miss a factor of Delta_t (day length in seconds) in the numerator.

        To confirm this, see the derivation of equation 2.2.21 in Liou and note that

        omega = 2 pi (radian/day)

        or

        omega = (2 pi / 86400) (radian/second).

        The correct equation should say

        S_Phi = A * B^2 * (h_phi * sin(phi) * sin(delta) + cos(phi) * cos(delta) * sin(h_phi)),

        where

        A = (S0 * Delta_t) / (Delta_t_Phi * pi),
        B = d_bar / d.

        Note that we do not know Delta_t_phi but we can use equation 2 in Zeitz et al (or
        equation 11 in Krebs-Kanzow et al) to get

        Delta_t_phi = h_phi * Delta_t / pi.

        This gives

        S_Phi = C * B^2 * (h_phi * sin(phi) * sin(delta) + cos(phi) * cos(delta) * sin(h_phi))

        with

        C = (S0 * Delta_t * pi) / (h_phi * Delta_t * pi)

        or

        C = S0 / h_phi.


        Parameters
        ----------
        solar_constant : numpy.ndarray
            solar constant (W/m^2)
        distance_factor : numpy.ndarray
            distance factor (no units)
        hour_angle : numpy.ndarray
            hour angle (radians)
        latitude : numpy.ndarray
            latitude (radians)
        declination : numpy.ndarray
            declination (radians)

        Returns
        ----------
        insolation numpy.ndarray
            Solar declination paleo (radians)

        Examples
        ----------

        >>>    import numpy as np
        >>>    from pism_emulator.models.debm import DEBMModel

        >>>    debm = DEBMModel()

        >>>    solar_constant = np.array([1361.0])
        >>>    distance_factor = np.array([1.1])
        >>>    hour_angle = np.array([0.8])
        >>>    latitude = np.array([np.pi / 4])
        >>>    declination = np.array([np.pi / 8])
        >>>    insolation = debm.insolation(solar_constant, distance_factor, hour_angle, latitude, declination)
        >>>    insolation
        array([1282.10500694])
        """

        return (
            np.divide(solar_constant, hour_angle, where=hour_angle != 0)
            * distance_factor
            * (
                hour_angle * np.sin(latitude) * np.sin(declination)
                + np.cos(latitude) * np.cos(declination) * np.sin(hour_angle)
            )
        )

    def orbital_parameters(self, time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate orbital parameters (declination, distance_factor) given a time

        Parameters
        ----------
        time numpy.ndarray
            Time

        Returns
        ----------
        orbital_parameters tuple[numpy.ndarray, numpy.ndarray]
            Orbital parameters declination, distance_factor

        Examples
        ----------

        >>>    import numpy as np
        >>>    from pism_emulator.models.debm import DEBMModel

        >>>    time = np.array([2022.25])
        >>>    debm = DEBMModel()

        >>>    debm.orbital_parameters(time)
        (array([0.083785]), array([1.000671]))

        >>>    debm = DEBMModel(paleo_enabled=True)

        >>>    debm.orbital_parameters(time)
        (array([0.07843061]), array([0.99889585]))

        """

        year_fraction = self.year_fraction(time)

        if self.paleo_enabled:
            eccentricity = self.paleo_eccentricity
            perihelion_longitude = np.deg2rad(self.paleo_perihelion_longitude)
            solar_longitude = self.solar_longitude(
                year_fraction, eccentricity, perihelion_longitude
            )
            distance_factor = self.distance_factor_paleo(
                eccentricity, perihelion_longitude, solar_longitude
            )
            declination = self.solar_declination_paleo(
                np.deg2rad(self.paleo_obliquity), solar_longitude
            )

        else:
            declination = self.solar_declination_present_day(year_fraction)
            distance_factor = self.distance_factor_present_day(year_fraction)

        return declination, distance_factor

    def seconds_per_year(self) -> float:
        """
        Return the number of seconds in a year
        """

        return 3.15569259747e7

    def albedo(self, melt_rate: np.ndarray) -> np.ndarray:
        """
        Albedo parameterized as a function of the melt rate

        See equation 7 in Zeitz et al.

        This function converts meltrate from m/yr to m/s before using it in the albedo
        so we can keep the same albedo parameters as in Zeitz et al.

        Parameters
        ----------
        melt_rate numpy.ndarray
            Melt rate (m/yr)

        Returns
        ----------
        albedo numpy.ndarray
            Albedo (1)

        Examples
        ----------

        >>>    import numpy as np
        >>>    from pism_emulator.models.debm import DEBMModel

        >>>    melt_rate = np.array([1.0])

        >>>    debm = DEBMModel()
        >>>    albedo = debm.albedo(melt_rate)

        array([0.79704371])
        """

        return np.maximum(
            self.albedo_max + self.albedo_slope * melt_rate * self.ice_density,
            np.zeros_like(melt_rate) + self.albedo_min,
        )

    def atmosphere_transmissivity(self, elevation: np.ndarray) -> np.ndarray:
        """
        Atmosphere transmissivity (no units; acts as a scaling factor)

        See appendix A2 in Zeitz et al 2021.

        Parameters
        ----------
        elevation : numpy.ndarray
            elevation above the geoid (meters)

        Returns
        ----------
        transmissivity numpy.ndarray
            Transmissivity (1)

        Examples
        ----------

        >>>    import numpy as np
        >>>    from pism_emulator.models.debm import DEBMModel

        >>>    debm = DEBMModel()

        >>>    elevation = np.array([0.0, 1000.0, 2000.0])
        >>>    transmissivity = debm.atmosphere_transmissivity(elevation)
        >>>    transmissivity
        array([0.65 , 0.682, 0.714])
        """
        return self.tau_a_intercept + self.tau_a_slope * elevation
