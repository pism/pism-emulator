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

    def __call__(self, temp, prec) -> dict:
        """Run the DEBM model.
        Use temperature, precipitation to compute accumulation and melt
        surface mass fluxes, and the resulting surface mass balance.
        *temp*: array_like
            Input near-surface air temperature in degrees Celcius.
        *prec*: array_like
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
        # FIXME use data arrays instead
        temp = np.asarray(temp)
        prec = np.asarray(prec)

        # expand arrays to the largest shape
        # FIXME use xarray auto-broadcasting instead
        maxshape = max(temp.shape, prec.shape, stdv.shape)
        temp = self._expand(temp, maxshape)
        prec = self._expand(prec, maxshape)

        # interpolate time-series
        # FIXME propagate data arrays, coordinates
        if (self.interpolate_n > 1) and (self.interpolate_n != temp.shape[0]):
            temp = self._interpolate(temp)
            prec = self._interpolate(prec)

        # compute accumulation and pdd
        accumulation_rate = self.accumulation_rate(temp, prec)

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

    def year_fraction(self, time: np.ndarray) -> np.ndarray:
        """
        Fractional part of a year

        Parameters
        ----------
        time : float, list of floats or numpy.ndarray
            Decimal time
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

        >>>    sigma = 4.2
        >>>    temperature = 2.1

        >>>    cgi = debm.CalovGreveIntegrand(sigma, temperature)
        >>>    cgi
        array(2.93074554)

        >>>    sigma = [4.2, 2.0]
        >>>    temperature = [1.0, 2.1]

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
        array(0.839038303283583)
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

        * @param[in] eccentricity eccentricity of the earth’s orbit (no units)
        * @param[in] perihelion_longitude perihelion longitude (radians)
        * @param[in] solar_longitude solar longitude (radians)
        """

        E = eccentricity

        assert np.isin(E, 1).any() == False, f"Division by zero, eccentricity is {E}"

        return (
            1.0 + E * np.cos(solar_longitude - perhelion_longitude) / (1.0 - E**2)
        ) ** 2.0

    def solar_declination_present_day(self, year_fraction: np.ndarray) -> np.ndarray:
        """
        * Solar declination (radian)
        *
        * Implements equation 2.2.10 from Liou (2002)
        """

        # These coefficients come from Table 2.2 in Liou 2002
        a0 = (0.006918,)
        a1 = (-0.399912,)
        a2 = (-0.006758,)
        a3 = (-0.002697,)
        b0 = (0.0,)
        b1 = (0.070257,)
        b2 = (0.000907,)
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
        * Solar declination (radians). This is the "paleo" version used when
        * the trigonometric expansion (equation 2.2.10 in Liou 2002) is not valid.
        *
        * The return value is in the range [-pi/2, pi/2].
        *
        * Implements equation in the text just above equation A1 in Zeitz et al.
        *
        * See also equation 2.2.4 of Liou (2002).
        """

        return np.arcsin(np.sin(obliquity * np.sin(solar_longitude)))

    def insolation(
        self,
        solar_constant: np.ndarray,
        distance_factor: np.ndarray,
        hour_angle: np.ndarray,
        latitude: np.ndarray,
        declination: np.ndarray,
    ) -> np.ndarray:
        """

        * Average top of atmosphere insolation (rate) during the daily melt period, in W/m^2.
        *
        * This should be equation 5 in Zeitz et al or equation 12 in Krebs-Kanzow et al, but both
        * of these miss a factor of Delta_t (day length in seconds) in the numerator.
        *
        * To confirm this, see the derivation of equation 2.2.21 in Liou and note that
        *
        * omega = 2 * pi (radian/day)
        *
        * or
        *
        * omega = (2 * pi / 86400) (radian/second).
        *
        * The correct equation should say
        *
        * S_Phi = A * B^2 * (h_phi * sin(phi) * sin(delta) + cos(phi) * cos(delta) * sin(h_phi)),
        *
        * where
        *
        * A = (S0 * Delta_t) / (Delta_t_Phi * pi),
        * B = d_bar / d.
        *
        * Note that we do not know Delta_t_phi but we can use equation 2 in Zeitz et al (or
        * equation 11 in Krebs-Kanzow et al) to get
        *
        * Delta_t_phi = h_phi * Delta_t / pi.
        *
        * This gives
        *
        * S_Phi = C * B^2 * (h_phi * sin(phi) * sin(delta) + cos(phi) * cos(delta) * sin(h_phi))
        *
        * with
        *
        * C = (S0 * Delta_t * pi) / (h_phi * Delta_t * pi)
        *
        * or
        *
        * C = S0 / h_phi.
        *
        * @param[in] solar constant solar constant, W/m^2
        * @param[in] distance_factor square of the ratio of the mean sun-earth distance to the current sun-earth distance (no units)
        * @param[in] hour_angle hour angle (radians) when the sun reaches the critical angle Phi
        * @param[in] latitude latitude (radians)
        * @param[in] declination declination (radians)
        *
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
        Calculate orbital parameters (declination, distance_factor) given a given time
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
        else:
            declination = self.solar_declination_present_day(year_fraction)
            distance_factor = self.distance_factor_present_day(year_fraction)
        return declination, distance_factor
