# Copyright (C) 2023 Andy Aschwanden, Maria Zeitz
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

from collections import OrderedDict

import torch


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
        interpolate_n: int = 12,
        device="cpu",
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
        self.device = device

    def CalovGreveIntegrand(sigma, temperature):
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

        if sigma == 0:
            temperature = torch.maximum(temperature, 0.0)

        Z = temperature / (torch.sqrt(2.0) * sigma)
        return (sigma / torch.sqrt(2.0 * torch.pi)) * torch.exp(-Z * Z) + (
            temperature / 2.0
        ) * torch.erfc(-Z)

    def hour_angle(phi, latitude, declination):
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


   def solar_longitude(year_fraction, eccentricity, perihelion_longitude):
       """
       * Solar longitude (radians) at current time in the year.
       *
       * @param[in] year_fraction year fraction (between 0 and 1)
       * @param[in] eccentricity eccentricity of the earth’s orbit (no units)
       * @param[in] perihelion_longitude perihelion longitude (radians)
       *
       * Implements equation A2 in Zeitz et al.
       """
       
       E   = eccentricity
       E2  = E * E
       E3  = E * E * E
       L_p = perihelion_longitude

       # Note: lambda = 0 at March equinox (80th day of the year)
       equinox_day_number = 80.0
       delta_lambda  = 2.0 * torch.pi * (year_fraction - equinox_day_number / 365.0);
       beta          = sqrt(1.0 - E2);

       lambda_m = (-2.0 * ((E / 2.0 + E3 / 8.0) * (1.0 + beta) * sin(-L_p) -
                         E2 / 4.0 * (1.0 / 2.0 + beta) * sin(-2.0 * L_p) +
                         E3 / 8.0 * (1.0 / 3.0 + beta) * sin(-3.0 * L_p)) +
                 delta_lambda)

       return (lambda_m +
               (2.0 * E - E3 / 4.0) * torch.sin(lambda_m - L_p) +
               (5.0 / 4.0)   * E2 * torch.sin(2.0 * (lambda_m - L_p)) +
               (13.0 / 12.0) * E3 * torch.sin(3.0 * (lambda_m - L_p)))
