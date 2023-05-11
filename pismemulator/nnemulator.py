# Copyright (C) 2021-22 Andy Aschwanden, Douglas C Brinkerhoff
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

import lightning as pl
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torchmetrics import Metric
from torchmetrics.utilities.checks import _check_same_shape
import xarray as xr

from pismemulator.metrics import (
    AbsoluteError,
    AreaAbsoluteError,
    absolute_error,
    area_absolute_error,
)


class PDDEmulator(pl.LightningModule):
    """
    The Neural Network emulator is adapted from Aschwanden & Brinkerhoff (2022),
    sans Principal Component Analysis.
    but maps (T, P, beta) -> A, M, R where
    T: temperature (monthly or daily)
    P: precipitation (monthly or daily)
    S: standard deviation (monthly or daily)
    beta: f_snow, f_ice, refreeze_snow, refreeze_ice, temp_snow, temp_rain

    A: Accumulation (annual)
    M: Melt (annual)
    R: Runoff (annual)
    F: Refreeze (annual)
    B: SMB (annual)
    """

    def __init__(
        self,
        n_parameters: int,
        n_outputs: int,
        hparams,
        *args,
        **kwargs,
    ):
        super().__init__()
        hparams["n_paramters"] = n_parameters
        hparams["n_outputs"] = n_outputs
        n_hidden = hparams["n_hidden"]
        n_layers = hparams["n_layers"]
        self.save_hyperparameters(hparams)

        if isinstance(n_hidden, int):
            n_hidden = [n_hidden] * (n_layers - 1)
        p = [0] + [0.5] * (n_layers - 2) + [0.3]

        # Inputs to hidden layer linear transformation
        self.l_first = nn.Linear(n_parameters, n_hidden[0])
        self.norm_first = nn.LayerNorm(n_hidden[0])
        self.dropout_first = nn.Dropout(p=p[0])

        models = []
        for n in range(n_layers - 2):
            models.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("Linear", nn.Linear(n_hidden[n], n_hidden[n + 1])),
                            ("LayerNorm", nn.LayerNorm(n_hidden[n + 1])),
                            ("Dropout", nn.Dropout(p=p[n + 1])),
                        ]
                    )
                )
            )
        self.dnn = nn.ModuleList(models)
        self.l_last = nn.Linear(n_hidden[-1], n_outputs, bias=False)

        self.train_ae = AbsoluteError()
        self.test_ae = AbsoluteError()

    def forward(self, x, add_mean=False):
        # Pass the input tensor through each of our operations

        a = self.l_first(x)
        a = self.norm_first(a)
        a = self.dropout_first(a)
        z = torch.relu(a)

        for dnn in self.dnn:
            a = dnn(z)
            z = torch.relu(a) + z

        return self.l_last(z)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("PDDEmulator")
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--n_hidden", type=int, default=128)
        parser.add_argument("--n_layers", type=int, default=5)
        parser.add_argument("--learning_rate", type=float, default=0.001)

        return parent_parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), self.hparams.learning_rate, weight_decay=0.0
        )
        # This is an approximation to Doug's version:
        scheduler = {
            "scheduler": ExponentialLR(optimizer, 0.99, verbose=False),
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, f, o, _ = batch
        f_pred = self.forward(x)
        loss = absolute_error(f_pred, f, o)

        return loss

    def validation_step(self, batch, batch_idx):
        x, f, o, o_0 = batch
        f_pred = self.forward(x)

        self.log("train_loss", self.train_ae(f_pred, f, o), sync_dist=True)
        self.log("test_loss", self.test_ae(f_pred, f, o_0), sync_dist=True)

        return {"x": x, "f": f, "f_pred": f_pred, "o": o, "o_0": o_0}

    def validation_epoch_end(self, outputs):

        self.log(
            "train_loss",
            self.train_ae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "test_loss",
            self.test_ae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )


class DNNEmulator(pl.LightningModule):
    def __init__(
        self,
        n_parameters: int,
        n_eigenglaciers: int,
        V_hat: Tensor,
        F_mean: Tensor,
        area: Tensor,
        hparams,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        n_layers = self.hparams.n_layers
        n_hidden = self.hparams.n_hidden

        if isinstance(n_hidden, int):
            n_hidden = [n_hidden] * n_layers
        p = [0] + [0.5] * (n_layers - 2) + [0.3]

        # Inputs to hidden layer linear transformation
        self.l_first = nn.Linear(n_parameters, n_hidden[0])
        self.norm_first = nn.LayerNorm(n_hidden[0])
        self.dropout_first = nn.Dropout(p=p[0])

        models = []
        for n in range(n_layers - 1):
            models.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("Linear", nn.Linear(n_hidden[n], n_hidden[n + 1])),
                            ("LayerNorm", nn.LayerNorm(n_hidden[n + 1])),
                            ("Dropout", nn.Dropout(p=p[n + 1])),
                        ]
                    )
                )
            )
        self.dnn = nn.ModuleList(models)
        self.l_last = nn.Linear(n_hidden[-1], n_eigenglaciers, bias=False)

        self.V_hat = torch.nn.Parameter(V_hat, requires_grad=False)
        self.F_mean = torch.nn.Parameter(F_mean, requires_grad=False)

        self.register_buffer("area", area)

        self.train_ae = AreaAbsoluteError()
        self.test_ae = AreaAbsoluteError()

    def forward(self, x, add_mean=False):
        # Pass the input tensor through each of our operations

        a = self.l_first(x)
        a = self.norm_first(a)
        a = self.dropout_first(a)
        z = torch.relu(a)

        for dnn in self.dnn:
            a = dnn(z)
            z = torch.relu(a) + z

        z_last = self.l_last(z)

        if add_mean:
            F_pred = z_last @ self.V_hat.T + self.F_mean
        else:
            F_pred = z_last @ self.V_hat.T

        return F_pred

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("NNEmulator")
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--n_hidden", type=int, default=128)
        parser.add_argument("--n_layers", type=int, default=4)
        parser.add_argument("--learning_rate", type=float, default=0.01)

        return parent_parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), self.hparams.learning_rate, weight_decay=0.0
        )
        # This is an approximation to Doug's version:
        scheduler = {
            "scheduler": ExponentialLR(optimizer, 0.9975, verbose=False),
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, f, o, _ = batch
        f_pred = self.forward(x)
        loss = area_absolute_error(f_pred, f, o, self.area)

        return loss

    def validation_step(self, batch, batch_idx):
        x, f, o, o_0 = batch
        f_pred = self.forward(x)

        self.log("train_loss", self.train_ae(f_pred, f, o, self.area))
        self.log("test_loss", self.test_ae(f_pred, f, o_0, self.area))

        return {"x": x, "f": f, "f_pred": f_pred, "o": o, "o_0": o_0}

    def validation_epoch_end(self, outputs):

        self.log(
            "train_loss",
            self.train_ae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test_loss",
            self.test_ae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )


class NNEmulator(pl.LightningModule):
    def __init__(
        self,
        n_parameters,
        n_eigenglaciers,
        V_hat,
        F_mean,
        area,
        hparams,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        n_hidden_1 = self.hparams.n_hidden_1
        n_hidden_2 = self.hparams.n_hidden_2
        n_hidden_3 = self.hparams.n_hidden_3
        n_hidden_4 = self.hparams.n_hidden_4

        # Inputs to hidden layer linear transformation
        self.l_1 = nn.Linear(n_parameters, n_hidden_1)
        self.norm_1 = nn.LayerNorm(n_hidden_1)
        self.dropout_1 = nn.Dropout(p=0.0)
        self.l_2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.norm_2 = nn.LayerNorm(n_hidden_2)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.l_3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.norm_3 = nn.LayerNorm(n_hidden_3)
        self.dropout_3 = nn.Dropout(p=0.5)
        self.l_4 = nn.Linear(n_hidden_3, n_hidden_4)
        self.norm_4 = nn.LayerNorm(n_hidden_3)
        self.dropout_4 = nn.Dropout(p=0.3)
        self.l_5 = nn.Linear(n_hidden_4, n_eigenglaciers, bias=False)

        self.V_hat = torch.nn.Parameter(V_hat, requires_grad=False)
        self.F_mean = torch.nn.Parameter(F_mean, requires_grad=False)

        self.register_buffer("area", area)

        self.train_ae = AreaAbsoluteError()
        self.test_ae = AreaAbsoluteError()

    def forward(self, x, add_mean=False):
        # Pass the input tensor through each of our operations

        a_1 = self.l_1(x)
        a_1 = self.norm_1(a_1)
        a_1 = self.dropout_1(a_1)
        z_1 = torch.relu(a_1)

        a_2 = self.l_2(z_1)
        a_2 = self.norm_2(a_2)
        a_2 = self.dropout_2(a_2)
        z_2 = torch.relu(a_2) + z_1

        a_3 = self.l_3(z_2)
        a_3 = self.norm_3(a_3)
        a_3 = self.dropout_3(a_3)
        z_3 = torch.relu(a_3) + z_2

        a_4 = self.l_4(z_3)
        a_4 = self.norm_3(a_4)
        a_4 = self.dropout_3(a_4)
        z_4 = torch.relu(a_4) + z_3

        z_5 = self.l_5(z_4)
        if add_mean:
            F_pred = z_5 @ self.V_hat.T + self.F_mean
        else:
            F_pred = z_5 @ self.V_hat.T

        return F_pred

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("NNEmulator")
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--n_hidden_1", type=int, default=128)
        parser.add_argument("--n_hidden_2", type=int, default=128)
        parser.add_argument("--n_hidden_3", type=int, default=128)
        parser.add_argument("--n_hidden_4", type=int, default=128)
        parser.add_argument("--learning_rate", type=float, default=0.01)

        return parent_parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), self.hparams.learning_rate, weight_decay=0.0
        )
        # This is an approximation to Doug's version:
        scheduler = {
            "scheduler": ExponentialLR(optimizer, 0.9975, verbose=False),
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, f, o, _ = batch
        f_pred = self.forward(x)
        loss = area_absolute_error(f_pred, f, o, self.area)

        return loss

    def validation_step(self, batch, batch_idx):
        x, f, o, o_0 = batch
        f_pred = self.forward(x)

        self.log("train_loss", self.train_ae(f_pred, f, o, self.area))
        self.log("test_loss", self.test_ae(f_pred, f, o_0, self.area))

        return {"x": x, "f": f, "f_pred": f_pred, "o": o, "o_0": o_0}

    def validation_epoch_end(self, outputs):

        self.log(
            "train_loss",
            self.train_ae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test_loss",
            self.test_ae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )


class PDDStddevModel(torch.nn.modules.Module):
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
    *std_dev* : float
        Standard deviation of temperature in expectation integral.
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
        std_dev: float = 0.0,
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
        self.std_dev = std_dev
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

    @property
    def std_dev(self):
        return self._std_dev

    @refreeze_ice.setter
    def std_dev(self, value):
        self._std_deve = value

    def forward(self, temp, prec, stdv=0.0):
        """Run the positive degree day model.

        Use temperature, precipitation, and standard deviation of temperature
        to compute the number of positive degree days, accumulation and melt
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

        Return the number of positive degree days ('pdd'), surface mass balance
        ('smb'), and many other output variables in a dictionary.
        """

        device = self.device
        # ensure numpy arrays
        temp = torch.asarray(temp, device=device)
        prec = torch.asarray(prec, device=device)

        # expand arrays to the largest shape
        maxshape = max(temp.shape, prec.shape)
        temp = self._expand(temp, maxshape)
        prec = self._expand(prec, maxshape)

        # interpolate time-series
        if self.interpolate_n >= 1:
            temp = self._interpolate(temp)
            prec = self._interpolate(prec)

        # compute accumulation and pdd
        accu_rate = self.accu_rate(temp, prec)
        inst_pdd = self.inst_pdd(temp)

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
            if i > 0:
                snow_depth[i] = snow_depth[i - 1]
            snow_depth[i] += accu_rate[i]
            # parse model parameters for readability

            pot_snow_melt = ddf_snow * inst_pdd[i]
            # effective snow melt can't exceed amount of snow
            s_min = torch.minimum(snow_depth[i], pot_snow_melt)
            snow_melt_rate[i] = s_min

            # ice melt is proportional to excess snow melt
            ice_melt_rate[i] = (pot_snow_melt - snow_melt_rate[i]) * ddf_ice / ddf_snow

            snow_depth[i] -= snow_melt_rate[i]

        melt_rate = snow_melt_rate + ice_melt_rate
        snow_refreeze_rate = self.refreeze_snow * snow_melt_rate
        ice_refreeze_rate = self.refreeze_ice * ice_melt_rate
        refreeze_rate = snow_refreeze_rate + ice_refreeze_rate
        runoff_rate = melt_rate - refreeze_rate
        inst_smb = accu_rate - runoff_rate

        # output
        return {
            "temp": temp,
            "prec": prec,
            "stdv": stdv,
            "inst_pdd": inst_pdd,
            "accu_rate": accu_rate,
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
            "accu": self._integrate(accu_rate),
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
            res = np.asarray([array[0]] * shape[0])
        elif array.shape == shape[1:]:
            res = np.asarray([array] * shape[0])
        elif array.shape == ():
            res = array * torch.ones(shape)
        else:
            raise ValueError(
                "could not expand array of shape %s to %s" % (array.shape, shape)
            )
        return res

    def _integrate(self, array):
        """Integrate an array over one year"""
        return torch.sum(array, axis=0) / (self.interpolate_n - 1)

    def _interpolate(self, array):
        """Interpolate an array through one year."""

        from scipy.interpolate import interp1d

        rule = self.interpolate_rule
        npts = self.interpolate_n
        oldx = (torch.arange(len(array) + 2, device=self.device) - 0.5) / len(array)
        oldy = torch.vstack((array[-1], array, array[0]))
        newx = (torch.arange(npts) + 0.5) / npts  # use 0.0 for PISM-like behaviour
        newy = interp1d(oldx.cpu(), oldy.cpu(), kind=rule, axis=0)(newx)

        return torch.from_numpy(newy).to(self.device)

    def inst_pdd(self, temp):
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
        stdv = torch.tensor(self.std_dev)
        normtemp = temp / (torch.sqrt(torch.tensor(2)) * stdv)
        calovgreve = stdv / torch.sqrt(torch.tensor(2) * torch.pi) * torch.exp(
            -(normtemp**2)
        ) + temp / 2 * torch.erfc(-normtemp)

        # use positive part where sigma is zero and Calov and Greve elsewhere
        teff = torch.where(stdv == 0.0, positivepart, calovgreve)

        # convert to degree-days
        return teff * 365.242198781

    def accu_rate(self, temp, prec):
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
        snowfrac = torch.clip(reduced_temp, 0, 1)

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
        ddf_snow = self.pdd_factor_snow / 1000
        ddf_ice = self.pdd_factor_ice / 1000

        # compute a potential snow melt
        pot_snow_melt = ddf_snow * pdd

        # effective snow melt can't exceed amount of snow
        snow_melt = torch.minimum(snow, pot_snow_melt)

        # ice melt is proportional to excess snow melt
        ice_melt = (pot_snow_melt - snow_melt) * ddf_ice / ddf_snow

        # return melt rates
        return (snow_melt, ice_melt)


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

        device = self.device
        # ensure numpy arrays
        temp = torch.asarray(temp, device=device)
        prec = torch.asarray(prec, device=device)
        stdv = torch.asarray(stdv, device=device)

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

        # compute accumulation and pdd
        accu_rate = self.accu_rate(temp, prec)
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
            if i > 0:
                snow_depth[i] = snow_depth[i - 1]
            snow_depth[i] += accu_rate[i]
            # parse model parameters for readability

            pot_snow_melt = ddf_snow * inst_pdd[i]
            # effective snow melt can't exceed amount of snow
            s_min = torch.minimum(snow_depth[i], pot_snow_melt)
            snow_melt_rate[i] = s_min

            # ice melt is proportional to excess snow melt
            ice_melt_rate[i] = (pot_snow_melt - snow_melt_rate[i]) * ddf_ice / ddf_snow

            snow_depth[i] -= snow_melt_rate[i]

        melt_rate = snow_melt_rate + ice_melt_rate
        snow_refreeze_rate = self.refreeze_snow * snow_melt_rate
        ice_refreeze_rate = self.refreeze_ice * ice_melt_rate
        refreeze_rate = snow_refreeze_rate + ice_refreeze_rate
        runoff_rate = melt_rate - refreeze_rate
        inst_smb = accu_rate - runoff_rate

        # output
        return {
            "temp": temp,
            "prec": prec,
            "stdv": stdv,
            "inst_pdd": inst_pdd,
            "accu_rate": accu_rate,
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
            "accu": self._integrate(accu_rate),
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
            res = np.asarray([array[0]] * shape[0])
        elif array.shape == shape[1:]:
            res = np.asarray([array] * shape[0])
        elif array.shape == ():
            res = array * torch.ones(shape)
        else:
            raise ValueError(
                "could not expand array of shape %s to %s" % (array.shape, shape)
            )
        return res

    def _integrate(self, array):
        """Integrate an array over one year"""
        return torch.sum(array, axis=0) / (self.interpolate_n - 1)

    def _interpolate(self, array):
        """Interpolate an array through one year."""

        from scipy.interpolate import interp1d

        rule = self.interpolate_rule
        npts = self.interpolate_n
        oldx = (torch.arange(len(array) + 2, device=self.device) - 0.5) / len(array)
        oldy = torch.vstack((array[-1], array, array[0]))
        newx = (torch.arange(npts) + 0.5) / npts  # use 0.0 for PISM-like behaviour
        newy = interp1d(oldx.cpu(), oldy.cpu(), kind=rule, axis=0)(newx)

        return torch.from_numpy(newy).to(self.device)

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
        ) + temp / 2 * torch.erfc(-normtemp)

        # use positive part where sigma is zero and Calov and Greve elsewhere
        teff = torch.where(stdv == 0.0, positivepart, calovgreve)

        # convert to degree-days
        return teff * 365.242198781

    def accu_rate(self, temp, prec):
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
        snowfrac = torch.clip(reduced_temp, 0, 1)

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
        ddf_snow = self.pdd_factor_snow / 1000
        ddf_ice = self.pdd_factor_ice / 1000

        # compute a potential snow melt
        pot_snow_melt = ddf_snow * pdd

        # effective snow melt can't exceed amount of snow
        snow_melt = torch.minimum(snow, pot_snow_melt)

        # ice melt is proportional to excess snow melt
        ice_melt = (pot_snow_melt - snow_melt) * ddf_ice / ddf_snow

        # return melt rates
        return (snow_melt, ice_melt)


class PDDModel():
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
        interpolate_n: int = 12,
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

        # ensure numpy arrays
        temp = np.asarray(temp)
        prec = np.asarray(prec)
        stdv = np.asarray(stdv)

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

        # compute accumulation and pdd
        accu_rate = self.accu_rate(temp, prec)
        inst_pdd = self.inst_pdd(temp, stdv)

        # initialize snow depth, melt and refreeze rates
        snow_depth = np.zeros_like(temp)
        snow_melt_rate = np.zeros_like(temp)
        ice_melt_rate = np.zeros_like(temp)
        snow_refreeze_rate = np.zeros_like(temp)
        ice_refreeze_rate = np.zeros_like(temp)

        # parse model parameters for readability
        ddf_snow = self.pdd_factor_snow / 1000
        ddf_ice = self.pdd_factor_ice / 1000

        for i in range(len(temp)):
            if i > 0:
                snow_depth[i] = snow_depth[i - 1]
            snow_depth[i] += accu_rate[i]
            # parse model parameters for readability

            pot_snow_melt = ddf_snow * inst_pdd[i]
            # effective snow melt can't exceed amount of snow
            s_min = np.minimum(snow_depth[i], pot_snow_melt)
            snow_melt_rate[i] = s_min

            # ice melt is proportional to excess snow melt
            ice_melt_rate[i] = (pot_snow_melt - snow_melt_rate[i]) * ddf_ice / ddf_snow

            snow_depth[i] -= snow_melt_rate[i]

        melt_rate = snow_melt_rate + ice_melt_rate
        snow_refreeze_rate = self.refreeze_snow * snow_melt_rate
        ice_refreeze_rate = self.refreeze_ice * ice_melt_rate
        refreeze_rate = snow_refreeze_rate + ice_refreeze_rate
        runoff_rate = melt_rate - refreeze_rate
        inst_smb = accu_rate - runoff_rate

        # output
        return {
            "temp": temp,
            "prec": prec,
            "stdv": stdv,
            "inst_pdd": inst_pdd,
            "accu_rate": accu_rate,
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
            "accu": self._integrate(accu_rate),
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
        oldy = np.vstack((array[-1], array, array[0]))
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
        normtemp = temp / (np.sqrt(2) * stdv)
        calovgreve = stdv / np.sqrt(2 * np.pi) * np.exp(
            -(normtemp**2)
        ) + temp / 2 * scipy.special.erfc(-normtemp)

        # use positive part where sigma is zero and Calov and Greve elsewhere
        teff = np.where(stdv == 0.0, positivepart, calovgreve)

        # convert to degree-days
        return teff * 365.242198781

    def accu_rate(self, temp, prec):
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
        ddf_snow = self.pdd_factor_snow / 1000
        ddf_ice = self.pdd_factor_ice / 1000

        # compute a potential snow melt
        pot_snow_melt = ddf_snow * pdd

        # effective snow melt can't exceed amount of snow
        snow_melt = np.minimum(snow, pot_snow_melt)

        # ice melt is proportional to excess snow melt
        ice_melt = (pot_snow_melt - snow_melt) * ddf_ice / ddf_snow

        # return melt rates
        return (snow_melt, ice_melt)
