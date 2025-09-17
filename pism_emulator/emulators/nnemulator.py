# Copyright (C) 2021-23 Andy Aschwanden, Douglas C Brinkerhoff
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

from argparse import ArgumentParser
from collections import OrderedDict
from typing import Dict, List, Tuple, Any
import math

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR, _LRScheduler

from pism_emulator.metrics import AreaAbsoluteError, area_absolute_error


def _kaiming_init(module: nn.Module) -> None:
    """Kaiming-uniform init for Linear layers; zero-init biases."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class MLPBlock(nn.Module):
    """A single MLP block: Linear -> Norm -> Dropout -> Activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        p_dropout: float = 0.0,
        norm: str = "batch",
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=True)

        if norm == "batch":
            self.norm: nn.Module = nn.BatchNorm1d(out_features)
        elif norm == "layer":
            self.norm = nn.LayerNorm(out_features)
        elif norm == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unknown norm '{norm}'")

        if activation.lower() == "relu":
            self.act: nn.Module = nn.ReLU()
        elif activation.lower() == "silu":
            self.act = nn.SiLU()
        elif activation.lower() == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation '{activation}'")

        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        """Forward pass."""
        return self.act(self.drop(self.norm(self.lin(x))))


# ---------- Deep residual emulator (configurable depth) ----------


class DNNEmulator(pl.LightningModule):
    """
    A deeper residual MLP emulator that maps parameter vectors ``x``
    to glacier field coefficients then back to physical space via
    a fixed basis ``V_hat`` and optional mean ``F_mean``.

    Notes
    -----
    - Uses residual skip inside each block (pre-activation style).
    - ``V_hat`` and ``F_mean`` are stored as non-trainable buffers so they land in state_dict but are not optimized.
    """

    def __init__(
        self,
        n_parameters,
        n_eigenglaciers,
        V_hat,
        F_mean: Tensor,
        area,
        hparams,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["n_parameters", "n_eigenglaciers", "V_hat", "F_mean", "area"]
        )

        hp = self.hparams["hparams"]
        if isinstance(hp, dict):
            self.hparams.update(hp)
        elif hasattr(hp, "__dict__"):
            self.hparams.update(vars(hp))

        width: int = int(self.hparams.get("width", 256))
        depth: int = int(self.hparams.get("depth", 4))
        p_drop: float = float(self.hparams.get("dropout", 0.1))
        norm: str = str(self.hparams.get("norm", "batch"))
        activation: str = str(self.hparams.get("activation", "relu"))

        # input projection
        self.inp = MLPBlock(
            n_parameters, width, p_drop, norm=norm, activation=activation
        )

        # residual stack
        self.blocks = nn.ModuleList(
            [
                MLPBlock(width, width, p_drop, norm=norm, activation=activation)
                for _ in range(depth)
            ]
        )

        # output head to coefficient space, then project back with V_hat
        self.head = nn.Linear(width, n_eigenglaciers, bias=False)

        # buffers (non-trainable, but saved with state)
        self.register_buffer("V_hat", V_hat, persistent=True)
        self.register_buffer("F_mean", F_mean, persistent=True)
        self.register_buffer("area", area, persistent=True)

        # metrics
        self.train_ae = AreaAbsoluteError()
        self.test_ae = AreaAbsoluteError()

        # init
        self.apply(_kaiming_init)

        # optional torch.compile for speed on PyTorch ≥ 2.0
        self._compiled = False
        if bool(self.hparams.get("compile", False)) and hasattr(torch, "compile"):
            try:
                self.forward = torch.compile(self.forward, dynamic=True)  # type: ignore[assignment]
                self._compiled = True
            except Exception:
                # Fall back silently if not supported on the current platform
                pass

    def forward(self, x: Tensor, add_mean: bool = False) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shape (batch, n_parameters).
        add_mean : bool
            If True, add ``F_mean`` back to the reconstruction.

        Returns
        -------
        Tensor
            Reconstructed fields ``F_pred`` of shape (batch, n_nodes).
        """
        z = self.inp(x)
        for block in self.blocks:
            z = block(z) + z  # residual
        coeffs = self.head(z)  # (batch, n_eigenglaciers)
        F_pred = coeffs @ self.V_hat.T  # (batch, n_nodes)
        if add_mean:
            F_pred = F_pred + self.F_mean
        return F_pred

    # ----- Lightning plumbing -----

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("DNNEmulator")
        parser.add_argument("--width", type=int, default=256)
        parser.add_argument("--depth", type=int, default=4)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument(
            "--norm", type=str, default="batch", choices=["batch", "layer", "none"]
        )
        parser.add_argument(
            "--activation", type=str, default="relu", choices=["relu", "silu", "gelu"]
        )
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument(
            "--compile", action="store_true", help="Use torch.compile if available"
        )
        return parent_parser

    def configure_optimizers(
        self,
    ) -> Tuple[List[Optimizer], List[Dict[str, _LRScheduler]]]:
        opt = torch.optim.Adam(
            self.parameters(), lr=float(self.hparams.learning_rate), weight_decay=0.0
        )
        sch = {"scheduler": ExponentialLR(opt, gamma=0.9975)}
        return [opt], [sch]

    def _shared_step(
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x, f, o, _ = batch
        f_pred = self.forward(x)
        # o has shape (..., 2) with (t0, t1); area_absolute_error wants o_0
        o_0 = o[..., 0]
        loss = area_absolute_error(f_pred, f, o_0, self.area)
        return x, f, f_pred, o, loss

    def training_step(
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        x, f, f_pred, o, loss = (*self._shared_step(batch),)
        o_0 = o[..., 0]
        self.log(
            "train_loss",
            self.train_ae(f_pred, f, o_0, self.area),
            sync_dist=True,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        x, f, f_pred, o, loss = (*self._shared_step(batch),)
        o_0 = o[..., 0]
        self.log(
            "val_loss",
            self.test_ae(f_pred, f, o_0, self.area),
            sync_dist=True,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return {"loss": loss}

    def test_step(
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        x, f, f_pred, o, loss = (*self._shared_step(batch),)
        o_0 = o[..., 0]
        self.log(
            "test_loss",
            self.test_ae(f_pred, f, o_0, self.area),
            sync_dist=True,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return {"loss": loss}


class NNEmulator(pl.LightningModule):
    def __init__(
        self,
        n_parameters,
        n_eigenglaciers,
        V_hat,
        F_mean,
        area,
        hparams,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["n_parameters", "n_eigenglaciers", "V_hat", "F_mean", "area"]
        )
        hp = self.hparams["hparams"]
        if isinstance(hp, dict):
            self.hparams.update(hp)
        elif hasattr(hp, "__dict__"):
            self.hparams.update(vars(hp))

        n_hidden_1 = self.hparams.get("n_hidden_1")
        n_hidden_2 = self.hparams.get("n_hidden_2")
        n_hidden_3 = self.hparams.get("n_hidden_3")
        n_hidden_4 = self.hparams.get("n_hidden_4")

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

        self.register_buffer("V_hat", V_hat, persistent=True)
        self.register_buffer("F_mean", F_mean, persistent=True)
        self.register_buffer("area", area, persistent=True)

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
        a_4 = self.norm_4(a_4)
        a_4 = self.dropout_4(a_4)
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
            "scheduler": ExponentialLR(optimizer, 0.9975),
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, f, o, _ = batch
        f_pred = self.forward(x)
        area = self.area
        loss = area_absolute_error(f_pred, f, o, area)

        return loss

    def validation_step(self, batch, batch_idx):
        x, f, o, o_0 = batch
        f_pred = self.forward(x)
        area = self.area

        self.log("train_loss", self.train_ae(f_pred, f, o, area))
        self.log("test_loss", self.test_ae(f_pred, f, o_0, area))

        return {"x": x, "f": f, "f_pred": f_pred, "o": o, "o_0": o_0}

    def on_validation_epoch_end(self):
        self.log(
            "train_loss",
            self.train_ae,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test_loss",
            self.test_ae,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_after_backward(self):
        if self.global_rank == 0:
            unused = [
                n
                for n, p in self.named_parameters()
                if p.requires_grad and p.grad is None
            ]
            if unused:
                print(
                    "UNUSED PARAMS THIS STEP:",
                    unused[:10],
                    "..." if len(unused) > 10 else "",
                )


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
        device="auto",
        all_vars: bool = False,
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
        self.all_vars = all_vars

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
            if i == 0:
                intermediate_snow_depth = accu_rate[i]
            else:
                intermediate_snow_depth = snow_depth[i - 1] + accu_rate[i]
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
        inst_smb = accu_rate - runoff_rate

        # output
        if not self.all_vars:
            output = {
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
        else:
            output = {
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
        return output

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
