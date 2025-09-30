import argparse
from types import SimpleNamespace
from typing import Sequence, Tuple

import lightning as pl
import numpy as np
import pint_xarray
import torch
import xarray as xr
from torch import Tensor

from pism_emulator.models.pdd import TorchPDDModel


class PDD(pl.LightningModule):
    """
    Positive Degree-Day (PDD) surface mass-balance component for Lightning.

    """

    def __init__(
        self,
        temp,
        precip,
        stdv,
        *,
        n_interpolate: int = 12,
        interpolate_rule: str = "linear",
        temp_snow: float = -1.0,
        temp_rain: float = 3.0,
    ):
        """
        Initialize the PDD module.

        Parameters
        ----------
        temp : array_like
            Near-surface air temperature time series (shape T×Y×X, 1×Y×X, Y×X, or scalar).
        precip : array_like
            Precipitation rate (same broadcastable shapes as ``temp``).
        stdv : array_like
            Standard deviation of near-surface air temperature (Kelvin).
        n_interpolate : int, optional
            Number of time points to interpolate to over a year (>=1). Default is 12.
        interpolate_rule : str, optional
            Interpolation kind for 1D interpolation (e.g., ``"linear"``, ``"cubic"``).
        temp_snow : float, optional
            Temperature (°C) at/below which all precipitation is snow. Default -1.0.
        temp_rain : float, optional
            Temperature (°C) at/above which all precipitation is rain. Default 3.0.
        """
        super().__init__()

        # Store only the small scalar hyperparameters; ignore the big arrays.
        self.save_hyperparameters(ignore=["temp", "precip", "stdv"])

        # Tensors
        temp_t = torch.as_tensor(temp)
        precip_t = torch.as_tensor(precip)
        stdv_t = torch.as_tensor(stdv)

        self.register_buffer("temp", temp_t)
        self.register_buffer("precip", precip_t)
        self.register_buffer("stdv", stdv_t)

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
        """
        parser = parent_parser.add_argument_group("PDD")
        parser.add_argument("--n_interpolate", type=int, default=12)

        return parent_parser

    def forward(self, x: Sequence[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Forward pass placeholder.

        Parameters
        ----------
        x : sequence of Tensor
            Parameter tensors in the order:
            ``(pdd_factor_snow, pdd_factor_ice, refreeze_snow, refreeze_ice, temp_snow, temp_rain)``.

        Returns
        -------
        tuple of Tensor
            Example return of (pdd, accumulation). Implement as needed.
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

        # parse model parameters for readability
        ddf_snow = pdd_factor_snow / 1000
        ddf_ice = pdd_factor_ice / 1000

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
        snow_refreeze_rate = refreeze_snow * snow_melt_rate
        ice_refreeze_rate = refreeze_ice * ice_melt_rate
        refreeze_rate = snow_refreeze_rate + ice_refreeze_rate
        runoff_rate = melt_rate - refreeze_rate
        inst_smb = accumulation_rate - runoff_rate

        # output
        return {
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

    def inst_pdd(self, temp: Tensor, stdv: Tensor) -> Tensor:
        """
        Compute instantaneous PDD (Calov & Greve, 2005).

        Parameters
        ----------
        temp : Tensor
            Near-surface air temperature (°C).
        stdv : Tensor
            Standard deviation of near-surface air temperature (K).

        Returns
        -------
        Tensor
            Instantaneous positive degree-days (°C·days).
        """
        positivepart = torch.clamp(temp, min=0.0)
        normtemp = temp / (torch.sqrt(torch.tensor(2.0, device=temp.device)) * stdv)
        calovgreve = stdv / torch.sqrt(
            torch.tensor(2.0 * torch.pi, device=temp.device)
        ) * torch.exp(-(normtemp**2)) + temp * 0.5 * (1.0 - torch.erf(-normtemp))
        teff = torch.where(stdv == 0.0, positivepart, calovgreve)
        return torch.clamp(teff, min=0.0) * 365.242198781

    def accumulation_rate(
        self, temp: Tensor, precip: Tensor, temp_snow, temp_rain
    ) -> Tensor:
        """
        Compute snowfall accumulation rate from temperature and precipitation.

        The snow fraction decreases linearly from 1→0 between ``temp_snow`` and ``temp_rain``.

        Parameters
        ----------
        temp : Tensor
            Near-surface air temperature (°C).
        precip : Tensor
            Precipitation rate (m yr⁻¹).

        Returns
        -------
        Tensor
            Accumulation rate (m yr⁻¹).
        """
        reduced_temp = (temp_rain - temp) / (temp_rain - temp_snow)
        snowfrac = torch.clamp(reduced_temp, 0.0, 1.0)
        return snowfrac * precip

    def _integrate(self, array: Tensor) -> Tensor:
        """
        Integrate a time series over one year.

        Parameters
        ----------
        array : Tensor
            Series with leading time axis.

        Returns
        -------
        Tensor
            Time-integrated field on the current device.
        """
        return torch.sum(array, dim=0) / max(self.hparams.n_interpolate - 1, 1)


def preprocess_time(ds: xr.Dataset) -> xr.Dataset:
    units = ds.time.attrs["units"]
    year = units.split(" since ")[1].split("-")[0]
    time = xr.date_range(f"{year}-01-01", periods=len(ds.time), freq="MS")
    ds["time"] = time
    return ds


ds = xr.open_mfdataset(
    "/Users/andy/base/pism-ragis/data/climate/mar/MARv3.14-monthly-ERA5-1975.nc",
    decode_times=False,
    preprocess=preprocess_time,
)
var_mapping = {
    "RUcorr": "runoff",
    "T2Mcorr": "temp",
    "SMBcorr": "smb",
    "MEcorr": "melt",
    "RF": "precip",
    "SF": "snowfall",
}
ds = ds.rename_vars(var_mapping)
ds = ds[list(var_mapping.values())]
ds["precip"].attrs.update({"units": "mm/month"})
ds["smb"].attrs.update({"units": "mm/month"})
ds["runoff"].attrs.update({"units": "mm/month"})
ds["snowfall"].attrs.update({"units": "mm/month"})
ds["melt"].attrs.update({"units": "mm/month"})
ds = ds.fillna(0).pint.quantify()
ds["precip"] = ds["precip"].pint.to("m/yr") * 12
ds["snowfall"] = ds["snowfall"].pint.to("m/yr")
ds["runoff"] = ds["runoff"].pint.to("m/yr")
ds["melt"] = ds["melt"].pint.to("m/yr")
ds["smb"] = ds["smb"].pint.to("m/yr")

pdd = TorchPDDModel()
temp = ds.temp.to_numpy()
precip = ds.precip.to_numpy()
stdv = np.zeros_like(temp)

tpdd = PDD(temp, precip, stdv)
