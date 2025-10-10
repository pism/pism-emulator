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
