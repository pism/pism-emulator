# Copyright (C) 2025 Andy Aschwanden
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

# pylint: disable=redefined-builtin

"""
Download and process observed speeds.
"""

import re
from pathlib import Path

import earthaccess
import rioxarray as rxr
import xarray as xr


def download_earthaccess(
    filter_str: str | None = None, result_dir: Path | str = ".", **kwargs
) -> list:
    """
    Download datasets via Earthaccess.

    Parameters
    ----------
    filter_str : str, optional
        A string to filter the search results. Default is None.
    result_dir : Union[Path, str], optional
        The directory where the downloaded files will be saved. Default is ".".
    **kwargs : dict
        Additional keyword arguments to pass to the Earthaccess search function.

    Returns
    -------
    List
        A list of paths to the downloaded files.
    """
    p = Path(result_dir)
    p.mkdir(parents=True, exist_ok=True)

    earthaccess.login()
    results = earthaccess.search_data(**kwargs)
    if filter_str is not None:
        results = [
            granule
            for granule in results
            if filter_str
            in granule["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]
        ]
    earthaccess.get_s3_credentials(results=results)
    return earthaccess.download(results, p)


def main():
    """
    Download and prepare observed velocities.
    """
    print("Preparing Velocities")
    filter_str = "greenland_vel_mosaic250"
    result_dir = Path("observed_speeds")
    short_name = "NSIDC-0670"
    results = download_earthaccess(
        short_name=short_name, filter_str=filter_str, result_dir=result_dir
    )

    pat = re.compile(r"_mosaic\d+_(?P<comp>[a-z]{2})(?=_)")
    das = []
    for r in results:
        if (m := pat.search(str(r))) is None:
            raise ValueError(f"No component code found in {r!r}")
        varname = m.group("comp")
        da = (
            rxr.open_rasterio(r)
            .rio.write_nodata(-2e9, inplace=True)
            .squeeze()
            .drop_vars("band")
        )
        da = da.where(da != da.rio.nodata)
        da.name = varname
        da.attrs.update({"units": "m/yr"})
        das.append(da)
    ds = xr.merge(das, compat="override")
    ds["velsurf_mag"] = (ds["vx"] ** 2 + ds["vy"] ** 2) ** (1.0 / 2)
    ds["velsurf_mag_error"] = (ds["ex"] ** 2 + ds["ey"] ** 2) ** (1.0 / 2)
    ds["velsurf_mag"].attrs.update({"units": "m/yr"})
    ds["velsurf_mag_error"].attrs.update({"units": "m/yr"})

    ds_t = xr.open_dataset(
        "speeds_v2/velsurf_mag_gris_g1800m_v4_id_0_0_50.nc"
    ).squeeze()
    ds_t.rio.write_crs("EPSG:3413", inplace=True)
    ds_p = ds.rio.reproject_match(ds_t)
    ds_p.to_netcdf("observed_speeds/greenland_vel_mosaic250_v1_g1800m.nc")


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
