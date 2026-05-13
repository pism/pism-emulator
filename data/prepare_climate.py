# Copyright (C) 2024 Andy Aschwanden
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

"""
Prepare Climate.
"""

# pylint: disable=unused-import,broad-exception-caught,too-many-positional-arguments,redefined-builtin,redefined-outer-name,too-many-statements,no-member
# mypy: ignore-errors

import tarfile
import time as time_m
import zipfile
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import cf_xarray
import numpy as np
import pandas as pd
import pint  # pylint: disable=unused-import
import pint_xarray  # noqa: F401  (registers accessor) # pylint: disable=unused-import
import requests
import xarray as xr
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm


def unzip_files(
    files: list[str | Path],
    output_dir: str | Path = ".",
    overwrite: bool = False,
    max_workers: int = 4,
) -> list[Path]:
    """
    Unzip files in parallel.

    Parameters
    ----------
    files : list[str |  Path]
        List of file paths to unzip.
    output_dir : Union[str, Path], optional
        The directory where the unzipped files will be saved, by default ".".
    overwrite : bool, optional
        Whether to overwrite existing files, by default False.
    max_workers : int, optional
        The maximum number of threads to use for unzipping, by default 4.

    Returns
    -------
    List[Path]
        List of paths to the unzipped files.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for f in files:
            futures.append(
                executor.submit(unzip_file, f, str(output_dir), overwrite=overwrite)
            )
        for future in as_completed(futures):
            try:
                future.result()
            except (IOError, ValueError) as e:
                print(f"An error occurred: {e}", unzip_file)

    responses = list(Path(output_dir).rglob("*.nc"))
    return responses


def unzip_file(zip_path: str, extract_to: str, overwrite: bool = False) -> None:
    """
    Unzip a file to a specified directory with a progress bar and optional overwrite.

    Parameters
    ----------
    zip_path : str
        The path to the ZIP file.
    extract_to : str
        The directory where the contents will be extracted.
    overwrite : bool, optional
        Whether to overwrite existing files, by default False.
    """
    # Ensure the extract_to directory exists
    Path(extract_to).mkdir(parents=True, exist_ok=True)

    # Open the ZIP file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Get the list of file names in the zip file
        file_list = zip_ref.namelist()

        # Iterate over the file names with a progress bar
        for file in tqdm(file_list, desc="Extracting files", unit="file"):
            file_path = Path(extract_to) / file
            if not file_path.exists() or overwrite:
                zip_ref.extract(member=file, path=extract_to)


def process_hirham(
    data_dir: str | Path,
    output_file: str | Path,
    base_url: str,
    overwrite: bool = False,
    max_workers: int = 4,
    start_year: int = 1980,
    end_year: int = 2021,
) -> None:
    """
    Prepare and process HIRHAM data and save the output to a NetCDF file.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing the input data.
    output_file : str | Path
        Path to the output NetCDF file.
    base_url : str
        Base URL for downloading HIRHAM data.
    overwrite : bool, optional
        Whether to overwrite existing files, by default False.
    max_workers : int, optional
        Maximum number of parallel workers, by default 4.
    start_year : int, optional
        Starting year for processing, by default 1980.
    end_year : int, optional
        Ending year for processing, by default 2021.
    """
    print("Processing HIRHAM")

    hirham_dir = data_dir / Path("hirham")
    hirham_dir.mkdir(parents=True, exist_ok=True)
    hirham_nc_dir = hirham_dir / Path("nc")
    hirham_nc_dir.mkdir(parents=True, exist_ok=True)
    hirham_zip_dir = hirham_dir / Path("zip")
    hirham_zip_dir.mkdir(parents=True, exist_ok=True)

    responses = download_hirham(
        base_url,
        start_year,
        end_year,
        output_dir=hirham_zip_dir,
        max_workers=max_workers,
    )

    responses = unzip_files(
        responses,
        output_dir=hirham_nc_dir,
        overwrite=overwrite,
        max_workers=max_workers,
    )
    responses = sorted(responses)
    rho_w = xr.DataArray(1000).pint.quantify("kg m^-3")
    rho_w.name = "water_density"

    ds = xr.open_mfdataset(
        responses,
        preprocess=preprocess_time,
        parallel=False,
        engine="netcdf4",
        chunks={"time": 365, "rlat": -1, "rlon": -1},
        combine="nested",
        concat_dim="time",
    )
    ds.lat.attrs["units"] = "degree"
    ds.lon.attrs["units"] = "degree"
    ds = ds[["tas", "rainfall", "snfall", "rogl", "gld", "rfrz", "sn", "snmel"]]
    ds = ds.rename({"ncl4": "rlat", "ncl5": "rlon", "y": "rlat", "x": "rlon"})

    ds["sn"].attrs.update({"units": "m"})
    ds["rfrz"].attrs.update({"units": "m day^-1"})
    ds["rogl"].attrs.update({"units": "kg m^-2 day^-1"})
    ds["gld"].attrs.update({"units": "kg m^-2 day^-1"})
    ds["rainfall"].attrs.update({"units": "kg m^-2 day^-1"})
    ds["snfall"].attrs.update({"units": "kg m^-2 day^-1"})
    ds["snmel"].attrs.update({"units": "kg m^-2 day^-1"})
    ds = ds.pint.quantify()
    ds["tas"] = ds["tas"].pint.to("celsius")
    ds["precipitation"] = ds["rainfall"] + ds["snfall"]
    ds["precipitation"].attrs.update({"long_name": "precipitation_flux"})
    ds = ds.pint.dequantify()

    encoding = {var: {"_FillValue": False} for var in ["rlat", "rlon"]}
    # Use complevel=1 for faster writes (was 2)
    comp = {"zlib": True, "complevel": 1, "_FillValue": None}

    encoding_compression = {
        var: comp
        for var in ds.data_vars
        if var not in ("time", "time_bounds", "time_bnds")
    }
    encoding.update(encoding_compression)

    print("Computing dataset...")
    with ProgressBar():
        ds_computed = ds.compute()

    print(f"Writing to {output_file}")
    ds_computed.to_netcdf(output_file, encoding=encoding)


def preprocess_time(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """
    Replace a numeric time coordinate.

    This helper expects ``ds.time`` values to be either:
    1. Floating-point values where the integer part encodes the calendar date as ``YYYYMMDD``
    2. Datetime objects (in which case they're already processed)

    The fractional part is ignored and the resulting timestamps are set to
    **12:00 (noon)** on each date.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Input object with a ``time`` coordinate.
        Values must be numeric and interpretable as ``YYYYMMDD`` or already be datetime objects.

    Returns
    -------
    xr.Dataset or xr.DataArray
        A copy of ``ds`` with its ``time`` coordinate replaced by datetime values at noon.

    Raises
    ------
    KeyError
        If ``ds`` has no ``time`` coordinate.
    ValueError
        If time values cannot be parsed as ``YYYYMMDD``.

    Notes
    -----
    - The fractional day in ``ds.time`` is not used. The output time is always
      set to noon (12:00) to avoid edge cases around time zone offsets and
      midnight.
    - Preserves the full time dimension length.

    Examples
    --------
    >>> ds = ds.assign_coords(time=("time", [19800101.875, 19800102.875]))
    >>> out = preprocess_time(ds)
    >>> str(out.time.values[0])[:19]
    '1980-01-01T12:00:00'
    """
    time_values = ds.time.to_numpy()

    # Check if time is already datetime objects
    if np.issubdtype(time_values.dtype, np.datetime64):
        # Already datetimes, just set to noon
        new_times = pd.DatetimeIndex(time_values).normalize() + pd.to_timedelta(
            12, unit="h"
        )
    else:
        # Numeric format (YYYYMMDD.fraction) - process all values
        date_ints = np.floor(time_values).astype(int)
        # Parse YYYYMMDD for all dates
        base_dates = pd.to_datetime(date_ints.astype(str), format="%Y%m%d")
        # Set all to noon
        new_times = base_dates + pd.to_timedelta(12, unit="h")

    ds = ds.assign_coords(time=new_times)
    return ds


def download_file(url: str, output_path: Path) -> None:
    """
    Download a file from a URL with a progress bar.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    output_path : Path
        The local path where the downloaded file will be saved.
    """

    if output_path.exists():
        return
    response = requests.get(url, stream=True, timeout=10)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kilobyte
    with (
        open(output_path, "wb") as file,
        tqdm(total=total_size, unit="iB", unit_scale=True) as progress_bar,
    ):
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)


def download_hirham(
    base_url: str,
    start_year: int,
    end_year: int,
    output_dir: str | Path = ".",
    max_workers: int = 4,
) -> list[Path]:
    """
    Download HIRHAM files in parallel.

    Parameters
    ----------
    base_url : str
        The base URL for downloading HIRHAM data.
    start_year : int
        The starting year of the files to download.
    end_year : int
        The ending year of the files to download.
    output_dir : str | Path, optional
        The directory where the downloaded files will be saved, by default ".".
    max_workers : int, optional
        The maximum number of threads to use for downloading, by default 4.

    Returns
    -------
    list[Path]
        List of paths to the downloaded files.
    """
    print(f"Downloading HIRHAM5 from {base_url}")
    responses = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for year in range(start_year, end_year + 1):
            year_file = f"{year}.zip"
            url = base_url + year_file
            output_path = output_dir / Path(year_file)
            futures.append(executor.submit(download_file, url, output_path))
            responses.append(output_path)
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

    return responses


hirham_url = "http://ensemblesrt3.dmi.dk/data/prudence/temp/nichan/Daily2D_GrIS/"

xr.set_options(keep_attrs=True)


if __name__ == "__main__":
    __spec__ = None

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Prepare climate forcing."
    parser.add_argument("--years", nargs=2, type=int, default=[1980, 1989])
    parser.add_argument(
        "--n_jobs", help="""Number of parallel jobs.""", type=int, default=8
    )
    options = parser.parse_args()
    max_workers = options.n_jobs
    years = options.years

    result_dir = Path("climate")
    result_dir.mkdir(parents=True, exist_ok=True)

    start_year, end_year = years
    output_file = result_dir / Path(f"HIRHAM5-daily-ERA5_{start_year}_{end_year}.nc")
    process_hirham(
        data_dir=result_dir,
        start_year=start_year,
        end_year=end_year,
        output_file=output_file,
        base_url=hirham_url,
        max_workers=max_workers,
    )
