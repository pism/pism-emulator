"""Tests for masking ice-free cells with the target's ``landice`` variable."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pism_emulator.datasets import PISMInterpolatedDataset

NY, NX, DX = 6, 8, 500.0  # model grid
FINE = 5  # observation grid is 5x finer (100 m)


def _write_case(tmp: Path, with_landice: bool = True) -> tuple[list[str], str, str]:
    """
    Write two model runs on a 500 m grid and a 100 m target with an ice mask.

    Parameters
    ----------
    tmp : pathlib.Path
        Directory for the files.
    with_landice : bool, optional
        Whether the target file contains ``landice``.

    Returns
    -------
    tuple[list[str], str, str]
        Training files, samples file and target file.
    """
    x = DX / 2 + DX * np.arange(NX)
    y = DX / 2 + DX * np.arange(NY)
    files = []
    for run in (0, 1):
        speed = np.full((1, NY, NX), 10.0 * (run + 1), dtype=np.float32)
        ds = xr.Dataset(
            {"velsurf_mag": (("time", "y", "x"), speed)},
            coords={"time": [0.0], "y": y, "x": x},
        )
        f = tmp / f"state_g500m_RGI-01_id_0_uq_{run}_0001-01-01.nc"
        ds.to_netcdf(f)
        files.append(str(f))

    xf = DX / FINE / 2 + DX / FINE * np.arange(NX * FINE)
    yf = DX / FINE / 2 + DX / FINE * np.arange(NY * FINE)
    v = np.full((NY * FINE, NX * FINE), 5.0, dtype=np.float32)  # speed also off ice
    ice = np.zeros_like(v, dtype=np.int8)
    ice[: 3 * FINE, : 4 * FINE] = 1  # ice covers the 3 x 4 lower-left model cells
    data = {"v": (("y", "x"), v)}
    if with_landice:
        data["landice"] = (("y", "x"), ice)
    target = tmp / "obs.nc"
    xr.Dataset(data, coords={"y": yf, "x": xf}).to_netcdf(target)

    samples = tmp / "uq.csv"
    pd.DataFrame({"uq": [0, 1], "a": [0.1, 0.2], "b": [1.0, 3.0]}).to_csv(
        samples, index=False
    )
    return files, str(samples), str(target)


def _dataset(files, samples, target, **kw) -> PISMInterpolatedDataset:
    return PISMInterpolatedDataset(
        training_files=files,
        samples_file=samples,
        target_file=target,
        target_var="v",
        y_lim=(1e-1, 100e3),
        parallel=False,
        **kw,
    )


def test_landice_masks_ice_free_cells(tmp_path: Path) -> None:
    """
    Only the 3 x 4 ice-covered model cells enter the dataset.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    files, samples, target = _write_case(tmp_path)
    ds = _dataset(files, samples, target)
    assert ds.target.sparse_idx_1d.size == 12
    assert ds.samples.Y.shape == (2, 12)
    rows, cols = ds.target.sparse_idx_2d
    assert rows.max() < 3 and cols.max() < 4
    assert ds.target.mask_2d.sum() == NY * NX - 12


@pytest.mark.parametrize("with_landice, mask_var", [(True, None), (False, "landice")])
def test_no_ice_mask_keeps_all_cells(
    tmp_path: Path, with_landice: bool, mask_var: str | None
) -> None:
    """
    Disabling the mask, or a target without ``landice``, keeps every cell.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    with_landice : bool
        Whether the target file contains ``landice``.
    mask_var : str or None
        Value passed as ``target_mask_var``.
    """
    files, samples, target = _write_case(tmp_path, with_landice=with_landice)
    ds = _dataset(files, samples, target, target_mask_var=mask_var)
    assert ds.target.sparse_idx_1d.size == NY * NX
