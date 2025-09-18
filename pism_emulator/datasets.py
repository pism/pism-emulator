from __future__ import annotations

import os
import re
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from itertools import repeat
from os import PathLike
from os.path import join
from pathlib import Path
from time import time
from typing import Any, Final, Union, cast
from torch.utils.data import get_worker_info

import dask

# add at top-level
import netCDF4 as nc
import numpy as np
import pandas as pd
import torch
import xarray as xr
from numpy.typing import NDArray
from tqdm.auto import tqdm as _tqdm

from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only


ID_RE: Final[re.Pattern[str]] = re.compile(r"id_(?P<id>\d+)_")
from torch.utils.data import get_worker_info


def _is_global_zero() -> bool:
    # works with torchrun and Lightning
    r = os.environ.get("RANK", os.environ.get("GLOBAL_RANK", "0"))
    try:
        return int(r) == 0
    except ValueError:
        return True  # be permissive if unset


def _is_worker_zero() -> bool:
    wi = get_worker_info()
    return (wi is None) or (wi.id == 0)


def tqdm_rank0(*args, **kwargs):
    """tqdm that only shows on global rank 0 and worker 0."""
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("leave", False)
    # disable everywhere except global rank 0 + worker 0
    if not (_is_global_zero() and _is_worker_zero()):
        kwargs["disable"] = True
    return _tqdm(*args, **kwargs)


def id_key(path: str | os.PathLike[str]) -> int:
    """Return the integer id from a filename like '*id_123_*'. Raises if missing."""
    m = ID_RE.search(Path(path).name)
    if m is None:
        raise ValueError(
            f"Could not parse id from {path!s} using pattern {ID_RE.pattern!r}"
        )
    return int(m.group("id"))


def parse_id_from_path(p: str | Path) -> int:
    """Extract integer id from a filename like '...id_123_...'. Raises on failure."""
    m = ID_RE.search(Path(p).name)
    if not m:
        raise ValueError(
            f"Could not parse id from {p!s} using pattern {ID_RE.pattern!r}"
        )
    return int(m.group("id"))


def _read_one_nc4(
    path: str | PathLike[str],
    var: str,
    step: int,
    idx1d: NDArray[np.intp],
    eps: float,
) -> NDArray[np.float32]:
    """
    CF-compatible fast reader: auto mask/scale ON, masked->NaN, then NaN->eps.
    Matches xarray ``open_dataset(..., decode_cf=True)`` semantics used in PISMDataset.
    """
    if step < 1:
        raise ValueError(f"'step' must be >= 1, got {step}")

    ds = nc.Dataset(path, "r")
    try:
        v = ds.variables[var]
        # Match xarray CF decode: enable both masking and scale/offset application
        v.set_auto_maskandscale(True)

        dims: tuple[str, ...] = v.dimensions
        sizes = {d: ds.dimensions[d].size for d in dims}
        try:
            yi = dims.index("y")
            xi = dims.index("x")
        except ValueError as e:
            raise KeyError(
                f'"{var}" in {path} is missing "y" or "x" dim: {dims}'
            ) from e

        slicers: list[int | slice] = []
        for k, d in enumerate(dims):
            if k == yi:
                slicers.append(slice(0, None, step))
            elif k == xi:
                slicers.append(slice(0, None, step))
            else:
                n = sizes[d]
                slicers.append(0 if n == 1 else slice(0, 1))  # first entry if >1
        arr = v[tuple(slicers)]
        # If a non-(y,x) leading axis of length 1 was kept, squeeze it now
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[-2], arr.shape[-1])  # (y, x)

        # Convert MaskedArray → ndarray with NaNs where masked, then to float32
        if np.ma.isMaskedArray(arr):
            arr = np.ma.filled(arr, np.nan)
        arr = np.asarray(arr, dtype=np.float32, order="C")

    finally:
        ds.close()

    # Replace NaNs with epsilon, exactly like your old code
    np.nan_to_num(arr, nan=eps, copy=False)

    # Gather sparse nodes in C-order (same as data[self.sparse_idx_2d].flatten())
    out = np.take(arr.ravel(), idx1d)
    return cast(NDArray[np.float32], out)


def _sniff_engine(path: str) -> str:
    """Choose an xarray engine from the file signature."""
    # HDF5 magic: \x89HDF\r\n\x1a\n ; NetCDF3: b"CDF\001" or b"CDF\002"
    with open(path, "rb") as f:
        sig = f.read(8)
    if sig.startswith(b"\x89HDF") or sig.startswith(b"HDF"):
        return (
            "h5netcdf"  # NetCDF-4/HDF5 -> open with h5netcdf over netcdf4 for stability
        )
    if sig.startswith(b"CDF"):
        return "scipy"  # NetCDF-3 -> scipy engine
    # fallback: try h5netcdf first
    return "h5netcdf"


class PISMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        training_files,
        samples_file,
        target_file,
        training_var="velsurf_mag",
        target_var="velsurf_mag",
        target_corr_var="thickness",
        target_error_var="velsurf_mag_error",
        target_corr_threshold=25.0,
        thin=1,
        normalize_x=True,
        log_y=True,
        threshold=100e3,
        epsilon=0.0,
        verbose=False,
        target_engine=None,
        training_engine=None,
        parallel=True,
        chunks_after={"y": 512, "x": 512, "exp_id": 1},
        dask_scheduler=None,
    ):
        self.training_files = sorted(
            map(
                str,
                (
                    training_files
                    if isinstance(training_files, (list, tuple))
                    else Path().glob(training_files)
                ),
            )
        )
        self.samples_file = samples_file
        self.target_file = target_file
        self.training_var = training_var
        self.target_var = target_var
        self.target_corr_var = target_corr_var
        self.target_error_var = target_error_var
        self.target_corr_threshold = float(target_corr_threshold)
        self.thin = int(thin)
        self.normalize_x = normalize_x
        self.log_y = log_y
        self.threshold = float(threshold)
        self.epsilon = float(epsilon)
        self.verbose = verbose

        if dask_scheduler:
            import dask

            dask.config.set(scheduler=dask_scheduler)

        # Engines: sniff if not provided
        self.target_engine = target_engine or _sniff_engine(self.target_file)
        # Avoid netcdf4 for multi-file; prefer h5netcdf
        self.training_engine = training_engine or "h5netcdf"

        self.parallel = bool(parallel)
        self.chunks_after = chunks_after or {}

        self._load_target_and_mask()
        self._load_training_and_samples()

    # ---------------------------
    # PyTorch Dataset interface
    # ---------------------------
    def __getitem__(self, i):
        return self.X[i], self.Y[i]

    def __len__(self):
        return min(len(self.X), len(self.Y))

    # ---------------------------
    # Helpers
    # ---------------------------
    def _thin_sel(self):
        s = self.thin
        return {"x": slice(None, None, s), "y": slice(None, None, s)}

    def _parse_id(self, path):
        m = re.search(r"id_(\d+)_", Path(path).name)
        if not m:
            raise ValueError(f"Could not parse id from filename: {path}")
        return int(m.group(1))

    # ---------------------------
    # Target & mask
    # ---------------------------
    def _load_target_and_mask(self):
        if self.verbose:
            rank_zero_info(
                f"Loading target {self.target_file} (engine={self.target_engine})"
            )

        # open without forcing chunks to avoid “separate stored chunks” warnings
        ds = xr.open_dataset(
            self.target_file, decode_times=False, engine=self.target_engine
        ).isel(**self._thin_sel())

        if self.target_var not in ds:
            raise KeyError(f"'{self.target_var}' not found in target file")

        targ_da = ds[self.target_var].squeeze(drop=True)
        mask = targ_da.isnull()

        self.target_has_error = self.target_error_var in ds
        self.target_has_corr = self.target_corr_var in ds

        if self.target_has_corr:
            corr_da = ds[self.target_corr_var].squeeze(drop=True)
            mask = xr.where(corr_da < self.target_corr_threshold, True, mask)

        # Compute mask once; use integer indices afterward
        mask_np = mask.compute().values.astype(bool)
        self.mask_2d = mask_np
        self.ny = int(ds.sizes["y"])
        self.nx = int(ds.sizes["x"])

        self.sparse_idx_2d = np.where(~mask_np)
        self.sparse_idx_1d = np.ravel_multi_index(
            self.sparse_idx_2d, (self.ny, self.nx)
        )

        targ_vec = (
            targ_da.stack(node=("y", "x"))
            .isel(node=self.sparse_idx_1d)
            .fillna(self.epsilon)
            .astype("float32")
            .compute()
            .values
        )
        self.Y_target = torch.from_numpy(targ_vec)

        if self.target_has_error:
            err_vec = (
                ds[self.target_error_var]
                .squeeze(drop=True)
                .stack(node=("y", "x"))
                .isel(node=self.sparse_idx_1d)
                .fillna(self.epsilon)
                .astype("float32")
                .compute()
                .values
            )
            self.Y_target_error = torch.from_numpy(err_vec)
            self.Y_target_error_2d = (
                ds[self.target_error_var]
                .squeeze(drop=True)
                .fillna(self.epsilon)
                .astype("float32")
                .compute()
                .values
            )

        if self.target_has_corr:
            corr_vec = (
                ds[self.target_corr_var]
                .squeeze(drop=True)
                .stack(node=("y", "x"))
                .isel(node=self.sparse_idx_1d)
                .fillna(self.epsilon)
                .astype("float32")
                .compute()
                .values
            )
            self.Y_target_corr = torch.from_numpy(corr_vec)
            self.Y_target_corr_2d = (
                ds[self.target_corr_var]
                .squeeze(drop=True)
                .fillna(self.epsilon)
                .astype("float32")
                .compute()
                .values
            )

        self.grid_resolution = float(abs(ds["x"][1] - ds["x"][0]))

        y2d = np.zeros((self.ny, self.nx), dtype=np.float32)
        y2d.flat[self.sparse_idx_1d] = self.Y_target.numpy()
        self.Y_target_2d = np.ma.array(data=y2d, mask=self.mask_2d)

        ds.close()

    def _load_training_and_samples(self):
        if self.verbose:
            rank_zero_info("  Loading samples & training responses... (netCDF4 direct)")

        # Map files to ids and align with samples
        ids = [self._parse_id(p) for p in self.training_files]
        files_by_id = {i: p for i, p in zip(ids, self.training_files)}

        samples = (
            pd.read_csv(self.samples_file, delimiter=",", skipinitialspace=True)
            .sort_values("id")
            .set_index("id", drop=True)
        )

        keep_ids = [i for i in sorted(files_by_id) if i in samples.index]
        if self.verbose:
            missing = sorted(set(samples.index).difference(keep_ids))
            if missing:
                rank_zero_info(f"  Missing runs (dropping from samples): {missing}")

        samples = samples.loc[keep_ids]

        # Preallocate response: (n_runs, n_nodes)
        n_runs = len(keep_ids)
        n_nodes = self.sparse_idx_1d.size
        response = np.empty((n_runs, n_nodes), dtype=np.float32)

        step = self.thin
        idx1d = self.sparse_idx_1d

        # --- parallel fill, replacing the old for-loop ---
        files_iter = (files_by_id[i] for i in keep_ids)
        vars_iter = repeat(self.training_var)
        steps_iter = repeat(step)
        idx_iter = repeat(idx1d)
        eps_iter = repeat(self.epsilon)

        start_time = time()

        total = len(keep_ids)
        workers: int = min(8, os.cpu_count() or 1)

        with ProcessPoolExecutor(max_workers=workers) as ex:
            future_to_row = {
                ex.submit(
                    _read_one_nc4,
                    files_by_id[i],
                    self.training_var,
                    step,
                    idx1d,
                    self.epsilon,
                ): row
                for row, i in enumerate(keep_ids)
            }
            for fut in tqdm_rank0(
                as_completed(future_to_row),
                total=total,
                desc="Reading training files",
                unit="file",
            ):
                row = future_to_row[fut]
                response[row] = fut.result()
        end_time = time()
        rank_zero_info(f"Reading training data took {(end_time-start_time):.0f}s")

        # Filter by threshold (same as old code: on linear scale)
        good = response.max(axis=1) < float(self.threshold)

        # Log10 transform EXACTLY like the old code
        if self.log_y:
            response = np.log10(response)
            response[np.isneginf(response)] = 0  # -inf -> 0

        # Torch conversion + clamp negatives to 0 (old code behavior)
        X = torch.from_numpy(samples.to_numpy(dtype=np.float32))[good]
        Y = torch.from_numpy(response.astype(np.float32)[good])
        Y[Y < 0] = 0  # clamp negative logs (values < 1) to 0

        self.X_keys = list(samples.columns)
        self.X_mean = X.mean(dim=0)
        self.X_std = X.std(dim=0)
        self.X = (X - self.X_mean) / (self.X_std) if self.normalize_x else X
        self.Y = Y

        self.n_parameters = self.X.shape[1]
        self.n_samples = self.Y.shape[0]
        self.n_grid_points = self.Y.shape[1]

        self.normed_area = torch.ones(self.n_grid_points, dtype=torch.float32)
        self.normed_area /= self.normed_area.sum()
