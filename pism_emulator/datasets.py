from __future__ import annotations

import os
import re
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from itertools import repeat
from os import PathLike
from os.path import join
from pathlib import Path
from time import time
from typing import Any, Final, Union, cast

import dask

# add at top-level
import netCDF4 as nc
import numpy as np
import pandas as pd
import torch
import xarray as xr
from numpy.typing import NDArray
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

ID_RE: Final[re.Pattern[str]] = re.compile(r"id_(?P<id>\d+)_")


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
    Read a single NetCDF file quickly and extract thinned values at given nodes.

    This function bypasses xarray and uses ``netCDF4`` directly for speed. It disables
    CF mask/scale on the variable, applies a stride (``step``) along the ``y`` and ``x``
    dimensions, squeezes any extra leading singleton dimensions (e.g., time=1), and
    returns a 1-D vector containing the values at precomputed raveled indices.

    Parameters
    ----------
    path : str or os.PathLike[str]
        Path to the NetCDF file to read.
    var : str
        Name of the variable to read. The variable must have ``y`` and ``x`` dimensions.
    step : int
        Thinning stride to apply along ``y`` and ``x`` (e.g., ``step=2`` selects every
        second grid point). Must be >= 1.
    idx1d : numpy.ndarray of intp, shape (N,)
        Precomputed raveled indices (on the **thinned** grid) selecting the desired
        nodes from ``arr.ravel()`` after slicing. Typically produced via
        ``np.ravel_multi_index`` on the thinned grid or by stacking then converting
        to flat indices.
    eps : float
        Replacement value for NaNs (used via ``np.nan_to_num``).

    Returns
    -------
    numpy.ndarray of float32, shape (N,)
        Vector of selected values at the requested nodes, dtype ``float32``.

    Raises
    ------
    ValueError
        If ``step < 1``.
    KeyError
        If the variable does not contain ``"y"`` and ``"x"`` dimensions.

    Notes
    -----
    * CF mask/scale is disabled (``set_auto_maskandscale(False)``) for performance.
      If your dataset relies on scale/offset, apply them manually.
    * Any non-``(y, x)`` leading dimension of length 1 (e.g., time=1) is squeezed.
      If such a dimension has length > 1, only the first entry is read.
    * Uses ``np.take(arr.ravel(), idx1d)`` to minimize indexing overhead.

    Examples
    --------
    >>> vals = _read_one_nc4("run_id_001.nc", "velsurf_mag", step=2, idx1d=idx, eps=0.0)
    >>> vals.shape
    (idx.size,)
    """
    if step < 1:
        raise ValueError(f"'step' must be >= 1, got {step}")

    ds = nc.Dataset(path, "r")
    try:
        v = ds.variables[var]
        # Disable CF mask/scale to avoid Python-side conversions
        v.set_auto_maskandscale(False)

        # Dimensions and sizes
        dims: tuple[str, ...] = v.dimensions
        sizes: dict[str, int] = {d: ds.dimensions[d].size for d in dims}

        # Find y/x positions
        try:
            yi = dims.index("y")
            xi = dims.index("x")
        except ValueError as e:
            raise KeyError(
                f'"{var}" in {path} is missing "y" or "x" dim: {dims}'
            ) from e

        # Build slicers: ints or slices per dim
        slicers: list[int | slice] = []
        for k, d in enumerate(dims):
            if k == yi:
                slicers.append(slice(0, None, step))
            elif k == xi:
                slicers.append(slice(0, None, step))
            else:
                n = sizes[d]
                # squeeze if singleton, otherwise read first entry (common: time=1)
                slicers.append(0 if n == 1 else slice(0, 1))

        arr = v[tuple(slicers)]

        # If a non-(y,x) leading axis of length 1 was kept, squeeze it now
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[-2], arr.shape[-1])  # (y, x)
    finally:
        ds.close()

    # Ensure dtype and NaN handling without extra copies
    arr = np.asarray(arr, dtype=np.float32)
    np.nan_to_num(arr, nan=eps, copy=False)

    # Gather sparse nodes (fast flat gather)
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


class PISMDatasetXRP(torch.utils.data.Dataset):
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
            print(f"Loading target {self.target_file} (engine={self.target_engine})")

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
            print("  Loading samples & training responses... (netCDF4 direct)")

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
                print(f"  Missing runs (dropping from samples): {missing}")

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
            for fut in tqdm(
                as_completed(future_to_row),
                total=total,
                desc="Reading training files",
                unit="file",
            ):
                row = future_to_row[fut]
                response[row] = fut.result()
        end_time = time()
        print(f"Reading training data took {(end_time-start_time):.0f}s")

        # Filter by threshold
        good = response.max(axis=1) < float(self.threshold)

        # Log10 transform w/out extra allocations
        if self.log_y:
            with np.errstate(divide="ignore", invalid="ignore"):
                # write into response in place to avoid a new array
                np.log10(response, out=response, where=(response > 0))
                # zeros where <=0 already handled by where; fill -inf with 0
                response[~np.isfinite(response)] = 0.0

        # Torch conversion
        Y = torch.from_numpy(response[good])
        X = torch.from_numpy(samples.to_numpy(dtype=np.float32))[good]

        self.X_keys = list(samples.columns)
        self.X_mean = X.mean(dim=0)
        self.X_std = X.std(dim=0)
        self.X = (X - self.X_mean) / (self.X_std + 1e-12) if self.normalize_x else X
        self.Y = Y

        self.n_parameters = self.X.shape[1]
        self.n_samples = self.Y.shape[0]
        self.n_grid_points = self.Y.shape[1]

        self.normed_area = torch.ones(self.n_grid_points, dtype=torch.float32)
        self.normed_area /= self.normed_area.sum()


class PISMDatasetXR(torch.utils.data.Dataset):
    def __init__(
        self,
        training_files,  # list[str] or glob string
        samples_file,
        target_file,
        target_var="velsurf_mag",
        target_corr_threshold=25.0,
        target_corr_var="thickness",
        target_error_var="velsurf_mag_error",
        training_var="velsurf_mag",
        thin=1,
        normalize_x=True,
        log_y=True,
        threshold=100e3,
        epsilon=0.0,
        verbose=False,
        # I/O & performance knobs
        engine="netcdf4",  # often faster & lock-free
        chunks={"y": 512, "x": 512},  # tune for your filesystem
        dask_scheduler=None,  # e.g., "threads" or "processes" if you want to set globally
        persist_training=False,  # call .persist() for repeated epochs
    ):
        self.training_files = self._normalize_file_list(training_files)
        self.samples_file = samples_file
        self.target_file = target_file
        self.target_var = target_var
        self.target_corr_threshold = target_corr_threshold
        self.target_corr_var = target_corr_var
        self.target_error_var = target_error_var
        self.thin = thin
        self.threshold = threshold
        self.training_var = training_var
        self.epsilon = epsilon
        self.log_y = log_y
        self.normalize_x = normalize_x
        self.verbose = verbose
        self.engine = engine
        self.chunks = chunks
        self.persist_training = persist_training

        if dask_scheduler:
            dask.config.set(scheduler=dask_scheduler)

        # 1) Build target-derived mask & vectors (only once)
        self._load_target_and_mask()

        # 2) Load samples & training responses with open_mfdataset
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
    @staticmethod
    def _normalize_file_list(pattern_or_list):
        if isinstance(pattern_or_list, (list, tuple)):
            files = list(pattern_or_list)
        else:
            files = sorted(map(str, Path().glob(pattern_or_list)))
        # de-dup while preserving order
        return list(OrderedDict.fromkeys(files))

    @staticmethod
    def _parse_id(path):
        m = re.search(r"id_(\d+)_", Path(path).name)
        if not m:
            raise ValueError(f"Could not parse id from filename: {path}")
        return int(m.group(1))

    def _thin_sel(self):
        # thin via isel to keep compatibility across xarray versions
        step = self.thin
        return {"x": slice(None, None, step), "y": slice(None, None, step)}

    # ---------------------------
    # Target & mask
    # ---------------------------

    def _load_target_and_mask(self):
        if self.verbose:
            print(f"Loading target {self.target_file}")

        ds = xr.open_dataset(
            self.target_file, decode_times=False, engine=self.engine, chunks=self.chunks
        ).isel(**self._thin_sel())

        if self.target_var not in ds:
            raise KeyError(f"'{self.target_var}' not found in target file")

        targ_da = ds[self.target_var].squeeze(drop=True)

        # Start with NaN mask from target
        mask = targ_da.isnull()

        # Optional correlation filter → expand mask
        self.target_has_corr = self.target_corr_var in ds
        if self.target_has_corr:
            corr_da = ds[self.target_corr_var].squeeze(drop=True)
            mask = xr.where(corr_da < float(self.target_corr_threshold), True, mask)

        # --- KEY CHANGE: compute boolean mask to NumPy BEFORE using as indexer ---
        mask_np = mask.compute().values  # 2D NumPy boolean
        self.mask_2d = mask_np
        self.ny = int(ds.dims["y"])
        self.nx = int(ds.dims["x"])

        # Sparse indices (y, x) and 1D (stacked) indices
        self.sparse_idx_2d = np.where(~mask_np)
        self.sparse_idx_1d = np.ravel_multi_index(
            self.sparse_idx_2d, (self.ny, self.nx)
        )

        # Stack ("y","x") → 1D, then select valid nodes by integer positions
        targ_vec = (
            targ_da.stack(node=("y", "x"))
            .isel(node=self.sparse_idx_1d)
            .fillna(self.epsilon)
            .astype("float32")
            .compute()
            .values
        )
        self.Y_target = torch.from_numpy(targ_vec)

        # Optional error
        self.target_has_error = self.target_error_var in ds
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

        # Optional corr output
        if self.target_has_corr:
            corr_vec = (
                corr_da.stack(node=("y", "x"))
                .isel(node=self.sparse_idx_1d)
                .fillna(self.epsilon)
                .astype("float32")
                .compute()
                .values
            )
            self.Y_target_corr = torch.from_numpy(corr_vec)
            self.Y_target_corr_2d = (
                corr_da.fillna(self.epsilon).astype("float32").compute().values
            )

        # Grid resolution
        self.grid_resolution = float(abs(ds["x"][1] - ds["x"][0]))

        # Optional 2D masked array (for visualization)
        y2d = np.zeros((self.ny, self.nx), dtype=np.float32)
        y2d.flat[self.sparse_idx_1d] = self.Y_target.numpy()
        self.Y_target_2d = np.ma.array(data=y2d, mask=self.mask_2d)

        ds.close()

    # ---------------------------
    # Training & samples via open_mfdataset
    # ---------------------------
    def _load_training_and_samples(self):
        if self.verbose:
            print("  Loading samples & training responses...")

        # parse and sort by id
        ids = [self._parse_id(f) for f in self.training_files]
        order = np.argsort(ids)
        files_sorted = [self.training_files[i] for i in order]
        ids_sorted = [ids[i] for i in order]

        # samples CSV
        samples = (
            pd.read_csv(self.samples_file, delimiter=",", skipinitialspace=True)
            .sort_values(by="id")
            .set_index("id", drop=True)
        )

        # drop missing ids (runs that failed)
        missing = set(samples.index) - set(ids_sorted)
        if missing and self.verbose:
            print(f"  Missing runs (dropping from samples): {sorted(missing)}")
        samples = samples.drop(index=list(missing), errors="ignore")

        # keep only rows we have files for
        samples = samples.loc[sorted(set(samples.index) & set(ids_sorted))]

        # build preprocess: thin, keep training_var, add exp_id for concat
        thin_sel = self._thin_sel()

        def _preprocess(ds):
            if self.training_var not in ds:
                raise KeyError(
                    f"'{self.training_var}' not in {ds.encoding.get('source', 'dataset')}"
                )
            ds = ds.isel(**thin_sel)
            da = ds[self.training_var].squeeze(drop=True)
            # attach exp_id from filename to concat over it
            src = ds.encoding.get("source", "")
            exp_id = self._parse_id(src)
            da = da.expand_dims(exp_id=[exp_id])
            # ensure float32 early to cut memory by half
            return da.astype("float32").to_dataset(name=self.training_var)

        t0 = time()
        ds_all = xr.open_mfdataset(
            files_sorted,
            combine="nested",
            concat_dim="exp_id",
            preprocess=_preprocess,
            decode_times=False,
            engine=self.engine,
            chunks={
                "exp_id": 1,
                **self.chunks,
            },  # small exp_id chunks, spatial chunks tunable
            parallel=True,
        )

        # ds_all[training_var]: (exp_id, y, x)
        resp = ds_all[self.training_var]

        # select only sample ids we actually have (keeps sample order)
        have_ids = np.array(list(resp["exp_id"].values))
        keep_ids = np.array([i for i in samples.index if i in have_ids])
        resp = resp.sel(exp_id=keep_ids)

        # stack to (exp_id, node) and select valid nodes
        resp = (
            resp.stack(node=("y", "x")).isel(node=self.valid_nodes).fillna(self.epsilon)
        )

        # threshold filter on max per exp_id (lazy until compute)
        valid_mask = (resp.max(dim="node") < float(self.threshold)).compute()
        resp = resp.sel(exp_id=resp["exp_id"].values[valid_mask.values])

        # (optional) persist training responses for repeated epochs
        if self.persist_training:
            resp = resp.persist()

        # align samples with remaining exp_ids and convert to torch
        samples = samples.loc[resp["exp_id"].values]

        # log transform
        if self.log_y:
            resp = xr.apply_ufunc(
                np.log10, resp, dask="parallelized", output_dtypes=[resp.dtype]
            )
            resp = xr.where(np.isfinite(resp), resp, 0.0)

        # compute to NumPy and torch
        self.Y = torch.from_numpy(
            resp.astype("float32").compute().values
        )  # (n_samples, n_nodes)

        X = torch.from_numpy(samples.to_numpy(dtype=np.float32))
        self.X_keys = list(samples.columns)

        # normalize X
        self.X_mean = X.mean(dim=0)
        self.X_std = X.std(dim=0)
        if self.normalize_x:
            X = (X - self.X_mean) / (self.X_std + 1e-12)

        self.X = X

        # shapes & bookkeeping
        self.n_parameters = self.X.shape[1]
        self.n_samples, self.n_grid_points = self.Y.shape

        # simple uniform area weights (placeholder)
        self.normed_area = torch.ones(self.n_grid_points, dtype=torch.float32)
        self.normed_area /= self.normed_area.sum()

        if self.verbose:
            print(f"  Loaded: {self.n_samples} samples × {self.n_grid_points} nodes")
            print(f"  Reading training data took {time()-t0:.1f}s")

    # ---------------------------
    # Utilities
    # ---------------------------
    def return_original(self):
        if self.normalize_x:
            return self.X * self.X_std + self.X_mean
        return self.X


def preprocess(ds, thin: int = 1, mapplane_vars: list[str] = ["x", "y"]):
    """
    Select slices from dataset
    """
    slices = {key: slice(0, value, thin) for key, value in ds.sizes.items()}
    drop_dims = [key for (key, val) in slices.items() if key not in mapplane_vars]
    for d in drop_dims:
        del slices[d]
    return ds.isel(slices)


class PISMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir="path/to/dir",
        samples_file="path/to/file",
        target_file=None,
        target_var="velsurf_mag",
        target_corr_threshold=25.0,
        target_corr_var="thickness",
        target_error_var="velsurf_mag_error",
        training_var="velsurf_mag",
        thin=1,
        normalize_x=True,
        log_y=True,
        threshold=100e3,
        epsilon=0,
        verbose=False,
    ):
        self.data_dir = data_dir
        self.samples_file = samples_file
        self.target_file = target_file
        self.target_var = target_var
        self.target_corr_threshold = target_corr_threshold
        self.target_corr_var = target_corr_var
        self.target_error_var = target_error_var
        self.thin = thin
        self.threshold = threshold
        self.training_var = training_var
        self.epsilon = epsilon
        self.log_y = log_y
        self.normalize_x = normalize_x
        self.verbose = verbose
        self.load_target()
        self.load_data()

    def __getitem__(self, i):
        return tuple(d[i] for d in [self.X, self.Y])

    def __len__(self):
        return min(len(d) for d in [self.X, self.Y])

    def load_target(self):
        epsilon = self.epsilon
        thin = self.thin
        print(f"Loading target {self.target_file}")
        ds = xr.open_dataset(self.target_file, decode_times=False)
        ds = preprocess(ds, thin=thin)
        data = ds[self.target_var].squeeze()
        mask = data.isnull()
        data = np.nan_to_num(
            data.values,
            nan=epsilon,
        )
        ny, nx = data.shape
        self.target_has_error = False
        if self.target_error_var in ds.variables:
            data_error = ds[self.target_error_var].squeeze()
            data_error = np.nan_to_num(
                data_error.values,
                nan=epsilon,
            )
            self.target_has_error = True

        self.target_has_corr = False
        if self.target_corr_var in ds.variables:
            data_corr = ds[self.target_corr_var].squeeze()
            data_corr = np.nan_to_num(
                data_corr.values,
                nan=epsilon,
            )
            self.target_has_corr = True
            mask = mask.where(data_corr >= self.target_corr_threshold, True)
        mask = mask.values

        grid_resolution = np.abs(np.diff(ds["x"][0:2]))[0]
        self.grid_resolution = grid_resolution
        ds.close()

        idx = (mask == 0).nonzero()

        data = data[idx]
        Y_target = torch.from_numpy(np.array(data.flatten(), dtype=np.float32))
        self.Y_target = Y_target
        if self.target_has_error:
            data_error = data_error[idx]
            Y_target_error_2d = data_error
            Y_target_error = torch.from_numpy(
                np.array(data_error.flatten(), dtype=np.float32)
            )

            self.Y_target_error = Y_target_error
            self.Y_target_error_2d = Y_target_error_2d
        if self.target_has_corr:
            data_corr = data_corr[idx]
            Y_target_corr_2d = data_corr
            Y_target_corr = torch.from_numpy(
                np.array(data_corr.flatten(), dtype=np.float32)
            )

            self.Y_target_corr = Y_target_corr
            self.Y_target_corr_2d = Y_target_corr_2d
        self.mask_2d = mask
        self.sparse_idx_2d = idx
        self.sparse_idx_1d = np.ravel_multi_index(idx, mask.shape)
        data_2d = np.zeros((ny, nx))
        data_2d.put(self.sparse_idx_1d, data)
        Y_target_2d = np.ma.array(data=data_2d, mask=self.mask_2d)
        self.Y_target_2d = Y_target_2d

    def load_data(self):
        epsilon = self.epsilon
        thin = self.thin

        identifier_name = "id"
        training_var = self.training_var
        training_files = glob(join(self.data_dir, "*.nc"))
        training_files = list(OrderedDict.fromkeys(training_files))

        ids = [parse_id_from_path(f) for f in training_files]
        samples = (
            pd.read_csv(self.samples_file, delimiter=",", skipinitialspace=True)
            .squeeze("columns")
            .sort_values(by=identifier_name)
        )
        samples.index = samples[identifier_name]
        samples.index.name = None

        ids_df = pd.DataFrame(data=ids, columns=["id"])
        ids_df.index = ids_df[identifier_name]
        ids_df.index.name = None

        # It is possible that not all ensemble simulations succeeded and returned a value
        # so we must search for missing response values
        missing_ids = list(set(samples["id"]).difference(ids_df["id"]))
        if missing_ids:
            if self.verbose:
                print(f"The following simulations are missing:\n   {missing_ids}")
                print("  ... adjusting priors")
            # and remove the missing samples and responses
            samples_missing_removed = samples[~samples["id"].isin(missing_ids)]
            samples = samples_missing_removed

        samples = samples.drop(samples.columns[0], axis=1)
        m_samples, n_parameters = samples.shape
        self.X_keys = samples.keys()

        ds0 = xr.open_dataset(training_files[0], decode_times=False)
        ds0 = preprocess(ds0, thin=thin)
        _, ny, nx = ds0.variables[self.target_var].values.shape

        ds0.close()
        self.nx = nx
        self.ny = ny
        response = np.zeros((m_samples, len(self.sparse_idx_1d)))

        print("  Loading data sets...")
        training_files.sort(key=id_key)
        start_time = time()
        for idx, m_file in tqdm(enumerate(training_files), total=len(training_files)):
            ds = xr.open_dataset(m_file, decode_times=False)
            ds = preprocess(ds, thin=thin)
            data = np.squeeze(
                np.nan_to_num(
                    ds.variables[training_var].values,
                    nan=epsilon,
                )
            )
            response[idx, :] = data[self.sparse_idx_2d].flatten()
            ds.close()
        end_time = time()
        self.training_files = training_files
        print(f"Reading training data took {(end_time-start_time):.0f}s")

        p = response.max(axis=1) < self.threshold
        if self.log_y:
            response = np.log10(response)
            response[np.isneginf(response)] = 0

        X = torch.from_numpy(np.array(samples[p], dtype=np.float32))
        Y = torch.from_numpy(np.array(response[p], dtype=np.float32))
        Y[Y < 0] = 0

        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        self.X_mean = X_mean
        self.X_std = X_std

        if self.normalize_x:
            X = (X - X_mean) / X_std

        self.X = X
        self.Y = Y

        n_parameters = X.shape[1]
        self.n_parameters = n_parameters
        n_samples, n_grid_points = Y.shape
        self.n_samples = n_samples
        self.n_grid_points = n_grid_points

        normed_area = np.ones(n_grid_points, dtype=np.float32)
        normed_area = torch.tensor(normed_area)
        normed_area /= normed_area.sum()
        self.normed_area = normed_area

    def return_original(self):
        if self.normalize_x:
            return self.X * self.X_std + self.X_mean
        else:
            return self.X
