# Copyright (C) 2021-25 Andy Aschwanden, Douglas C Brinkerhoff
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

# pylint: disable=too-many-instance-attributes
"""
Dataset Module.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Final, Sequence, cast

import dask

# add at top-level
import netCDF4 as nc
import numpy as np
import pandas as pd
import torch
import xarray as xr
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from numpy.typing import NDArray
from torch.utils.data import Dataset, get_worker_info
from tqdm.auto import tqdm as _tqdm

ID_RE: Final[re.Pattern[str]] = re.compile(r"id_(?P<id>\d+)_")


def _sniff_engine(path: str) -> str:
    """
    Heuristically choose an xarray engine based on file signature.

    Reads the first 8 bytes and returns an engine string suitable for
    ``xarray.open_dataset``:

    - NetCDF-4/HDF5 (magic ``\\x89HDF\\r\\n\\x1a\\n`` or ``b\"HDF\"``) → ``"h5netcdf"``
    - NetCDF-3 (magic ``b\"CDF\\x01\"`` or ``b\"CDF\\x02\"``) → ``"scipy"``
    - Fallback → ``"h5netcdf"``

    Parameters
    ----------
    path : str
        Path to the NetCDF file.

    Returns
    -------
    str
        Engine name: one of ``"h5netcdf"`` or ``"scipy"``.

    Notes
    -----
    ``"h5netcdf"`` is preferred over ``"netcdf4"`` for NetCDF-4/HDF5 files for
    improved stability and fewer dependency issues in many environments.
    """
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


def _is_global_zero() -> bool:
    """
    Check whether the current process is global rank 0.

    Works with both Torch distributed (`torchrun`) and Lightning by inspecting
    the ``RANK`` or ``GLOBAL_RANK`` environment variables.

    Returns
    -------
    bool
        ``True`` if the parsed rank is 0 or the variables are unset/invalid,
        ``False`` otherwise.

    Notes
    -----
    If the environment variable cannot be parsed as an integer, this function
    returns ``True`` (permissive default) so that progress output does not get
    unintentionally suppressed in non-distributed contexts.
    """
    r = os.environ.get("RANK", os.environ.get("GLOBAL_RANK", "0"))
    try:
        return int(r) == 0
    except ValueError:
        return True  # be permissive if unset


def _is_worker_zero() -> bool:
    """
    Check whether the current DataLoader worker is worker 0.

    Uses :func:`torch.utils.data.get_worker_info` to detect the worker ID.

    Returns
    -------
    bool
        ``True`` if running in the main process (no worker) or in worker ``id == 0``,
        ``False`` otherwise.
    """
    wi = get_worker_info()
    return (wi is None) or (wi.id == 0)


def tqdm_rank0(*args, **kwargs):
    """
    Create a ``tqdm`` progress bar that only renders on global rank 0, worker 0.

    Parameters
    ----------
    *args
        Positional arguments forwarded to :class:`tqdm.tqdm`.
    **kwargs
        Keyword arguments forwarded to :class:`tqdm.tqdm`. The following defaults
        are set if not provided:
        - ``dynamic_ncols=True`` for auto column width.
        - ``leave=False`` so bars do not persist after completion.
        Additionally, ``disable=True`` is injected automatically unless both
        the global rank is 0 and the worker ID is 0.

    Returns
    -------
    tqdm.tqdm
        A tqdm instance that is disabled on non-zero ranks/workers.

    Notes
    -----
    This helper prevents duplicate progress bars under DDP/multiprocessing by
    showing the bar only once (rank 0, worker 0).
    """
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("leave", False)
    # disable everywhere except global rank 0 + worker 0
    if not (_is_global_zero() and _is_worker_zero()):
        kwargs["disable"] = True
    return _tqdm(*args, **kwargs)


def id_key(path: str | Path) -> int:
    """
    Extract the integer ``id`` embedded in a filename.

    The filename is expected to contain a substring like ``id_<number>`` and the
    search is performed against :data:`ID_RE` on the basename.

    Parameters
    ----------
    path : str or pathlib.Path
        Path whose basename will be searched for the ``id`` pattern.

    Returns
    -------
    int
        The parsed integer ID.

    Raises
    ------
    ValueError
        If the pattern is not found in the basename.

    """
    m = ID_RE.search(Path(path).name)
    if m is None:
        raise ValueError(
            f"Could not parse id from {path!s} using pattern {ID_RE.pattern!r}"
        )
    return int(m.group("id"))


def parse_id_from_path(p: str | Path) -> int:
    """
    Extract the integer ``id`` embedded in a filename.

    This is equivalent to :func:`id_key`. The filename is expected to contain
    a substring like ``id_<number>`` and the search is performed against
    :data:`ID_RE` on the basename.

    Parameters
    ----------
    p : str or pathlib.Path
        Path whose basename will be searched for the ``id`` pattern.

    Returns
    -------
    int
        The parsed integer ID.

    Raises
    ------
    ValueError
        If the pattern is not found in the basename.
    """
    m = ID_RE.search(Path(p).name)
    if not m:
        raise ValueError(
            f"Could not parse id from {p!s} using pattern {ID_RE.pattern!r}"
        )
    return int(m.group("id"))


def _read_one_nc4(
    path: str | Path,
    var: str,
    step: int,
    idx1d: NDArray[np.intp],
    eps: float,
) -> NDArray[np.float32]:
    """
    Read a CF-compliant NetCDF variable and return a sparsified, downsampled 1-D view.

    This fast reader:
    1) opens the file with :mod:`netCDF4`,
    2) enables CF-style mask/scale application (matching ``xarray.open_dataset(..., decode_cf=True)``),
    3) converts masked values to ``NaN`` and then replaces ``NaN`` with ``eps``,
    4) strides the ``y`` and ``x`` dimensions by ``step``, squeezes any leading singleton,
    5) gathers values at linear indices ``idx1d`` from the flattened array.

    Parameters
    ----------
    path : str or Path
        Path to a NetCDF file.
    var : str
        Name of the variable to read from ``path``.
    step : int
        Spatial stride for the ``y`` and ``x`` dimensions. Must be ``>= 1``.
    idx1d : numpy.ndarray (dtype=intp, 1-D)
        Linear indices (C-order) used to gather a sparse subset from the flattened
        downsampled array.
    eps : float
        Epsilon used to replace ``NaN`` values after mask/scale decoding.

    Returns
    -------
    numpy.ndarray (dtype=float32, shape=(len(idx1d),))
        Gathered values from the flattened (``y``, ``x``) array at positions
        ``idx1d``, with masked/NaN values replaced by ``eps``.

    Raises
    ------
    ValueError
        If ``step < 1``.
    KeyError
        If the variable does not have both ``"y"`` and ``"x"`` dimensions.

    Notes
    -----
    - Non-(``y``, ``x``) leading dimensions are reduced by selecting the first
      entry when the size is greater than 1 (or kept if exactly 1).
    - The return uses C-order flattening (row-major), consistent with
      ``arr.ravel(order='C')`` and ``xarray`` default memory layout for NumPy
      backends.
    """
    if step < 1:
        raise ValueError(f"'step' must be >= 1, got {step}")

    ds = nc.Dataset(path, "r")  # pylint: disable=no-member
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


@dataclass
class TargetData:
    """
    Target field, mask, and grid metadata derived from the target NetCDF.

    Attributes
    ----------
    ny : int
        Number of grid points in the ``y`` dimension after thinning.
    nx : int
        Number of grid points in the ``x`` dimension after thinning.
    mask_2d : numpy.ndarray
        Boolean mask of shape ``(ny, nx)`` where ``True`` indicates masked cells
        (NaN or below correlation threshold).
    sparse_idx_2d : tuple[numpy.ndarray, numpy.ndarray]
        Tuple of 1-D index arrays giving unmasked ``(y, x)`` coordinates.
    sparse_idx_1d : numpy.ndarray
        Linear (C-order) indices of unmasked nodes with shape ``(nnodes,)``.
    Y_target : torch.Tensor
        Target values stacked over unmasked nodes with shape ``(nnodes,)``.
    Y_target_2d : numpy.ma.MaskedArray
        2-D masked array view of the target with shape ``(ny, nx)``.
    Y_target_error : torch.Tensor or None, default=None
        Per-node uncertainty (if available) stacked over unmasked nodes.
    Y_target_error_2d : numpy.ndarray or None, default=None
        2-D array of per-node uncertainties aligned with ``Y_target_2d``.
    Y_target_corr : torch.Tensor or None, default=None
        Correlation variable values stacked over unmasked nodes (if available).
    Y_target_corr_2d : numpy.ndarray or None, default=None
        2-D array of correlation values aligned with ``Y_target_2d``.
    grid_resolution : float, default=0.0
        Grid spacing in the ``x`` direction (absolute difference between the
        first two coordinates), units match the input dataset.
    """

    ny: int
    nx: int
    mask_2d: np.ndarray  # bool      (ny, nx)
    sparse_idx_2d: tuple[np.ndarray, np.ndarray]
    sparse_idx_1d: np.ndarray  # intp      (nnodes,)
    Y_target: torch.Tensor  # (nnodes,)
    Y_target_2d: np.ma.MaskedArray  # (ny, nx)
    Y_target_error: torch.Tensor | None = None
    Y_target_error_2d: np.ndarray | None = None
    Y_target_corr: torch.Tensor | None = None
    Y_target_corr_2d: np.ndarray | None = None
    grid_resolution: float = 0.0


@dataclass
class SamplesData:
    """
    Prepared features and responses aligned across training runs.

    Attributes
    ----------
    X : torch.Tensor
        Feature matrix of shape ``(n_runs, n_params)``. If
        :data:`DatasetConfig.normalize_x` is ``True``, features are z-scored.
    Y : torch.Tensor
        Response matrix of shape ``(n_runs, n_nodes)`` (sparse-node ordering).
    X_keys : list[str]
        Column names from the samples CSV corresponding to ``X`` columns.
    X_mean : torch.Tensor
        Per-column mean used for normalization (or the raw mean if disabled).
    X_std : torch.Tensor
        Per-column standard deviation used for normalization
        (or the raw std if normalization disabled).
    n_parameters : int
        Number of feature columns (``X.shape[1]``).
    n_samples : int
        Number of retained runs after filtering (``X.shape[0]`` / ``Y.shape[0]``).
    n_grid_points : int
        Number of unmasked nodes per response (``Y.shape[1]``).
    normed_area : torch.Tensor
        Normalized per-node area weights of shape ``(n_nodes,)`` that sum to 1.
    """

    X: torch.Tensor  # (nruns, nparams) (normalized if requested)
    Y: torch.Tensor  # (nruns, nnodes)
    X_keys: list[str]
    X_mean: torch.Tensor
    X_std: torch.Tensor
    n_parameters: int
    n_samples: int
    n_grid_points: int
    normed_area: torch.Tensor  # (nnodes,)


@dataclass
class DatasetConfig:
    """
    Immutable configuration for building a :class:`PISMDataset`.

    Attributes
    ----------
    training_files : list[str]
        Sorted list of paths to training NetCDF files. Each filename must
        contain an ``id_<N>`` token used to align with the samples CSV.
    samples_file : str
        Path to the CSV containing parameter samples. Must include an ``id``
        column matching the training file IDs.
    target_file : str
        Path to the target (observational) NetCDF file.
    training_var : str, default="velsurf_mag"
        Variable name to read from each training file.
    target_var : str, default="velsurf_mag"
        Variable name to read from the target file.
    target_corr_var : str, default="thickness"
        Optional correlation variable used to mask low-confidence grid cells.
    target_error_var : str, default="velsurf_mag_error"
        Optional per-node uncertainty variable in the target file.
    target_corr_threshold : float, default=25.0
        Threshold applied to ``target_corr_var``; values below this are masked.
    thin : int, default=1
        Spatial stride for downsampling along ``y`` and ``x`` when reading arrays.
    normalize_x : bool, default=True
        If ``True``, z-score normalize feature columns (per column mean/std).
    log_y : bool, default=True
        If ``True``, apply ``log10`` to responses after reading; ``-inf`` values
        are set to 0 to match legacy behavior.
    threshold : float, default=1e5
        Run-level filter on the **max** response (in linear scale). Runs exceeding
        this value are dropped.
    epsilon : float, default=0.0
        Value used to replace NaNs after CF decode and masking.
    verbose : bool, default=False
        If ``True``, emit rank-0 progress messages.
    target_engine : str or None, default=None
        xarray engine for ``target_file``. If ``None``, detected via file signature.
    training_engine : str or None, default=None
        Engine hint for training files. Defaults to ``"h5netcdf"`` for stability.
    parallel : bool, default=True
        If ``True``, read training files in parallel via a process pool.
    chunks_after : dict[str, int] or None, default=None
        Placeholder for chunking controls kept for API compatibility.
    dask_scheduler : str or None, default=None
        If provided, sets the Dask scheduler (e.g., ``"threads"``, ``"processes"``).
    """

    training_files: list[str]
    samples_file: str
    target_file: str
    training_var: str = "velsurf_mag"
    target_var: str = "velsurf_mag"
    target_corr_var: str = "thickness"
    target_error_var: str = "velsurf_mag_error"
    target_corr_threshold: float = 25.0
    thin: int = 1
    normalize_x: bool = True
    log_y: bool = True
    threshold: float = 100e3
    epsilon: float = 0.0
    verbose: bool = False
    target_engine: str | None = None
    training_engine: str | None = None
    parallel: bool = True
    chunks_after: dict[str, int] | None = None
    dask_scheduler: str | None = None


class PISMDataset(Dataset):
    """
    PISM training dataset (refactored).

    This dataset:
    - loads and masks the target field, building a sparse node index,
    - reads many training NetCDFs in parallel (downsampled on (y,x)),
    - aligns runs with a CSV of parameter samples,
    - applies thresholding and optional log10 transform,
    - normalizes features if requested.

    Parameters
    ----------
    training_files : Sequence[str] | str
        List/glob of NetCDF training files; each file name must contain ``id_<N>``.
    samples_file : str
        CSV with parameter samples; must contain an ``id`` column matching file ids.
    target_file : str
        NetCDF target file (observations).
    training_var : str, default="velsurf_mag"
        Variable to read from training files.
    target_var : str, default="velsurf_mag"
        Variable to read from the target file.
    target_corr_var : str, default="thickness"
        Optional correlation variable; values below ``target_corr_threshold`` are masked.
    target_error_var : str, default="velsurf_mag_error"
        Optional per-node error variable in the target file.
    target_corr_threshold : float, default=25.0
        Threshold for masking based on ``target_corr_var``.
    thin : int, default=1
        Spatial stride (downsampling) for the y/x dimensions.
    normalize_x : bool, default=True
        If True, z-score normalize the features (per column).
    log_y : bool, default=True
        If True, apply ``log10`` to responses (match previous behavior).
    threshold : float, default=100e3
        Filter out runs whose **max** response (linear scale) exceeds this value.
    epsilon : float, default=0.0
        Replace NaNs with this epsilon when reading arrays.
    verbose : bool, default=False
        Print rank-0 progress.
    target_engine : str or None, default=None
        xarray engine for the target file; if None, chosen via signature sniffing.
    training_engine : str or None, default=None
        Engine for multi-file training reads; defaults to ``"h5netcdf"``.
    parallel : bool, default=True
        If True, read training files in parallel with a process pool.
    chunks_after : dict or None, default=None
        Unused in this refactor (placeholder to match old signature).
    dask_scheduler : str or None, default=None
        If provided, sets ``dask.config.set(scheduler=...)``.

    Notes
    -----
    - Instance attributes are kept minimal: ``cfg``, ``target``, ``samples``.
    - Constructor is keyword-only to avoid pylint R0917.
    """

    # ↓↓↓ keep instance attribute count low: only three top-level fields
    def __init__(
        self,
        *,
        training_files: Sequence[str] | str,
        samples_file: str,
        target_file: str,
        training_var: str = "velsurf_mag",
        target_var: str = "velsurf_mag",
        target_corr_var: str = "thickness",
        target_error_var: str = "velsurf_mag_error",
        target_corr_threshold: float = 25.0,
        thin: int = 1,
        normalize_x: bool = True,
        log_y: bool = True,
        threshold: float = 100e3,
        epsilon: float = 0.0,
        verbose: bool = False,
        target_engine: str | None = None,
        training_engine: str | None = None,
        parallel: bool = True,
        chunks_after: dict[str, int] | None = None,
        dask_scheduler: str | None = None,
    ) -> None:

        if isinstance(training_files, (list, tuple)):
            tfiles = [str(p) for p in training_files]
        elif isinstance(training_files, (str, Path)):
            pattern = os.fspath(training_files)
            tfiles = [str(p) for p in Path().glob(pattern)]
        else:
            raise TypeError(
                "training_files must be a str/glob pattern or a sequence of str"
            )

        if dask_scheduler:

            dask.config.set(scheduler=dask_scheduler)

        cfg = DatasetConfig(
            training_files=tfiles,
            samples_file=samples_file,
            target_file=target_file,
            training_var=training_var,
            target_var=target_var,
            target_corr_var=target_corr_var,
            target_error_var=target_error_var,
            target_corr_threshold=float(target_corr_threshold),
            thin=int(thin),
            normalize_x=bool(normalize_x),
            log_y=bool(log_y),
            threshold=float(threshold),
            epsilon=float(epsilon),
            verbose=bool(verbose),
            target_engine=target_engine or _sniff_engine(target_file),
            training_engine=training_engine or "h5netcdf",
            parallel=bool(parallel),
            chunks_after=chunks_after or {},
            dask_scheduler=dask_scheduler,
        )
        self.cfg = cfg

        # build target & samples blocks
        self.target: TargetData = self._load_target_and_mask()
        self.samples: SamplesData = self._load_training_and_samples()

    # ------------- PyTorch Dataset protocol -------------

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples.X[i], self.samples.Y[i]

    def __len__(self) -> int:
        return min(self.samples.n_samples, self.samples.Y.shape[0])

    # ------------- internals -------------

    def _thin_sel(self) -> dict[str, slice]:
        s = self.cfg.thin
        return {"x": slice(None, None, s), "y": slice(None, None, s)}

    @staticmethod
    def _parse_id(path: str) -> int:
        m = re.search(r"id_(\d+)_", Path(path).name)
        if not m:
            raise ValueError(f"Could not parse id from filename: {path}")
        return int(m.group(1))

    # ---------- target/mask ----------

    def _load_target_and_mask(self) -> TargetData:
        cfg = self.cfg
        if cfg.verbose:
            rank_zero_info(
                f"Loading target {cfg.target_file} (engine={cfg.target_engine})"
            )

        ds = xr.open_dataset(
            cfg.target_file, decode_times=False, engine=cfg.target_engine
        ).isel(**self._thin_sel())

        if cfg.target_var not in ds:
            raise KeyError(f"'{cfg.target_var}' not found in target file")

        targ_da = ds[cfg.target_var].squeeze(drop=True)
        mask = targ_da.isnull()

        has_err = cfg.target_error_var in ds
        has_corr = cfg.target_corr_var in ds

        if has_corr:
            corr_da = ds[cfg.target_corr_var].squeeze(drop=True)
            mask = xr.where(corr_da < cfg.target_corr_threshold, True, mask)

        mask_np = mask.compute().values.astype(bool)
        ny = int(ds.sizes["y"])
        nx = int(ds.sizes["x"])

        sparse_idx_2d = np.where(~mask_np)
        sparse_idx_1d = np.ravel_multi_index(sparse_idx_2d, (ny, nx))

        targ_vec = (
            targ_da.stack(node=("y", "x"))
            .isel(node=sparse_idx_1d)
            .fillna(cfg.epsilon)
            .astype("float32")
            .compute()
            .values
        )
        Y_target = torch.from_numpy(targ_vec.astype(np.float32))

        Y_target_error = Y_target_error_2d = None
        if has_err:
            err_vec = (
                ds[cfg.target_error_var]
                .squeeze(drop=True)
                .stack(node=("y", "x"))
                .isel(node=sparse_idx_1d)
                .fillna(cfg.epsilon)
                .astype("float32")
                .compute()
                .values
            )
            Y_target_error = torch.from_numpy(err_vec.astype(np.float32))
            Y_target_error_2d = (
                ds[cfg.target_error_var]
                .squeeze(drop=True)
                .fillna(cfg.epsilon)
                .astype("float32")
                .compute()
                .values
            )

        Y_target_corr = Y_target_corr_2d = None
        if has_corr:
            corr_vec = (
                ds[cfg.target_corr_var]
                .squeeze(drop=True)
                .stack(node=("y", "x"))
                .isel(node=sparse_idx_1d)
                .fillna(cfg.epsilon)
                .astype("float32")
                .compute()
                .values
            )
            Y_target_corr = torch.from_numpy(corr_vec.astype(np.float32))
            Y_target_corr_2d = (
                ds[cfg.target_corr_var]
                .squeeze(drop=True)
                .fillna(cfg.epsilon)
                .astype("float32")
                .compute()
                .values
            )

        grid_resolution = float(abs(ds["x"][1] - ds["x"][0]))

        y2d = np.zeros((ny, nx), dtype=np.float32)
        y2d.flat[sparse_idx_1d] = Y_target.numpy()
        Y_target_2d = np.ma.array(data=y2d, mask=mask_np)

        ds.close()

        return TargetData(
            ny=ny,
            nx=nx,
            mask_2d=mask_np,
            sparse_idx_2d=sparse_idx_2d,
            sparse_idx_1d=sparse_idx_1d,
            Y_target=Y_target,
            Y_target_2d=Y_target_2d,
            Y_target_error=Y_target_error,
            Y_target_error_2d=Y_target_error_2d,
            Y_target_corr=Y_target_corr,
            Y_target_corr_2d=Y_target_corr_2d,
            grid_resolution=grid_resolution,
        )

    # ---------- training + samples ----------

    def _load_training_and_samples(self) -> SamplesData:
        cfg, tgt = self.cfg, self.target

        if cfg.verbose:
            rank_zero_info("  Loading samples & training responses... (netCDF4 direct)")

        ids = [self._parse_id(p) for p in cfg.training_files]
        files_by_id = dict(zip(ids, cfg.training_files))

        samples = (
            pd.read_csv(cfg.samples_file, delimiter=",", skipinitialspace=True)
            .sort_values("id")
            .set_index("id", drop=True)
        )

        keep_ids = [i for i in sorted(files_by_id) if i in samples.index]
        if cfg.verbose:
            missing = sorted(set(samples.index).difference(keep_ids))
            if missing:
                rank_zero_info(f"  Missing runs (dropping from samples): {missing}")

        samples = samples.loc[keep_ids]

        n_runs = len(keep_ids)
        n_nodes = tgt.sparse_idx_1d.size
        response = np.empty((n_runs, n_nodes), dtype=np.float32)

        step = cfg.thin
        idx1d = tgt.sparse_idx_1d

        start_time = time()
        total = len(keep_ids)

        if cfg.parallel:
            workers: int = min(8, os.cpu_count() or 1)
            with ProcessPoolExecutor(max_workers=workers) as ex:
                future_to_row = {
                    ex.submit(
                        _read_one_nc4,
                        files_by_id[i],
                        cfg.training_var,
                        step,
                        idx1d,
                        cfg.epsilon,
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
        else:
            for row, i in enumerate(
                tqdm_rank0(keep_ids, desc="Reading training files", unit="file")
            ):
                response[row] = _read_one_nc4(
                    files_by_id[i], cfg.training_var, step, idx1d, cfg.epsilon
                )

        end_time = time()
        rank_zero_info(f"Reading training data took {(end_time - start_time):.0f}s")

        good = response.max(axis=1) < float(cfg.threshold)

        if cfg.log_y:
            response = np.log10(response)
            response[np.isneginf(response)] = 0  # -inf -> 0

        X = torch.from_numpy(samples.to_numpy(dtype=np.float32))[good]
        Y = torch.from_numpy(response.astype(np.float32)[good])
        Y[Y < 0] = 0  # clamp negatives post-log

        X_keys = list(samples.columns)
        X_mean = X.mean(dim=0)
        X_std = X.std(dim=0)
        Xn = (X - X_mean) / X_std if cfg.normalize_x else X

        n_parameters = Xn.shape[1]
        n_samples = Y.shape[0]
        n_grid_points = Y.shape[1]

        normed_area = torch.ones(n_grid_points, dtype=torch.float32)
        normed_area /= normed_area.sum()

        return SamplesData(
            X=Xn,
            Y=Y,
            X_keys=X_keys,
            X_mean=X_mean,
            X_std=X_std,
            n_parameters=n_parameters,
            n_samples=n_samples,
            n_grid_points=n_grid_points,
            normed_area=normed_area,
        )
