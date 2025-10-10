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
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from glob import glob
from os.path import join
from pathlib import Path
from re import Pattern
from time import time
from typing import Final, Sequence, cast

import dask

# add at top-level
import netCDF4 as nc
import numpy as np
import pandas as pd
import rioxarray
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


def build_config_arrays(var) -> tuple[np.ndarray, np.ndarray]:
    """Read attrs from variable 'pism_config' and return sorted (keys, values) arrays."""
    suffixes_to_exclude = ("_doc", "_type", "_units", "_option", "_choices")

    attrs = {k: getattr(var, k) for k in var.ncattrs()}
    config = {
        k: v
        for k, v in attrs.items()
        if not any(k.endswith(suf) for suf in suffixes_to_exclude)
    }
    # default if missing
    config.setdefault("geometry.front_retreat.prescribed.file", "false")

    config_sorted = OrderedDict(sorted(config.items()))
    pc_keys = np.array(list(config_sorted.keys()), dtype="U")  # Unicode strings
    pc_vals = np.array(list(config_sorted.values()), dtype="U")
    return pc_keys, pc_vals


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
        Number of grid points in the ``y`` dimension.
    nx : int
        Number of grid points in the ``x`` dimension.
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
    normalize_x : bool, default=True
        If ``True``, z-score normalize feature columns (per column mean/std).
    log_y : bool, default=True
        If ``True``, apply ``log10`` to responses after reading; ``-inf`` values
        are set to 0 to match legacy behavior.
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
    normalize_x: bool = True
    log_y: bool = True
    y_lim: tuple = (1, 100e3)
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
    - applies optional log10 transform,
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
    normalize_x : bool, default=True
        If True, z-score normalize the features (per column).
    log_y : bool, default=True
        If True, apply ``log10`` to responses (match previous behavior).
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
        normalize_x: bool = True,
        log_y: bool = True,
        y_lim: tuple = (1, 100e3),
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
            normalize_x=bool(normalize_x),
            log_y=bool(log_y),
            y_lim=tuple(y_lim),
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

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples.X[i], self.samples.Y[i]

    def __len__(self) -> int:
        return min(self.samples.n_samples, self.samples.Y.shape[0])

    def __str__(self) -> str:
        """Human-friendly one-liner."""
        cfg = self.cfg
        tgt = self.target
        smp = self.samples
        return (
            "PISMDataset("
            f"n_samples={smp.n_samples}, "
            f"n_parameters={smp.n_parameters}, "
            f"grid={tgt.ny}x{tgt.nx}, "
            f"observed_nodes={tgt.sparse_idx_1d.size}, "
            f"normalize_x={cfg.normalize_x}, "
            f"log_y={cfg.log_y}, "
            f"training_var='{cfg.training_var}', "
            f"target_var='{cfg.target_var}', "
            f"engines=(target='{cfg.target_engine}', training='{cfg.training_engine}'))"
        )

    def __repr__(self) -> str:
        """Detailed, multi-line summary (compact even with many files)."""
        cfg = self.cfg
        tgt = self.target
        smp = self.samples

        def _short_list(items, *, max_items: int = 3) -> str:
            items = list(items)
            n = len(items)
            if n == 0:
                return "[]"
            if n <= max_items:
                body = ", ".join(repr(s) for s in items)
                return f"[{body}]"
            head = ", ".join(repr(s) for s in items[:max_items])
            return f"[{head}, ...]  # {n} items"

        return (
            "PISMDataset(\n"
            f"  training_files={_short_list(cfg.training_files)},\n"
            f"  samples_file={repr(cfg.samples_file)},\n"
            f"  target_file={repr(cfg.target_file)},\n"
            f"  training_var={repr(cfg.training_var)}, target_var={repr(cfg.target_var)},\n"
            f"  target_corr_var={repr(cfg.target_corr_var)}, target_error_var={repr(cfg.target_error_var)},\n"
            f"  target_corr_threshold={cfg.target_corr_threshold}, y_lim={cfg.y_lim},\n"
            f"  normalize_x={cfg.normalize_x}, log_y={cfg.log_y}, epsilon={cfg.epsilon}, parallel={cfg.parallel},\n"
            f"  engines=(target={repr(cfg.target_engine)}, training={repr(cfg.training_engine)}),\n"
            f"  # Derived/runtime\n"
            f"  n_samples={smp.n_samples}, n_parameters={smp.n_parameters}, n_grid_points={smp.n_grid_points},\n"
            f"  grid_shape=({tgt.ny}, {tgt.nx}), observed_nodes={tgt.sparse_idx_1d.size}, "
            f"grid_resolution={tgt.grid_resolution}\n"
            ")"
        )

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
        )

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
        Y_target = torch.clamp(Y_target, *cfg.y_lim)
        if cfg.log_y:
            Y_target = torch.log10(Y_target)

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
                        1,
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

        good = response.max(axis=1) < cfg.y_lim[1]

        X = torch.from_numpy(samples.to_numpy(dtype=np.float32))[good]
        Y = torch.from_numpy(response.astype(np.float32)[good])
        if cfg.log_y:
            Y = torch.log10(torch.clamp(Y, *cfg.y_lim))

        X_keys = list(samples.columns)
        X_mean = X.mean(dim=0)
        X_std = X.std(dim=0)
        Xn = (X - X_mean) / X_std if cfg.normalize_x else X

        n_parameters = Xn.shape[1]
        n_samples = Y.shape[0]
        n_grid_points = Y.shape[1]

        normed_area = torch.ones(n_grid_points, dtype=torch.float32)
        normed_area /= normed_area.sum()

        if self.cfg.verbose:
            rank_zero_info(f"X: {X.shape}")
            rank_zero_info(f"Y {Y.shape}")
            rank_zero_info(f"normed_area: {normed_area.shape}")

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


class LegacyPISMDataset(torch.utils.data.Dataset):
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
        thinning_factor=1,
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
        self.thinning_factor = thinning_factor
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
        thinning_factor = self.thinning_factor
        rank_zero_info(f"Loading target {self.target_file}")
        ds = xr.open_dataset(self.target_file, decode_times=False)
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
        thinning_factor = self.thinning_factor

        identifier_name = "id"
        training_var = self.training_var
        training_files = glob(join(self.data_dir, "*.nc"))
        training_files = list(OrderedDict.fromkeys(training_files))
        pat: Pattern[str] = re.compile(r"id_(\d+)_")

        ids: list[int] = []
        for f in training_files:
            if (m := pat.search(f)) is None:
                raise ValueError(f"Could not find id_..._ in filename: {f!r}")
            ids.append(int(m.group(1)))
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
                rank_zero_info(
                    f"The following simulations are missing:\n   {missing_ids}"
                )
                rank_zero_info("  adjusting priors")
            # and remove the missing samples and responses
            samples_missing_removed = samples[~samples["id"].isin(missing_ids)]
            samples = samples_missing_removed

        samples = samples.drop(samples.columns[0], axis=1)
        m_samples, n_parameters = samples.shape
        self.X_keys = samples.keys()

        ds0 = xr.open_dataset(training_files[0], decode_times=False)
        _, ny, nx = ds0.variables[self.target_var].values.shape

        ds0.close()
        self.nx = nx
        self.ny = ny
        response = np.zeros((m_samples, len(self.sparse_idx_1d)))

        rank_zero_info("  Loading data sets...")
        pat = re.compile(r"id_(\d+)_")

        def _id_key(path: str) -> int:
            m = pat.search(path)
            if m is None:
                raise ValueError(f"Could not find id_..._ in filename: {path!r}")
            return int(m.group(1))

        training_files.sort(key=_id_key)
        start_time = time()
        for idx, m_file in tqdm_rank0(
            enumerate(training_files), total=len(training_files)
        ):
            ds = xr.open_dataset(m_file, decode_times=False)
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
        rank_zero_info(f"Reading training data took {(end_time-start_time):.0f}s")

        p = response.max(axis=1) < 100e3
        if self.log_y:
            response = np.log10(response)
            response[np.isneginf(response)] = 0

        X = torch.from_numpy(np.array(samples[p], dtype=np.float32))
        Y = torch.from_numpy(np.array(response[p], dtype=np.float32))

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


class PISMInterpolatedDataset(Dataset):
    """
    Like PISMDataset, but interpolates the target onto the first training file's grid
    using xarray.interp_like, then builds the sparse node index
    from the interpolated target.

    Behavior mirrors PISMDataset for:
      - y_lim-based filtering (upper limit for run filtering; clamped for log),
      - log_y transform (safe log10 with clamp to y_lim),
      - feature normalization,
      - DatasetConfig/TargetData/SamplesData usage.
    """

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
        normalize_x: bool = True,
        log_y: bool = True,
        y_lim: tuple = (1, 100e3),
        epsilon: float = 0.0,
        target_engine: str | None = None,
        training_engine: str | None = None,
        parallel: bool = True,
        chunks_after: dict[str, int] | None = None,
        dask_scheduler: str | None = None,
    ) -> None:
        # Resolve training files (same logic as PISMDataset)
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
            normalize_x=bool(normalize_x),
            log_y=bool(log_y),
            y_lim=tuple(y_lim),
            epsilon=float(epsilon),
            target_engine=target_engine or _sniff_engine(target_file),
            training_engine=training_engine or "h5netcdf",
            parallel=bool(parallel),
            chunks_after=chunks_after or {},
            dask_scheduler=dask_scheduler,
        )
        self.cfg = cfg

        # Build target (interpolated) & samples blocks
        self.target: TargetData = self._load_target_interp_and_mask()
        self.samples: SamplesData = self._load_training_and_samples()

    # -------------------- Dataset API --------------------
    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples.X[i], self.samples.Y[i]

    def __len__(self) -> int:
        return min(self.samples.n_samples, self.samples.Y.shape[0])

    def __str__(self) -> str:
        cfg = self.cfg
        tgt = self.target
        smp = self.samples
        return (
            "PISMInterpolatedDataset("
            f"n_samples={smp.n_samples}, n_parameters={smp.n_parameters}, "
            f"grid={tgt.ny}x{tgt.nx}, observed_nodes={tgt.sparse_idx_1d.size}, "
            f"normalize_x={cfg.normalize_x}, log_y={cfg.log_y}, "
            f"training_var='{cfg.training_var}', target_var='{cfg.target_var}', "
            f"engines=(target='{cfg.target_engine}', training='{cfg.training_engine}'))"
        )

    @staticmethod
    def _parse_id(path: str) -> int:
        m = re.search(r"id_(\d+)_", Path(path).name)
        if not m:
            raise ValueError(f"Could not parse id from filename: {path}")
        return int(m.group(1))

    # ---------- target / mask (with interpolation) ----------
    def _load_target_interp_and_mask(self) -> TargetData:
        cfg = self.cfg

        rank_zero_info(f"Loading target {cfg.target_file} (engine={cfg.target_engine})")
        rank_zero_info("  Establishing reference grid from first training file")

        if not cfg.training_files:
            raise FileNotFoundError("No training files provided")

        # 1) Open first training file -> reference grid
        ref_path = cfg.training_files[0]
        dref = xr.open_dataset(ref_path, decode_times=False, engine=cfg.training_engine)
        if cfg.training_var not in dref:
            dref.close()
            raise KeyError(f"'{cfg.training_var}' not found in {ref_path}")
        ref_da = dref[cfg.training_var]

        # Determine y/x dims robustly using last two dims
        ref_y_dim, ref_x_dim = ref_da.dims[-2], ref_da.dims[-1]
        ny = int(ref_da.sizes[ref_y_dim])
        nx = int(ref_da.sizes[ref_x_dim])

        # 2) Load target and interp_like onto ref grid
        dtgt = xr.open_dataset(
            cfg.target_file, decode_times=False, engine=cfg.target_engine
        )
        if cfg.target_var not in dtgt:
            dref.close()
            dtgt.close()
            raise KeyError(f"'{cfg.target_var}' not found in target file")

        targ_da = dtgt[cfg.target_var].squeeze()

        # Rename last two dims to match ref if needed
        tgt_y_dim, tgt_x_dim = targ_da.dims[-2], targ_da.dims[-1]
        if (tgt_y_dim, tgt_x_dim) != (ref_y_dim, ref_x_dim):
            targ_da = targ_da.rename({tgt_y_dim: ref_y_dim, tgt_x_dim: ref_x_dim})

        # Interpolate (linear) onto the ref grid
        targ_interp = targ_da.interp_like(ref_da.squeeze())

        # Optional extra masking via correlation field (interp it too, if present)
        mask = targ_interp.isnull()
        has_err = cfg.target_error_var in dtgt
        has_corr = cfg.target_corr_var in dtgt

        if has_corr:
            corr_da = dtgt[cfg.target_corr_var].squeeze(drop=True)
            cy, cx = corr_da.dims[-2], corr_da.dims[-1]
            if (cy, cx) != (ref_y_dim, ref_x_dim):
                corr_da = corr_da.rename({cy: ref_y_dim, cx: ref_x_dim})
            corr_interp = corr_da.interp_like(ref_da, method="nearest")
            mask = xr.where(corr_interp < cfg.target_corr_threshold, True, mask)

        mask_np = mask.compute().values.astype(bool)

        # 3) Vectorize the interpolated target on observed nodes
        sparse_idx_2d = np.where(~mask_np)
        sparse_idx_1d = np.ravel_multi_index(sparse_idx_2d, (ny, nx))

        targ_vec = (
            targ_interp.transpose(..., ref_y_dim, ref_x_dim)
            .fillna(cfg.epsilon)
            .astype("float32")
            .compute()
            .values.reshape(ny * nx)
        )
        targ_vec = targ_vec[sparse_idx_1d]
        Y_target = torch.from_numpy(targ_vec.astype(np.float32))

        # Apply same log scaling policy as PISMDataset
        if cfg.log_y:
            # Clamp to y_lim then log10, identical to PISMDataset
            Y_target = torch.log10(torch.clamp(Y_target, *cfg.y_lim))

        # Optional error/corr 2D fields (already on ref grid)
        Y_target_error = Y_target_error_2d = None
        if has_err:
            err_interp = (
                dtgt[cfg.target_error_var]
                .squeeze(drop=True)
                .rename({tgt_y_dim: ref_y_dim, tgt_x_dim: ref_x_dim})
                .interp_like(ref_da, method="nearest")
                .fillna(cfg.epsilon)
                .astype("float32")
                .compute()
                .values
            )
            Y_target_error_2d = err_interp
            err_vec = err_interp.reshape(ny * nx)[sparse_idx_1d]
            Y_target_error = torch.from_numpy(err_vec.astype(np.float32))

        Y_target_corr = Y_target_corr_2d = None
        if has_corr:
            corr_arr = (
                corr_interp.fillna(cfg.epsilon).astype("float32").compute().values
            )
            Y_target_corr_2d = corr_arr
            corr_vec = corr_arr.reshape(ny * nx)[sparse_idx_1d]
            Y_target_corr = torch.from_numpy(corr_vec.astype(np.float32))

        # Grid resolution (assumes uniform x spacing on ref grid)
        try:
            xcoord = dref[ref_x_dim].isel({ref_x_dim: slice(None, None, 1)})
            grid_resolution = float(abs(xcoord[1] - xcoord[0]))
        except Exception:
            grid_resolution = float("nan")

        # Build a masked 2D view of Y_target for convenience
        y2d = np.zeros((ny, nx), dtype=np.float32)
        y2d.flat[sparse_idx_1d] = Y_target.numpy()
        Y_target_2d = np.ma.array(data=y2d, mask=mask_np)

        # Close datasets
        dref.close()
        dtgt.close()

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

    # ---------- training + samples (same policy as PISMDataset) ----------
    def _load_training_and_samples(self) -> SamplesData:
        cfg, tgt = self.cfg, self.target

        rank_zero_info("  Loading samples & training responses (netCDF4 direct)")

        ids = [self._parse_id(p) for p in cfg.training_files]
        files_by_id = dict(zip(ids, cfg.training_files))

        samples = (
            pd.read_csv(cfg.samples_file, delimiter=",", skipinitialspace=True)
            .sort_values("id")
            .set_index("id", drop=True)
        )

        keep_ids = [i for i in sorted(files_by_id) if i in samples.index]

        missing = sorted(set(samples.index).difference(keep_ids))
        if missing:
            rank_zero_info(f"  Missing runs (dropping from samples): {missing}")

        samples = samples.loc[keep_ids]

        n_runs = len(keep_ids)
        n_nodes = tgt.sparse_idx_1d.size
        response = np.empty((n_runs, n_nodes), dtype=np.float32)

        idx1d = tgt.sparse_idx_1d

        start_time = time()
        total = len(keep_ids)

        if cfg.parallel:
            workers: int = min(8, os.cpu_count() or 1)
            with ProcessPoolExecutor(max_workers=workers) as ex:
                future_to_row = {
                    ex.submit(
                        _read_one_nc4,  # <-- same helper you already use
                        files_by_id[i],
                        cfg.training_var,
                        1,
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
        # Same run filtering policy: use upper y_lim bound in *physical* space proxy.
        good = response.max(axis=1) < cfg.y_lim[1]

        X = torch.from_numpy(samples.to_numpy(dtype=np.float32))[good]
        Y = torch.from_numpy(response.astype(np.float32)[good])
        Y = torch.clamp(Y, *cfg.y_lim)
        # Same scaling policy: clamp to y_lim and log10 if requested
        if cfg.log_y:
            Y = torch.log10(Y)

        X_keys = list(samples.columns)
        X_mean = X.mean(dim=0)
        X_std = X.std(dim=0)
        eps = 1e-6
        X_std = torch.where(X_std < eps, torch.ones_like(X_std), X_std)
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
