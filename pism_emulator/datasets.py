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

# pylint: disable=too-many-instance-attributes,too-many-lines,too-many-positional-arguments,too-many-statements
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
from typing import Final, Literal, Sequence, cast

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

YTransformName = Literal["none", "log10", "robust"]


def inverse_y_transform_np(
    Yt: np.ndarray,
    *,
    name: str,
    y_lim: tuple[float, float] | None = None,
    params: dict[str, object] | None = None,
) -> np.ndarray:
    """
    Invert the Y transform.

    Parameters
    ----------
    Yt : numpy.ndarray
        Transformed array with shape ``(..., n_nodes)``.
    name : {"none", "log10", "robust"}
        Transform name to invert.
    y_lim : tuple[float, float] or None, optional
        Optional clamp range in physical units applied after inversion.
    params : dict[str, object] or None, optional
        Transform parameters. For ``"robust"`` this must contain:

        - ``"center"`` : array-like, shape ``(n_nodes,)``
        - ``"scale"``  : array-like, shape ``(n_nodes,)``

    Returns
    -------
    numpy.ndarray
        Inverted array in physical units with shape ``(..., n_nodes)``.

    Raises
    ------
    ValueError
        If ``name`` is unknown or if ``name=="robust"`` but required parameters are missing.
    """

    params = params or {}

    if name == "none":
        Y = Yt

    elif name == "log10":
        # Forward was log10(clamp(Y, *y_lim)); inverse is 10**Yt
        Y = np.power(10.0, Yt)

    elif name == "robust":
        center = params.get("center", None)
        scale = params.get("scale", None)
        if center is None or scale is None:
            raise ValueError(
                "robust inverse needs params['center'] and params['scale']"
            )
        center = np.asarray(center)
        scale = np.asarray(scale)
        Y = Yt * scale + center

    else:
        raise ValueError(f"Unknown y_transform={name!r}")

    if y_lim is not None:
        Y = np.clip(Y, y_lim[0], y_lim[1])

    return Y


def _fit_robust_params(
    Y: torch.Tensor,
    *,
    with_centering: bool = True,
    with_scaling: bool = True,
    quantile_range: tuple[float, float] = (25.0, 75.0),
    unit_variance: bool = False,  # kept for API parity; see notes
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fit robust centering and scaling parameters for a per-feature affine transform.

    This computes a per-feature (e.g., per-grid-node) ``center`` and ``scale`` for
    robust normalization of a response matrix ``Y``. It is designed to mimic the
    core behavior of :class:`sklearn.preprocessing.RobustScaler`:

    * If ``with_centering=True``, the center is the per-feature median.
    * If ``with_scaling=True``, the scale is the inter-quantile range (IQR-like)
      computed from ``quantile_range``.

    Parameters
    ----------
    Y : torch.Tensor
        Input tensor of responses with shape ``(n_runs, n_features)`` (e.g., runs × nodes).
    with_centering : bool, optional
        If True, compute ``center`` as the per-feature median. If False, ``center``
        is all zeros. Default is True.
    with_scaling : bool, optional
        If True, compute ``scale`` as ``q_hi - q_lo`` per feature using
        ``quantile_range``. If False, ``scale`` is all ones. Default is True.
    quantile_range : tuple[float, float], optional
        Quantile range (in percent) used to compute the scaling. Default is
        ``(25.0, 75.0)`` (i.e., IQR).
    unit_variance : bool, optional
        Placeholder for API parity with scikit-learn's RobustScaler. This
        implementation currently does **not** apply the additional rescaling to
        achieve unit variance under a normality assumption. Default is False.
    eps : float, optional
        Minimum allowed scale. Features with ``scale < eps`` are assigned ``1``
        to avoid division by near-zero values. Default is 1e-6.

    Returns
    -------
    center : torch.Tensor
        Per-feature centering vector with shape ``(n_features,)``.
    scale : torch.Tensor
        Per-feature scaling vector with shape ``(n_features,)``.

    Raises
    ------
    ValueError
        If ``quantile_range`` is invalid (e.g., not length 2, or lo >= hi).

    Notes
    -----
    ``quantile_range`` is interpreted as percentiles in ``[0, 100]`` and passed to
    :func:`torch.quantile` after conversion to fractions in ``[0, 1]``.

    Examples
    --------
    >>> Y = torch.tensor([[0.0, 1.0], [2.0, 100.0], [4.0, 3.0]])
    >>> center, scale = _fit_robust_params(Y)
    >>> center.shape, scale.shape
    (torch.Size([2]), torch.Size([2]))
    """
    if len(quantile_range) != 2:
        raise ValueError("quantile_range must be a tuple of (low, high) percentiles")
    q_lo, q_hi = quantile_range
    if not (0.0 <= q_lo < q_hi <= 100.0):
        raise ValueError("quantile_range must satisfy 0 <= low < high <= 100")

    q_lo_t = q_lo / 100.0
    q_hi_t = q_hi / 100.0

    center = torch.zeros(Y.shape[1], dtype=Y.dtype, device=Y.device)
    scale = torch.ones(Y.shape[1], dtype=Y.dtype, device=Y.device)

    if with_centering:
        center = torch.median(Y, dim=0).values

    if with_scaling:
        lo = torch.quantile(Y, q_lo_t, dim=0)
        hi = torch.quantile(Y, q_hi_t, dim=0)
        scale = hi - lo
        scale = torch.where(scale < eps, torch.ones_like(scale), scale)

    # NOTE: sklearn's RobustScaler(unit_variance=True) rescales to unit variance
    # based on a normal distribution assumption; implementing that exactly is optional.
    # If you want it later, we can add the same constant factor sklearn uses.

    _ = unit_variance  # explicitly unused; kept for signature/API parity

    return center, scale


def _apply_affine(
    Y: torch.Tensor, center: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """
    Apply an elementwise affine normalization.

    Parameters
    ----------
    Y : torch.Tensor
        Input tensor to transform.
    center : torch.Tensor
        Centering tensor to subtract from ``Y``. Must be broadcast-compatible with ``Y``.
    scale : torch.Tensor
        Scaling tensor to divide by after centering. Must be broadcast-compatible
        with ``Y`` and should be non-zero.

    Returns
    -------
    torch.Tensor
        Transformed tensor ``(Y - center) / scale``.
    """
    return (Y - center) / scale


def _apply_y_transform(
    Y: torch.Tensor,
    *,
    name: str,
    y_lim: tuple[float, float],
    params: dict[str, object],
) -> tuple[torch.Tensor, dict[str, object]]:
    """
    Apply a named transform to response/target tensors.

    Supported transforms are:

    - ``"none"``: no transform (identity).
    - ``"log10"``: base-10 log of values clamped to ``y_lim``.
    - ``"robust"``: robust affine scaling using a median/quantile-based center and
      scale (similar to :class:`sklearn.preprocessing.RobustScaler`).

    For ``"robust"``, this function can *fit* the transform parameters on the
    provided ``Y`` if they are not already present in ``params``. The returned
    ``params_used`` can then be reused to transform other tensors (e.g., the
    observational target) consistently.

    Parameters
    ----------
    Y : torch.Tensor
        Input response tensor to transform. Typically has shape ``(N, G)`` for
        training data (runs × grid points), but any shape is accepted.
    name : str
        Transform name. Must be one of ``{"none", "log10", "robust"}``.
    y_lim : tuple[float, float]
        Clamp range ``(ymin, ymax)`` used for the ``"log10"`` transform (and often
        applied upstream in physical space). Values are clamped before taking
        ``log10`` to avoid ``-inf``.
    params : dict[str, object]
        Transform configuration and/or fitted parameters.

        For ``"robust"``, the following keys may be provided:

        * ``with_centering`` : bool, optional
        * ``with_scaling`` : bool, optional
        * ``quantile_range`` : tuple[float, float], optional
        * ``unit_variance`` : bool, optional
        * ``center`` : torch.Tensor, optional (fitted)
        * ``scale`` : torch.Tensor, optional (fitted)

        If ``center`` and ``scale`` are absent, they are fit from ``Y``.

    Returns
    -------
    Y_transformed : torch.Tensor
        Transformed tensor.
    params_used : dict[str, object]
        Dictionary of parameters actually used. For ``"robust"`` this will include
        fitted ``center`` and ``scale`` when they were computed.

    Raises
    ------
    ValueError
        If ``name`` is not one of ``"none"``, ``"log10"``, or ``"robust"``.

    Notes
    -----
    For ``"log10"``, this function applies::

        log10(clamp(Y, ymin, ymax))

    To invert the transform, use ``10 ** Y_transformed`` (or ``torch.pow(10, ...)``).
    """
    if name == "none":
        return Y, params

    if name == "log10":
        return torch.log10(torch.clamp(Y, *y_lim)), params

    if name == "robust":
        # fit once on training Y, then reuse for target
        if "center" not in params or "scale" not in params:
            center, scale = _fit_robust_params(
                Y,
                with_centering=bool(params.get("with_centering", True)),
                with_scaling=bool(params.get("with_scaling", True)),
                quantile_range=tuple(params.get("quantile_range", (25.0, 75.0))),  # type: ignore[arg-type]
                unit_variance=bool(params.get("unit_variance", False)),
            )
            params = dict(params)
            params["center"] = center
            params["scale"] = scale
        return _apply_affine(Y, params["center"], params["scale"]), params

    raise ValueError(
        f"Unknown y_transform={name!r}; expected 'none', 'log10', or 'robust'"
    )


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
    """
    Build sorted ``(keys, values)`` arrays from a NetCDF ``pism_config`` variable's attributes.

    This helper reads all NetCDF attributes on ``var`` and filters out metadata
    attributes whose names end with common suffixes (documentation/type/units/etc.).
    The remaining attributes are treated as PISM configuration key/value pairs,
    sorted by key, and returned as NumPy unicode arrays.

    Parameters
    ----------
    var : netCDF4.Variable or xarray.DataArray
        NetCDF variable object representing ``pism_config``. The object must
        provide a ``ncattrs()`` method and allow attribute access via
        ``getattr(var, name)``.

    Returns
    -------
    pc_keys : numpy.ndarray
        Sorted configuration keys as a 1D unicode array (dtype ``"U"``).
    pc_vals : numpy.ndarray
        Configuration values aligned with ``pc_keys`` as a 1D unicode array
        (dtype ``"U"``).

    Notes
    -----
    The following attribute-name suffixes are excluded from the configuration set:

    ``("_doc", "_type", "_units", "_option", "_choices")``

    If the key ``"geometry.front_retreat.prescribed.file"`` is missing, it is
    added with the default value ``"false"``.
    """
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
    engine : str, default="netcdf4"
        xarray engine.
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
    log_y: bool = False
    y_transform: YTransformName | None = None
    y_transform_kwargs: dict[str, object] | None = None
    y_lim: tuple = (1, 100e3)
    epsilon: float = 0.0
    verbose: bool = False
    engine: str = "netcdf4"
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
    y_lim : tuple, optional
        Physical-space clamp range ``(ymin, ymax)`` applied to training responses
        and the target vector *before* transformation. Default is ``(1e-1, 100e3)``.
    epsilon : float, default=0.0
        Replace NaNs with this epsilon when reading arrays.
    verbose : bool, default=False
        Print rank-0 progress.
    engine : str, default="netcdf4"
        The xarray engine.
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
        engine: str = "netcdf4",
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
            engine=engine,
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
        """
        Human-friendly one-liner.

        Returns
        -------
        str
            Dataset Summary.
        """
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
            f"engines=(target='{cfg.engine}', training='{cfg.engine}'))"
        )

    def __repr__(self) -> str:
        """
        Detailed, multi-line summary (compact even with many files).

        Returns
        -------
        str
            Dataset Summary.
        """
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
            f"  engines=(target={repr(cfg.engine)}, training={repr(cfg.engine)}),\n"
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
            rank_zero_info(f"Loading target {cfg.target_file} (engine={cfg.engine})")

        ds = xr.open_dataset(cfg.target_file, decode_times=False, engine=cfg.engine)

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
                    files_by_id[i], cfg.training_var, 1, idx1d, cfg.epsilon
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
    """
    Legacy PyTorch Dataset for PISM ensemble training data and an observational target.

    This class loads:

    * A *target* field from ``target_file`` (e.g., observed velocity) and builds a
      sparse index selecting valid grid cells (optionally filtered by a correlation/
      quality field such as thickness).
    * An ensemble of *training* fields from NetCDF files in ``data_dir`` and
      extracts the values at the target's observed grid cells.
    * A parameter/sample table from ``samples_file`` (CSV) aligned by run ``id``.
    * Optional feature normalization for ``X`` and optional log10 transform for ``Y``.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing training NetCDF files (globbed as ``*.nc``).
        Default is ``"path/to/dir"``.
    samples_file : str, optional
        Path to a CSV file containing ensemble parameters and an ``id`` column.
        Default is ``"path/to/file"``.
    target_file : str or None, optional
        Path to a NetCDF file containing the target/observational field.
        If None, :meth:`load_target` will fail. Default is None.
    target_var : str, optional
        Variable name in ``target_file`` used as the target field (e.g.,
        ``"velsurf_mag"``). Default is ``"velsurf_mag"``.
    target_corr_threshold : float, optional
        Threshold applied to ``target_corr_var`` to mask unreliable target cells.
        Cells with ``target_corr_var < target_corr_threshold`` are masked.
        Default is 25.0.
    target_corr_var : str, optional
        Variable name in ``target_file`` used to build an additional mask
        (e.g., thickness). Default is ``"thickness"``.
    target_error_var : str, optional
        Optional variable name in ``target_file`` containing target error
        (e.g., ``"velsurf_mag_error"``). Default is ``"velsurf_mag_error"``.
    training_var : str, optional
        Variable name in training NetCDF files used as the training response.
        Default is ``"velsurf_mag"``.
    normalize_x : bool, optional
        If True, standardize ``X`` using mean/std computed from the retained
        samples. Default is True.
    log_y : bool, optional
        If True, apply a base-10 log transform to training responses ``Y``.
        Default is True.
    threshold : float, optional
        Upper bound used to filter out runs with extreme responses. Runs are
        retained if their maximum response satisfies ``max(Y) < threshold``.
        Default is 100e3.
    epsilon : float, optional
        Fill value used when replacing NaNs in target/training fields.
        Default is 0.0.
    verbose : bool, optional
        If True, print additional info (e.g., missing run ids). Default is False.

    Notes
    -----
    This is retained for backward compatibility. The newer dataset classes
    (e.g., ``PISMDataset`` / ``PISMInterpolatedDataset``) provide a more explicit
    configuration interface and more robust preprocessing.
    """

    def __init__(
        self,
        data_dir: str = "path/to/dir",
        samples_file: str = "path/to/file",
        target_file: str | None = None,
        target_var: str = "velsurf_mag",
        target_corr_threshold: float = 25.0,
        target_corr_var: str = "thickness",
        target_error_var: str = "velsurf_mag_error",
        training_var: str = "velsurf_mag",
        normalize_x: bool = True,
        log_y: bool = True,
        threshold: float = 100e3,
        epsilon: float = 0.0,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the dataset and load target + training data.

        Parameters
        ----------
        data_dir : str, optional
            Directory containing training NetCDF files (globbed as ``*.nc``).
            Default is ``"path/to/dir"``.
        samples_file : str, optional
            Path to a CSV file containing ensemble parameters and an ``id`` column.
            Default is ``"path/to/file"``.
        target_file : str or None, optional
            Path to a NetCDF file containing the target/observational field.
            If None, :meth:`load_target` will fail. Default is None.
        target_var : str, optional
            Variable name in ``target_file`` used as the target field (e.g.,
            ``"velsurf_mag"``). Default is ``"velsurf_mag"``.
        target_corr_threshold : float, optional
            Threshold applied to ``target_corr_var`` to mask unreliable target cells.
            Cells with ``target_corr_var < target_corr_threshold`` are masked.
            Default is 25.0.
        target_corr_var : str, optional
            Variable name in ``target_file`` used to build an additional mask
            (e.g., thickness). Default is ``"thickness"``.
        target_error_var : str, optional
            Optional variable name in ``target_file`` containing target error
            (e.g., ``"velsurf_mag_error"``). Default is ``"velsurf_mag_error"``.
        training_var : str, optional
            Variable name in training NetCDF files used as the training response.
            Default is ``"velsurf_mag"``.
        normalize_x : bool, optional
            If True, standardize ``X`` using mean/std computed from the retained
            samples. Default is True.
        log_y : bool, optional
            If True, apply a base-10 log transform to training responses ``Y``.
            Default is True.
        threshold : float, optional
            Upper bound used to filter out runs with extreme responses. Runs are
            retained if their maximum response satisfies ``max(Y) < threshold``.
            Default is 100e3.
        epsilon : float, optional
            Fill value used when replacing NaNs in target/training fields.
            Default is 0.0.
        verbose : bool, optional
            If True, print additional info (e.g., missing run ids). Default is False.
        """
        self.data_dir = data_dir
        self.samples_file = samples_file
        self.target_file = target_file
        self.target_var = target_var
        self.target_corr_threshold = target_corr_threshold
        self.target_corr_var = target_corr_var
        self.target_error_var = target_error_var
        self.threshold = threshold
        self.training_var = training_var
        self.epsilon = epsilon
        self.log_y = log_y
        self.normalize_x = normalize_x
        self.verbose = verbose

        self.load_target()
        self.load_data()

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        """
        Get one sample.

        Parameters
        ----------
        i : int
            Sample index.

        Returns
        -------
        tuple of torch.Tensor
            ``(X[i], Y[i])`` where ``X`` are normalized (if enabled) parameters and
            ``Y`` are the corresponding training responses at observed grid cells.
        """
        return tuple(d[i] for d in [self.X, self.Y])

    def __len__(self) -> int:
        """
        Return the number of samples.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return min(len(d) for d in [self.X, self.Y])

    def load_target(self) -> None:
        """
        Load the target field and build sparse indices for observed grid cells.

        This method reads ``target_var`` from ``target_file`` and creates a mask
        of invalid cells using NaNs and (optionally) ``target_corr_var``. It then
        stores:

        * ``Y_target`` : target values at valid grid cells (1D tensor)
        * ``Y_target_2d`` : masked 2D target field (numpy masked array)
        * ``sparse_idx_2d`` / ``sparse_idx_1d`` : indices of valid cells
        * Optional error/correlation fields if present.

        Notes
        -----
        The target values are **not** transformed (e.g., no log scaling) in this
        legacy implementation; only training ``Y`` may be log-transformed in
        :meth:`load_data`.
        """
        epsilon = self.epsilon
        rank_zero_info(f"Loading target {self.target_file}")

        ds = xr.open_dataset(self.target_file, decode_times=False)
        data = ds[self.target_var].squeeze()
        mask = data.isnull()
        data = np.nan_to_num(data.values, nan=epsilon)

        ny, nx = data.shape

        self.target_has_error = False
        if self.target_error_var in ds.variables:
            data_error = ds[self.target_error_var].squeeze()
            data_error = np.nan_to_num(data_error.values, nan=epsilon)
            self.target_has_error = True

        self.target_has_corr = False
        if self.target_corr_var in ds.variables:
            data_corr = ds[self.target_corr_var].squeeze()
            data_corr = np.nan_to_num(data_corr.values, nan=epsilon)
            self.target_has_corr = True
            mask = mask.where(data_corr >= self.target_corr_threshold, True)

        mask = mask.values

        grid_resolution = np.abs(np.diff(ds["x"][0:2]))[0]
        self.grid_resolution = grid_resolution

        ds.close()

        idx = (mask == 0).nonzero()

        data = data[idx]
        self.Y_target = torch.from_numpy(np.array(data.flatten(), dtype=np.float32))

        if self.target_has_error:
            data_error = data_error[idx]
            self.Y_target_error_2d = data_error
            self.Y_target_error = torch.from_numpy(
                np.array(data_error.flatten(), dtype=np.float32)
            )

        if self.target_has_corr:
            data_corr = data_corr[idx]
            self.Y_target_corr_2d = data_corr
            self.Y_target_corr = torch.from_numpy(
                np.array(data_corr.flatten(), dtype=np.float32)
            )

        self.mask_2d = mask
        self.sparse_idx_2d = idx
        self.sparse_idx_1d = np.ravel_multi_index(idx, mask.shape)

        data_2d = np.zeros((ny, nx))
        data_2d.put(self.sparse_idx_1d, data)
        self.Y_target_2d = np.ma.array(data=data_2d, mask=self.mask_2d)

    def load_data(self) -> None:
        """
        Load training responses and parameter samples, align by run id, and preprocess.

        This method:

        1. Finds all training NetCDF files in ``data_dir`` and extracts integer run
           ids from filenames matching ``id_(\\d+)_``.
        2. Loads the samples CSV from ``samples_file`` and drops any rows with ids
           not present in the training file set.
        3. Extracts the training response at the target's valid grid cells for each run.
        4. Filters out runs whose maximum response exceeds ``threshold``.
        5. Optionally log10-transforms responses (``log_y=True``) and normalizes
           the sample parameters (``normalize_x=True``).
        """
        epsilon = self.epsilon

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

        missing_ids = list(set(samples["id"]).difference(ids_df["id"]))
        if missing_ids:
            if self.verbose:
                rank_zero_info(
                    f"The following simulations are missing:\n   {missing_ids}"
                )
                rank_zero_info("  adjusting priors")
            samples = samples[~samples["id"].isin(missing_ids)]

        samples = samples.drop(samples.columns[0], axis=1)
        self.X_keys = samples.keys()

        ds0 = xr.open_dataset(training_files[0], decode_times=False)
        _, ny, nx = ds0.variables[self.target_var].values.shape
        ds0.close()

        self.nx = nx
        self.ny = ny

        response = np.zeros((len(samples), len(self.sparse_idx_1d)))

        rank_zero_info("  Loading data sets...")

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
                np.nan_to_num(ds.variables[training_var].values, nan=epsilon)
            )
            response[idx, :] = data[self.sparse_idx_2d].flatten()
            ds.close()

        end_time = time()
        self.training_files = training_files
        rank_zero_info(f"Reading training data took {(end_time-start_time):.0f}s")

        keep = response.max(axis=1) < self.threshold

        if self.log_y:
            response = np.log10(response)
            response[np.isneginf(response)] = 0

        X = torch.from_numpy(np.array(samples[keep], dtype=np.float32))
        Y = torch.from_numpy(np.array(response[keep], dtype=np.float32))

        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0)

        if self.normalize_x:
            X = (X - self.X_mean) / self.X_std

        self.X = X
        self.Y = Y

        self.n_parameters = X.shape[1]
        self.n_samples, self.n_grid_points = Y.shape

        normed_area = torch.ones(self.n_grid_points, dtype=torch.float32)
        self.normed_area = normed_area / normed_area.sum()

    def return_original(self) -> Tensor:
        """
        Return parameters in original (unnormalized) space.

        Returns
        -------
        torch.Tensor
            Parameter tensor ``X`` in physical/original units. If normalization
            was disabled, this simply returns ``self.X``.
        """
        X = self.X
        if self.normalize_x:
            X = X * self.X_std + self.X_mean
        return X


class PISMInterpolatedDataset(Dataset):
    """
    Dataset for PISM emulator training with an interpolated observational target.

    This dataset is similar to :class:`PISMDataset`, but first interpolates the
    target field onto the grid of the first training file using
    :meth:`xarray.DataArray.interp_like`. After interpolation, it builds a sparse
    index of "observed" grid cells (non-masked) and extracts training responses at
    those cells for each ensemble run.

    The dataset supports:
    * feature normalization of parameters ``X`` (standard score per column),
    * response/target transforms via ``y_transform`` (applied consistently to both
      training responses and the target vector),
    * masking based on missing target values and an optional correlation/quality
      variable (e.g., thickness),
    * optional extraction of target error and correlation fields.

    Parameters
    ----------
    training_files : Sequence[str] or str
        Either a sequence of NetCDF paths, or a glob pattern that resolves to the
        training files. The first file is used as the reference grid for target
        interpolation.
    samples_file : str
        Path to a CSV file containing ensemble parameters. Must include an ``id``
        column matching training file run ids parsed from filenames.
    target_file : str
        Path to a NetCDF file containing the target/observational field.
    training_var : str, optional
        Variable name in training files used as the response field. Default is
        ``"velsurf_mag"``.
    target_var : str, optional
        Variable name in ``target_file`` used as the target field. Default is
        ``"velsurf_mag"``.
    target_corr_var : str, optional
        Optional variable name in ``target_file`` used to mask target cells based
        on a correlation/quality threshold (e.g., ``"thickness"``). Default is
        ``"thickness"``.
    target_error_var : str, optional
        Optional variable name in ``target_file`` containing target uncertainties.
        Default is ``"velsurf_mag_error"``.
    target_corr_threshold : float, optional
        Threshold applied to ``target_corr_var`` when present. Cells with
        ``target_corr_var < target_corr_threshold`` are masked. Default is 25.0.
    normalize_x : bool, optional
        If True, standardize ``X`` using mean/std over the retained samples.
        Default is True.
    y_transform : YTransformName or None, optional
        Name of the response/target transform. If None, no transform is applied.
        Typical values include ``"none"``, ``"log10"``, and ``"robust"``.
        Default is None.
    y_transform_kwargs : dict[str, object] or None, optional
        Optional keyword arguments/parameters for the chosen transform. For
        example, robust scaling may use pre-fit parameters (center/scale) or
        configuration options. Default is None.
    y_lim : tuple, optional
        Physical-space clamp range ``(ymin, ymax)`` applied to training responses
        and the target vector *before* transformation. Default is ``(1e-1, 100e3)``.
    epsilon : float, optional
        Fill value used when replacing missing target values or training values.
        Default is 0.0.
    engine : str, optional
        Xarray backend engine used to read NetCDF files (e.g., ``"netcdf4"``).
        Default is ``"netcdf4"``.
    parallel : bool, optional
        If True, read training files in parallel using a process pool. Default is True.
    chunks_after : dict[str, int] or None, optional
        Optional chunking dictionary applied after load (reserved for dask/xarray
        workflows). Default is None.
    dask_scheduler : str or None, optional
        Optional Dask scheduler name to set via ``dask.config.set``. Default is None.

    Attributes
    ----------
    cfg : DatasetConfig
        Configuration object capturing file paths, variable names, and preprocessing
        options.
    target : TargetData
        Target data container (mask, indices, target vector, and optional error/corr).
    samples : SamplesData
        Sample/response container storing normalized features ``X`` and responses ``Y``.

    Notes
    -----
    * The target is clamped to ``y_lim`` during target loading, but **not**
      transformed there; transforms are applied later in
      :meth:`_load_training_and_samples` so that training and target transforms are
      guaranteed to match.
    * Run filtering uses the maximum value of each run's response in physical
      space: a run is kept if ``max(response) < y_lim[1]``.
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
        y_transform: YTransformName | None = None,
        y_transform_kwargs: dict[str, object] | None = None,
        y_lim: tuple = (1e-1, 100e3),
        epsilon: float = 0.0,
        engine: str = "netcdf4",
        parallel: bool = True,
        chunks_after: dict[str, int] | None = None,
        dask_scheduler: str | None = None,
    ) -> None:
        """
        Initialize the dataset, load the target (interpolated) and training samples.

        See class docstring for parameter descriptions.

        Parameters
        ----------
        training_files : Sequence[str] or str
            Either a sequence of NetCDF paths, or a glob pattern that resolves to the
            training files. The first file is used as the reference grid for target
            interpolation.
        samples_file : str
            Path to a CSV file containing ensemble parameters. Must include an ``id``
            column matching training file run ids parsed from filenames.
        target_file : str
            Path to a NetCDF file containing the target/observational field.
        training_var : str, optional
            Variable name in training files used as the response field. Default is
            ``"velsurf_mag"``.
        target_var : str, optional
            Variable name in ``target_file`` used as the target field. Default is
            ``"velsurf_mag"``.
        target_corr_var : str, optional
            Optional variable name in ``target_file`` used to mask target cells based
            on a correlation/quality threshold (e.g., ``"thickness"``). Default is
            ``"thickness"``.
        target_error_var : str, optional
            Optional variable name in ``target_file`` containing target uncertainties.
            Default is ``"velsurf_mag_error"``.
        target_corr_threshold : float, optional
            Threshold applied to ``target_corr_var`` when present. Cells with
            ``target_corr_var < target_corr_threshold`` are masked. Default is 25.0.
        normalize_x : bool, optional
            If True, standardize ``X`` using mean/std over the retained samples.
            Default is True.
        y_transform : YTransformName or None, optional
            Name of the response/target transform. If None, no transform is applied.
            Typical values include ``"none"``, ``"log10"``, and ``"robust"``.
            Default is None.
        y_transform_kwargs : dict[str, object] or None, optional
            Optional keyword arguments/parameters for the chosen transform. For
            example, robust scaling may use pre-fit parameters (center/scale) or
            configuration options. Default is None.
        y_lim : tuple, optional
            Physical-space clamp range ``(ymin, ymax)`` applied to training responses
            and the target vector *before* transformation. Default is ``(1e-1, 100e3)``.
        epsilon : float, optional
            Fill value used when replacing missing target values or training values.
            Default is 0.0.
        engine : str, optional
            Xarray backend engine used to read NetCDF files (e.g., ``"netcdf4"``).
            Default is ``"netcdf4"``.
        parallel : bool, optional
            If True, read training files in parallel using a process pool. Default is True.
        chunks_after : dict[str, int] or None, optional
            Optional chunking dictionary applied after load (reserved for dask/xarray
            workflows). Default is None.
        dask_scheduler : str or None, optional
            Optional Dask scheduler name to set via ``dask.config.set``. Default is None.

        Raises
        ------
        TypeError
            If ``training_files`` is not a sequence of paths or a glob pattern string.
        FileNotFoundError
            If no training files are provided or resolved.
        KeyError
            If required variables are missing from the reference training file or
            target file.
        ValueError
            If run ids cannot be parsed from training filenames.
        """

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
            y_lim=tuple(y_lim),
            y_transform=y_transform if y_transform is not None else "none",
            epsilon=float(epsilon),
            engine=engine,
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
            f"normalize_x={cfg.normalize_x}, y_transform={cfg.y_transform}, "
            f"training_var='{cfg.training_var}', target_var='{cfg.target_var}', "
            f"engines=(target='{cfg.engine}', training='{cfg.engine}'))"
        )

    @staticmethod
    def _parse_id(path: str) -> int:
        """
        Parse the integer run id from a training filename.

        Parameters
        ----------
        path : str
            Path to a training file. The filename must contain a substring
            matching ``id_(\\d+)_``.

        Returns
        -------
        int
            Parsed run id.

        Raises
        ------
        ValueError
            If the id cannot be parsed from the filename.
        """
        m = re.search(r"id_(\d+)_", Path(path).name)
        if not m:
            raise ValueError(f"Could not parse id from filename: {path}")
        return int(m.group(1))

    def _load_target_interp_and_mask(self) -> TargetData:
        """
        Load the target file, interpolate it to the reference grid, and build masks/indices.

        This method:
        1. Opens the first training file to establish a reference grid.
        2. Opens the target file and interpolates ``target_var`` to the reference grid
           using :meth:`xarray.DataArray.interp_like`.
        3. Constructs a boolean mask from NaNs and, if present, masks additional
           cells where ``target_corr_var < target_corr_threshold``.
        4. Builds sparse indices (2D and 1D) for non-masked grid cells and vectorizes
           the interpolated target field at those cells.
        5. Optionally extracts and vectorizes target error and correlation fields.

        Returns
        -------
        TargetData
            Target container including:
            * ``Y_target`` (1D torch tensor on observed nodes),
            * ``Y_target_2d`` (masked 2D numpy array),
            * mask and sparse indices,
            * optional error and correlation vectors/2D arrays.

        Raises
        ------
        FileNotFoundError
            If no training files are available.
        KeyError
            If ``training_var`` is missing from the reference training file or if
            ``target_var`` is missing from the target file.
        """
        cfg = self.cfg

        rank_zero_info(f"Loading target {cfg.target_file} (engine={cfg.engine})")
        rank_zero_info("   Establishing reference grid from first training file")

        if not cfg.training_files:
            raise FileNotFoundError("No training files provided")

        ref_path = cfg.training_files[0]
        dref = xr.open_dataset(ref_path, decode_times=False, engine=cfg.engine)
        if cfg.training_var not in dref:
            dref.close()
            raise KeyError(f"'{cfg.training_var}' not found in {ref_path}")
        ref_da = dref[cfg.training_var]

        ref_y_dim, ref_x_dim = ref_da.dims[-2], ref_da.dims[-1]
        ny = int(ref_da.sizes[ref_y_dim])
        nx = int(ref_da.sizes[ref_x_dim])

        dtgt = xr.open_dataset(cfg.target_file, decode_times=False, engine=cfg.engine)
        if cfg.target_var not in dtgt:
            dref.close()
            dtgt.close()
            raise KeyError(f"'{cfg.target_var}' not found in target file")

        targ_da = dtgt[cfg.target_var].squeeze()

        tgt_y_dim, tgt_x_dim = targ_da.dims[-2], targ_da.dims[-1]
        if (tgt_y_dim, tgt_x_dim) != (ref_y_dim, ref_x_dim):
            targ_da = targ_da.rename({tgt_y_dim: ref_y_dim, tgt_x_dim: ref_x_dim})

        targ_interp = targ_da.interp_like(ref_da.squeeze())

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
        Y_target = torch.clamp(Y_target, *cfg.y_lim)
        # DO NOT transform here anymore

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
        """
        Load training responses and parameter samples, apply filtering and transforms.

        This method:
        1. Parses run ids from training filenames and aligns them with the samples CSV.
        2. Reads training responses at the target's observed nodes (optionally in parallel).
        3. Filters out runs with extreme values using the physical-space upper bound
           ``y_lim[1]``.
        4. Builds tensors ``X`` (parameters) and ``Y`` (responses), clamps ``Y`` to
           ``y_lim``, and applies ``y_transform`` to both ``Y`` and the target vector.
        5. Optionally standardizes ``X`` (mean/std) when ``normalize_x=True``.

        Returns
        -------
        SamplesData
            Container with fields:
            * ``X`` : feature tensor (normalized if enabled), shape ``(N, P)``
            * ``Y`` : response tensor (transformed if enabled), shape ``(N, G_obs)``
            * ``X_mean`` / ``X_std`` : feature normalization statistics
            * ``normed_area`` : uniform normalized weights over observed nodes

        Notes
        -----
        The same transform parameters ``params`` returned by ``_apply_y_transform`` are
        reused to transform the target vector, ensuring the model is trained and
        evaluated in a consistent response space.
        """
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
                    files_by_id[i], cfg.training_var, 1, idx1d, cfg.epsilon
                )

        end_time = time()
        rank_zero_info(f"Reading training data took {(end_time - start_time):.0f}s")
        # Same run filtering policy: use upper y_lim bound in *physical* space proxy.
        good = response.max(axis=1) < cfg.y_lim[1]

        X = torch.from_numpy(samples.to_numpy(dtype=np.float32))[good]
        Y = torch.from_numpy(response.astype(np.float32)[good])
        Y = torch.clamp(Y, *cfg.y_lim)

        name = cfg.y_transform
        params = dict(cfg.y_transform_kwargs or {})

        Y, params = _apply_y_transform(Y, name=name, y_lim=cfg.y_lim, params=params)

        # Apply same transform to the target vector (already clamped in _load_target_and_mask)
        tY = self.target.Y_target
        tY, _ = _apply_y_transform(tY, name=name, y_lim=cfg.y_lim, params=params)
        self.target.Y_target = tY

        # Rebuild Y_target_2d so it matches transformed Y_target
        y2d = np.zeros((self.target.ny, self.target.nx), dtype=np.float32)
        y2d.flat[self.target.sparse_idx_1d] = (
            self.target.Y_target.detach().cpu().numpy()
        )
        self.target.Y_target_2d = np.ma.array(data=y2d, mask=self.target.mask_2d)

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
