# Copyright (C) 2026 Andy Aschwanden
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
Training one emulator on several glaciers.

Each glacier has its own model grid, observations and ensemble output. The emulator
and the MCMC sampler are glacier-agnostic: they see one long vector per ensemble
member, made of each glacier's observed (ice) cells one glacier after another.
:class:`GlacierIndex` records where each glacier sits in that vector, so that
evaluation and plots can split it back into per-glacier maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import xarray as xr
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch.utils.data import Dataset

from pism_emulator.datasets import (
    PISMInterpolatedDataset,
    SamplesData,
    YTransformName,
    _apply_y_transform,
)

#: Default location of the state files inside a glacier directory.
DEFAULT_STATE_GLOB = "output/state/state_*_uq_*.nc"
#: Default location of the ensemble parameter file inside a glacier directory.
DEFAULT_SAMPLES_FILE = "output/uq.csv"
#: File name of the saved :class:`GlacierIndex` in an emulator directory.
GLACIER_INDEX_FILE = "glacier_index.nc"


@dataclass
class GlacierSpec:
    """
    Input files of one glacier.

    Attributes
    ----------
    rgi_id : str
        Glacier identifier (e.g. ``"RGI2000-v7.0-C-01-03383"``).
    training_files : list[str]
        Ensemble output files of this glacier.
    target_file : str
        Observations of this glacier.
    samples_file : str
        Ensemble parameters (CSV with a ``uq`` or ``id`` column).
    """

    rgi_id: str
    training_files: list[str]
    target_file: str
    samples_file: str


def discover_glaciers(
    glacier_dir: str | Path,
    target_template: str,
    *,
    state_glob: str = DEFAULT_STATE_GLOB,
    samples_file: str = DEFAULT_SAMPLES_FILE,
    glaciers: Sequence[str] | None = None,
) -> list[GlacierSpec]:
    """
    Find the glaciers of a calibration directory and their input files.

    Every subdirectory of ``glacier_dir`` that starts with ``RGI`` is a glacier.
    Glaciers without state files, samples file or observations are skipped with a
    message.

    Parameters
    ----------
    glacier_dir : str or pathlib.Path
        Directory with one subdirectory per glacier.
    target_template : str
        Path of a glacier's observations, with ``{rgi_id}`` as placeholder, e.g.
        ``"~/glacier_s4f_input/{rgi_id}/input/obs_{rgi_id}.nc"``.
    state_glob : str, optional
        Glob, relative to a glacier directory, matching its ensemble output.
    samples_file : str, optional
        Path, relative to a glacier directory, of its ensemble parameters.
    glaciers : sequence of str or None, optional
        Only use these glaciers. Default: all.

    Returns
    -------
    list[GlacierSpec]
        Usable glaciers, sorted by id.

    Raises
    ------
    FileNotFoundError
        If ``glacier_dir`` does not exist, a requested glacier is missing, or no
        glacier is usable.
    """
    root = Path(glacier_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Glacier directory {root} does not exist")
    found = sorted(
        p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("RGI")
    )
    if glaciers:
        missing = sorted(set(glaciers) - set(found))
        if missing:
            raise FileNotFoundError(f"Glaciers not found in {root}: {missing}")
        found = [g for g in found if g in set(glaciers)]

    specs = []
    for rgi_id in found:
        gdir = root / rgi_id
        files = sorted(glob(str(gdir / state_glob)))
        sfile = gdir / samples_file
        tfile = Path(target_template.format(rgi_id=rgi_id)).expanduser()
        problems = [
            msg
            for ok, msg in (
                (files, f"no files matching {state_glob}"),
                (sfile.is_file(), f"no {samples_file}"),
                (tfile.is_file(), f"no observations {tfile}"),
            )
            if not ok
        ]
        if problems:
            rank_zero_info(f"  Skipping {rgi_id}: {'; '.join(problems)}")
            continue
        specs.append(GlacierSpec(rgi_id, files, str(tfile), str(sfile)))
    if not specs:
        raise FileNotFoundError(f"No usable glaciers in {root}")
    return specs


@dataclass
class GlacierBlock:  # pylint: disable=too-many-instance-attributes
    """
    Where one glacier sits in the stacked node vector, and its grid.

    Attributes
    ----------
    rgi_id : str
        Glacier identifier.
    offset : int
        First node of this glacier in the stacked vector.
    n_nodes : int
        Number of observed nodes of this glacier.
    ny, nx : int
        Model grid size.
    sparse_idx_1d : numpy.ndarray
        Flat (C-order) indices of the nodes in the ``(ny, nx)`` grid.
    x, y : numpy.ndarray
        Model grid coordinates.
    available_ids : list[int]
        Ensemble members with output on this glacier.
    """

    rgi_id: str
    offset: int
    n_nodes: int
    ny: int
    nx: int
    sparse_idx_1d: np.ndarray
    x: np.ndarray
    y: np.ndarray
    available_ids: list[int] = field(default_factory=list)

    @property
    def nodes(self) -> slice:
        """
        Slice of this glacier in the stacked node vector.

        Returns
        -------
        slice
            ``slice(offset, offset + n_nodes)``.
        """
        return slice(self.offset, self.offset + self.n_nodes)


class GlacierIndex:
    """
    Layout of the stacked node vector over several glaciers.

    Parameters
    ----------
    blocks : sequence of GlacierBlock
        One block per glacier, in stacking order.
    member_ids : sequence of int
        Ensemble members used for training (available on every glacier).
    """

    def __init__(self, blocks: Sequence[GlacierBlock], member_ids: Sequence[int]):
        self.blocks = list(blocks)
        self.member_ids = [int(i) for i in member_ids]
        self._by_id = {b.rgi_id: b for b in self.blocks}

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, rgi_id: str) -> GlacierBlock:
        return self._by_id[rgi_id]

    @property
    def rgi_ids(self) -> list[str]:
        """
        Glacier ids in stacking order.

        Returns
        -------
        list[str]
            Glacier ids.
        """
        return [b.rgi_id for b in self.blocks]

    @property
    def n_nodes(self) -> int:
        """
        Total number of nodes over all glaciers.

        Returns
        -------
        int
            Length of the stacked vector.
        """
        return sum(b.n_nodes for b in self.blocks)

    def glacier_of_node(self) -> np.ndarray:
        """
        Glacier number (position in :attr:`rgi_ids`) of every node.

        Returns
        -------
        numpy.ndarray
            Integer array of length :attr:`n_nodes`.
        """
        return np.repeat(np.arange(len(self.blocks)), [b.n_nodes for b in self.blocks])

    def split(self, vec) -> dict[str, np.ndarray]:
        """
        Split stacked values along their last axis into one array per glacier.

        Parameters
        ----------
        vec : array-like
            Array with last axis of length :attr:`n_nodes`.

        Returns
        -------
        dict[str, numpy.ndarray]
            ``{rgi_id: vec[..., nodes]}``.
        """
        arr = (
            vec.detach().cpu().numpy()
            if isinstance(vec, torch.Tensor)
            else np.asarray(vec)
        )
        if arr.shape[-1] != self.n_nodes:
            raise ValueError(
                f"Last axis has {arr.shape[-1]} nodes, expected {self.n_nodes}"
            )
        return {b.rgi_id: arr[..., b.nodes] for b in self.blocks}

    def to_2d(self, vec, rgi_id: str) -> np.ma.MaskedArray:
        """
        Map one glacier's values (or the stacked vector) back to its grid.

        Parameters
        ----------
        vec : array-like
            1-D array of length :attr:`n_nodes` or of that glacier's ``n_nodes``.
        rgi_id : str
            Glacier to map.

        Returns
        -------
        numpy.ma.MaskedArray
            Array of shape ``(ny, nx)``, masked off the glacier's nodes.
        """
        b = self[rgi_id]
        arr = (
            vec.detach().cpu().numpy()
            if isinstance(vec, torch.Tensor)
            else np.asarray(vec)
        )
        if arr.shape[-1] == self.n_nodes and self.n_nodes != b.n_nodes:
            arr = arr[..., b.nodes]
        if arr.shape != (b.n_nodes,):
            raise ValueError(
                f"Expected {b.n_nodes} values for {rgi_id}, got {arr.shape}"
            )
        out = np.zeros(b.ny * b.nx, dtype=arr.dtype)
        mask = np.ones(b.ny * b.nx, dtype=bool)
        out[b.sparse_idx_1d] = arr
        mask[b.sparse_idx_1d] = False
        return np.ma.array(out.reshape(b.ny, b.nx), mask=mask.reshape(b.ny, b.nx))

    def to_dataset(self) -> xr.Dataset:
        """
        Encode the index as an :class:`xarray.Dataset` (see :meth:`save`).

        Returns
        -------
        xarray.Dataset
            Dataset with per-glacier, per-node and per-member variables.
        """
        all_ids = sorted(
            {i for b in self.blocks for i in b.available_ids} | set(self.member_ids)
        )
        available = np.array(
            [[i in set(b.available_ids) for b in self.blocks] for i in all_ids],
            dtype=np.int8,
        )
        return xr.Dataset(
            {
                "rgi_id": ("glacier", np.array(self.rgi_ids, dtype=object)),
                "offset": (
                    "glacier",
                    np.array([b.offset for b in self.blocks], np.int64),
                ),
                "n_nodes": (
                    "glacier",
                    np.array([b.n_nodes for b in self.blocks], np.int64),
                ),
                "ny": ("glacier", np.array([b.ny for b in self.blocks], np.int64)),
                "nx": ("glacier", np.array([b.nx for b in self.blocks], np.int64)),
                "sparse_idx_1d": (
                    "node",
                    np.concatenate([b.sparse_idx_1d for b in self.blocks]).astype(
                        np.int64
                    ),
                ),
                "x": ("x_all", np.concatenate([b.x for b in self.blocks])),
                "y": ("y_all", np.concatenate([b.y for b in self.blocks])),
                "member_available": (("member", "glacier"), available),
                "member_used": (
                    "member",
                    np.isin(all_ids, self.member_ids).astype(np.int8),
                ),
            },
            coords={"member": np.array(all_ids, np.int64)},
            attrs={
                "title": "pism-emulator glacier index",
                "comment": "Glaciers are stacked in order along 'node'; x and y are "
                "concatenated per glacier (nx resp. ny values each).",
            },
        )

    def save(self, path: str | Path) -> None:
        """
        Write the index to a netCDF file.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file.
        """
        self.to_dataset().to_netcdf(path)

    @classmethod
    def load(cls, path: str | Path) -> GlacierIndex:
        """
        Read an index written by :meth:`save`.

        Parameters
        ----------
        path : str or pathlib.Path
            File written by :meth:`save`.

        Returns
        -------
        GlacierIndex
            The index.
        """
        with xr.open_dataset(path) as ds:
            ds = ds.load()
        members = ds["member"].values
        blocks, xo, yo = [], 0, 0
        for g in range(ds.sizes["glacier"]):
            off, n = int(ds["offset"][g]), int(ds["n_nodes"][g])
            ny, nx = int(ds["ny"][g]), int(ds["nx"][g])
            blocks.append(
                GlacierBlock(
                    rgi_id=str(ds["rgi_id"].values[g]),
                    offset=off,
                    n_nodes=n,
                    ny=ny,
                    nx=nx,
                    sparse_idx_1d=ds["sparse_idx_1d"].values[off : off + n],
                    x=ds["x"].values[xo : xo + nx],
                    y=ds["y"].values[yo : yo + ny],
                    available_ids=[
                        int(m) for m in members[ds["member_available"].values[:, g] > 0]
                    ],
                )
            )
            xo, yo = xo + nx, yo + ny
        used = [int(m) for m in members[ds["member_used"].values > 0]]
        return cls(blocks, used)

    def same_layout(self, other: GlacierIndex) -> bool:
        """
        Whether two indices stack the same nodes of the same glaciers.

        Parameters
        ----------
        other : GlacierIndex
            Index to compare with.

        Returns
        -------
        bool
            ``True`` if glaciers, node counts and node positions agree.
        """
        return self.rgi_ids == other.rgi_ids and all(
            a.n_nodes == b.n_nodes and np.array_equal(a.sparse_idx_1d, b.sparse_idx_1d)
            for a, b in zip(self.blocks, other.blocks)
        )


def _speed_error(dtgt: xr.Dataset, ref: xr.DataArray) -> xr.DataArray | None:
    """
    Speed uncertainty from the velocity components and their errors.

    ``sigma_v = sqrt((vx * sigma_x)**2 + (vy * sigma_y)**2) / v`` (first-order error
    propagation), sampled onto the model grid by nearest neighbour.

    Parameters
    ----------
    dtgt : xarray.Dataset
        Observations with ``vx``, ``vy``, ``vx_error`` and ``vy_error``.
    ref : xarray.DataArray
        2-D field on the model grid.

    Returns
    -------
    xarray.DataArray or None
        The error on the model grid, or ``None`` if a component is missing.
    """
    names = ("vx", "vy", "vx_error", "vy_error")
    if not all(n in dtgt for n in names):
        return None
    vx, vy, ex, ey = (dtgt[n].squeeze(drop=True) for n in names)
    speed = np.hypot(vx, vy)
    err = np.sqrt((vx * ex) ** 2 + (vy * ey) ** 2) / speed.where(speed > 0)
    # where the speed is 0, the speed error is the larger component error
    err = err.fillna(np.maximum(ex, ey))
    dy, dx = err.dims[-2], err.dims[-1]
    ry, rx = ref.dims[-2], ref.dims[-1]
    if (dy, dx) != (ry, rx):
        err = err.rename({dy: ry, dx: rx})
    return err.interp_like(ref, method="nearest")


class MultiGlacierDataset(Dataset):  # pylint: disable=too-many-instance-attributes
    """
    Ensemble output and observations of several glaciers, stacked per member.

    Each glacier is read with :class:`~pism_emulator.datasets.PISMInterpolatedDataset`
    (target interpolated to its model grid, ice-free cells masked). Members are
    matched by run id; a member that is missing on any glacier is dropped
    everywhere, so every row of ``Y`` holds all glaciers. The parameters of a member
    must be the same on every glacier. The response transform and the ``X``
    normalization are fitted on all glaciers together, and every node is weighted
    by its cell area.

    Parameters
    ----------
    glaciers : sequence of GlacierSpec
        Glaciers to stack, e.g. from :func:`discover_glaciers`.
    training_var : str, optional
        Response variable in the ensemble output.
    target_var : str, optional
        Observed variable in the target files.
    target_error_var : str, optional
        Uncertainty of ``target_var``. If a target file lacks it, the error is
        derived from ``vx``, ``vy``, ``vx_error`` and ``vy_error`` when present.
    target_mask_var : str or None, optional
        Ice mask in the target files (cells where it is 0 are left out).
    y_lim : tuple[float, float], optional
        Clamp range of responses and targets (physical units).
    y_transform : {"none", "log10", "robust"}, optional
        Response transform.
    y_transform_kwargs : dict or None, optional
        Extra transform parameters.
    normalize_x : bool, optional
        Standardize the parameters.
    engine : str, optional
        Backend used by xarray to open the files.
    parallel : bool, optional
        Read each glacier's files in parallel.

    Raises
    ------
    ValueError
        If no member is available on every glacier, or if glaciers disagree on a
        member's parameters.
    """

    def __init__(
        self,
        glaciers: Sequence[GlacierSpec],
        *,
        training_var: str = "velsurf_mag",
        target_var: str = "v",
        target_error_var: str = "v_error",
        target_mask_var: str | None = "landice",
        y_lim: tuple[float, float] = (1.0, 100e3),
        y_transform: YTransformName = "log10",
        y_transform_kwargs: dict[str, object] | None = None,
        normalize_x: bool = True,
        engine: str = "netcdf4",
        parallel: bool = True,
    ) -> None:
        if not glaciers:
            raise ValueError("No glaciers given")
        self.glaciers = list(glaciers)
        self.y_lim = (float(y_lim[0]), float(y_lim[1]))
        self.y_transform = y_transform
        per = []
        for spec in self.glaciers:
            rank_zero_info(f"Glacier {spec.rgi_id}: {len(spec.training_files)} files")
            ds = PISMInterpolatedDataset(
                training_files=spec.training_files,
                samples_file=spec.samples_file,
                target_file=spec.target_file,
                training_var=training_var,
                target_var=target_var,
                target_error_var=target_error_var,
                target_mask_var=target_mask_var,
                target_corr_threshold=0.0,
                normalize_x=False,
                y_transform="none",
                y_lim=self.y_lim,
                engine=engine,
                parallel=parallel,
            )
            per.append(ds)

        # members available on every glacier, with identical parameters
        common = sorted(set.intersection(*(set(d.samples.ids) for d in per)))
        if not common:
            raise ValueError("No ensemble member has output on every glacier")
        rows = [[d.samples.ids.index(i) for i in common] for d in per]
        keys = per[0].samples.X_keys
        X0 = per[0].samples.X[rows[0]]
        for spec, d, r in zip(self.glaciers[1:], per[1:], rows[1:]):
            if d.samples.X_keys != keys or not torch.allclose(d.samples.X[r], X0):
                raise ValueError(
                    f"{spec.rgi_id}: parameters differ from {self.glaciers[0].rgi_id} "
                    f"({spec.samples_file} vs {self.glaciers[0].samples_file})"
                )
        union = sorted(set().union(*(d.samples.ids for d in per)))
        rank_zero_info(
            f"  Members used on all {len(per)} glaciers: {len(common)} of {len(union)}"
        )
        for spec, d in zip(self.glaciers, per):
            lost = sorted(set(d.samples.ids) - set(common))
            absent = sorted(set(union) - set(d.samples.ids))
            if lost or absent:
                rank_zero_info(
                    f"    {spec.rgi_id}: missing {absent or 'none'}; "
                    f"dropped because missing elsewhere {lost or 'none'}"
                )

        # stack glaciers
        blocks, offset = [], 0
        for spec, d in zip(self.glaciers, per):
            x, y = self._grid_coords(spec.training_files[0], training_var, engine)
            n = int(d.target.sparse_idx_1d.size)
            blocks.append(
                GlacierBlock(
                    rgi_id=spec.rgi_id,
                    offset=offset,
                    n_nodes=n,
                    ny=d.target.ny,
                    nx=d.target.nx,
                    sparse_idx_1d=np.asarray(d.target.sparse_idx_1d, dtype=np.int64),
                    x=x,
                    y=y,
                    available_ids=list(d.samples.ids),
                )
            )
            offset += n
        self.index = GlacierIndex(blocks, common)

        Y = torch.cat([d.samples.Y[r] for d, r in zip(per, rows)], dim=1)
        Y_target = torch.cat([d.target.Y_target for d in per])
        params = dict(y_transform_kwargs or {})
        Y, params = _apply_y_transform(
            Y, name=y_transform, y_lim=self.y_lim, params=params
        )
        Y_target, _ = _apply_y_transform(
            Y_target, name=y_transform, y_lim=self.y_lim, params=params
        )
        self.y_params = params
        self.Y_target = Y_target
        self.Y_target_error = self._stack_errors(per, target_error_var, engine)

        X = X0
        X_mean, X_std = X.mean(dim=0), X.std(dim=0)
        X_std = torch.where(X_std < 1e-6, torch.ones_like(X_std), X_std)
        Xn = (X - X_mean) / X_std if normalize_x else X

        # area weights: every node is one model cell
        cell_area = torch.cat(
            [
                torch.full((b.n_nodes,), float(d.target.grid_resolution) ** 2)
                for b, d in zip(blocks, per)
            ]
        )
        if not torch.isfinite(cell_area).all():
            cell_area = torch.ones_like(cell_area)
        self.samples = SamplesData(
            X=Xn,
            Y=Y,
            X_keys=list(keys),
            X_mean=X_mean,
            X_std=X_std,
            n_parameters=Xn.shape[1],
            n_samples=Y.shape[0],
            n_grid_points=Y.shape[1],
            normed_area=cell_area / cell_area.sum(),
            ids=common,
        )
        rank_zero_info(
            f"  Stacked {len(blocks)} glaciers: {self.index.n_nodes} nodes, "
            f"{len(common)} members"
        )

    @staticmethod
    def _grid_coords(path: str, var: str, engine: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Model grid coordinates of a glacier.

        Parameters
        ----------
        path : str
            An ensemble output file of the glacier.
        var : str
            Response variable (its last two dims are y and x).
        engine : str
            Backend used by xarray.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(x, y)``; index ranges if the file has no coordinate variables.
        """
        with xr.open_dataset(path, decode_times=False, engine=engine) as ds:
            da = ds[var]
            ydim, xdim = da.dims[-2], da.dims[-1]
            x = ds[xdim].values if xdim in ds.coords else np.arange(da.sizes[xdim])
            y = ds[ydim].values if ydim in ds.coords else np.arange(da.sizes[ydim])
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)

    def _stack_errors(
        self, per, target_error_var: str, engine: str
    ) -> torch.Tensor | None:
        """
        Stack per-glacier target errors, deriving the speed error where needed.

        Parameters
        ----------
        per : list[PISMInterpolatedDataset]
            Per-glacier datasets.
        target_error_var : str
            Error variable name.
        engine : str
            Backend used by xarray.

        Returns
        -------
        torch.Tensor or None
            Stacked errors, or ``None`` if a glacier has no error information.
        """
        out = []
        for spec, d in zip(self.glaciers, per):
            if d.target.Y_target_error is not None:
                out.append(d.target.Y_target_error)
                continue
            ref_file, tgt_file = spec.training_files[0], spec.target_file
            with (
                xr.open_dataset(ref_file, decode_times=False, engine=engine) as dref,
                xr.open_dataset(tgt_file, decode_times=False, engine=engine) as dtgt,
            ):
                err = _speed_error(dtgt, dref[d.cfg.training_var].squeeze(drop=True))
                if err is None:
                    rank_zero_info(
                        f"  {spec.rgi_id}: no '{target_error_var}' and no vx/vy errors"
                    )
                    return None
                vals = err.fillna(0.0).values.astype(np.float32).reshape(-1)
            out.append(torch.from_numpy(vals[d.target.sparse_idx_1d]))
        for spec, err in zip(self.glaciers, out):
            frac = float((err <= 0).float().mean()) if err.numel() else 0.0
            if frac > 0.1:
                rank_zero_info(
                    f"  Warning: {spec.rgi_id}: target error is 0 on {frac:.0%} of the "
                    "ice cells (check the observation file before sampling)"
                )
        return torch.cat(out)

    def __len__(self) -> int:
        return self.samples.n_samples

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples.X[i], self.samples.Y[i]

    def member_table(self) -> pd.DataFrame:
        """
        Which members have output on which glacier, and which were used.

        Returns
        -------
        pandas.DataFrame
            One row per member, one boolean column per glacier, plus ``used``.
        """
        ds = self.index.to_dataset()
        table = pd.DataFrame(
            ds["member_available"].values.astype(bool),
            index=pd.Index(ds["member"].values, name="member"),
            columns=self.index.rgi_ids,
        )
        table["used"] = ds["member_used"].values.astype(bool)
        return table
