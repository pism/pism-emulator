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
"""
Data Module.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


def seed_worker(worker_id: int) -> None:  # pylint: disable=unused-argument
    """Seed NumPy and Python RNGs for a DataLoader worker.

    Parameters
    ----------
    worker_id : int
        The integer id of the worker process/thread. The value is not
        directly used; it exists to match the signature expected by
        ``DataLoader(worker_init_fn=...)``.

    Notes
    -----
    This function uses :func:`torch.initial_seed` to derive a 32-bit worker
    seed, then seeds both :mod:`numpy.random` and :mod:`random` so that
    each worker has a reproducible, independent RNG stream.
    """
    # Derive a 32-bit seed from PyTorch's worker-specific seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


@dataclass
class DataConfig:
    """
    Configuration bundle for dataset and DataLoader parameters.

    Attributes
    ----------
    X : torch.Tensor
        Input feature matrix with shape ``(N_samples, N_features)``.
    F : torch.Tensor
        Response matrix with shape ``(N_samples, N_nodes)`` (e.g., fields to be
        compressed/expanded).
    omegas : torch.Tensor
        Per-sample weights with shape ``(N_samples,)`` or ``(N_samples, 1)`` used
        in mean/variance weighting and loss aggregation.
    omegas_0 : torch.Tensor
        Optional auxiliary per-sample weights with same length as ``omegas``.
    batch_size : int, default=128
        Batch size used by train/validation DataLoaders.
    train_size : float, default=0.9
        Fraction of samples assigned to the training split (remainder to validation).
    num_workers : int, default=0
        Number of worker processes for DataLoaders. On macOS/MPS, ``0`` is often
        fastest due to process spawn overhead.
    """

    X: Tensor
    F: Tensor
    omegas: Tensor
    omegas_0: Tensor
    batch_size: int = 128
    train_size: float = 0.9
    num_workers: int = 0


@dataclass
class EigCache:
    """
    In-memory cache for eigenglacier/eigenbasis computation.

    Attributes
    ----------
    ready : bool, default=False
        Flag indicating whether the cache is populated and can be reused without
        recomputation.
    V_hat : torch.Tensor or None, default=None
        Basis matrix of shape ``(N_nodes, q)`` whose columns are scaled by
        ``sqrt(lambda)`` (truncated eigenvectors / right singular vectors).
    F_bar : torch.Tensor or None, default=None
        Mean-centered responses with shape ``(N_samples, N_nodes)``.
    F_mean : torch.Tensor or None, default=None
        Per-node mean over samples with shape ``(N_nodes,)``.
    eigs_vals : torch.Tensor or None, default=None
        Vector of eigenvalues (or squared singular values) with length ``q``.
    lock : threading.Lock
        Mutual-exclusion lock used to guard one-time computation in concurrent
        contexts (e.g., multiple workers or ranks).
    """

    ready: bool = False
    V_hat: Tensor | None = None
    F_bar: Tensor | None = None
    F_mean: Tensor | None = None
    eigs_vals: Tensor | None = None
    lock: Lock = Lock()


# seed_worker and g assumed defined elsewhere in your module


class PISMDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for PISM emulator training/evaluation.

    Parameters
    ----------
    X, F, omegas, omegas_0 : torch.Tensor
        See original docstring.
    batch_size : int, default=128
    train_size : float, default=0.9
    num_workers : int, default=0
    """

    # pylint: disable=too-many-instance-attributes  # (remove after refactor if under threshold)

    def __init__(
        self,
        X: Tensor,
        F: Tensor,
        omegas: Tensor,
        omegas_0: Tensor,
        *,
        batch_size: int = 128,
        train_size: float = 0.9,
        num_workers: int = 0,
    ):
        super().__init__()
        self.cfg = DataConfig(
            X, F, omegas, omegas_0, batch_size, train_size, num_workers
        )
        self.eig = EigCache()

        # only the splits are kept; loaders are created on demand
        self._train = None
        self._val = None
        self._all = None

    def prepare_data(self, **kwargs) -> None:
        """Compute eigenglaciers once (or load from cache)."""
        V_hat, F_bar, F_mean, eigs_vals = self.get_eigenglaciers(**kwargs)
        self.eig.V_hat, self.eig.F_bar, self.eig.F_mean, self.eig.eigs_vals = (
            V_hat,
            F_bar,
            F_mean,
            eigs_vals,
        )

    def setup(self, stage: str | None = None) -> None:
        """Split into train/val and stage datasets."""
        # use cached F_bar from prepare_data
        ds_all = TensorDataset(
            self.cfg.X, self.eig.F_bar, self.cfg.omegas, self.cfg.omegas_0
        )
        self._all = ds_all

        train_ds, val_ds = train_test_split(
            ds_all, train_size=self.cfg.train_size, random_state=0
        )
        self._train, self._val = train_ds, val_ds

    def _build_loader(self, ds: TensorDataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset=ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            worker_init_fn=seed_worker,
            persistent_workers=True,
            generator=g,
        )

    def train_dataloader(self) -> DataLoader:
        return self._build_loader(self._train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._build_loader(self._val, shuffle=False)

    # -------------------- Eigenglaciers --------------------

    def get_eigenglaciers(
        self,
        *,
        q: int = 10,
        svd_lowrank: bool = True,
        cache_path: str | Path | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute/load the eigenglacier basis and mean-centered responses.

        All parameters are keyword-only to avoid R0917.

        Returns
        -------
        V_hat, F_bar, F_mean, eigs_vals : tuple[Tensor, Tensor, Tensor, Tensor]
        """
        if self.eig.ready:
            return self.eig.V_hat, self.eig.F_bar, self.eig.F_mean, self.eig.eigs_vals

        if cache_path is not None:
            cache = Path(cache_path)
            if cache.exists():
                pack = torch.load(cache, map_location="cpu")
                self.eig = EigCache(
                    ready=True,
                    V_hat=pack["V_hat"],
                    F_bar=pack["F_bar"],
                    F_mean=pack["F_mean"],
                    eigs_vals=pack["eigs_vals"],
                )
                return (
                    self.eig.V_hat,
                    self.eig.F_bar,
                    self.eig.F_mean,
                    self.eig.eigs_vals,
                )

        rank_zero_info(f"Generating eigenglaciers using the first {q} eigen values")
        with self.eig.lock:
            if self.eig.ready:
                return (
                    self.eig.V_hat,
                    self.eig.F_bar,
                    self.eig.F_mean,
                    self.eig.eigs_vals,
                )

            F = self.cfg.F
            omegas = self.cfg.omegas
            n_grid_points = F.shape[1]
            F_mean = (F * omegas).sum(axis=0)
            F_bar = F - F_mean  # Eq. 28

            if svd_lowrank:
                Z = torch.diag(torch.sqrt(omegas.squeeze() * n_grid_points))
                _, S, V = torch.svd_lowrank(Z @ F_bar, q=q)
                lamda = S**2 / (n_grid_points)
            else:
                S = F_bar.T @ torch.diag(omegas.squeeze()) @ F_bar
                lamda, V = torch.linalg.eigh(S)  # pylint: disable=not-callable
                lamda = lamda[:, 0].squeeze()

            V_hat = V.detach() @ torch.diag(torch.sqrt(lamda.detach()))

            if cache_path is not None:
                torch.save(
                    {
                        "V_hat": V_hat.cpu(),
                        "F_bar": F_bar.cpu(),
                        "F_mean": F_mean.cpu(),
                        "eigs_vals": lamda.cpu(),
                    },
                    cache_path,
                )

            self.eig = EigCache(True, V_hat, F_bar, F_mean, lamda)
            return V_hat, F_bar, F_mean, lamda


@dataclass
class PDDConfig:
    """
    Configuration bundle for PDD data and DataLoaders.

    Attributes
    ----------
    X : torch.Tensor
        Input features of shape ``(N_samples, N_features)``.
    Y : torch.Tensor
        Targets of shape ``(N_samples, ...)`` (scalars or fields).
    omegas : torch.Tensor
        Per-sample weights, shape ``(N_samples,)`` or ``(N_samples, 1)``.
    omegas_0 : torch.Tensor
        Optional auxiliary weights per sample (same length as ``omegas``).
    batch_size : int, default=128
        Batch size for train/val.
    train_size : float, default=0.9
        Fraction of samples for training (remainder for validation).
    num_workers : int, default=0
        DataLoader worker processes (``0`` is often fastest on macOS/MPS).
    """

    X: Tensor
    Y: Tensor
    omegas: Tensor
    omegas_0: Tensor
    batch_size: int = 128
    train_size: float = 0.9
    num_workers: int = 0


class PDDDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for PDD-style training/evaluation (refactored).

    This version:
    - Stores inputs in a small `PDDConfig` (fewer instance attributes).
    - Builds train/val DataLoaders lazily via a helper.
    - Keeps Lightning hook signatures (`prepare_data`, `setup`, loaders) clean.
    - Uses keyword-only parameters on internal helpers to avoid `R0917`.

    Parameters
    ----------
    X, Y, omegas, omegas_0 : torch.Tensor
        See `PDDConfig` for shapes.
    batch_size : int, default=128
        Batch size for all DataLoaders.
    train_size : float, default=0.9
        Train split fraction.
    num_workers : int, default=0
        DataLoader workers.
    """

    # pylint: disable=too-many-instance-attributes  # trimmed; but safe to remove if below threshold

    def __init__(
        self,
        X: Tensor,
        Y: Tensor,
        omegas: Tensor,
        omegas_0: Tensor,
        *,
        batch_size: int = 128,
        train_size: float = 0.9,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.cfg = PDDConfig(
            X=X,
            Y=Y,
            omegas=omegas,
            omegas_0=omegas_0,
            batch_size=batch_size,
            train_size=train_size,
            num_workers=num_workers,
        )
        # dataset splits; loaders built on demand
        self._all: TensorDataset | None = None
        self._train: TensorDataset | None = None
        self._val: TensorDataset | None = None

    # -------------------- Lightning hooks --------------------

    def prepare_data(self) -> None:
        """
        One-time data preparation hook (no-op here).

        Notes
        -----
        Kept for API symmetry with other modules; nothing to precompute.
        """
        return

    def setup(self, stage: str | None = None) -> None:
        """
        Split into train/val and stage datasets.

        Parameters
        ----------
        stage : str or None, optional
            Lightning stage hint (``'fit'``, ``'validate'``, etc.). Unused here.
        """
        all_data = TensorDataset(
            self.cfg.X, self.cfg.Y, self.cfg.omegas, self.cfg.omegas_0
        )
        self._all = all_data

        train_ds, val_ds = train_test_split(
            all_data, train_size=self.cfg.train_size, random_state=0
        )
        self._train, self._val = train_ds, val_ds

    # -------------------- DataLoaders (lazy) --------------------

    def _build_loader(self, ds: TensorDataset, *, shuffle: bool) -> DataLoader:
        """Create a DataLoader with consistent seeding/worker config."""
        return DataLoader(
            dataset=ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

    def train_dataloader(self) -> DataLoader:
        """
        Training DataLoader.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader over the training split.
        """
        if self._train is None:  # defensive, in case setup hasn't run
            self.setup("fit")
        return self._build_loader(self._train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """
        Validation DataLoader.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader over the validation split.
        """
        if self._val is None:
            self.setup("validate")
        return self._build_loader(self._val, shuffle=False)
