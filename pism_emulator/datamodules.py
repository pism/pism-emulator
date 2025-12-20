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
    """
    Seed NumPy and Python RNGs for a DataLoader worker.

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
    num_workers : int, default=0
        Number of worker processes for DataLoaders. On macOS/MPS, ``0`` is often
        fastest due to process spawn overhead.
    """

    X: Tensor
    F: Tensor
    omegas: Tensor
    omegas_0: Tensor
    batch_size: int = 128
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


class PISMDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for PISM emulator training and evaluation.

    This DataModule wraps in-memory tensors into a :class:`~torch.utils.data.TensorDataset`,
    computes an eigenglacier basis (and mean-centered fields) once in
    :meth:`prepare_data`, and provides deterministic DataLoaders.

    Parameters
    ----------
    X : torch.Tensor
        Input/parameter tensor with leading sample dimension ``N`` (shape ``(N, P)``).
    F : torch.Tensor
        Model response tensor with leading sample dimension ``N`` (shape ``(N, G)``),
        where ``G`` is the number of grid points/nodes.
    omegas : torch.Tensor
        Quadrature/area weights used to compute weighted means and inner products.
        Must be broadcast-compatible with ``F`` along the grid dimension
        (typically shape ``(G,)`` or ``(G, 1)``).
    omegas_0 : torch.Tensor
        Auxiliary weight tensor stored alongside samples for downstream use.
        Included in the dataset returned by DataLoaders.
    batch_size : int, optional
        Batch size for DataLoaders. Default is 128.
    seed : int, optional
        Seed used for deterministic DataLoader behavior. Default is 42.
    num_workers : int, optional
        Number of DataLoader worker processes. Default is 0.

    Attributes
    ----------
    cfg : DataConfig
        Configuration container storing tensors and DataLoader settings.
    eig : EigCache
        Cache holding the eigenglacier basis and derived tensors produced by
        :meth:`get_eigenglaciers`.

    Notes
    -----
    * :meth:`prepare_data` computes and caches eigenglacier quantities
      (or loads them from disk if ``cache_path`` is provided).
    * :meth:`setup` currently builds a single dataset; train/val splitting is not
      implemented in the shown code and both loaders iterate over the same dataset.
      If you intend a train/val split, you should add it in :meth:`setup`.
    """

    # pylint: disable=too-many-instance-attributes  # remove after refactor if under threshold

    def __init__(
        self,
        X: Tensor,
        F: Tensor,
        omegas: Tensor,
        omegas_0: Tensor,
        *,
        batch_size: int = 128,
        seed: int = 42,
        num_workers: int = 0,
    ) -> None:
        """
        Initialize the DataModule.

        Parameters
        ----------
        X, F, omegas, omegas_0 : torch.Tensor
            See class docstring.
        batch_size : int, optional
            Batch size for DataLoaders. Default is 128.
        seed : int, optional
            Seed for deterministic DataLoader behavior. Default is 42.
        num_workers : int, optional
            Number of worker processes. Default is 0.

        Raises
        ------
        ValueError
            If ``X`` and ``F`` have mismatched leading dimensions.
        """
        super().__init__()

        if X.shape[0] != F.shape[0]:
            raise ValueError("X and F must have the same number of samples")

        self.cfg = DataConfig(X, F, omegas, omegas_0, batch_size, num_workers)
        self.eig = EigCache()

        self._dl_generator = torch.Generator(device="cpu")
        self._dl_generator.manual_seed(seed)

        # only the splits are kept; loaders are created on demand
        self._train: TensorDataset | None = None
        self._val: TensorDataset | None = None
        self._all: TensorDataset | None = None

    def prepare_data(self, **kwargs) -> None:
        """
        Compute eigenglaciers once (or load from cache).

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`get_eigenglaciers`. This allows configuration of
            basis size, cutoff, SVD mode, and caching.

        Notes
        -----
        Lightning calls this hook once per node. Computed tensors are stored in
        :attr:`eig` for use in :meth:`setup`.
        """
        V_hat, F_bar, F_mean, eigs_vals = self.get_eigenglaciers(**kwargs)
        self.eig.V_hat, self.eig.F_bar, self.eig.F_mean, self.eig.eigs_vals = (
            V_hat,
            F_bar,
            F_mean,
            eigs_vals,
        )

    def setup(self, stage: str | None = None) -> None:
        """
        Build datasets for the requested stage.

        Parameters
        ----------
        stage : str or None, optional
            Lightning stage hint (e.g., ``"fit"``, ``"validate"``, ``"test"``).
            This implementation does not specialize by stage but accepts the
            argument for Lightning compatibility.

        Notes
        -----
        This method uses the cached mean-centered field ``F_bar`` computed in
        :meth:`prepare_data`. Train/validation splitting is not implemented here;
        both :meth:`train_dataloader` and :meth:`val_dataloader` will iterate over
        the same dataset unless you add a split.
        """
        ds_all = TensorDataset(
            self.cfg.X, self.eig.F_bar, self.cfg.omegas, self.cfg.omegas_0
        )
        self._all = ds_all
        self._train, self._val = ds_all, ds_all

    def _build_loader(self, ds: TensorDataset, *, shuffle: bool) -> DataLoader:
        """
        Construct a DataLoader with consistent seeding/worker configuration.

        Parameters
        ----------
        ds : torch.utils.data.TensorDataset
            Dataset to wrap.
        shuffle : bool
            Whether to shuffle samples each epoch.

        Returns
        -------
        torch.utils.data.DataLoader
            Configured DataLoader.
        """
        return DataLoader(
            dataset=ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            worker_init_fn=seed_worker,
            generator=self._dl_generator,
            persistent_workers=True,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        """
        Create the training DataLoader.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader over the training dataset.

        Notes
        -----
        This implementation currently sets ``shuffle=False``. If you want typical
        SGD behavior, set ``shuffle=True``.
        """
        if self._train is None:
            self.setup("fit")
        return self._build_loader(self._train, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        """
        Create the validation DataLoader.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader over the validation dataset.
        """
        if self._val is None:
            self.setup("validate")
        return self._build_loader(self._val, shuffle=False)

    # -------------------- Eigenglaciers --------------------

    def get_eigenglaciers(
        self,
        *,
        q: int = 10,
        cutoff: float | None = None,
        svd_lowrank: bool = True,
        cache_path: str | Path | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute or load the eigenglacier basis and mean-centered responses.

        Given a set of responses ``F`` and weights ``omegas``, this method computes
        the weighted mean field ``F_mean`` and mean-centered anomalies ``F_bar``.
        It then constructs an eigenglacier basis ``V_hat`` using either a low-rank
        SVD or an explicit eigen-decomposition.

        Parameters
        ----------
        q : int, optional
            Target rank/basis size when ``cutoff`` is None. Default is 10.
        cutoff : float, optional
            If provided, choose the number of modes such that the cumulative
            explained variance reaches ``cutoff`` (e.g., 0.95 for 95%).
            When set, ``q`` is ignored for truncation. Default is None.
        svd_lowrank : bool, optional
            If True, use :func:`torch.svd_lowrank` on a weighted anomaly matrix.
            If False, form the weighted covariance matrix and use
            :func:`torch.linalg.eigh`. Default is True.
        cache_path : str or pathlib.Path, optional
            If provided, attempt to load cached tensors from this path. If the
            file does not exist, computed results are saved to this location.

        Returns
        -------
        V_hat : torch.Tensor
            Eigenglacier basis matrix. Shape depends on truncation choice
            (typically ``(G, r)`` where ``G`` is number of grid points).
        F_bar : torch.Tensor
            Mean-centered anomalies, ``F - F_mean``. Shape ``(N, G)``.
        F_mean : torch.Tensor
            Weighted mean field across runs. Shape ``(G,)`` (or broadcastable).
        eigs_vals : torch.Tensor
            Eigenvalues (variance captured by each mode). Shape ``(r,)``.

        Notes
        -----
        * Results are cached in-memory in :attr:`eig`. If already computed in the
          current process, cached values are returned immediately.
        * When ``cutoff`` is provided, the effective rank is determined by the
          cumulative explained variance fraction.
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

            q_r = F.shape[0] if cutoff is not None else q

            if svd_lowrank:
                Z = torch.diag(torch.sqrt(omegas.squeeze() * n_grid_points))
                _, S, V = torch.svd_lowrank(Z @ F_bar, q=q_r)
                lamda = S**2 / n_grid_points
            else:
                S = F_bar.T @ torch.diag(omegas.squeeze()) @ F_bar
                lamda, V = torch.linalg.eigh(S)  # pylint: disable=not-callable
                lamda = lamda[:, 0].squeeze()

            if cutoff is not None:
                cutoff_index = torch.sum(torch.cumsum(lamda / lamda.sum(), 0) < cutoff)
                lamda_truncated = lamda.detach()[:cutoff_index]
                rank_zero_info(
                    "Generating eigenglaciers using the first "
                    f"{cutoff_index} eigenvalues for {cutoff * 100}% fidelity"
                )
                V_hat = V.detach()[:, :cutoff_index] @ torch.diag(
                    torch.sqrt(lamda_truncated)
                )
            else:
                rank_zero_info(
                    f"Generating eigenglaciers using the first {q_r} eigenvalues"
                )
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
    batch_size : int, default=128
        Batch size for train/val.
    train_size : float, default=0.9
        Fraction of samples for training (remainder for validation).
    num_workers : int, default=0
        DataLoader worker processes (``0`` is often fastest on macOS/MPS).
    """

    X: Tensor
    Y: Tensor
    batch_size: int = 128
    train_size: float = 0.9
    num_workers: int = 0


class PDDDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for PDD-style training and evaluation.

    This DataModule wraps in-memory tensors into a :class:`~torch.utils.data.TensorDataset`,
    splits them into training/validation subsets, and constructs PyTorch DataLoaders
    with deterministic seeding.

    Parameters
    ----------
    X : torch.Tensor
        Predictor/features tensor. Shape must be compatible with indexing along
        the leading sample dimension (``(N, ...)``).
    Y : torch.Tensor
        Target tensor aligned with ``X``. Must have the same leading sample
        dimension as ``X`` (``Y.shape[0] == X.shape[0]``).
    omegas : torch.Tensor
        Auxiliary tensor stored in the config for downstream use (not currently
        included in the DataLoaders in this implementation).
    omegas_0 : torch.Tensor
        Auxiliary tensor stored in the config for downstream use (not currently
        included in the DataLoaders in this implementation).
    batch_size : int, optional
        Batch size for all DataLoaders. Default is 128.
    seed : int, optional
        Seed used for deterministic DataLoader shuffling/worker seeding. Default is 42.
    train_size : float, optional
        Fraction of samples to use for training. Must be in ``(0, 1)``.
        Default is 0.9.
    num_workers : int, optional
        Number of DataLoader worker processes. Default is 0.

    Attributes
    ----------
    cfg : PDDConfig
        Configuration container holding input tensors and DataLoader settings.

    Notes
    -----
    * This module assumes tensors are already materialized in memory.
    * DataLoaders are built lazily; calling :meth:`train_dataloader` or
      :meth:`val_dataloader` will trigger :meth:`setup` if needed.
    * ``pin_memory=True`` is enabled; on CPU-only runs it is harmless but may not
      provide benefit.
    """

    # pylint: disable=too-many-instance-attributes  # safe to remove if below threshold

    def __init__(
        self,
        X: Tensor,
        Y: Tensor,
        omegas: Tensor,
        omegas_0: Tensor,
        *,
        batch_size: int = 128,
        seed: int = 42,
        train_size: float = 0.9,
        num_workers: int = 0,
    ) -> None:
        """
        Initialize the DataModule.

        Parameters
        ----------
        X, Y, omegas, omegas_0 : torch.Tensor
            See class docstring.
        batch_size : int, optional
            Batch size for DataLoaders. Default is 128.
        seed : int, optional
            Seed for deterministic DataLoader behavior. Default is 42.
        train_size : float, optional
            Train split fraction in ``(0, 1)``. Default is 0.9.
        num_workers : int, optional
            Number of worker processes. Default is 0.

        Raises
        ------
        ValueError
            If ``train_size`` is not in ``(0, 1)`` or if ``X`` and ``Y`` have
            mismatched leading dimensions.
        """
        super().__init__()

        if not (0.0 < train_size < 1.0):
            raise ValueError("train_size must be in (0, 1)")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of samples")

        self.cfg = PDDConfig(
            X=X,
            Y=Y,
            omegas=omegas,
            omegas_0=omegas_0,
            batch_size=batch_size,
            train_size=train_size,
            num_workers=num_workers,
        )

        self._dl_generator = torch.Generator(device="cpu")
        self._dl_generator.manual_seed(seed)

        # dataset splits; loaders built on demand
        self._all: TensorDataset | None = None
        self._train: TensorDataset | None = None
        self._val: TensorDataset | None = None

    # -------------------- Lightning hooks --------------------

    def prepare_data(self) -> None:
        """
        Perform one-time data preparation (no-op for in-memory tensors).

        Notes
        -----
        Kept for API symmetry with other Lightning modules. All tensors are passed
        directly at initialization, so there is nothing to download or preprocess.
        """
        return

    def setup(self, stage: str | None = None) -> None:
        """
        Create the underlying dataset and split into train/validation subsets.

        Parameters
        ----------
        stage : str or None, optional
            Lightning stage hint (e.g., ``"fit"``, ``"validate"``, ``"test"``).
            This implementation does not specialize behavior by stage, but the
            argument is accepted for Lightning compatibility.

        Notes
        -----
        Splitting is performed using :func:`sklearn.model_selection.train_test_split`.
        """
        all_data = TensorDataset(self.cfg.X, self.cfg.Y)
        self._all = all_data

        train_ds, val_ds = train_test_split(
            all_data, train_size=self.cfg.train_size, random_state=0
        )
        self._train, self._val = train_ds, val_ds

    # -------------------- DataLoaders (lazy) --------------------

    def _build_loader(self, ds: TensorDataset, *, shuffle: bool) -> DataLoader:
        """
        Construct a DataLoader with consistent seeding and worker configuration.

        Parameters
        ----------
        ds : torch.utils.data.TensorDataset
            Dataset to wrap.
        shuffle : bool
            Whether to shuffle samples each epoch.

        Returns
        -------
        torch.utils.data.DataLoader
            Configured DataLoader.
        """
        return DataLoader(
            dataset=ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=self._dl_generator,
        )

    def train_dataloader(self) -> DataLoader:
        """
        Create the training DataLoader.

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
        Create the validation DataLoader.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader over the validation split.
        """
        if self._val is None:
            self.setup("validate")
        return self._build_loader(self._val, shuffle=False)


class LegacyPISMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        X,
        F,
        omegas,
        omegas_0,
        batch_size: int = 128,
        train_size: float = 0.9,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.X = X
        self.F = F
        self.omegas = omegas
        self.omegas_0 = omegas_0
        self.batch_size = batch_size
        self.train_size = train_size
        self.num_workers = num_workers
        self._dl_generator = torch.Generator(device="cpu")
        self._dl_generator.manual_seed(seed)

    def setup(self, stage: Optional[str] = None):
        all_data = TensorDataset(self.X, self.F_bar, self.omegas, self.omegas_0)
        self.all_data = all_data

        training_data, val_data = train_test_split(
            all_data, train_size=self.train_size, random_state=0
        )
        self.training_data = training_data
        self.test_data = training_data

        self.val_data = val_data
        train_all_loader = DataLoader(
            dataset=all_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=self._dl_generator,
        )
        self.train_all_loader = train_all_loader
        val_all_loader = DataLoader(
            dataset=all_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=self._dl_generator,
        )
        self.val_all_loader = val_all_loader
        train_loader = DataLoader(
            dataset=training_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=self._dl_generator,
        )
        self.train_loader = train_loader
        self.test_loader = train_loader
        val_loader = DataLoader(
            dataset=val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=self._dl_generator,
        )
        self.val_loader = val_loader

    def prepare_data(self, **kwargs):
        V_hat, F_bar, F_mean = self.get_eigenglaciers(**kwargs)
        n_eigenglaciers = V_hat.shape[1]
        self.V_hat = V_hat
        self.F_bar = F_bar
        self.F_mean = F_mean
        self.n_eigenglaciers = n_eigenglaciers

    def get_eigenglaciers(self, **kwargs):
        rank_zero_info("Generating eigenglaciers")
        defaultKwargs = {
            "cutoff": 1.0,
            "q": 10,
            "svd_lowrank": True,
            "eigenvalues": False,
        }
        if len(kwargs) > 0:
            kwargs = {**defaultKwargs, **kwargs}
        else:
            kwargs = defaultKwargs

        q = kwargs["q"]

        F = self.F
        omegas = self.omegas
        n_grid_points = F.shape[1]
        F_mean = (F * omegas).sum(axis=0)
        F_bar = F - F_mean  # Eq. 28
        if kwargs["svd_lowrank"]:
            Z = torch.diag(torch.sqrt(omegas.squeeze() * n_grid_points))
            U, S, V = torch.svd_lowrank(Z @ F_bar, q=q)
            lamda = S**2 / (n_grid_points)
        else:
            S = F_bar.T @ torch.diag(omegas.squeeze()) @ F_bar  # Eq. 27

            lamda, V = torch.linalg.eig(S)  # Eq. 26
            lamda = lamda[:, 0].squeeze()

        rank_zero_info(f"...using the first {q} eigen values")
        lamda_truncated = lamda.detach()
        V = V.detach()
        V_hat = V @ torch.diag(torch.sqrt(lamda))

        if kwargs["eigenvalues"]:
            return V_hat, F_bar, F_mean, lamda
        else:
            return V_hat, F_bar, F_mean

    def train_dataloader(self):
        return self.train_loader

    def validation_dataloader(self):
        return self.val_loader
