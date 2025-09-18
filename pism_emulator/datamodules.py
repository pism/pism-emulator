import random
from pathlib import Path
from threading import Lock

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def seed_worker(worker_id: int) -> None:
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


class PISMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        X,
        F,
        omegas,
        omegas_0,
        batch_size: int = 128,
        train_size: float = 0.9,
        num_workers: int = 0,
    ):
        super().__init__()
        self.X = X
        self.F = F
        self.omegas = omegas
        self.omegas_0 = omegas_0
        self.batch_size = batch_size
        self.train_size = train_size
        self.num_workers = num_workers
        self._eigs_lock = Lock()
        self._eigs_ready = False
        self.V_hat = None
        self.F_bar = None
        self.F_mean = None
        self.eigs_vals = None

    def setup(self, stage: str | None = None):
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
            worker_init_fn=seed_worker,
            generator=g,
        )
        self.train_all_loader = train_all_loader
        val_all_loader = DataLoader(
            dataset=all_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
        self.val_all_loader = val_all_loader
        train_loader = DataLoader(
            dataset=training_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
        self.train_loader = train_loader
        self.test_loader = train_loader
        val_loader = DataLoader(
            dataset=val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
        self.val_loader = val_loader

    def prepare_data(self, **kwargs):
        V_hat, F_bar, F_mean, eigs_vals = self.get_eigenglaciers(**kwargs)
        n_eigenglaciers = V_hat.shape[1]
        self.V_hat = V_hat
        self.F_bar = F_bar
        self.F_mean = F_mean
        self.eigs_vals = eigs_vals
        self.n_eigenglaciers = n_eigenglaciers

    def get_eigenglaciers(
        self,
        cutoff: float = 1.0,
        q: int = 10,
        svd_lowrank: bool = True,
        cache_path: str | Path | None = None,
    ):
        if self._eigs_ready:
            return self.V_hat, self.F_bar, self.F_mean, self.eigs_vals

        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                pack = torch.load(cache_path, map_location="cpu")
                self.V_hat, self.F_bar, self.F_mean, self.eigs_vals = (
                    pack["V_hat"],
                    pack["F_bar"],
                    pack["F_mean"],
                    pack["eigs_vals"],
                )
                self._eigs_ready = True

                return self.V_hat, self.F_bar, self.F_mean, self.eigs_vals

        rank_zero_info("Generating eigenglaciers")
        # Only one caller does the compute
        with self._eigs_lock:
            # another thread may have finished while we waited
            if self._eigs_ready:
                return self.V_hat, self.F_bar, self.F_mean, self.eigs_vals

            F = self.F
            omegas = self.omegas
            n_grid_points = F.shape[1]
            F_mean = (F * omegas).sum(axis=0)
            F_bar = F - F_mean  # Eq. 28
            if svd_lowrank:
                Z = torch.diag(torch.sqrt(omegas.squeeze() * n_grid_points))
                U, S, V = torch.svd_lowrank(Z @ F_bar, q=q)
                lamda = S**2 / (n_grid_points)
            else:
                S = F_bar.T @ torch.diag(omegas.squeeze()) @ F_bar  # Eq. 27

                lamda, V = torch.linalg.eig(S)  # Eq. 26
                lamda = lamda[:, 0].squeeze()

            rank_zero_info(f"    using the first {q} eigen values")
            lamda_truncated = lamda.detach()
            V = V.detach()
            V_hat = V @ torch.diag(torch.sqrt(lamda))

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

            self.V_hat = V_hat
            self.F_bar = F_bar
            self.F_mean = F_mean
            self._eigs_ready = True

            return V_hat, F_bar, F_mean, lamda

    def train_dataloader(self):
        return self.train_loader

    def validation_dataloader(self):
        return self.val_loader


class PDDDataModule(pl.LightningDataModule):
    def __init__(
        self,
        X,
        Y,
        omegas,
        omegas_0,
        batch_size: int = 128,
        train_size: float = 0.9,
        num_workers: int = 0,
    ):
        super().__init__()
        self.X = X
        self.Y = Y
        self.omegas = omegas
        self.omegas_0 = omegas_0
        self.batch_size = batch_size
        self.train_size = train_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None):
        all_data = TensorDataset(self.X, self.Y, self.omegas, self.omegas_0)
        self.all_data = all_data

        training_data, val_data = train_test_split(
            all_data, train_size=self.train_size, random_state=0
        )
        self.training_data = training_data
        self.val_data = val_data

        train_loader = DataLoader(
            dataset=training_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
        self.train_loader = train_loader
        val_loader = DataLoader(
            dataset=val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
        self.val_loader = val_loader

    def prepare_data(self, **kwargs):
        pass

    def train_dataloader(self):
        return self.train_loader

    def validation_dataloader(self):
        return self.val_loader
