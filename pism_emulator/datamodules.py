import random
from typing import Optional

import lightning as pl
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def seed_worker(worker_id):
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
            generator=g,
        )
        self.train_all_loader = train_all_loader
        val_all_loader = DataLoader(
            dataset=all_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        self.val_all_loader = val_all_loader
        train_loader = DataLoader(
            dataset=training_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
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
        V_hat, F_bar, F_mean = self.get_eigenglaciers(**kwargs)
        n_eigenglaciers = V_hat.shape[1]
        self.V_hat = V_hat
        self.F_bar = F_bar
        self.F_mean = F_mean
        self.n_eigenglaciers = n_eigenglaciers

    def get_eigenglaciers(self, **kwargs):
        print("Generating eigenglaciers")
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

        print(f"...using the first {q} eigen values")
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

    def setup(self, stage: Optional[str] = None):
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
            pin_memory=True,
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