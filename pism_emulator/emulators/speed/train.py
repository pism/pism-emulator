#!/bin/env python3

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

# pylint: disable=too-many-statements,redefined-builtin
"""
Surrogate model training.
"""

import random
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Mapping

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback, Timer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only
from pyfiglet import Figlet
from scipy.stats import dirichlet
from tqdm import tqdm

from pism_emulator.datamodules import PISMDataModule
from pism_emulator.datasets import PISMInterpolatedDataset as PISMDataset
from pism_emulator.emulators.nnemulator import (
    DNNEmulator,
    LegacyNNEmulator,
    NN5Emulator,
    NNEmulator,
)
from pism_emulator.plotting import plot_eigenglaciers

EMULATORS: Mapping[str, type[pl.LightningModule]] = {
    "NN": NNEmulator,
    "NN5": NN5Emulator,
    "DNN": DNNEmulator,
    "LegacyNN": LegacyNNEmulator,
}


torch.use_deterministic_algorithms(True)
warnings.filterwarnings("ignore", ".*does not have many workers.*")


class EpochProgressBar(Callback):
    """
    A simple tqdm progress bar that advances once per training epoch.

    The bar is created on rank 0 at training start, updates at each epoch end
    with the latest `train_loss` and `val`/`test` loss (if present in
    ``trainer.callback_metrics``), and closes at training end.

    Parameters
    ----------
    desc : str, optional
        Text shown to the left of the bar, by default ``"Training"``.
    ncols : int, optional
        Fixed width of the progress bar in terminal columns. If you want
        automatic resizing, set this high and change ``dynamic_ncols=True``
        in the implementation, by default ``120``.

    Attributes
    ----------
    _bar : tqdm or None
        The underlying tqdm progress bar instance (rank 0 only) or ``None``
        when not active.
    _desc : str
        Description shown next to the bar.
    _ncols : int
        Fixed width (columns) of the bar.
    """

    def __init__(self, *, desc: str = "Training", ncols: int = 120) -> None:
        """
        Initialize the progress bar callback.

        Parameters
        ----------
        desc : str, optional
            Text shown to the left of the bar, by default ``"Training"``.
        ncols : int, optional
            Fixed width of the progress bar in terminal columns, by default ``120``.
        """
        super().__init__()
        self._bar: tqdm | None = None
        self._desc = desc
        self._ncols = ncols

    @rank_zero_only
    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        """
        Create and display the tqdm bar at the start of training (rank 0 only).

        Parameters
        ----------
        trainer : lightning.pytorch.Trainer
            The current Trainer instance.
        pl_module : lightning.pytorch.LightningModule
            The model being trained.

        Notes
        -----
        The total number of epochs is derived from ``trainer.max_epochs``.
        If ``max_epochs`` is ``None``, the total is set to 0 and tqdm will
        display an indeterminate-length bar.
        """
        total = int(trainer.max_epochs) if trainer.max_epochs is not None else 0
        self._bar = tqdm(
            total=total,
            desc=self._desc,
            ncols=self._ncols,
            unit="epoch",
            leave=True,
            dynamic_ncols=False,
        )

    @rank_zero_only
    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """
        Advance the bar by one epoch and print latest losses (rank 0 only).

        Parameters
        ----------
        trainer : lightning.pytorch.Trainer
            The current Trainer instance. Uses ``trainer.callback_metrics`` to
            fetch the most recent metrics.
        pl_module : lightning.pytorch.LightningModule
            The model being trained.

        Notes
        -----
        Looks for ``"train_loss"`` and ``"test_loss"`` (or validation loss if
        you map it to ``"test_loss"``) in ``trainer.callback_metrics``. Missing
        metrics are displayed as ``nan``.
        """
        if self._bar is None:
            return
        m = trainer.callback_metrics
        train_loss = float(m.get("train_loss", float("nan")))
        test_loss = float(m.get("test_loss", float("nan")))
        self._bar.set_postfix_str(f"train={train_loss:.4g}, val={test_loss:.4g}")
        self._bar.update(1)

    @rank_zero_only
    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        """
        Close and clear the tqdm bar at the end of training (rank 0 only).

        Parameters
        ----------
        trainer : lightning.pytorch.Trainer
            The current Trainer instance.
        pl_module : lightning.pytorch.LightningModule
            The model that was trained.
        """
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def main():
    """
    Main.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--emulator", choices=["NN", "NN5", "DNN", "LegacyNN"], default="DNN"
    )
    tmp, _ = parser.parse_known_args()

    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cutoff", type=float, default=None)
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--emulator-dir", default="emulator_ensemble")
    parser.add_argument("--engine", default="netcdf4")
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("-q", type=int, default=100)
    parser.add_argument(
        "--samples-file", default="../data/samples/velocity_calibration_samples_50.csv"
    )
    parser.add_argument(
        "--strategy",
        choices=["ddp_spawn", "ddp", "auto", "single_device"],
        default="auto",
    )
    parser.add_argument(
        "--target-file",
        default="../data/observed_speeds/greenland_vel_mosaic250_v1_g9000m.nc",
    )
    parser.add_argument("--target-var", type=str, default="velsurf_mag")
    parser.add_argument("--target-error-var", type=str, default="velsurf_mag_error")
    parser.add_argument("--y-lim", type=float, nargs=2, default=[1, 10e3])
    parser.add_argument("--y-transform", default="log10")
    parser.add_argument("TRAINING_FILES", nargs="*", help="PISM netCDF files")

    cls = EMULATORS[tmp.emulator]
    cls.add_model_specific_args(parser)
    Emulator = cls  # type: type[pl.LightningModule]
    # let the chosen model extend the parser
    if tmp.emulator == "NN":
        Emulator = NNEmulator
    elif tmp.emulator == "DNN":
        Emulator = DNNEmulator
    elif tmp.emulator == "LegacyNN":
        Emulator = LegacyNNEmulator

    args = parser.parse_args()
    hparams = vars(args)

    accelerator = args.accelerator
    batch_size = args.batch_size
    cutoff = args.cutoff
    devices = args.devices
    emulator_dir = args.emulator_dir
    engine = args.engine
    y_transform = args.y_transform
    model_index = args.model_index
    num_workers = args.num_workers
    max_epochs = args.max_epochs
    q = args.q
    samples_file = args.samples_file
    strategy = args.strategy
    target_file = args.target_file
    target_var = args.target_var
    target_error_var = args.target_error_var
    tb_logs_dir = f"{emulator_dir}/tb_logs"
    training_files = args.TRAINING_FILES
    y_lim = args.y_lim

    g = torch.Generator()
    g.manual_seed(0)

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    pl.seed_everything(0, workers=True)

    callbacks: list = [EpochProgressBar(desc="Training")]

    dataset = PISMDataset(
        training_files=training_files,
        samples_file=samples_file,
        target_file=target_file,
        target_var=target_var,
        target_error_var=target_error_var,
        engine=engine,
        y_lim=y_lim,
        y_transform=y_transform,
        parallel=True,
    )
    X = dataset.samples.X
    F = dataset.samples.Y
    area = dataset.samples.normed_area
    n_parameters = dataset.samples.n_parameters
    n_samples = dataset.samples.n_samples

    emulator_dir = Path(emulator_dir)
    emulator_dir.mkdir(parents=True, exist_ok=True)
    result_dir = emulator_dir / Path("emulator")
    result_dir.mkdir(parents=True, exist_ok=True)

    f = Figlet(font="standard")
    banner = f.renderText("pism-emulator")
    rank_zero_info("=" * 80)
    rank_zero_info(banner)
    rank_zero_info("=" * 80)

    rank_zero_info(f"Training model {model_index}")
    omegas = torch.Tensor(dirichlet.rvs(np.ones(n_samples), random_state=model_index)).T
    omegas = omegas.type_as(X)
    omegas_0 = torch.ones_like(omegas) / len(omegas)

    dm = PISMDataModule(
        X,
        F,
        omegas,
        omegas_0,
        num_workers=num_workers,
        batch_size=batch_size,
        seed=model_index,
    )

    svd_dir = Path("svd_cache")
    svd_dir.mkdir(parents=True, exist_ok=True)
    svd_cache = svd_dir / Path("svd.h5")
    dm.prepare_data(cutoff=cutoff, q=q, cache_path=svd_cache)
    dm.setup(stage="fit")
    V_hat = dm.eig.V_hat
    F_mean = dm.eig.F_mean
    plot_eigenglaciers(dataset, dm, model_index, emulator_dir, q=q)

    dl = dm.train_dataloader()
    rank_zero_info(
        f"N={len(dl.dataset)}, batch_size={getattr(dl, 'batch_size', '?')}, "
        f"batches/epoch={len(dl)}"
    )
    logger = TensorBoardLogger(tb_logs_dir, name=f"Emulator {model_index}")

    timer = Timer()
    callbacks.append(timer)

    e = Emulator(
        n_parameters,
        V_hat,
        F_mean,
        area,
        **hparams,
    )

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        log_every_n_steps=1,
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=False,
        enable_checkpointing=False,
        strategy=strategy,
    )

    trainer.fit(e, datamodule=dm)
    final_ckpt = result_dir / Path(f"emulator_{model_index}.ckpt")
    trainer.save_checkpoint(final_ckpt)
    rank_zero_info(f"Training took {timer.time_elapsed():.0f}s")


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
