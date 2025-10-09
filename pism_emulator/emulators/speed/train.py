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

import os
import warnings
from argparse import ArgumentParser
from os.path import abspath, dirname, join, realpath
from typing import Mapping

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, Timer
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
from pism_emulator.utils import plot_eigenglaciers

EMULATORS: Mapping[str, type[pl.LightningModule]] = {
    "NN": NNEmulator,
    "NN5": NN5Emulator,
    "DNN": DNNEmulator,
    "LegacyNN": LegacyNNEmulator,
}


torch.use_deterministic_algorithms(True)
torch.set_float32_matmul_precision("high")  # faster GEMMs on Ada/L40S
torch.backends.cudnn.benchmark = True

warnings.filterwarnings("ignore", ".*does not have many workers.*")


def current_script_directory():
    import inspect

    filename = inspect.stack(0)[0][1]
    return realpath(dirname(filename))


script_directory = current_script_directory()


class EpochProgressBar(Callback):
    """A simple tqdm bar that advances once per epoch."""

    def __init__(self, *, desc: str = "Training", ncols: int = 120):
        super().__init__()
        self._bar: tqdm | None = None
        self._desc = desc
        self._ncols = ncols

    @rank_zero_only
    def on_train_start(self, trainer, pl_module) -> None:
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
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self._bar is None:
            return
        # Grab latest metrics if available
        m = trainer.callback_metrics
        train_loss = float(m.get("train_loss", float("nan")))
        test_loss = float(m.get("test_loss", float("nan")))
        self._bar.set_postfix_str(f"train={train_loss:.4g}, val={test_loss:.4g}")
        self._bar.update(1)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--emulator", choices=["NN", "NN5", "DNN", "LegacyNN"], default="NN"
    )
    tmp, _ = parser.parse_known_args()

    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--emulator_dir", default="emulator_ensemble")
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--model_index", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("-q", type=int, default=100)
    parser.add_argument("--drop_out", type=float, default=0.1)
    parser.add_argument("--y_lim", type=float, nargs=2, default=[1, 10e3])
    parser.add_argument(
        "--samples_file",
        default=abspath(
            join(
                script_directory, "../data/samples/velocity_calibration_samples_50.csv"
            )
        ),
    )
    parser.add_argument(
        "--strategy",
        choices=["ddp_spawn", "ddp", "auto", "single_device"],
        default="auto",
    )
    parser.add_argument(
        "--target_file",
        default=abspath(
            join(
                script_directory,
                "../data/observed_speeds/greenland_vel_mosaic250_v1_g9000m.nc",
            )
        ),
    )
    parser.add_argument("--target_var", type=str, default="velsurf_mag")
    parser.add_argument("--target_error_var", type=str, default="velsurf_mag_error")
    parser.add_argument("--thin", type=int, default=1)
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
    devices = args.devices
    emulator_dir = args.emulator_dir
    model_index = args.model_index
    num_workers = args.num_workers
    max_epochs = args.max_epochs
    q = args.q
    p = args.drop_out
    samples_file = args.samples_file
    strategy = args.strategy
    target_file = args.target_file
    target_var = args.target_var
    target_error_var = args.target_error_var
    thin = args.thin
    tb_logs_dir = f"{emulator_dir}/tb_logs"
    training_files = args.TRAINING_FILES
    y_lim = args.y_lim

    callbacks: list = [EpochProgressBar(desc="Training")]

    rank_zero_info(y_lim)
    dataset = PISMDataset(
        training_files=training_files,
        samples_file=samples_file,
        target_file=target_file,
        target_var=target_var,
        target_error_var=target_error_var,
        y_lim=y_lim,
        log_y=True,
        parallel=True,
    )
    X = dataset.samples.X
    F = dataset.samples.Y
    area = dataset.samples.normed_area
    n_grid_points = dataset.samples.n_grid_points
    n_parameters = dataset.samples.n_parameters
    n_samples = dataset.samples.n_samples

    torch.manual_seed(0)
    np.random.seed(model_index)

    if not os.path.isdir(emulator_dir):
        os.makedirs(emulator_dir)
        os.makedirs(os.path.join(emulator_dir, "emulator"))

    f = Figlet(font="standard")
    banner = f.renderText("pism-emulator")
    print("=" * 80)
    print(banner)
    print("=" * 80)

    rank_zero_info(f"Training model {model_index}")
    omegas = torch.Tensor(dirichlet.rvs(np.ones(n_samples), random_state=model_index)).T
    omegas = omegas.type_as(X)
    omegas_0 = torch.ones_like(omegas) / len(omegas)

    dm = PISMDataModule(
        X, F, omegas, omegas_0, num_workers=num_workers, batch_size=batch_size
    )

    dm.prepare_data(cutoff=0.9999)
    dm.setup(stage="fit")
    V_hat = dm.eig.V_hat
    F_mean = dm.eig.F_mean
    print(V_hat.shape)
    plot_eigenglaciers(dataset, dm, model_index, emulator_dir, q=q)

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=f"{emulator_dir}/emulator",
    #     filename="emulator_{model_index}",
    #     save_last=True,  # write only the final checkpoint
    #     every_n_epochs=None,  # disable periodic-by-epoch saving
    #     every_n_train_steps=None,  # disable periodic-by-step saving
    #     train_time_interval=None,  # disable time-based saving
    #     save_top_k=0,  # disable "best" checkpoints (no monitor)
    #     save_on_train_epoch_end=False,  # don't save at each epoch end
    # )
    # checkpoint_callback.CHECKPOINT_NAME_LAST = f"emulator_{model_index}"
    # callbacks.append(checkpoint_callback)

    dl = dm.train_dataloader()
    print(
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
    final_ckpt = f"{emulator_dir}/emulator/emulator_{model_index}.ckpt"
    trainer.save_checkpoint(final_ckpt)
    rank_zero_info(f"Training took {timer.time_elapsed():.0f}s")


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
