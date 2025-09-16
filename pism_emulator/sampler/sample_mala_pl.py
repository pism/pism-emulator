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
from pathlib import Path
import time
from argparse import ArgumentParser
from os.path import join
from typing import Literal
import arviz as az
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from lightning import LightningModule
from scipy.stats import beta
from tqdm.auto import tqdm
import matplotlib.pylab as plt

from pism_emulator.datasets import PISMDatasetXRP as PISMDataset
from pism_emulator.nnemulator import NNEmulator


from typing import Callable, Sequence
import numpy as np
import torch
from torch import Tensor

from pism_emulator.sampler.mala import MALASamplerModule, ChainInitDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter

from pathlib import Path
import torch
import pytorch_lightning as pl


class DiskPredictionWriter(BasePredictionWriter):
    """Write each chain's samples to disk during predict (works with DDP spawn/fork)."""

    def __init__(self, out_dir: str, write_interval: str = "batch"):
        # write_interval: "batch" | "epoch"
        super().__init__(write_interval=write_interval)
        self.out_dir = Path(out_dir)

    def write_on_batch_end(  # called every batch when write_interval="batch"
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction,  # whatever predict_step returned
        batch_indices,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if prediction is None:
            return

        # Lightning may give a single dict or a list of dicts
        preds = prediction if isinstance(prediction, (list, tuple)) else [prediction]
        rank = int(getattr(trainer, "global_rank", 0))

        self.out_dir.mkdir(parents=True, exist_ok=True)
        for p in preds:
            chain = int(p["chain"])
            samples = p["samples"]  # (S, D) CPU tensor
            path = self.out_dir / f"rank{rank:02d}_chain{chain:06d}.pt"
            torch.save({"chain": chain, "rank": rank, "samples": samples}, path)


def load_pred_dir(pred_dir: str, expected_chains: int | None = None) -> torch.Tensor:
    pred_dir = Path(pred_dir)
    files = sorted(pred_dir.glob("rank*_chain*.pt"))
    records = [torch.load(f) for f in files]
    if not records:
        raise RuntimeError(f"No prediction files found in {pred_dir}")
    records.sort(key=lambda r: r["chain"])
    if expected_chains is not None and len(records) != expected_chains:
        raise RuntimeError(f"Expected {expected_chains} chains, found {len(records)}.")
    return torch.stack([r["samples"] for r in records])  # (C, S, D)


def make_trainer_for_chains(accelerator: str, n_chains: int) -> pl.Trainer:
    """
    CPU: run n_chains processes in parallel (DDP spawn).
    GPU/MPS: run 1 process (1 chain).
    """
    if accelerator.lower() == "cpu" and n_chains > 1:
        devices = n_chains
        strategy = "ddp_spawn"  # safe on macOS/Windows; uses spawn
    else:
        devices = 1  # one chain per (single) GPU/MPS
        strategy = "auto"

    return pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=False,
        enable_checkpointing=False,
        inference_mode=False,  # we need autograd for MALA
        num_sanity_val_steps=0,
    )


def run_sampling(sampler, inits, accelerator="cpu", tmp_dir="./_preds"):
    # Dataset should yield (chain_id, init_vector)
    dl = DataLoader(ChainInitDataset(inits), batch_size=1, shuffle=False, num_workers=0)

    n_chains = inits.shape[0]
    multi_cpu = accelerator == "cpu" and n_chains > 1

    if multi_cpu:
        # DDP spawn/fork: must NOT use return_predictions=True
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=n_chains,
            strategy="ddp_spawn",
            logger=False,
            enable_checkpointing=False,
            inference_mode=False,  # you need autograd
            num_sanity_val_steps=0,
            enable_progress_bar=False,  # use your own per-rank tqdm
            callbacks=[DiskPredictionWriter(tmp_dir, write_interval="batch")],
        )
        _ = trainer.predict(sampler, dl, return_predictions=False)

        # Load everything that each rank just wrote:
        from pathlib import Path

        files = sorted(Path(tmp_dir).glob("rank*_chain*.pt"))
        if not files:
            raise RuntimeError(f"No prediction files found in {tmp_dir}")
        records = [torch.load(f) for f in files]
        records.sort(key=lambda r: r["chain"])
        chains = torch.stack([r["samples"] for r in records])  # (C, S, D)
    else:
        # single GPU or single CPU process
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=1,
            logger=False,
            enable_checkpointing=False,
            inference_mode=False,
            num_sanity_val_steps=0,
            enable_progress_bar=False,
        )
        outs = trainer.predict(sampler, dl, return_predictions=True)
        outs = [o for o in outs if o is not None]
        outs.sort(key=lambda d: int(d["chain"]))
        chains = torch.stack([d["samples"] for d in outs])

    return chains


def main():
    parser = ArgumentParser()
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--checkpoint", default=False, action="store_true")
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--emulator_dir", default="emulator_ensemble")
    parser.add_argument("--model_index", type=int, default=0)
    parser.add_argument("--num_iterations", type=int, default=100000)
    parser.add_argument("--out_format", choices=["csv", "parquet"], default="parquet")
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument(
        "--samples_file", default="../data/samples/velocity_calibration_samples_100.csv"
    )
    parser.add_argument(
        "--target_file",
        default="../data/observed_speeds/greenland_vel_mosaic250_v1_g9000m.nc",
    )
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("TRAINING_FILES", nargs="*", help="PISM netCDF files")

    parser = NNEmulator.add_model_specific_args(parser)
    args = parser.parse_args()
    hparams = vars(args)

    accelerator = args.accelerator
    checkpoint = args.checkpoint
    emulator_dir = args.emulator_dir
    alpha = args.alpha
    model_index = args.model_index
    chains = args.chains
    samples = args.samples
    burn = args.burn
    out_format = args.out_format
    samples_file = args.samples_file
    target_file = args.target_file
    thin = args.thin
    training_files = args.TRAINING_FILES

    posterior_dir = f"{emulator_dir}/posterior_samples/"
    if not os.path.isdir(posterior_dir):
        os.makedirs(posterior_dir)

    dataset = PISMDataset(
        training_files=training_files,
        samples_file=samples_file,
        target_file=target_file,
        thin=thin,
        target_corr_threshold=0,
        target_error_var="velsurf_mag_error",
        target_var="velsurf_mag",
    )

    X = dataset.X
    X_min = X.cpu().numpy().min(axis=0) - 1e-3
    X_max = X.cpu().numpy().max(axis=0) + 1e-3
    n_parameters = dataset.n_parameters
    Y_target = dataset.Y_target

    torch.manual_seed(0)
    np.random.seed(0)
    emulator_file = join(emulator_dir, "emulator", f"emulator_{model_index}.h5")

    state_dict = torch.load(emulator_file, weights_only=True)
    e = NNEmulator(
        state_dict["l_1.weight"].shape[1],
        state_dict["V_hat"].shape[1],
        state_dict["V_hat"],
        state_dict["F_mean"],
        state_dict["area"],
        hparams,
    )
    e.load_state_dict(state_dict)

    if dataset.target_has_error:
        sigma = dataset.Y_target_error
        sigma[sigma < 10] = 10
    else:
        sigma = 10

    rho = 1.0 / (1e4**2)
    point_area = (dataset.grid_resolution * thin) ** 2
    K = point_area * rho
    sigma_hat = np.sqrt(sigma**2 / K**2)

    # Eq 23 in SI
    # this is 2.0 in the paper
    alpha_b = 3.0
    beta_b = 3.0
    X_prior = (
        beta.rvs(alpha_b, beta_b, size=(samples, n_parameters)) * (X_max - X_min)
        + X_min
    )
    # Initial condition for MAP. Note that using 0 yields similar results
    X_0 = torch.tensor(X_prior.mean(axis=0), requires_grad=True, dtype=torch.float)

    start = time.process_time()
    sampler = MALASamplerModule(
        e,
        X_min,
        X_max,
        Y_target,
        sigma_hat,
        metric_mode="current",
        delayed_accept=False,
        hess_refresh=1,
        burn=1000,
        samples=samples,
        h0=0.1,
        acc_target=0.25,
    )

    X_map = sampler.find_MAP(
        X_0,
        n_iters=25,
        lr=0.1,
    )
    X_map = X_map.detach().to(dtype=torch.float32, device="cpu")
    X_mean = np.asarray(dataset.X_mean.cpu().numpy(), dtype=np.float32)
    X_std = np.asarray(dataset.X_std.cpu().numpy(), dtype=np.float32)

    inits = X_map.unsqueeze(0).repeat(chains, 1).contiguous()
    chains = run_sampling(sampler, inits, accelerator=accelerator)
    print(time.process_time() - start)

    chains_np = [np.asarray(c) for c in chains]  # each (S, D)
    arr = np.stack(chains_np, axis=0)  # (C, S, D)

    # Denorm once
    X_mean = np.asarray(dataset.X_mean.cpu().numpy(), dtype=np.float32)
    X_std = np.asarray(dataset.X_std.cpu().numpy(), dtype=np.float32)
    arr_denorm = arr * X_std[None, None, :] + X_mean[None, None, :]

    C, S, D = arr_denorm.shape
    coords = {"chain": np.arange(C), "draw": np.arange(S)}
    dims = {name: ["chain", "draw"] for name in dataset.X_keys}

    posterior = {name: arr_denorm[:, :, i] for i, name in enumerate(dataset.X_keys)}

    idata = az.from_dict(
        posterior=posterior,
        coords=coords,
        dims=dims,
    )

    # (Optional) sanity check
    print("posterior dims:", idata.posterior.sizes)  # should show chain=C, draw=S

    # Save to Zarr (overwrite)
    out_dir = Path(posterior_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_zarr = out_dir / f"X_posterior_model_{model_index}.zarr"
    idata.to_datatree().to_zarr(str(out_zarr), mode="w")  # overwrite

    # Robust plotting: drop (near-)constant vars and use hist with fewer bins
    plot_traces = True
    if plot_traces:
        # variance across chain & draw
        var_all = np.nanvar(arr_denorm, axis=(0, 1))
        keep = var_all > 1e-12
        if np.any(keep):
            var_names = [dataset.X_keys[i] for i in np.flatnonzero(keep)]
            az.plot_trace(
                idata, var_names=var_names, hist_kwargs={"bins": 50}
            )  # <-- key fix: kind/hist_kwargs at top level
            out_png = out_dir / f"X_posterior_model_{model_index}.trace.png"
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close("all")
        else:
            print("All parameters are (near) constant; skipping trace plot.")


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
