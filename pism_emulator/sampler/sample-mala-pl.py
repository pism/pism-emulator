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
        samples=3000,
        h0=0.1,
        acc_target=0.25,
    )

    X_map = sampler.find_MAP(
        X_0,
        X_keys=dataset.X_keys,  # optional pretty printing
        X_mean=dataset.X_mean.cpu(),
        X_std=dataset.X_std.cpu(),
        n_iters=25,
        lr=0.1,
    )

    inits = torch.stack([X_map.clone() for _ in range(chains)])
    dl = DataLoader(ChainInitDataset(inits), batch_size=1, num_workers=0, shuffle=False)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=chains,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        inference_mode=False,  # <-- IMPORTANT
        num_sanity_val_steps=0,  # optional: skip sanity steps
    )
    chains = trainer.predict(
        sampler, dl
    )  # list length = n_chains, each (samples, dim) tensor
    print(time.process_time() - start)

    # result: iterable of chains, each array (draw, dim)
    arr = np.stack(chains, axis=0)  # (chain, draw, dim)

    # Denormalize ONCE (no in-place *= / += in a loop)
    X_mean = np.asarray(dataset.X_mean.cpu().numpy(), dtype=np.float32)
    X_std = np.asarray(dataset.X_std.cpu().numpy(), dtype=np.float32)
    arr_denorm = arr * X_std[None, None, :] + X_mean[None, None, :]

    # Build one InferenceData with all chains
    posterior = {name: arr_denorm[:, :, i] for i, name in enumerate(dataset.X_keys)}
    idata = az.from_dict(posterior=posterior)  # infers chain/draw from (C, S)

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
