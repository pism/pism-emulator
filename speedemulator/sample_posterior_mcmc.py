#!/bin/env python3

import time
from argparse import ArgumentParser
from os.path import join

import numpy as np
import pylab as plt
import seaborn as sns
import torch
from scipy.stats import beta

from pismemulator.models import StudentT
from pismemulator.nnemulator import NNEmulator, PISMDataset
from pismemulator.sampler import mMALA_Sampler
from pismemulator.utils import param_keys_dict as keys_dict

if __name__ == "__main__":
    __spec__ = None

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", default=False, action="store_true")
    parser.add_argument("--data_dir", default="../tests/training_data")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--emulator_dir", default="emulator_ensemble")
    parser.add_argument("--model_index", type=int, default=0)
    parser.add_argument("--num_chains", type=int, default=1)
    parser.add_argument("--out_format", choices=["csv", "parquet"], default="parquet")
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--step_size", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument(
        "--samples_file", default="../data/samples/velocity_calibration_samples_100.csv"
    )
    parser.add_argument(
        "--target_file",
        default="../data/observed_speeds/greenland_vel_mosaic250_v1_g9000m.nc",
    )
    parser.add_argument("--thinning_factor", type=int, default=1)

    parser = NNEmulator.add_model_specific_args(parser)
    args = parser.parse_args()
    hparams = vars(args)

    checkpoint = args.checkpoint
    data_dir = args.data_dir
    device = args.device
    emulator_dir = args.emulator_dir
    alpha = args.alpha
    model_index = args.model_index
    num_chains = args.num_chains
    samples = args.samples
    burn = args.burn
    out_format = args.out_format
    samples_file = args.samples_file
    step_size = args.step_size
    target_file = args.target_file
    thinning_factor = args.thinning_factor

    dataset = PISMDataset(
        data_dir=data_dir,
        samples_file=samples_file,
        target_file=target_file,
        thinning_factor=thinning_factor,
        target_corr_threshold=0,
    )

    X = dataset.X
    X_min = X.cpu().numpy().min(axis=0) - 1e-3
    X_max = X.cpu().numpy().max(axis=0) + 1e-3
    n_parameters = dataset.n_parameters

    torch.manual_seed(0)
    np.random.seed(0)
    emulator_file = join(emulator_dir, "emulator", f"emulator_{model_index}.h5")

    state_dict = torch.load(emulator_file)
    e = NNEmulator(
        state_dict["l_1.weight"].shape[1],
        state_dict["V_hat"].shape[1],
        state_dict["V_hat"],
        state_dict["F_mean"],
        state_dict["area"],
        hparams,
    )
    e.load_state_dict(state_dict)
    e.to(device)

    Y_target = dataset.Y_target
    if dataset.target_has_error:
        sigma = dataset.Y_target_error
        sigma[sigma < 10] = 10
    else:
        sigma = 10

    rho = 1.0 / (1e4**2)
    point_area = (dataset.grid_resolution * thinning_factor) ** 2
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
    X_0 = torch.tensor(X_prior.mean(axis=0), dtype=torch.float, device=device)

    student = StudentT(
        e,
        X_0,
        X_min,
        X_max,
        Y_target,
        sigma_hat,
        X_mean=dataset.X_mean,
        X_std=dataset.X_std,
        X_keys=dataset.X_keys,
        alpha=alpha,
    )
    start = time.process_time()
    sampler = mMALA_Sampler(
        probmodel=student,
        params=["X"],
        step_size=step_size,
        num_steps=samples,
        num_chains=num_chains,
        save_interval=100,
        save_dir=emulator_dir,
        save_format="parquet",
        burn_in=burn,
        tune=False,
        pretrain=True,
    )
    sampler.sample_chains()
    print(time.process_time() - start)
    X_posterior = torch.vstack(
        [sampler.chain.samples[k]["X"] for k in range(len(sampler.chain.samples))]
    )
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
    fig.subplots_adjust(wspace=0.05, hspace=0.5)
    for k in range(X_posterior.shape[1]):
        ax = axs.ravel()[k]
        sns.kdeplot(X_posterior[:, k] * dataset.X_std[k] + dataset.X_mean[k], ax=ax)
        sns.despine(ax=ax, left=True, bottom=False)
        ax.set_xlabel(keys_dict[dataset.X_keys[k]])
        ax.set_ylabel(None)
        ax.axes.yaxis.set_visible(False)
    fig.tight_layout()
