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

# pylint: disable=too-many-statements,too-many-branches,redefined-builtin


"""
MALA Speed sampler.
"""
import time
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import arviz as az
import lightning as pl
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from pyfiglet import Figlet
from scipy.stats import beta

from pism_emulator.datasets import PISMInterpolatedDataset as PISMDataset
from pism_emulator.emulators.nnemulator import DNNEmulator, NNEmulator
from pism_emulator.sampler.mala import MALASamplerModule, run_sampling
from pism_emulator.utils import param_keys_dict as keys_dict

EMULATORS: Mapping[str, type[pl.LightningModule]] = {
    "NN": NNEmulator,
    "DNN": DNNEmulator,
}

warnings.filterwarnings(
    "ignore",
    message=r".*'predict_dataloader' does not have many workers.*",
    category=UserWarning,
    module=r"lightning\.pytorch",
)

rcparams = {
    "axes.linewidth": 0.15,
    "xtick.major.size": 2.0,
    "xtick.major.width": 0.15,
    "ytick.major.size": 2.0,
    "ytick.major.width": 0.15,
    "hatch.linewidth": 0.15,
    "font.size": 6,
}


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """
    Run MALA sampling for the speed posterior and assemble an ArviZ InferenceData.

    This function is the programmatic entry point. It parses command-line style
    arguments, runs the sampler, and returns a dictionary containing the resulting
    `arviz.InferenceData` object (and optionally other intermediate artifacts).

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments **excluding** the program name (i.e., like
        ``sys.argv[1:]``). If ``None`` (default), arguments are taken from the
        current process' ``sys.argv[1:]``. Passing ``argv=[]`` is recommended
        when calling from a Jupyter notebook to avoid ipykernel arguments.

    Returns
    -------
    dict[str, Any]
        Results dictionary. At minimum this includes:

        - ``"idata"``: :class:`arviz.InferenceData`
          InferenceData containing ``posterior`` and ``prior`` groups, and
          optionally a ``sample_stats`` group (e.g., log-probability, step sizes,
          acceptance indicators) if available.

        Additional keys may be included depending on the configuration and
        sampler implementation (for example, raw prior/posterior arrays, runtime
        metadata, or filenames of saved outputs).
    """

    parser = ArgumentParser()
    parser.add_argument("--emulator", choices=["NN", "DNN"], default="DNN")
    tmp, _ = parser.parse_known_args()
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--emulator-dir", default="emulator_ensemble")
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument(
        "--samples-file", default="../data/samples/velocity_calibration_samples_100.csv"
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
    parser.add_argument("MODEL_FILE", nargs=1, help="Emulator ckpt")

    cls = EMULATORS[tmp.emulator]
    cls.add_model_specific_args(parser)
    Emulator = cls  # type: type[pl.LightningModule]
    # let the chosen model extend the parser
    if tmp.emulator == "NN":
        Emulator = NNEmulator
    elif tmp.emulator == "DNN":
        Emulator = DNNEmulator

    args = parser.parse_args()

    accelerator = args.accelerator
    emulator_dir = args.emulator_dir
    alpha = args.alpha
    model_index = args.model_index
    chains = args.chains
    samples = args.samples
    y_lim = args.y_lim
    y_transform = args.y_transform
    burn = args.burn
    samples_file = args.samples_file
    target_file = args.target_file
    training_files = args.TRAINING_FILES
    model_file = args.MODEL_FILE[0]
    target_var = args.target_var
    target_error_var = args.target_error_var

    emulator_dir = Path(emulator_dir)
    emulator_dir.mkdir(parents=True, exist_ok=True)
    posterior_dir = emulator_dir / Path("posterior")
    posterior_dir.mkdir(parents=True, exist_ok=True)

    f = Figlet(font="standard")
    banner = f.renderText("pism-emulator")
    rank_zero_info("=" * 80)
    rank_zero_info(banner)
    rank_zero_info("=" * 80)
    rank_zero_info("MALA Sampler")
    rank_zero_info("-" * 80)
    rank_zero_info("")

    dataset = PISMDataset(
        training_files=training_files,
        samples_file=samples_file,
        target_file=target_file,
        target_corr_threshold=0,
        target_error_var=target_error_var,
        target_var=target_var,
        y_lim=y_lim,
        y_transform=y_transform,
    )

    X = dataset.samples.X
    X_min = X.cpu().numpy().min(axis=0) - 1e-3
    X_max = X.cpu().numpy().max(axis=0) + 1e-3
    X_mean = np.asarray(dataset.samples.X_mean.cpu().numpy(), dtype=np.float32)
    X_std = np.asarray(dataset.samples.X_std.cpu().numpy(), dtype=np.float32)
    n_parameters = dataset.samples.n_parameters
    Y_target = dataset.target.Y_target

    torch.manual_seed(0)
    np.random.seed(0)

    model = Emulator.load_from_checkpoint(
        model_file,
        map_location="cpu",
    )
    if dataset.target.Y_target_error is not None:
        sigma = dataset.target.Y_target_error
    else:
        sigma = 10
    sigma = torch.clamp(sigma, min=1e-4)

    rho = 1.0 / (1e4**2)
    point_area = (dataset.target.grid_resolution) ** 2
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

    start = time.time()
    sampler = MALASamplerModule(
        model,
        X_min,
        X_max,
        Y_target,
        sigma_hat,
        alpha=alpha,
        metric_mode="current",
        delayed_accept=False,
        hess_refresh=1,
        burn=burn,
        samples=samples,
        h0=0.1,
        acc_target=0.25,
    )

    X_map = sampler.find_MAP(
        X_0,
    )

    X_map = X_map.detach().to(dtype=torch.float32)
    rank_zero_info("-" * 80)
    rank_zero_info("MAP Point")
    rank_zero_info("-" * 80)
    rank_zero_info(
        "".join(
            [
                f"{keys_dict[key]}: {(val * std + mean):.3f}\n"
                for key, val, std, mean in zip(
                    dataset.samples.X_keys,
                    X_map,
                    X_std,
                    X_mean,
                )
            ]
        )
    )

    inits = X_map.unsqueeze(0).repeat(chains, 1).contiguous()
    stats = run_sampling(sampler, inits, accelerator=accelerator)
    samples = stats["samples"]  # (C, S, D)
    lp = stats.get("lp")  # (C, S) or None
    step = stats.get("step_size")  # (C, S) or None
    accept = stats.get("accept")  # (C, S) or None
    rank_zero_info("\n\n\n\n")
    end = time.time()
    time_elapsed = end - start
    rank_zero_info(f"Sampling took {time_elapsed:.0f}s")

    chains_np = [np.asarray(c) for c in samples]  # each (S, D)
    arr = np.stack(chains_np, axis=0)  # (C, S, D)

    # Denorm once
    X_mean = np.asarray(dataset.samples.X_mean.cpu().numpy(), dtype=np.float32)
    X_std = np.asarray(dataset.samples.X_std.cpu().numpy(), dtype=np.float32)
    arr_denorm = arr * X_std[None, None, :] + X_mean[None, None, :]

    C, S, D = arr_denorm.shape
    coords = {"chain": np.arange(C), "draw": np.arange(S)}
    dims = {name: ["chain", "draw"] for name in dataset.samples.X_keys}

    posterior = {
        name: arr_denorm[:, :, i] for i, name in enumerate(dataset.samples.X_keys)
    }

    S_prior_total, D = X_prior.shape
    C_prior = C  # match posterior chains
    assert S_prior_total % C_prior == 0, "prior samples must split evenly across chains"
    S_prior = S_prior_total // C_prior

    X_prior_reshaped = (
        X_prior.reshape(C_prior, S_prior, D) * X_std[None, None, :]
        + X_mean[None, None, :]
    )

    prior_coords = {"chain": np.arange(C_prior), "draw": np.arange(S_prior)}
    prior_dims = {name: ["chain", "draw"] for name in dataset.samples.X_keys}

    prior = {
        name: X_prior_reshaped[:, :, i]  # -> (C_prior, S_prior)
        for i, name in enumerate(dataset.samples.X_keys)
    }

    sample_stats: dict[str, np.ndarray] = {}

    if lp is not None:
        lp_t = cast(torch.Tensor, lp)
        sample_stats["lp"] = lp_t.detach().cpu().numpy()

    if step is not None:
        step_t = cast(torch.Tensor, step)
        sample_stats["step"] = step_t.detach().cpu().numpy()

    if accept is not None:
        accept_t = cast(torch.Tensor, accept)
        sample_stats["accept"] = accept_t.detach().cpu().numpy()

    idata = az.from_dict(
        posterior=posterior,
        prior=prior,
        sample_stats=sample_stats if sample_stats else None,
    )

    # Save to Zarr (overwrite)
    out_dir = Path(posterior_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional: cast to float32 to shrink size
    for grp in ("posterior", "prior"):
        if hasattr(idata, grp):
            ds = getattr(idata, grp)
            setattr(idata, grp, ds.astype({v: "float32" for v in ds.data_vars}))

    # Add useful metadata
    idata.attrs.update(
        {
            "created": pd.Timestamp.utcnow().isoformat(),
            "model": type(model).__name__,
            "emulator_dir": str(posterior_dir),
            "n_chains": int(idata.posterior.sizes["chain"]),
            "n_draws": int(idata.posterior.sizes["draw"]),
        }
    )

    # Save + load
    out_nc = out_dir / f"X_posterior_model_{model_index}.nc"
    az.to_netcdf(idata, out_nc)  # write

    # Robust plotting: drop (near-)constant vars and use hist with fewer bins
    az.style.use(["arviz-white", "arviz-greenish"])
    with mpl.rc_context(rc=rcparams):
        # variance across chain & draw
        var_all = np.nanvar(arr_denorm, axis=(0, 1))
        keep = var_all > 1e-12
        if np.any(keep):
            var_names = [dataset.samples.X_keys[i] for i in np.flatnonzero(keep)]
            axes = az.plot_trace(
                idata, var_names=var_names, hist_kwargs={"bins": 50}, figsize=(6.4, 6.4)
            )  # <-- key fix: kind/hist_kwargs at top level
            fig = axes.flatten()[0].get_figure()
            fig.suptitle("Posterior Traces")
            out_png = out_dir / f"X_posterior_model_{model_index}.trace.png"
            plt.savefig(out_png, dpi=300, bbox_inches="tight")
            plt.close("all")
        else:
            rank_zero_info("All parameters are (near) constant; skipping trace plot.")

    return {"idata": idata, "prior": prior, "posterior": posterior}


def cli(argv: Sequence[str] | None = None) -> int:
    """
    Console entry point.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments (without the program name). If None, uses sys.argv.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    _ = main(argv=argv)
    return 0


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    raise SystemExit(cli())
