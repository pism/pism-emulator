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

import datetime
import datetime as dt
import os
import time
import warnings
from argparse import ArgumentParser
from os.path import join
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

import arviz as az
import lightning as pl
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from joblib import Parallel, delayed
from lightning import LightningModule
from lightning.pytorch.callbacks import Timer
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only
from pyDOE3 import lhs
from pyfiglet import Figlet
from scipy.stats import beta
from scipy.stats.distributions import uniform
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pism_emulator.models.pdd import PDD
from pism_emulator.sampler.mala import ChainInitDataset, MALASamplerModule, run_sampling

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


def make_fake_climate_2d(filename=None):
    """Create an artificial temperature and precipitation file.

    This function is used if pypdd.py is called as a script without an input
    file. The file produced contains an idealized, three-dimensional (t, x, y)
    distribution of near-surface air temperature, precipitation rate and
    standard deviation of near-surface air temperature to be read by
    `PDDModel.nco`.

    filename: str, optional
        Name of output file.
    """

    ATTRIBUTES = {
        # coordinate variables
        "x": {
            "axis": "X",
            "long_name": "x-coordinate in Cartesian system",
            "standard_name": "projection_x_coordinate",
            "units": "m",
        },
        "y": {
            "axis": "Y",
            "long_name": "y-coordinate in Cartesian system",
            "standard_name": "projection_y_coordinate",
            "units": "m",
        },
        "time": {
            "axis": "T",
            "long_name": "time",
            "standard_name": "time",
            "bounds": "time_bounds",
            "units": "yr",
        },
        "time_bounds": {},
        # climatic variables
        "temp": {"long_name": "near-surface air temperature", "units": "degC"},
        "prec": {"long_name": "ice-equivalent precipitation rate", "units": "m yr-1"},
        "stdv": {
            "long_name": "standard deviation of near-surface air temperature",
            "units": "K",
        },
        # cumulative quantities
        "smb": {
            "standard_name": "land_ice_surface_specific_mass_balance",
            "long_name": "cumulative ice-equivalent surface mass balance",
            "units": "m yr-1",
        },
        "pdd": {
            "long_name": "cumulative number of positive degree days",
            "units": "degC day",
        },
        "accu": {
            "long_name": "cumulative ice-equivalent surface accumulation",
            "units": "m",
        },
        "snow_melt": {
            "long_name": "cumulative ice-equivalent surface melt of snow",
            "units": "m",
        },
        "ice_melt": {
            "long_name": "cumulative ice-equivalent surface melt of ice",
            "units": "m",
        },
        "melt": {"long_name": "cumulative ice-equivalent surface melt", "units": "m"},
        "runoff": {
            "long_name": "cumulative ice-equivalent surface meltwater runoff",
            "units": "m yr-1",
        },
        # instantaneous quantities
        "inst_pdd": {
            "long_name": "instantaneous positive degree days",
            "units": "degC day",
        },
        "accu_rate": {
            "long_name": "instantaneous ice-equivalent surface accumulation rate",
            "units": "m yr-1",
        },
        "snow_melt_rate": {
            "long_name": "instantaneous ice-equivalent surface melt rate of snow",
            "units": "m yr-1",
        },
        "ice_melt_rate": {
            "long_name": "instantaneous ice-equivalent surface melt rate of ice",
            "units": "m yr-1",
        },
        "melt_rate": {
            "long_name": "instantaneous ice-equivalent surface melt rate",
            "units": "m yr-1",
        },
        "runoff_rate": {
            "long_name": "instantaneous ice-equivalent surface runoff rate",
            "units": "m yr-1",
        },
        "inst_smb": {
            "long_name": "instantaneous ice-equivalent surface mass balance",
            "units": "m yr-1",
        },
        "snow_depth": {"long_name": "depth of snow cover", "units": "m"},
    }

    # FIXME code could be simplified a lot more but we need a better test not
    # relying on exact reproducibility of this toy climate data.

    # assign coordinate values
    lx = ly = 750000
    x = xr.DataArray(np.linspace(-lx, lx, 201, dtype="f4"), dims="x")
    y = xr.DataArray(np.linspace(-ly, ly, 201, dtype="f4"), dims="y")
    time = xr.DataArray((np.arange(12, dtype="f4") + 0.5) / 12, dims="time")
    tboundsvar = np.empty((12, 2), dtype="f4")
    tboundsvar[:, 0] = time[:] - 1.0 / 24
    tboundsvar[:, 1] = time[:] + 1.0 / 24

    # seasonality index from winter to summer
    season = xr.DataArray(-np.cos(np.arange(12) * 2 * np.pi / 12), dims="time")

    # order of operation is dictated by test md5sum and legacy f4 dtype
    temp = 5 * season - 10 * x / lx + 0 * y
    prec = y / ly * (season.astype("f4") + 0 * x + np.sign(y))
    stdv = (2 + y / ly - x / lx) * (1 + season)

    # this is also why transpose is needed here, and final type conversion
    temp = temp.transpose("time", "x", "y").astype("f4")
    prec = prec.transpose("time", "x", "y").astype("f4")
    stdv = stdv.transpose("time", "x", "y").astype("f4")

    # assign variable attributes
    temp.attrs.update(ATTRIBUTES["temp"])
    prec.attrs.update(ATTRIBUTES["prec"])
    stdv.attrs.update(ATTRIBUTES["stdv"])

    # make a dataset
    ds = xr.Dataset(
        data_vars={"temp": temp, "prec": prec, "stdv": stdv},
        coords={
            "time": time,
            "x": x,
            "y": y,
            "time_bounds": (["time", "nv"], tboundsvar[:]),
        },
    )

    # write dataset to file
    if filename is not None:
        ds.to_netcdf(filename)

    # return dataset
    return ds


def draw_samples(n_samples=1_0000, random_seed=2):
    """
    Draw samples.
    """
    np.random.seed(random_seed)

    distributions = {
        "pdd_factor_snow": uniform(loc=1.0, scale=5.0),  # uniform between 1 and 6
        "pdd_factor_ice": uniform(loc=3.0, scale=12),  # uniform between 3 and 15
        "refreeze_snow": uniform(loc=0.0, scale=0.6),  # uniform between 0 and 1
        "refreeze_ice": uniform(loc=0.0, scale=0.01),  # uniform between 0 and 1
        "temp_snow": uniform(loc=-2.0, scale=2.0),  # uniform between 0 and 1
        "temp_rain": uniform(loc=0.0, scale=2.0),  # uniform between 0 and 1
    }

    # Names of all the variables
    keys = [x for x in distributions.keys()]

    # Describe the Problem
    problem = {"num_vars": len(keys), "names": keys, "bounds": [[0, 1]] * len(keys)}

    # Generate uniform samples (i.e. one unit hypercube)
    unif_sample = lhs(len(keys), n_samples)

    # To hold the transformed variables
    dist_sample = np.zeros_like(unif_sample)

    # Now transform the unit hypercube to the prescribed distributions
    # For each variable, transform with the inverse of the CDF (inv(CDF)=ppf)
    for i, key in enumerate(keys):
        dist_sample[:, i] = distributions[key].ppf(unif_sample[:, i])

    # Save to CSV file using Pandas DataFrame and to_csv method
    header = keys
    # Convert to Pandas dataframe, append column headers, output as csv
    df = pd.DataFrame(data=dist_sample, columns=header)

    return df


def main():
    """
    Main.
    """
    parser = ArgumentParser()
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--model_index", type=int, default=0)
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.01)

    args = parser.parse_args()
    hparams = vars(args)

    accelerator = args.accelerator
    alpha = args.alpha
    model_index = args.model_index
    chains = args.chains
    samples = args.samples
    burn = args.burn

    posterior_dir = "posterior_samples/"
    if not os.path.isdir(posterior_dir):
        os.makedirs(posterior_dir)

    f = Figlet(font="standard")
    banner = f.renderText("pism-emulator")
    print("=" * 80)
    print(banner)
    print("=" * 80)
    print(f"MALA Sampler")
    print("-" * 80)
    print("")

    prior_df = draw_samples(n_samples=10_000)
    X_keys = prior_df.columns

    ds = make_fake_climate_2d()
    predictor_vars = ["accumulation", "melt", "runoff", "refreeze", "smb"]

    temp = ds["temp"].to_numpy()
    precip = ds["prec"].to_numpy()
    sd = ds["stdv"].to_numpy()
    model_true = PDD(temp, precip, sd)
    model = PDD(temp, precip, sd, predictor_vars=predictor_vars)

    f_snow_val = 4.2
    f_ice_val = 8.0
    refreeze_snow_val = 0.6
    refreeze_ice_val = 0.2
    temp_snow_val = 0.0
    temp_rain_val = 2.0
    x_true = torch.tensor(
        [
            f_snow_val,
            f_ice_val,
            refreeze_snow_val,
            refreeze_ice_val,
            temp_snow_val,
            temp_rain_val,
        ]
    )

    obs = model_true.forward(x_true)
    obs_pred = [obs[k] for k in predictor_vars if k in obs]

    Y_true = torch.vstack((obs_pred)).T

    X_prior = torch.from_numpy(prior_df.values)
    X_min = X_prior.cpu().numpy().min(axis=0)
    X_max = X_prior.cpu().numpy().max(axis=0)

    sigma = 0.1
    sh = torch.ones_like(Y_true)
    sigma_hat = sh * torch.tensor([sigma])

    start = time.time()
    sampler = MALASamplerModule(
        model,
        X_min,
        X_max,
        Y_true,
        sigma_hat,
        log_y=False,
        metric_mode="current",
        delayed_accept=False,
        hess_refresh=1,
        burn=burn,
        samples=samples,
        h0=0.1,
        acc_target=0.25,
    )

    alpha_b = 3.0
    beta_b = 3.0
    X_prior = (
        beta.rvs(alpha_b, beta_b, size=(10_000, X_prior.shape[-1])) * (X_max - X_min)
        + X_min
    )
    X_0 = torch.tensor(X_prior.mean(axis=0), requires_grad=True, dtype=torch.float)

    X_map = sampler.find_MAP(
        X_0,
    )

    X_map = X_map.detach().to(dtype=torch.float32)
    rank_zero_info("-" * 80)
    rank_zero_info("MAP Point")
    rank_zero_info("-" * 80)
    rank_zero_info(X_map)

    inits = X_map.unsqueeze(0).repeat(chains, 1).contiguous()
    stats = run_sampling(sampler, inits, accelerator=accelerator)
    samples = stats["samples"]  # (C, S, D)
    lp = stats.get("lp")  # (C, S) or None
    step = stats.get("step_size")  # (C, S) or None
    accept = stats.get("accept")  # (C, S) or None
    rank_zero_info("\n\n\n\n\n\n\n\n\n\n\n\n")
    end = time.time()
    time_elapsed = end - start
    rank_zero_info(f"Sampling took {time_elapsed:.0f}s")

    chains_np = [np.asarray(c) for c in samples]  # each (S, D)
    arr = np.stack(chains_np, axis=0)  # (C, S, D)

    C, S, D = arr.shape
    coords = {"chain": np.arange(C), "draw": np.arange(S)}
    dims = {name: ["chain", "draw"] for name in X_keys}

    posterior = {name: arr[:, :, i] for i, name in enumerate(X_keys)}

    S_prior_total, D = X_prior.shape
    C_prior = C  # match posterior chains
    assert S_prior_total % C_prior == 0, "prior samples must split evenly across chains"
    S_prior = S_prior_total // C_prior

    X_prior_reshaped = X_prior.reshape(C_prior, S_prior, D)

    prior_coords = {"chain": np.arange(C_prior), "draw": np.arange(S_prior)}
    prior_dims = {name: ["chain", "draw"] for name in X_keys}

    prior = {
        name: X_prior_reshaped[:, :, i]  # -> (C_prior, S_prior)
        for i, name in enumerate(X_keys)
    }

    idata = az.from_dict(posterior=posterior, prior=prior)
    idata = az.from_dict(
        posterior=posterior,
        prior=prior,
        sample_stats={
            "lp": lp.numpy(),  # (chain, draw)
            "step": step.numpy(),  # (chain, draw)
            "accept": accept.numpy(),  # (chain, draw) -> bool
        },
        # optional: log_likelihood group if you want arviz to treat it as such
        # log_likelihood = {"lp": lp.numpy()},
    )
    # (Optional) sanity check
    rank_zero_info(
        "posterior dims:", idata.posterior.sizes
    )  # should show chain=C, draw=S

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
        var_all = np.nanvar(arr, axis=(0, 1))
        keep = var_all > 1e-12
        if np.any(keep):
            var_names = [X_keys[i] for i in np.flatnonzero(keep)]
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


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    main()
