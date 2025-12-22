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
MALA Sampling.
"""
import os
import time as time_m
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Sequence, cast

import arviz as az
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from pyDOE3 import lhs
from pyfiglet import Figlet
from scipy.stats import beta
from scipy.stats.distributions import uniform

from pism_emulator.models.pdd import VecPDD as PDD
from pism_emulator.sampler.mala import MALASamplerModule, run_sampling

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


def make_fake_climate_2d(filename: str | None = None) -> xr.Dataset:
    """
    Create an idealized 2D synthetic climate dataset for tests.

    This generates an artificial monthly (12-point) climatology on a Cartesian
    grid with dimensions ``(time, x, y)``. The resulting dataset contains
    near-surface air temperature (``temp``), precipitation rate (``prec``), and
    temperature standard deviation (``stdv``), along with CF-style coordinate
    metadata and a ``time_bounds`` coordinate.

    Parameters
    ----------
    filename : str, optional
        If provided, write the dataset to this NetCDF file via ``to_netcdf``.
        If None (default), no file is written.

    Returns
    -------
    xr.Dataset
        Dataset with data variables:

        - ``temp`` : near-surface air temperature (degC), shape ``(time, x, y)``
        - ``prec`` : ice-equivalent precipitation rate (m yr-1), shape ``(time, x, y)``
        - ``stdv`` : standard deviation of near-surface air temperature (K),
          shape ``(time, x, y)``

        And coordinates:

        - ``time`` : monthly midpoints in fractional years, shape ``(time,)``
        - ``x`` / ``y`` : Cartesian coordinates in meters, shapes ``(x,)`` and ``(y,)``
        - ``time_bounds`` : bounds for ``time``, shape ``(time, 2)``

    Notes
    -----
    The construction order, dtype casts, and transposes are intentionally kept
    stable to preserve legacy test behavior (e.g., reproducibility checksums).
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

    # code could be simplified a lot more but we need a better test not
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

    return ds


def draw_samples(n_samples: int = 10_000, random_seed: int = 2) -> pd.DataFrame:
    """
    Draw Latin-hypercube samples for PDD model parameters.

    Samples are generated in the unit hypercube using Latin Hypercube Sampling
    (LHS) and then transformed to user-specified parameter distributions using
    each distribution's percent-point function (PPF; inverse CDF).

    Parameters
    ----------
    n_samples : int, optional
        Number of parameter sets to draw. Default is 10_000.
    random_seed : int, optional
        Seed for NumPy's random number generator used by the sampling routine.
        Default is 2.

    Returns
    -------
    pandas.DataFrame
        DataFrame with shape ``(n_samples, 6)`` containing the sampled parameters.
        Columns (in order) are:

        - ``pdd_factor_snow`` : snow degree-day factor (uniform in [1, 6])
        - ``pdd_factor_ice``  : ice degree-day factor (uniform in [3, 15])
        - ``refreeze_snow``   : snow refreezing fraction (uniform in [0, 0.8])
        - ``refreeze_ice``    : ice refreezing fraction (uniform in [0, 0.8])
        - ``temp_snow``       : snow/rain transition lower bound in °C (uniform in [-2, 0])
        - ``temp_rain``       : snow/rain transition upper bound in °C (uniform in [0, 4])

    Notes
    -----
    * LHS produces stratified samples over each dimension of the unit hypercube,
      which can improve space-filling relative to simple Monte Carlo sampling.
    * This function calls ``np.random.seed`` to ensure deterministic output given
      ``random_seed``.
    """
    np.random.seed(random_seed)

    distributions = {
        "pdd_factor_snow": uniform(loc=1.0, scale=5.0),  # uniform between 1 and 6
        "pdd_factor_ice": uniform(loc=3.0, scale=12),  # uniform between 3 and 15
        "refreeze_snow": uniform(loc=0.0, scale=0.8),  # uniform between 0 and 0.8
        "refreeze_ice": uniform(loc=0.0, scale=0.8),  # uniform between 0 and 0.8
        "temp_snow": uniform(loc=-2.0, scale=2.0),  # uniform between -2 and 0
        "temp_rain": uniform(loc=0.0, scale=4.0),  # uniform between 0 and 4
    }

    keys = list(distributions.keys())

    unif_sample = lhs(len(keys), n_samples)
    dist_sample = np.zeros_like(unif_sample)

    for i, key in enumerate(keys):
        dist_sample[:, i] = distributions[key].ppf(unif_sample[:, i])

    return pd.DataFrame(data=dist_sample, columns=keys)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """
    Run MALA sampling for the PDD posterior and assemble an ArviZ InferenceData.

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

    Raises
    ------
    SystemExit
        If argument parsing fails (e.g., unknown arguments), consistent with
        :mod:`argparse`. When calling from notebooks, pass ``argv=[]`` (or an
        explicit list of arguments) to prevent ipykernel arguments from being
        parsed.

    Notes
    -----
    This function is intended to be used from Python:

    >>> from pism_emulator.sampler import mala_pdd
    >>> out = mala_pdd.main(argv=["--samples", "500", "--burn", "100"])
    >>> idata = out["idata"]

    For the console script, use a thin wrapper (e.g., ``cli()``) that calls
    :func:`main` and returns an integer exit code, to avoid printing the returned
    dictionary via ``sys.exit(main())``.
    """

    parser = ArgumentParser()
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.01)

    args = parser.parse_args(list(argv) if argv is not None else None)

    accelerator = args.accelerator
    alpha = args.alpha
    chains = args.chains
    samples = args.samples
    burn = args.burn

    posterior_dir = "posterior_samples/"
    if not os.path.isdir(posterior_dir):
        os.makedirs(posterior_dir)

    f = Figlet(font="standard")
    banner = f.renderText("pism-emulator")
    rank_zero_info("=" * 80)
    rank_zero_info(banner)
    rank_zero_info("=" * 80)
    rank_zero_info("MALA Sampler")
    rank_zero_info("-" * 80)
    rank_zero_info("")

    prior_df = draw_samples(n_samples=10_000)
    X_keys = prior_df.columns

    ds = make_fake_climate_2d()
    predictor_vars = ["accumulation", "melt", "runoff", "refreeze", "smb"]

    temp = ds["temp"].to_numpy()
    precip = ds["prec"].to_numpy()
    sd = ds["stdv"].to_numpy()
    model_true = PDD(temp, precip, sd)
    model = PDD(temp, precip, sd, predictor_vars=predictor_vars)

    true_vals = {
        "pdd_factor_snow": 4.2,
        "pdd_factor_ice": 8.0,
        "refreeze_snow": 0.6,
        "refreeze_ice": 0.2,
        "temp_snow": -0.5,
        "temp_rain": 1.6,
    }
    x_true = torch.tensor(
        [
            true_vals["pdd_factor_snow"],
            true_vals["pdd_factor_ice"],
            true_vals["refreeze_snow"],
            true_vals["refreeze_ice"],
            true_vals["temp_snow"],
            true_vals["temp_rain"],
        ]
    )

    obs = model_true.forward(x_true)
    obs_pred = [obs[k] for k in predictor_vars if k in obs]

    Y_true = torch.vstack((obs_pred)).T
    noise = 0.01 * Y_true * torch.randn_like(Y_true)
    Y_true += noise

    X_prior = torch.from_numpy(prior_df.values)
    X_min = X_prior.cpu().numpy().min(axis=0)
    X_max = X_prior.cpu().numpy().max(axis=0)

    sigma = 0.025
    sh = torch.ones_like(Y_true)
    sigma_hat = sh * torch.tensor([sigma])

    start = time_m.time()
    sampler = MALASamplerModule(
        model,
        X_min,
        X_max,
        Y_true,
        sigma_hat,
        alpha=alpha,
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
    rank_zero_info("\n")
    end = time_m.time()
    time_elapsed = end - start
    rank_zero_info(f"Sampling took {time_elapsed:.0f}s")

    chains_np = [np.asarray(c) for c in samples]  # each (S, D)
    arr = np.stack(chains_np, axis=0)  # (C, S, D)

    C, _, D = arr.shape

    posterior = {name: arr[:, :, i] for i, name in enumerate(X_keys)}

    S_prior_total, D = X_prior.shape
    C_prior = C  # match posterior chains
    assert S_prior_total % C_prior == 0, "prior samples must split evenly across chains"
    S_prior = S_prior_total // C_prior

    X_prior_reshaped = X_prior.reshape(C_prior, S_prior, D)

    prior = {
        name: X_prior_reshaped[:, :, i]  # -> (C_prior, S_prior)
        for i, name in enumerate(X_keys)
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

    # Save + load
    out_nc = out_dir / "X_posterior_model.nc"
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
            )

            for i, vname in enumerate(var_names):
                hist_ax = axes[i, 0]  # histogram axis (usually column 1)
                tv = true_vals.get(vname)

                hist_ax.axvline(
                    tv,
                    linestyle=":",  # dotted
                    linewidth=1.5,
                    alpha=0.9,
                )
            if hasattr(idata, "prior"):
                for i, vname in enumerate(var_names):
                    hist_ax = axes[i, 0]  # histogram axis (usually column 1)

                    # ---- PRIOR ----
                    prior_vals = np.asarray(
                        idata.prior[vname], dtype=np.float64
                    ).ravel()
                    prior_vals = prior_vals[np.isfinite(prior_vals)]
                    if prior_vals.size < 2:
                        continue

                    # ---- POSTERIOR ----
                    post_vals = np.asarray(
                        idata.posterior[vname], dtype=np.float64
                    ).ravel()
                    post_vals = post_vals[np.isfinite(post_vals)]
                    if post_vals.size < 2:
                        continue

                    # Common bins (important!)
                    lo = min(prior_vals.min(), post_vals.min())
                    hi = max(prior_vals.max(), post_vals.max())
                    if lo == hi:
                        continue
                    bins = np.linspace(lo, hi, 31)

                    prior_hist, edges = np.histogram(
                        prior_vals, bins=bins, density=True
                    )
                    post_hist, _ = np.histogram(post_vals, bins=bins, density=True)
                    centers = 0.5 * (edges[1:] + edges[:-1])

                    # ---- SCALE PRIOR ----
                    prior_max = prior_hist.max()
                    post_max = post_hist.max()
                    if prior_max > 0:
                        scale = 5 * post_max / prior_max
                    else:
                        scale = 1.0

                    # ---- PLOT ----
                    hist_ax.plot(
                        centers,
                        scale * prior_hist,
                        alpha=0.35,
                        label="prior (scaled)",
                    )

            fig = axes.flatten()[0].get_figure()
            fig.suptitle("Posterior Traces")
            out_png = out_dir / "X_posterior_model_trace.png"
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
