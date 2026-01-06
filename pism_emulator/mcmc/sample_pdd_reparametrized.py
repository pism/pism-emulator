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
import time as time_m
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Sequence, cast

import arviz as az
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import pint  # pylint: disable=unused-import
import pint_xarray  # noqa: F401  (registers accessor) # pylint: disable=unused-import
import torch
import xarray as xr
from dask.diagnostics import ProgressBar
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from pyfiglet import Figlet
from scipy.stats import beta

from pism_emulator.lhs.draw import draw_samples
from pism_emulator.mcmc.mala import ReparametrizedMALASamplerModule as MALA
from pism_emulator.mcmc.mala import run_sampling
from pism_emulator.models.pdd import PDD

xr.set_options(keep_attrs=True)

torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.conv.fp32_precision = "tf32"  # pylint: disable=no-member

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
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--result-dir", default="posterior")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--use-eig", action="store_true")
    parser.add_argument("--thin", type=int, default=4)
    parser.add_argument("--years", nargs=2, default=["1980", "1989"])
    parser.add_argument("CLIMATEFILE", nargs=1)
    args = parser.parse_args(list(argv) if argv is not None else None)

    accelerator = args.accelerator
    alpha = args.alpha
    burn = args.burn
    chains = args.chains
    samples = args.samples
    thin = args.thin
    use_eig = args.use_eig
    url = args.CLIMATEFILE[0]
    years = args.years

    posterior_dir = Path(args.result_dir)
    posterior_dir.mkdir(parents=True, exist_ok=True)

    f = Figlet(font="standard")
    banner = f.renderText("pism-emulator")
    rank_zero_info("=" * 80)
    rank_zero_info(banner)
    rank_zero_info("=" * 80)
    rank_zero_info("MALA Sampler")
    rank_zero_info("-" * 80)
    rank_zero_info("")

    prior_df = draw_samples(n_samples=10_000)
    _, prior_max = prior_df.min(), prior_df.max()

    X_keys = prior_df.columns

    rho_w = xr.DataArray(1000).pint.quantify("kg m^-3")
    rho_w.name = "water_density"
    day = xr.DataArray(1).pint.quantify("day")

    hh5_vars = {
        "tas": "temp",
        "precipitation": "precipitation",
        "rogl": "runoff",
        "gld": "smb",
        "rfrz": "refreeze",
        "sn": "snow",
        "snmel": "snow_melt",
    }

    ds = xr.open_dataset(
        url,
        chunks={"time": -1, "rlat": "auto", "rlon": "auto"},
    )

    ds = ds.sel(time=slice(*years))
    ds = ds.thin({"rlat": thin, "rlon": thin})
    ds = ds.rename_vars(hh5_vars)
    ds = ds.transpose("rlat", "rlon", "time")
    ds = ds.stack(z=("rlat", "rlon")).dropna(dim="z").unify_chunks()

    predictor_rate_vars = ["smb", "runoff", "refreeze", "snow_melt"]
    predictor_sum_vars = ["snow"]
    predictor_vars = predictor_rate_vars + predictor_sum_vars

    temp_monthly_mean = ds["temp"].resample(time="MS").mean("time")
    precipitation_monthly_sum = ds["precipitation"].resample(time="MS").sum("time")
    precipitation_monthly_sum *= day
    precipitation_monthly_sum /= rho_w

    temp_monthly_std = ds["temp"].resample(time="MS").std("time")
    temp_monthly_std.name = "temp_std"

    train = xr.merge([temp_monthly_mean, temp_monthly_std, precipitation_monthly_sum])
    train = (
        train.assign_coords(
            year=train["time"].dt.year,
            month=train["time"].dt.month,
        )
        .set_index(time=("year", "month"))
        .unstack("time")
    )
    rank_zero_info("Preparing training data")
    with ProgressBar():
        train = train.pint.dequantify().compute()

    predict_rates = ds[predictor_rate_vars].resample(time="YS").sum("time") * day
    predict_sums = ds[predictor_sum_vars]
    predict_sums = predict_sums.groupby("time.year") - predict_sums.groupby(
        "time.year"
    ).first("time")
    predict_sums = predict_sums.resample(time="YS").max("time")
    predict = xr.merge([predict_rates, predict_sums])

    predict = (
        predict.assign_coords(
            year=predict["time"].dt.year,
            month=predict["time"].dt.month,
        )
        .set_index(time=("year", "month"))
        .unstack("time")
    )
    rank_zero_info("Preparing prediction data")
    with ProgressBar():
        predict = predict.pint.dequantify().compute()

    temp = torch.from_numpy(train["temp"].to_numpy().astype(np.float32, copy=False))
    precip = torch.from_numpy(
        train["precipitation"].to_numpy().astype(np.float32, copy=False)
    )
    sd = torch.from_numpy(train["temp_std"].to_numpy().astype(np.float32, copy=False))
    model = PDD(temp, precip, sd, predictor_vars=predictor_vars)

    cols = [torch.from_numpy(predict[v].to_numpy()) for v in predictor_vars]
    cols2 = list(cols)
    Y_true = torch.cat(cols2, dim=-1)

    X_prior = torch.from_numpy(prior_df.values)
    X_min = X_prior.cpu().numpy().min(axis=0)
    X_max = X_prior.cpu().numpy().max(axis=0)

    sigma = 0.05
    sigma_hat = torch.clamp(torch.abs(sigma * Y_true), min=1e-4)

    start = time_m.time()
    sampler = MALA(
        model,
        X_min,
        X_max,
        Y_true,
        sigma_hat,
        alpha=alpha,
        metric_mode="current",
        delayed_accept=False,
        hess_refresh=1,
        burn=burn,
        samples=samples,
        h0=0.1,
        acc_target=0.25,
        use_eig=use_eig,
    )

    alpha_b = 3.0
    beta_b = 3.0
    X_prior = (
        beta.rvs(alpha_b, beta_b, size=(10_000, X_prior.shape[-1])) * (X_max - X_min)
        + X_min
    )
    X_0 = torch.tensor(X_prior.mean(axis=0), requires_grad=True, dtype=torch.float)

    # Convert X_0 to unbounded space (φ) for ReparametrizedMALASamplerModule
    phi_0 = sampler.X_to_phi(X_0)

    # Find MAP in unbounded space
    phi_map = sampler.find_MAP(
        phi_0,
    )

    # Convert back to bounded space for interpretation
    X_map = sampler.phi_to_X(phi_map).detach().to(dtype=torch.float32)
    X_map_n = X_map.numpy()

    # Display MAP in original parameter space (bounded, physical values)
    rank_zero_info("-" * 80)
    rank_zero_info("MAP Point")
    rank_zero_info("-" * 80)
    rank_zero_info(X_map_n)

    # Initialize chains in unbounded space (φ) for ReparametrizedMALASamplerModule
    phi_map_tensor = phi_map.detach().cpu()
    inits = phi_map_tensor.unsqueeze(0).repeat(chains, 1).contiguous()
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

    for grp in ("posterior", "prior"):
        if hasattr(idata, grp):
            ds = getattr(idata, grp)
            setattr(idata, grp, ds.astype({v: "float32" for v in ds.data_vars}))

    # Save + load
    out_nc = posterior_dir / Path("X_posterior_model.nc")
    az.to_netcdf(idata, out_nc, engine="netcdf4")

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
            out_png = posterior_dir / Path("posterior_trace.png")
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
