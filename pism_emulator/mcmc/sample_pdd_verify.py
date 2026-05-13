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
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from pyfiglet import Figlet
from scipy.stats import beta

from pism_emulator.lhs.draw import draw_samples
from pism_emulator.mcmc.mala import MALASamplerModule as MALA
from pism_emulator.mcmc.mala import run_sampling
from pism_emulator.models.pdd import PDD, make_fake_climate_2d

warnings.filterwarnings(
    "ignore",
    message=r".*'predict_dataloader' does not have many workers.*",
    category=UserWarning,
    module=r"lightning\.pytorch",
)
torch.set_float32_matmul_precision("medium")


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
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--result-dir", default="posterior")
    parser.add_argument("--use-eig", action="store_true")

    args = parser.parse_args(list(argv) if argv is not None else None)

    accelerator = args.accelerator
    alpha = args.alpha
    burn = args.burn
    chains = args.chains
    samples = args.samples
    use_eig = args.use_eig

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
    X_keys = prior_df.columns

    ds = make_fake_climate_2d(torch_order=True)
    predictor_vars = ["accumulation", "melt", "runoff", "refreeze", "smb", "snow"]

    # Convert to PyTorch tensors (required for autograd in MALA)
    temp = torch.from_numpy(ds["temp"].to_numpy().astype(np.float32, copy=False))
    precip = torch.from_numpy(ds["prec"].to_numpy().astype(np.float32, copy=False))
    sd = torch.from_numpy(ds["stdv"].to_numpy().astype(np.float32, copy=False))

    model_true = PDD(temp, precip, sd, predictor_vars=predictor_vars)
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

    # Set random seed for reproducible noise
    torch.manual_seed(42)
    Y_true: torch.Tensor = model_true.forward(x_true)
    noise = 0.05 * Y_true * torch.randn_like(Y_true)
    Y_true += noise

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
    # Set random seed for reproducible prior samples
    np.random.seed(42)
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
    rank_zero_info("MAP Point (Regular Sampler)")
    rank_zero_info("-" * 80)
    rank_zero_info(X_map)

    # Evaluate objective value
    X_map_test = X_map.clone().detach().requires_grad_(True)
    neg_log_p_regular = sampler.neg_log_prob(X_map_test)
    rank_zero_info(f"neg_log_prob(regular_MAP): {neg_log_p_regular.item():.4f}")

    # Also evaluate the Adam-found MAP for comparison
    X_adam_map = torch.tensor(
        [1.0031, 3.0143, 0.7898, 0.7976, -0.0255, 0.0096],
        requires_grad=True,
        dtype=torch.float32,
    )
    neg_log_p_adam = sampler.neg_log_prob(X_adam_map)
    rank_zero_info(f"neg_log_prob(adam_MAP): {neg_log_p_adam.item():.4f}")

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

    # Optional: cast to float32 to shrink size
    for grp in ("posterior", "prior"):
        if hasattr(idata, grp):
            ds = getattr(idata, grp)
            setattr(idata, grp, ds.astype({v: "float32" for v in ds.data_vars}))

    # Save + load
    out_nc = posterior_dir / Path("X_posterior_model.nc")
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
