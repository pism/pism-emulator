"""
Compare speed posteriors of several samplers from their _preds chain files.

usage: compare.py SMOKE_DIR OUT_PNG LABEL=RUN_DIR [LABEL=RUN_DIR ...]
SMOKE_DIR holds a finished run (posterior/*.nc and _preds) used to recover the
parameter names and the normalization x_phys = x_norm * std + mean. The first
LABEL is the reference for the "shift" columns (in its posterior sds).
"""

import sys
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr


def load_preds(run_dir: str) -> dict:
    """
    Load and stack the per-chain files of one run.

    Parameters
    ----------
    run_dir : str
        Run directory containing ``_preds/rank*_chain*.pt``.

    Returns
    -------
    dict
        ``samples`` (C, S, D), ``lp``, ``accept`` and ``step_size`` (C, S) arrays.
    """
    recs = sorted(
        (torch.load(f) for f in Path(run_dir, "_preds").glob("rank*_chain*.pt")),
        key=lambda rec: rec["chain"],
    )
    return {
        key: torch.stack([rec[key] for rec in recs]).numpy()
        for key in ("samples", "lp", "accept", "step_size")
    }


def main(argv: list[str]) -> None:
    """
    Print a comparison table and write an overlay plot of the marginals.

    Parameters
    ----------
    argv : list of str
        ``[SMOKE_DIR, OUT_PNG, LABEL=RUN_DIR, ...]``.
    """
    plt.switch_backend("Agg")
    smoke, png, *runs = argv
    post = xr.open_dataset(
        next(Path(smoke, "posterior").glob("*.nc")), group="posterior"
    )
    names = list(post.data_vars)
    norm = load_preds(smoke)["samples"]  # (C, S, D)
    fits = [
        np.polyfit(norm[..., i].ravel(), post[n].values.ravel(), 1)
        for i, n in enumerate(names)
    ]
    std = np.array([f[0] for f in fits])
    mean = np.array([f[1] for f in fits])

    data = {}
    for spec in runs:
        label, run_dir = spec.split("=", 1)
        rec = load_preds(run_dir)
        rec["phys"] = rec["samples"] * std + mean
        data[label] = rec
    labels = list(data)
    ref = data[labels[0]]["phys"]

    print(
        "chains x draws: "
        + ", ".join(
            f"{lab}: {data[lab]['phys'].shape[0]}x{data[lab]['phys'].shape[1]}"
            for lab in labels
        )
    )
    for lab in labels:
        rec = data[lab]
        print(
            f"{lab:>9s}: acceptance {rec['accept'].mean():.3f}, "
            f"NaN lp {np.isnan(rec['lp']).mean():.3f}, "
            f"final step size {rec['step_size'][:, -1].mean():.4g}"
        )
    print()
    hdr = f"{'parameter':26s}"
    hdr += "".join(f"{lab + ' mean':>13s}{lab + ' sd':>11s}" for lab in labels)
    hdr += "".join(f"{'shift ' + lab:>14s}" for lab in labels[1:])
    hdr += "".join(f"{'Rhat ' + lab:>11s}{'ESS ' + lab:>10s}" for lab in labels)
    print(hdr)
    for i, n in enumerate(names):
        row = f"{n:26s}"
        for lab in labels:
            x = data[lab]["phys"][..., i]
            row += f"{x.mean():13.4g}{x.std():11.3g}"
        xo = ref[..., i]
        for lab in labels[1:]:
            row += f"{(data[lab]['phys'][..., i].mean() - xo.mean()) / xo.std():14.2f}"
        for lab in labels:
            x = data[lab]["phys"][..., i]
            row += f"{float(az.rhat(x)):11.3f}{float(az.ess(x)):10.0f}"
        print(row)

    k = len(names)
    ncol = (k + 1) // 2
    fig, axes = plt.subplots(2, ncol, figsize=(3.1 * ncol, 5.4))
    for ax, (i, n) in zip(axes.ravel(), enumerate(names)):
        allx = np.concatenate([data[lab]["phys"][..., i].ravel() for lab in labels])
        bins = np.linspace(*np.percentile(allx, [0.5, 99.5]), 45)
        for lab in labels:
            ax.hist(
                data[lab]["phys"][..., i].ravel(),
                bins,
                density=True,
                histtype="step",
                lw=1.5,
                label=lab,
            )
        ax.set_title(n, fontsize=9)
        ax.set_yticks([])
    axes.ravel()[0].legend(fontsize=8, frameon=False)
    for ax in axes.ravel()[k:]:
        ax.set_visible(False)
    fig.suptitle(
        "Speed posterior, 1800 m Greenland (legacy DNN emulator, 20 epochs), "
        "4 chains x 3000 draws",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print("\nwrote", png)


if __name__ == "__main__":
    main(sys.argv[1:])
