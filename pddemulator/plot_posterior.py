#!/bin/env python3

import os
from argparse import ArgumentParser
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
import pylab as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import NullFormatter
from scipy.stats import beta, gaussian_kde

import seaborn as sns
from pismemulator.utils import param_keys_dict as keys_dict

fontsize = 8
lw = 1.0
aspect_ratio = 1
markersize = 1

params = {
    "backend": "ps",
    "axes.linewidth": 0.25,
    "lines.linewidth": lw,
    "axes.labelsize": fontsize,
    "font.size": fontsize,
    "xtick.direction": "in",
    "xtick.labelsize": fontsize,
    "xtick.major.size": 2.5,
    "xtick.major.width": 0.25,
    "ytick.direction": "in",
    "ytick.labelsize": fontsize,
    "ytick.major.size": 2.5,
    "ytick.major.width": 0.25,
    "legend.fontsize": fontsize,
    "lines.markersize": markersize,
    "font.size": fontsize,
}

plt.rcParams.update(params)

if __name__ == "__main__":
    __spec__ = None

    parser = ArgumentParser()
    parser.add_argument("--emulator_dir", default="emulator_ensemble")
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--validate", default=False, action="store_true")
    parser.add_argument("--out_format", choices=["csv", "parquet"], default="parquet")

    args = parser.parse_args()

    emulator_dir = args.emulator_dir
    frac = args.fraction
    out_format = args.out_format
    validate = args.validate

    X_list = []
    if validate:
        p = Path(f"{emulator_dir}/posterior_samples_validate/")
    else:
        p = Path(f"{emulator_dir}/posterior_samples/")
    print("Loading posterior samples\n")
    infiles = f"X_posterior_model_*.{out_format}"
    for m, m_file in enumerate(sorted(p.glob(infiles))):
        print(f"  -- {m_file}")
        if out_format == "csv":
            df = pd.read_csv(m_file)
        elif out_format == "parquet":
            df = pd.read_parquet(m_file)
        else:
            raise NotImplementedError(f"{out_format} not implemented")

        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
        X_list.append(df)

    print(f"Merging posteriors into dataframe")
    posterior_df = pd.concat(X_list).reset_index(drop=True)

    X_prior = {
        "f_snow": [1, 6],  # uniform between 1 and 6
        "f_ice": [3, 15],  # uniform between 3 and 15
        "refreeze_snow": [0, 1],  # uniform between 0 and 1
        "refreeze_ice": [0, 1],  # uniform between 0 and 1
        "temp_snow": [-2, 0],  # uniform between 0 and 1
        "temp_rain": [0, 4],  # uniform between 0 and 1
    }
    n_params = len(X_prior)
    g = sns.PairGrid(
        posterior_df.sample(frac=frac),
        diag_sharey=False,
        hue="Committee Member",
        palette="icefire",
        height=1.0,
    )
    alpha_b = 3.0
    beta_b = 3.0
    X_min = np.array([x[0] for x in X_prior.values()])
    X_max = np.array([x[1] for x in X_prior.values()])
    rv = beta(alpha_b, beta_b)
    x = np.linspace(0, 1, 101)
    prior = rv.pdf(x)

    g.map_upper(sns.scatterplot, s=2)
    g.map_lower(sns.kdeplot, levels=4)
    g.map_diag(sns.kdeplot, lw=1)
    [
        g.axes[k, k].plot(
            x * (X_max[k] - X_min[k]) + X_min[k],
            (prior * (X_max[k] - X_min[k]) + X_min[k]) / 2,
            lw=2,
            color="r",
        )
        for k, _ in enumerate(X_prior.values())
    ]
    [
        ax[k].set_xlim(X_min[k % n_params], X_max[k % n_params])
        for k, ax in enumerate(g.axes)
    ]
    [
        ax[k].set_ylim(X_min[k % n_params], X_max[k % n_params])
        for k, ax in enumerate(g.axes)
    ]
    if validate:
        g.fig.savefig(join(emulator_dir, "posterior_validation.pdf"))
    else:
        g.fig.savefig(join(emulator_dir, "posterior.pdf"))
