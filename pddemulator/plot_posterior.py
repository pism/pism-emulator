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

fontsize = 6
lw = 1.0
aspect_ratio = 1
markersize = 2

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
    parser.add_argument("--out_format", choices=["csv", "parquet"], default="parquet")
    parser.add_argument(
        "--samples_file", default="../data/samples/velocity_calibration_samples_100.csv"
    )
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--validate", default=False, action="store_true")

    args = parser.parse_args()

    emulator_dir = args.emulator_dir
    frac = args.fraction
    out_format = args.out_format
    samples_file = args.samples_file
    validate = args.validate

    print("Loading prior samples\n")
    samples = pd.read_csv(samples_file).drop(columns=["id"])

    X = samples.values
    X_mean = samples.mean(axis=0)
    X_std = samples.std(axis=0)
    X_keys = samples.keys()

    n_samples = int(X.shape[0])
    n_parameters = int(X.shape[1])

    X_min = (((X.min(axis=0) - X_mean) / X_std - 1e-3) * X_std + X_mean).values
    X_max = (((X.max(axis=0) - X_mean) / X_std + 1e-3) * X_std + X_mean).values

    alpha_b = 3.0
    beta_b = 3.0
    X_prior = (
        beta.rvs(alpha_b, beta_b, size=(100000, n_parameters)) * (X_max - X_min) + X_min
    )

    color_post_0 = "#00B25F"
    color_post_1 = "#132DD6"
    color_prior = "#2171b5"
    color_posterior = "k"
    color_ensemble = "#BA9B00"
    color_other = "#20484E0"

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

        df = df.sample(frac=frac)
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
        model = m_file.name.split("_")[-1].split(".")[0]
        df["Model"] = int(model)
        X_list.append(df)

    print(f"Merging posteriors into dataframe")
    posterior_df = pd.concat(X_list)

    X_posterior = posterior_df.drop(columns=["Model"]).values

    g = sns.PairGrid(
        posterior_df.reset_index(drop=True).sample(frac=0.1),
        diag_sharey=False,
        hue="Model",
    )
    g.map_upper(sns.scatterplot, s=5)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=2)
    if validate:
        g.fig.savefig(join(emulator_dir, "posterior_validation.pdf"))
    else:
        g.fig.savefig(join(emulator_dir, "posterior.pdf"))
