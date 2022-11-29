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
    parser.add_argument(
        "--samples_file", default="../data/samples/velocity_calibration_samples_100.csv"
    )
    parser.add_argument("--fraction", type=float, default=0.1)

    args = parser.parse_args()

    emulator_dir = args.emulator_dir
    frac = args.fraction
    samples_file = args.samples_file

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
    p = Path(f"{emulator_dir}/posterior_samples/")
    print("Loading posterior samples\n")
    for m, m_file in enumerate(sorted(p.glob("X_posterior_model_*.parquet"))):
        print(f"  -- {m_file}")
        df = pd.read_parquet(m_file).sample(frac=frac)
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
        model = m_file.name.split("_")[-1].split(".")[0]
        df["Model"] = int(model)
        X_list.append(df)

    print(f"Merging posteriors into dataframe")
    posterior_df = pd.concat(X_list)

    X_posterior = posterior_df.drop(columns=["Model"]).values

    print("P16", np.percentile(X_posterior, 16, axis=0))
    print("P50", np.percentile(X_posterior, 50, axis=0))
    print("P84", np.percentile(X_posterior, 84, axis=0))

    C_0 = np.corrcoef((X_posterior - X_posterior.mean(axis=0)).T)
    Cn_0 = (np.sign(C_0) * C_0**2 + 1) / 2.0

    fig, axs = plt.subplots(nrows=n_parameters, ncols=n_parameters, figsize=(5.4, 5.6))
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    for i in range(n_parameters):
        for j in range(n_parameters):
            if i > j:

                axs[i, j].scatter(
                    X_posterior[:, j],
                    X_posterior[:, i],
                    c="#31a354",
                    s=0.05,
                    alpha=0.01,
                    label="Posterior",
                    rasterized=True,
                )

                min_val = min(X_prior[:, i].min(), X_posterior[:, i].min())
                max_val = max(X_prior[:, i].max(), X_posterior[:, i].max())
                bins_y = np.linspace(min_val, max_val, 30)

                min_val = min(X_prior[:, j].min(), X_posterior[:, j].min())
                max_val = max(X_prior[:, j].max(), X_posterior[:, j].max())
                bins_x = np.linspace(min_val, max_val, 30)

                v = gaussian_kde(X_posterior[:, [j, i]].T)
                bx = 0.5 * (bins_x[1:] + bins_x[:-1])
                by = 0.5 * (bins_y[1:] + bins_y[:-1])
                Bx, By = np.meshgrid(bx, by)

                axs[i, j].contour(
                    Bx,
                    By,
                    v(np.vstack((Bx.ravel(), By.ravel()))).reshape(Bx.shape),
                    7,
                    linewidths=0.5,
                    colors="black",
                )

                axs[i, j].set_xlim(X_prior[:, j].min(), X_prior[:, j].max())
                axs[i, j].set_ylim(X_prior[:, i].min(), X_prior[:, i].max())

            elif i < j:
                patch_upper = Polygon(
                    np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]),
                    facecolor=plt.cm.seismic(Cn_0[i, j]),
                )
                axs[i, j].add_patch(patch_upper)
                if C_0[i, j] > -0.5:
                    color = "black"
                else:
                    color = "white"
                axs[i, j].text(
                    0.5,
                    0.5,
                    "{0:.2f}".format(C_0[i, j]),
                    fontsize=6,
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axs[i, j].transAxes,
                    color=color,
                )

            elif i == j:
                min_val = min(X_prior[:, i].min(), X_posterior[:, i].min())
                max_val = max(X_prior[:, i].max(), X_posterior[:, i].max())
                bins = np.linspace(min_val, max_val, 30)
                X_prior_hist, b = np.histogram(X_prior[:, i], bins, density=True)
                X_posterior_hist, _ = np.histogram(
                    X_posterior[:, i], bins, density=True
                )
                b = 0.5 * (b[1:] + b[:-1])

                axs[i, j].plot(
                    b,
                    X_prior_hist,
                    color=color_prior,
                    linewidth=lw,
                    label="Prior",
                    linestyle="solid",
                )

                all_models = posterior_df["Model"].unique()
                for k, m_model in enumerate(all_models):
                    m_df = posterior_df[posterior_df["Model"] == m_model].drop(
                        columns=["Model"]
                    )
                    X_model_posterior = m_df.values
                    X_model_posterior_hist, _ = np.histogram(
                        X_model_posterior[:, i], _, density=True
                    )
                    if k == 0:
                        axs[i, j].plot(
                            b,
                            X_model_posterior_hist * 0.5,
                            color="0.5",
                            linewidth=lw * 0.25,
                            linestyle="solid",
                            alpha=0.5,
                            label="Posterior (BayesBag)",
                        )
                    else:
                        axs[i, j].plot(
                            b,
                            X_model_posterior_hist * 0.5,
                            color="0.5",
                            linewidth=lw * 0.25,
                            linestyle="solid",
                            alpha=0.5,
                        )

                axs[i, j].plot(
                    b,
                    X_posterior_hist,
                    color="black",
                    linewidth=lw,
                    linestyle="solid",
                    label="Posterior",
                )
                p16 = np.percentile(X_posterior[:, i], 16)
                p50 = np.percentile(X_posterior[:, i], 50)
                p84 = np.percentile(X_posterior[:, i], 84)
                print(X_keys[i], p16, p50, p84)

                axs[i, j].axvline(p16, color="black", lw=0.5, ls="dotted")
                axs[i, j].axvline(p84, color="black", lw=0.5, ls="dotted")
                axs[i, j].axvline(p50, color="black", lw=1.0, ls="dotted")
                axs[i, j].set_xlim(min_val, max_val)

            else:
                axs[i, j].remove()

    for i, ax in enumerate(axs[:, 0]):
        ax.set_ylabel(keys_dict[X_keys[i]])

    for j, ax in enumerate(axs[-1, :]):
        ax.set_xlabel(keys_dict[X_keys[j]])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        plt.setp(ax.xaxis.get_minorticklabels(), rotation=45)
        if j > 0:
            ax.tick_params(axis="y", which="both", length=0)
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())

    for ax in axs[:-1, 0].ravel():
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="both", length=0)

    for ax in axs[:-1, 1:].ravel():
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="both", which="both", length=0)

    l_prior = Line2D([], [], c=color_prior, lw=lw, ls="solid", label="Prior")
    l_post = Line2D([], [], c="k", lw=lw, ls="solid", label="Posterior")
    l_post_b = Line2D(
        [], [], c="0.25", lw=lw * 0.25, ls="solid", label="Posterior (BayesBag)"
    )

    legend = fig.legend(
        handles=[l_prior, l_post, l_post_b], bbox_to_anchor=(0.3, 0.955)
    )
    legend.get_frame().set_linewidth(0.0)
    legend.get_frame().set_alpha(0.0)

    figfile = f"{emulator_dir}/speed_emulator_posterior.pdf"
    print(f"Saving figure to {figfile}")
    fig.savefig(figfile)
