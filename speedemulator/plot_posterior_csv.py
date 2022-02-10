#!/bin/env python3

from argparse import ArgumentParser

import numpy as np
import os
from os.path import join

from pathlib import Path

import pandas as pd
import pylab as plt

from matplotlib.ticker import NullFormatter
from matplotlib.patches import Polygon
import seaborn as sns

from scipy.stats.distributions import truncnorm, gamma, uniform, randint

from pismemulator.utils import param_keys_dict as keys_dict

fontsize = 6
lw = 0.65
aspect_ratio = 1
markersize = 2
fig_width = 6.2  # inch
fig_height = aspect_ratio * fig_width  # inch
fig_size = [fig_width, fig_height]

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
    "figure.figsize": fig_size,
}

gris_calib_distributions = {
    "SIAE": uniform(
        loc=1.0, scale=3.0
    ),  # uniform between 1 and 4    AS16 best value: 1.25
    "SSAN": uniform(
        loc=3.0, scale=0.5
    ),  # uniform between 3 and 3.5  AS16 best value: 3.25
    "PPQ": uniform(loc=0.25, scale=0.7),  # uniform between 0.25 and 0.95
    "TEFO": uniform(loc=0.015, scale=0.035),  # uniform between 0.015 and 0.040
    "PHIMIN": uniform(loc=10.0, scale=20.0),  # uniform between  15 and 30
    "ZMIN": uniform(loc=-1000, scale=1000),  # uniform between -1000 and 0
    "ZMAX": uniform(loc=0, scale=1000),  # uniform between 0 and 1000
}

ant_calib_distributions = {
    "sia_e": uniform(loc=1.0, scale=3.0),  # uniform between 1 and 4
    "ssa_e": uniform(loc=0.5, scale=1.5),  # uniform between 1 and 2
    "ppq": uniform(loc=0.25, scale=0.7),  # uniform between 0.25 and 0.95
    "tefo": uniform(loc=0.015, scale=0.045),  # uniform between 0.015 and 0.050
    "phi_min": uniform(loc=10.0, scale=20.0),  # uniform between  10 and 30
    "z_min": uniform(loc=-1000, scale=1000),  # uniform between -1000 and 0
    "z_max": uniform(loc=0, scale=1000),  # uniform between 0 and 1000
    "pseudo_plastic_uthreshold": uniform(loc=0, scale=200),
}

prior_distributions = {"gris": gris_calib_distributions, "ant": ant_calib_distributions}

plt.rcParams.update(params)

if __name__ == "__main__":
    __spec__ = None

    parser = ArgumentParser()
    parser.add_argument("--emulator_dir", default="emulator_ensemble")
    parser.add_argument("--prior_distribution", choices=["ant", "gris"], default="gris")
    parser.add_argument(
        "--samples_file", default="../data/samples/velocity_calibration_samples_50.csv"
    )
    parser.add_argument("--fraction", type=float, default=1.0)

    args = parser.parse_args()

    emulator_dir = args.emulator_dir
    frac = args.fraction
    prior_dists = prior_distributions[args.prior_distribution]
    samples_file = args.samples_file

    samples = pd.read_csv(samples_file).drop(columns=["id"])

    X_prior = samples.values
    X_keys = samples.keys()

    n_parameters = X_prior.shape[1]

    X_min = X_prior.min(axis=0) - 1e-3
    X_max = X_prior.max(axis=0) + 1e-3

    color_post_0 = "#00B25F"
    color_post_1 = "#132DD6"
    color_prior = "#2171b5"
    color_posterior = "k"
    color_ensemble = "#BA9B00"
    color_other = "#20484E0"

    X_list = []
    p = Path(f"{emulator_dir}/posterior_samples/")
    for m, m_file in enumerate(sorted(p.glob("X_posterior_model_*.csv.gz"))):
        print(f"Loading {m_file}")
        df = pd.read_csv(m_file)
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
        model = m_file.name.split("_")[-1].split(".")[0]
        df["Model"] = int(model)
        X_list.append(df)

    print(f"Merging posteriors into dataframe")
    posterior_df = pd.concat(X_list)

    X_posterior = posterior_df.sample(frac=frac).drop(columns=["Model"]).values
    C_0 = np.corrcoef((X_posterior - X_posterior.mean(axis=0)).T)
    Cn_0 = (np.sign(C_0) * C_0 ** 2 + 1) / 2.0

    fig, axs = plt.subplots(ncols=int(n_parameters / 2), nrows=2, figsize=(6.2, 2.5))
    for i, ax in enumerate(fig.axes):
        m_key = df.drop(columns=["Model"]).keys()[i]
        min_val = min(X_prior[:, i].min(), X_posterior[:, i].min())
        max_val = max(X_prior[:, i].max(), X_posterior[:, i].max())
        bins = np.linspace(min_val, max_val, 30)
        x = np.linspace(X_min, X_max, 1000)
        X_prior_hist, b = np.histogram(X_prior[:, i], bins, density=True)
        b = 0.5 * (b[1:] + b[:-1])
        d = prior_dists[m_key]
        ax.plot(x, d.pdf(x), color=color_prior, lw=2)

        X_posterior_hist = np.histogram(X_posterior[:, i], bins, density=True)[0]
        ax.plot(b, X_prior_hist, color=color_prior, linewidth=0.8, label="Prior")

        if i == 0:
            legend = ax.legend(loc="upper left")
            legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)
        ax.set_xlabel(keys_dict[m_key])
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    fig.tight_layout()
    figfile = f"{emulator_dir}/prior.pdf"
    print(f"Saving figure to {figfile}")
    fig.savefig(f"{emulator_dir}/prior.pdf")

    fig, axs = plt.subplots(ncols=int(n_parameters / 2), nrows=2, figsize=(6.2, 2.5))
    for i, ax in enumerate(fig.axes):
        min_val = min(X_prior[:, i].min(), X_posterior[:, i].min())
        max_val = max(X_prior[:, i].max(), X_posterior[:, i].max())
        bins = np.linspace(min_val, max_val, 30)
        X_prior_hist, b = np.histogram(X_prior[:, i], bins, density=True)
        b = 0.5 * (b[1:] + b[:-1])
        X_posterior_hist = np.histogram(X_posterior[:, i], bins, density=True)[0]
        ax.plot(
            b,
            X_prior_hist,
            color=color_prior,
            linewidth=0.8,
            label="Prior",
            linestyle="dashed",
        )

        ax.plot(
            b,
            X_posterior_hist,
            color=color_posterior,
            linewidth=0.8,
            linestyle="solid",
            label="Posterior",
        )
        ax.histplot(
            X_posterior[:, i],
            bins,
            density=True,
            color=color_posterior,
            linewidth=0.8,
            linestyle="solid",
            label="Posterior",
        )
        if i == 0:
            legend = ax.legend(loc="upper left")
            legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)
        m_key = df.drop(columns=["Model"]).keys()[i]
        ax.set_xlabel(keys_dict[m_key])
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    fig.tight_layout()
    figfile = f"{emulator_dir}/prior_posterior.pdf"
    print(f"Saving figure to {figfile}")
    fig.savefig(f"{emulator_dir}/prior.pdf")

    fig, axs = plt.subplots(nrows=n_parameters, ncols=n_parameters, figsize=(6.2, 6.2))
    for i in range(n_parameters):
        for j in range(n_parameters):
            if i > j:

                axs[i, j].scatter(
                    X_posterior[:, j],
                    X_posterior[:, i],
                    c="#31a354",
                    s=0.25,
                    alpha=0.01,
                    label="Posterior",
                    rasterized=True,
                )
                min_val = X_posterior[:, i].min()
                max_val = X_posterior[:, i].max()
                bins_y = np.linspace(min_val, max_val, 30)

                min_val = X_posterior[:, j].min()
                max_val = X_posterior[:, j].max()
                bins_x = np.linspace(min_val, max_val, 30)

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
                b = 0.5 * (b[1:] + b[:-1])
                lw = 1
                all_models = posterior_df["Model"].unique()
                for m_model in all_models:
                    m_df = posterior_df[posterior_df["Model"] == m_model].drop(
                        columns=["Model"]
                    )
                    X_model_posterior = m_df.values
                    X_model_posterior_hist = np.histogram(
                        X_model_posterior[:, i], bins, density=True
                    )[0]
                    axs[i, j].plot(
                        b,
                        X_model_posterior_hist,
                        color="#31a354",
                        linewidth=0.2,
                        linestyle="solid",
                        alpha=0.5,
                    )

                X_posterior_hist = np.histogram(X_posterior[:, i], bins, density=True)[
                    0
                ]
                axs[i, j].plot(
                    b,
                    X_prior_hist,
                    color=color_prior,
                    linewidth=0.5 * lw,
                    label="Prior",
                    linestyle="dashed",
                )

                axs[i, j].plot(
                    b,
                    X_posterior_hist,
                    color="black",
                    linewidth=lw,
                    linestyle="solid",
                    label="Posterior",
                    alpha=0.7,
                )

                # if i == 1:
                #     legend = axs[i, j].legend(fontsize=6, loc="lower left")
                #     legend.get_frame().set_linewidth(0.0)
                #     legend.get_frame().set_alpha(0.0)

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

    # fig.subplots_adjust(hspace=0.025, wspace=0.025)
    # fig.tight_layout()
    figfile = f"{emulator_dir}/speed_emulator_posterior.pdf"
    print(f"Saving figure to {figfile}")
    fig.savefig(f"{emulator_dir}/speed_emulator_posterior.pdf")

    Prior = pd.DataFrame(data=X_prior, columns=X_keys).sample(frac=0.1)
    Prior["Type"] = "Prior"
    Posterior = pd.DataFrame(data=X_posterior, columns=X_keys).sample(frac=0.1)
    Posterior["Type"] = "Posterior"
    PP = pd.concat([Prior, Posterior])
