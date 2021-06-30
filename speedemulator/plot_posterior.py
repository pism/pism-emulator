#!/bin/env python3

from argparse import ArgumentParser

from glob import glob
import numpy as np
import os
from os.path import join
from scipy.special import gamma
from scipy.stats import beta


import pandas as pd
import pylab as plt


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

plt.rcParams.update(params)


if __name__ == "__main__":
    __spec__ = None

    parser = ArgumentParser()
    parser.add_argument("--emulator_dir", default="emulator_ensemble")
    parser.add_argument("--num_models", type=int, default=1)
    parser.add_argument("--samples_file", default="../data/samples/velocity_calibration_samples_100.csv")

    args = parser.parse_args()

    emulator_dir = args.emulator_dir
    num_models = args.num_models
    samples_file = args.samples_file

    samples = pd.read_csv(samples_file).drop(columns=["id"])
    X = samples.values
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_keys = samples.keys()

    n_parameters = X.shape[1]

    from matplotlib.ticker import NullFormatter
    from matplotlib.patches import Polygon

    X_min = X.min(axis=0) - 1e-3
    X_max = X.max(axis=0) + 1e-3
    # Eq 52
    # this is 2.0 in the paper

    alpha_b = 3.0
    beta_b = 3.0
    X_prior = beta.rvs(alpha_b, beta_b, size=(n_posterior_samples, n_parameters)) * (X_max - X_min) + X_min
    X_hat = X_prior * X_std + X_mean
    X_hat = X_prior

    color_post_0 = "#00B25F"
    color_post_1 = "#132DD6"
    color_prior = "#D81727"
    color_ensemble = "#BA9B00"
    color_other = "#20484E0"

    X_list = []

    for model_index in range(num_models):
        m_file = f"{emulator_dir}/posterior_samples/X_posterior_model_{model_index:03}.npy"
        print(f"Loading {m_file}")
        X_p = np.load(open(m_file, "rb"))
        X_list.append(X_p)

    X_posterior = np.vstack(X_list)
    X_posterior = X_posterior * X_std + X_mean

    df = pd.DataFrame(data=X_posterior, columns=X_keys)
    df.to_csv(f"{emulator_dir}/X_posterior.csv.gz")

    C_0 = np.corrcoef((X_posterior - X_posterior.mean(axis=0)).T)
    Cn_0 = (np.sign(C_0) * C_0 ** 2 + 1) / 2.0

    fig, axs = plt.subplots(nrows=n_parameters, ncols=n_parameters, figsize=(6.2, 6.2))
    for i in range(n_parameters):
        for j in range(n_parameters):
            if i > j:

                axs[i, j].scatter(
                    X_posterior[:, j], X_posterior[:, i], c="k", s=0.5, alpha=0.05, label="Posterior", rasterized=True
                )
                min_val = min(X_hat[:, i].min(), X_posterior[:, i].min())
                max_val = max(X_hat[:, i].max(), X_posterior[:, i].max())
                bins_y = np.linspace(min_val, max_val, 30)

                min_val = min(X_hat[:, j].min(), X_posterior[:, j].min())
                max_val = max(X_hat[:, j].max(), X_posterior[:, j].max())
                bins_x = np.linspace(min_val, max_val, 30)

                axs[i, j].set_xlim(X_hat[:, j].min(), X_hat[:, j].max())
                axs[i, j].set_ylim(X_hat[:, i].min(), X_hat[:, i].max())

            elif i < j:
                patch_upper = Polygon(
                    np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]), facecolor=plt.cm.seismic(Cn_0[i, j])
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
                min_val = min(X_hat[:, i].min(), X_posterior[:, i].min())
                max_val = max(X_hat[:, i].max(), X_posterior[:, i].max())
                bins = np.linspace(min_val, max_val, 30)
                X_hat_hist, b = np.histogram(X_hat[:, i], bins, density=True)
                b = 0.5 * (b[1:] + b[:-1])
                lw = 1
                for X_model_posterior in X_list:
                    X_model_posterior = X_model_posterior * X_std[i] + X_mean[i]
                    X_model_posterior_hist = np.histogram(X_model_posterior[:, i], bins, density=True)[0]
                    axs[i, j].plot(b, X_model_posterior_hist, color="0.5", linewidth=0.2, linestyle="solid", alpha=0.5)

                X_posterior_hist = np.histogram(X_posterior[:, i], bins, density=True)[0]
                axs[i, j].plot(b, X_hat_hist, color=color_prior, linewidth=0.5 * lw, label="Prior", linestyle="dashed")

                axs[i, j].plot(
                    b, X_posterior_hist, color="black", linewidth=lw, linestyle="solid", label="Posterior", alpha=0.7
                )

                if i == 1:
                    legend = axs[i, j].legend(fontsize=6, loc="upper left")
                    legend.get_frame().set_linewidth(0.0)
                    legend.get_frame().set_alpha(0.0)

                axs[i, j].set_xlim(min_val, max_val)

            else:
                axs[i, j].remove()

    keys = X_keys

    for i, ax in enumerate(axs[:, 0]):
        ax.set_ylabel(keys[i])

    for j, ax in enumerate(axs[-1, :]):
        ax.set_xlabel(keys[j])
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

    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    fig.savefig(f"{emulator_dir}/speed_emulator_posterior.pdf")

    # Prior = pd.DataFrame(data=X_hat, columns=dataset.X_keys).sample(frac=0.1)
    # Prior["Type"] = "Pior"
    # Posterior = pd.DataFrame(data=X_posterior, columns=dataset.X_keys).sample(frac=0.1)
    # Posterior["Type"] = "Posterior"
    # PP = pd.concat([Prior, Posterior])

    # from scipy.stats import pearsonr

    # def corrfunc(x, y, **kwds):
    #     cmap = kwds["cmap"]
    #     norm = kwds["norm"]
    #     ax = plt.gca()
    #     ax.tick_params(bottom=False, top=False, left=False, right=False)
    #     sns.despine(ax=ax, bottom=True, top=True, left=True, right=True)
    #     r, _ = pearsonr(x, y)
    #     facecolor = cmap(norm(r))
    #     ax.set_facecolor(facecolor)
    #     lightness = (max(facecolor[:3]) + min(facecolor[:3])) / 2
    #     ax.annotate(
    #         f"r={r:.2f}",
    #         xy=(0.5, 0.5),
    #         xycoords=ax.transAxes,
    #         color="white" if lightness < 0.7 else "black",
    #         size=6,
    #         ha="center",
    #         va="center",
    #     )

    # g = sns.PairGrid(PP, hue="Type", diag_sharey=False)
    # g.map_lower(sns.scatterplot, alpha=0.3, edgecolor="none")
    # g.map_upper(corrfunc, cmap=sns.color_palette("coolwarm", as_cmap=True), norm=plt.Normalize(vmin=-1, vmax=1))
    # g.map_diag(sns.kdeplot, lw=1)
    # g.savefig(f"{emulator_dir}/seaborn_test.pdf")
