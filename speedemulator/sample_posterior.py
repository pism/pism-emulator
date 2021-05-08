#!/bin/env python3

from argparse import ArgumentParser

import numpy as np
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from pismemulator.nnemulator import NNEmulator, PISMDataset, PISMDataModule


class MALASampler(object):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def find_MAP(self, X, Y_target, n_iters=50, print_interval=10):
        print("***********************************************")
        print("***********************************************")
        print("Finding MAP point")
        print("***********************************************")
        print("***********************************************")
        # Line search distances
        alphas = np.logspace(-4, 0, 11)
        # Find MAP point
        for i in range(n_iters):
            log_pi, g, H, Hinv, log_det_Hinv = self.get_log_like_gradient_and_hessian(
                X, Y_target, compute_hessian=True
            )
            p = Hinv @ -g
            alpha_index = np.nanargmin(
                [
                    self.get_log_like_gradient_and_hessian(X + alpha * p, Y_target, compute_hessian=False)
                    .detach()
                    .cpu()
                    .numpy()
                    for alpha in alphas
                ]
            )
            mu = X + alphas[alpha_index] * p
            X.data = mu.data
            if i % print_interval == 0:
                print("===============================================")
                print(
                    "iter: {0:d}, ln(P): {1:6.1f}, curr. m: {2:4.4f},{3:4.2f},{4:4.2f},{5:4.2f},{6:4.2f},{7:4.2f},{8:4.2f},{9:4.2f}".format(
                        i, log_pi, *X.data.cpu().numpy()
                    )
                )
                print("===============================================")
        return X

    def V(self, X, Y_target):
        # model result is in log space
        Y_pred = 10 ** self.model(X, add_mean=True)
        r = Y_pred - Y_target
        X_bar = (X - X_min) / (X_max - X_min)
        L1 = torch.sum(
            np.log(gamma((nu + 1) / 2.0))
            - np.log(gamma(nu / 2.0))
            - np.log(np.sqrt(np.pi * nu) * sigma_hat)
            - (nu + 1) / 2.0 * torch.log(1 + 1.0 / nu * (r / sigma_hat) ** 2)
        )
        L2 = torch.sum((alpha_b - 1) * torch.log(X_bar) + (beta_b - 1) * torch.log(1 - X_bar))

        return -(alpha * L1 + L2)

    def get_log_like_gradient_and_hessian(self, X, Y_target, eps=1e-2, compute_hessian=False):

        log_pi = self.V(X, Y_target)
        if compute_hessian:
            g = torch.autograd.grad(log_pi, X, retain_graph=True, create_graph=True)[0]
            H = torch.stack([torch.autograd.grad(e, X, retain_graph=True)[0] for e in g])
            lamda, Q = torch.eig(H, eigenvectors=True)
            lamda_prime = torch.sqrt(lamda[:, 0] ** 2 + eps)
            lamda_prime_inv = 1.0 / torch.sqrt(lamda[:, 0] ** 2 + eps)
            H = Q @ torch.diag(lamda_prime) @ Q.T
            Hinv = Q @ torch.diag(lamda_prime_inv) @ Q.T
            log_det_Hinv = torch.sum(torch.log(lamda_prime_inv))
            return log_pi, g, H, Hinv, log_det_Hinv
        else:
            return log_pi

    def draw_sample(self, mu, cov, eps=1e-10):
        L = torch.cholesky(cov + eps * torch.eye(cov.shape[0], device=device))
        return mu + L @ torch.randn(L.shape[0], device=device)

    def get_proposal_likelihood(self, Y, mu, inverse_cov, log_det_cov):
        return -0.5 * log_det_cov - 0.5 * (Y - mu) @ inverse_cov @ (Y - mu)

    def MALA_step(self, X, Y_target, h, local_data=None):
        if local_data is not None:
            pass
        else:
            local_data = self.get_log_like_gradient_and_hessian(X, Y_target, compute_hessian=True)

        log_pi, g, H, Hinv, log_det_Hinv = local_data

        X_ = self.draw_sample(X, 2 * h * Hinv).detach()
        X_.requires_grad = True

        log_pi_ = self.get_log_like_gradient_and_hessian(X_, Y_target, compute_hessian=False)

        logq = self.get_proposal_likelihood(X_, X, H / (2 * h), log_det_Hinv)
        logq_ = self.get_proposal_likelihood(X, X_, H / (2 * h), log_det_Hinv)

        log_alpha = -log_pi_ + logq_ + log_pi - logq
        alpha = torch.exp(min(log_alpha, torch.tensor([0.0], device=device)))
        u = torch.rand(1, device=device)
        if u <= alpha and log_alpha != np.inf:
            X.data = X_.data
            local_data = self.get_log_like_gradient_and_hessian(X, Y_target, compute_hessian=True)
            s = 1
        else:
            s = 0
        return X, local_data, s

    def MALA(
        self,
        X,
        Y_target,
        n_iters=10001,
        h=0.1,
        h_max=1.0,
        acc_target=0.25,
        k=0.01,
        beta=0.99,
        posterior_dir="./posterior_samples/",
        model_index=0,
        save_interval=1000,
        print_interval=50,
    ):
        print("***********************************************")
        print("***********************************************")
        print("Running Metropolis-Adjusted Langevin Algorithm for model index {0}".format(model_index))
        print("***********************************************")
        print("***********************************************")

        if not os.path.isdir(posterior_dir):
            os.makedirs(posterior_dir)

        local_data = None
        vars = []
        acc = acc_target
        print(n_iters)
        for i in range(n_iters):
            X, local_data, s = self.MALA_step(X, Y_target, h, local_data=local_data)
            vars.append(X.detach())
            acc = beta * acc + (1 - beta) * s
            h = min(h * (1 + k * np.sign(acc - acc_target)), h_max)
            if i % print_interval == 0:
                print("===============================================")
                print("sample: {0:d}, acc. rate: {1:4.2f}, log(P): {2:6.1f}".format(i, acc, local_data[0].item()))
                print(
                    "curr. m: {0:4.4f},{1:4.2f},{2:4.2f},{3:4.2f},{4:4.2f},{5:4.2f},{6:4.2f},{7:4.2f}".format(
                        *X.data.cpu().numpy()
                    )
                )
                print("===============================================")

            if i % save_interval == 0:
                print("///////////////////////////////////////////////")
                print("Saving samples for model {0:03d}".format(model_index))
                print("///////////////////////////////////////////////")
                X_posterior = torch.stack(vars).cpu().numpy()
                np.save(open(posterior_dir + "X_posterior_model_{0:03d}.npy".format(model_index), "wb"), X_posterior)
        X_posterior = torch.stack(vars).cpu().numpy()
        return X_posterior


if __name__ == "__main__":
    __spec__ = None

    parser = ArgumentParser()
    parser.add_argument("--emulator_dir", default="emulator_ensemble")
    parser.add_argument("--num_models", type=int, default=1)
    parser.add_argument("--thinning_factor", type=int, default=1)

    parser = NNEmulator.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    batch_size = args.batch_size
    emulator_dir = args.emulator_dir
    num_models = args.num_models
    thinning_factor = args.thinning_factor
    max_epochs = args.max_epochs

    device = "cpu"

    dataset = PISMDataset(
        data_dir="../data/speeds_v2/",
        samples_file="../data/samples/velocity_calibration_samples_100.csv",
        target_file="../data/validation/greenland_vel_mosaic250_v1_g1800m.nc",
        thinning_factor=thinning_factor,
    )

    X = dataset.X
    F = dataset.Y
    n_grid_points = dataset.n_grid_points
    n_parameters = dataset.n_parameters
    n_samples = dataset.n_samples
    normed_area = dataset.normed_area

    torch.manual_seed(0)
    pl.seed_everything(0)
    np.random.seed(0)

    models = []

    for model_index in range(num_models):
        state_dict = torch.load(f"{emulator_dir}/emulator_pl_lr_{model_index}.h5")
        e = NNEmulator(
            state_dict["l_1.weight"].shape[1],
            state_dict["V_hat"].shape[1],
            normed_area,
            state_dict["V_hat"],
            state_dict["F_mean"],
            args,
        )
        e.load_state_dict(state_dict)
        e.eval()
        models.append(e)

    import pylab as plt

    alpha = 0.01
    from scipy.special import gamma

    nu = 1.0

    sigma2 = 10 ** 2

    rho = 1.0 / (1e4 ** 2)
    point_area = (dataset.grid_resolution * thinning_factor) ** 2
    K = point_area * rho

    Tau = K * 1.0 / sigma2 * K

    sigma_hat = np.sqrt(sigma2 / K ** 2)

    from scipy.stats import beta

    alpha_b = 3.0
    beta_b = 3.0

    X_min = X.cpu().numpy().min(axis=0) - 1e-3
    X_max = X.cpu().numpy().max(axis=0) + 1e-3

    X_prior = beta.rvs(alpha_b, beta_b, size=(10000, X.shape[1])) * (X_max - X_min) + X_min

    # This is required for
    # X_bar = (X - X_min) / (X_max - X_min)
    # to work

    X_min = torch.tensor(X_min, dtype=torch.float32, device=device)
    X_max = torch.tensor(X_max, dtype=torch.float32, device=device)

    torch.manual_seed(0)
    np.random.seed(0)

    # Needs
    # alpha_b, beta_b: float
    # alpha: float
    # nu: float
    # gamma
    # sigma_hat
    U_target = dataset.Y_target

    X_posteriors = []
    for j, model in enumerate(models):
        X_0 = torch.tensor(X_prior.mean(axis=0), requires_grad=True, dtype=torch.float, device=device)
        mala = MALASampler(model)
        X_map = mala.find_MAP(X_0, U_target)
        # To reproduce the paper, n_iters should be 10^5
        X_posterior = mala.MALA(X_map, U_target, n_iters=10000, model_index=j, save_interval=1000, print_interval=100)
        X_posteriors.append(X_posterior)

    import pylab as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    # X_posterior = X_posterior*X_s.cpu().numpy() + X_m.cpu().numpy()
    X_hat = X_prior * dataset.X_std.cpu().numpy() + dataset.X_mean.cpu().numpy()

    color_post_0 = "#00B25F"
    color_post_1 = "#132DD6"
    color_prior = "#D81727"
    color_ensemble = "#BA9B00"
    color_other = "#20484E0"

    fig, axs = plt.subplots(nrows=8, ncols=8, figsize=(12, 12))
    X_list = []

    for model_index in range(num_models):
        X_list.append(np.load(open("./posterior_samples/X_posterior_model_{0:03d}.npy".format(model_index), "rb")))

        X_posterior = np.vstack(X_list)
        X_posterior = X_posterior * dataset.X_std.cpu().numpy() + dataset.X_mean.cpu().numpy()

        C_0 = np.corrcoef((X_posterior - X_posterior.mean(axis=0)).T)
        Cn_0 = (np.sign(C_0) * C_0 ** 2 + 1) / 2.0

    for i in range(8):
        for j in range(8):
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

                # v = st.gaussian_kde(X_posterior[:,[j,i]].T)
                # bx = 0.5*(bins_x[1:] + bins_x[:-1])
                # by = 0.5*(bins_y[1:] + bins_y[:-1])
                # Bx,By = np.meshgrid(bx,by)

                # axs[i,j].contour(10**Bx,10**By,v(np.vstack((Bx.ravel(),By.ravel()))).reshape(Bx.shape),7,alpha=0.7,colors='black')

                axs[i, j].set_xlim(X_hat[:, j].min(), X_hat[:, j].max())
                axs[i, j].set_ylim(X_hat[:, i].min(), X_hat[:, i].max())

                # axs[i,j].set_xscale('log')
                # axs[i,j].set_yscale('log')

            elif i < j:
                patch_upper = Polygon(
                    np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]), facecolor=plt.cm.seismic(Cn_0[i, j])
                )
                # patch_lower = Polygon(np.array([[0.,0.],[1.,0.],[1.,1.]]),facecolor=plt.cm.seismic(Cn_1[i,j]))
                axs[i, j].add_patch(patch_upper)
                # axs[i,j].add_patch(patch_lower)
                if C_0[i, j] > -0.5:
                    color = "black"
                else:
                    color = "white"
                axs[i, j].text(
                    0.5,
                    0.5,
                    "{0:.2f}".format(C_0[i, j]),
                    fontsize=12,
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axs[i, j].transAxes,
                    color=color,
                )
                # if C_1[i,j]>-0.5:
                #    color = 'black'
                # else:
                #    color = 'white'

                # axs[i,j].text(0.75,0.25,'{0:.2f}'.format(C_1[i,j]),fontsize=12,horizontalalignment='center',verticalalignment='center',transform=axs[i,j].transAxes,color=color)

            elif i == j:
                min_val = min(X_hat[:, i].min(), X_posterior[:, i].min())
                max_val = max(X_hat[:, i].max(), X_posterior[:, i].max())
                bins = np.linspace(min_val, max_val, 30)
                X_hat_hist, b = np.histogram(X_hat[:, i], bins, density=True)
                # X_prior_hist, b = np.histogram(X_prior[:, i], bins, density=True)
                X_posterior_hist = np.histogram(X_posterior[:, i], bins, density=True)[0]
                b = 0.5 * (b[1:] + b[:-1])
                lw = 3.0
                axs[i, j].plot(b, X_hat_hist, color=color_prior, linewidth=0.5 * lw, label="Prior", linestyle="dashed")

                axs[i, j].plot(
                    b, X_posterior_hist, color="black", linewidth=lw, linestyle="solid", label="Posterior", alpha=0.7
                )

                # for X_ind in X_stack:
                #    X_hist,_ = np.histogram(X_ind[:,i],bins,density=False)
                #    X_hist=X_hist/len(X_posterior)
                #    X_hist=X_hist/(bins[1]-bins[0])
                #    axs[i,j].plot(10**b,X_hist,'b-',alpha=0.2,lw=0.5)

                if i == 1:
                    axs[i, j].legend(fontsize=8)
                axs[i, j].set_xlim(min_val, max_val)
                # axs[i,j].set_xscale('log')

            else:
                axs[i, j].remove()

    keys = dataset.X_keys

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

    for ax in axs[:-1, 1:].ravel():
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="both", which="both", length=0)

    fig.savefig(f"{emulator_dir}/speed_emulator_posterior.pdf")

    Prior = pd.DataFrame(data=X_hat, columns=dataset.X_keys).sample(frac=0.1)
    Prior["Type"] = "Pior"
    Posterior = pd.DataFrame(data=X_posterior, columns=dataset.X_keys).sample(frac=0.1)
    Posterior["Type"] = "Posterior"
    PP = pd.concat([Prior, Posterior])

    from scipy.stats import pearsonr

    def corrfunc(x, y, **kwds):
        cmap = kwds["cmap"]
        norm = kwds["norm"]
        ax = plt.gca()
        ax.tick_params(bottom=False, top=False, left=False, right=False)
        sns.despine(ax=ax, bottom=True, top=True, left=True, right=True)
        r, _ = pearsonr(x, y)
        facecolor = cmap(norm(r))
        ax.set_facecolor(facecolor)
        lightness = (max(facecolor[:3]) + min(facecolor[:3])) / 2
        ax.annotate(
            f"r={r:.2f}",
            xy=(0.5, 0.5),
            xycoords=ax.transAxes,
            color="white" if lightness < 0.7 else "black",
            size=26,
            ha="center",
            va="center",
        )

    g = sns.PairGrid(PP, hue="Type", diag_sharey=False)
    g.map_lower(sns.scatterplot, cmap=sns.color_palette("rocket_r", as_cmap=True))
    g.map_upper(corrfunc, cmap=sns.color_palette("coolwarm", as_cmap=True), norm=plt.Normalize(vmin=-1, vmax=1))
    g.map_diag(sns.kdeplot, lw=3, cmap=sns.color_palette("rocket_r", as_cmap=True))
    g.savefig(f"{emulator_dir}/seaborn_test.pdf")
