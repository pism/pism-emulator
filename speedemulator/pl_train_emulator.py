#!/bin/env python3

from argparse import ArgumentParser

import numpy as np
import os
from scipy.stats import dirichlet
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from pismemulator.nnemulator import NNEmulator, PISMDataset, PISMDataModule


if __name__ == "__main__":
    __spec__ = None

    parser = ArgumentParser()
    parser.add_argument("--emulator_dir", default="emulator_ensemble")
    parser.add_argument("--num_models", type=int, default=1)
    parser.add_argument("--thinning_factor", type=int, default=1)

    parser = NNEmulator.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    # hparams = vars(args)

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

    if not os.path.isdir(emulator_dir):
        os.makedirs(emulator_dir)

    import pylab as plt

    for model_index in range(num_models):
        print(f"Training model {model_index} of {num_models}")
        omegas = torch.tensor(dirichlet.rvs(np.ones(n_samples)), dtype=torch.float).T
        omegas_0 = torch.ones_like(omegas) / len(omegas)

        data_loader = PISMDataModule(
            X,
            F,
            omegas,
            omegas_0,
        )
        data_loader.prepare_data()
        data_loader.setup(stage="fit")
        n_eigenglaciers = data_loader.n_eigenglaciers
        V_hat = data_loader.V_hat
        F_mean = data_loader.F_mean
        F_train = data_loader.F_bar

        # n_hidden_1 = 128
        # n_hidden_2 = 128
        # n_hidden_3 = 128
        # n_hidden_4 = 128

        # V_hat_doug, F_bar_doug, F_mean_doug = get_eigenglaciers(omegas, F)
        # # Doug's emulator
        # e_d = Emulator(
        #     n_parameters, n_eigenglaciers, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, V_hat_doug, F_mean_doug
        # )

        # train_surrogate(e_d, X, F_bar_doug, omegas, normed_area, epochs=max_epochs)
        # torch.save(e_d.state_dict(), "emulator_ensemble_doug_pl/emulator_{0:03d}.h5".format(model_index))

        # # 3000 epochs
        checkpoint_callback = ModelCheckpoint(dirpath=emulator_dir, filename="emulator_{epoch}_{model_index}")
        logger = TensorBoardLogger("tb_logs", name="PISM Speed Emulator")
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        e = NNEmulator(n_parameters, n_eigenglaciers, normed_area, V_hat, F_mean, args)
        trainer = pl.Trainer.from_argparse_args(
            args, callbacks=[lr_monitor, checkpoint_callback], logger=logger, deterministic=True
        )
        trainer.fit(e, data_loader.train_loader, data_loader.validation_loader)
        trainer.save_checkpoint(f"{emulator_dir}/emulator_{model_index:03d}.ckpt")
        torch.save(e.state_dict(), f"{emulator_dir}/emulator_{model_index:03d}.h5")

        e.eval()
        fig, axs = plt.subplots(nrows=5, ncols=2, sharex="col", sharey="row", figsize=(16, 40))
        for k in range(5):
            idx = np.random.randint(len(data_loader.all_data))
            X_val, F_val, _, _ = data_loader.all_data[idx]
            X_val_scaled = X_val * dataset.X_std + dataset.X_mean
            F_val = (F_val + F_mean).detach().numpy().reshape(dataset.ny, dataset.nx)
            F_pred = e(X_val, add_mean=True).detach().numpy().reshape(dataset.ny, dataset.nx)
            corr = np.corrcoef(F_val.flatten(), F_pred.flatten())[0, 1]
            axs[k, 0].imshow(F_val, origin="lower", vmin=0, vmax=3, cmap="viridis")
            axs[k, 1].imshow(F_pred, origin="lower", vmin=0, vmax=3, cmap="viridis")
            axs[k, 1].text(100, 25, f"Pearson r={corr:.3f}", c="white")
            axs[k, 0].text(
                100,
                25,
                "SIAE: {0:.2f}\nSSAN: {1:.2f}\nPPQ : {2:.2f}\nTEFO: {3:.2f}\nPHIM: {4:.2f}\nPHIX: {5:.2f}\nZMIN: {6:.2f}\nZMAX: {7:.2f}".format(
                    *X_val_scaled
                ),
                c="white",
            )
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.savefig(f"{emulator_dir}/val_{model_index}.pdf")
