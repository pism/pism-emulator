import re
from collections import OrderedDict
from glob import glob
from os.path import join

import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm.autonotebook import tqdm


class PISMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir="path/to/dir",
        samples_file="path/to/file",
        target_file=None,
        target_var="velsurf_mag",
        target_corr_threshold=25.0,
        target_corr_var="thickness",
        target_error_var="velsurf_mag_error",
        training_var="velsurf_mag",
        thinning_factor=1,
        normalize_x=True,
        log_y=True,
        threshold=100e3,
        epsilon=0,
        return_numpy=False,
        verbose=False,
    ):
        self.data_dir = data_dir
        self.samples_file = samples_file
        self.target_file = target_file
        self.target_var = target_var
        self.target_corr_threshold = target_corr_threshold
        self.target_corr_var = target_corr_var
        self.target_error_var = target_error_var
        self.thinning_factor = thinning_factor
        self.threshold = threshold
        self.training_var = training_var
        self.epsilon = epsilon
        self.log_y = log_y
        self.normalize_x = normalize_x
        self.return_numpy = return_numpy
        self.verbose = verbose
        self.load_target()
        self.load_data()

    def __getitem__(self, i):
        return tuple(d[i] for d in [self.X, self.Y])

    def __len__(self):
        return min(len(d) for d in [self.X, self.Y])

    def load_target(self):
        epsilon = self.epsilon
        return_numpy = self.return_numpy
        thinning_factor = self.thinning_factor
        print(f"Loading target {self.target_file}")
        ds = xr.open_dataset(self.target_file)
        data = ds.variables[self.target_var].squeeze()
        mask = data.isnull()
        mask = mask[::thinning_factor, ::thinning_factor]
        data = np.nan_to_num(
            data.values[::thinning_factor, ::thinning_factor],
            nan=epsilon,
        )
        ny, nx = data.shape
        self.target_has_error = False
        if self.target_error_var in ds.variables:
            data_error = ds.variables[self.target_error_var].squeeze()
            data_error = np.nan_to_num(
                data_error.values[::thinning_factor, ::thinning_factor],
                nan=epsilon,
            )
            self.target_has_error = True

        self.target_has_corr = False
        if self.target_corr_var in ds.variables:
            data_corr = ds.variables[self.target_corr_var].squeeze()
            data_corr = np.nan_to_num(
                data_corr.values[::thinning_factor, ::thinning_factor],
                nan=epsilon,
            )
            self.target_has_corr = True
            mask = mask.where(data_corr >= self.target_corr_threshold, True)
        mask = mask.values

        grid_resolution = np.abs(np.diff(ds.variables["x"][0:2]))[0]
        self.grid_resolution = grid_resolution
        ds.close()

        idx = (mask == 0).nonzero()

        data = data[idx]
        Y_target = np.array(data.flatten(), dtype=np.float32)
        if return_numpy is False:
            Y_target = torch.from_numpy(Y_target)
        self.Y_target = Y_target
        if self.target_has_error:
            data_error = data_error[idx]
            Y_target_error_2d = data_error
            Y_target_error = np.array(data_error.flatten(), dtype=np.float32)
            if not return_numpy:
                Y_target_error = torch.from_numpy(Y_target_error)

            self.Y_target_error = Y_target_error
            self.Y_target_error_2d = Y_target_error_2d
        if self.target_has_corr:
            data_corr = data_corr[idx]
            Y_target_corr_2d = data_corr
            Y_target_corr = np.array(data_corr.flatten(), dtype=np.float32)
            if not return_numpy:
                Y_target_corr = torch.from_numpy(Y_target_corr)

            self.Y_target_corr = Y_target_corr
            self.Y_target_corr_2d = Y_target_corr_2d
        self.mask_2d = mask
        self.sparse_idx_2d = idx
        self.sparse_idx_1d = np.ravel_multi_index(idx, mask.shape)
        data_2d = np.zeros((ny, nx))
        data_2d.put(self.sparse_idx_1d, data)
        Y_target_2d = np.ma.array(data=data_2d, mask=self.mask_2d)
        self.Y_target_2d = Y_target_2d

    def load_data(self):
        epsilon = self.epsilon
        return_numpy = self.return_numpy
        thinning_factor = self.thinning_factor

        identifier_name = "id"
        training_var = self.training_var
        training_files = glob(join(self.data_dir, "*.nc"))
        training_files = list(OrderedDict.fromkeys(training_files))
        ids = [int(re.search("id_(.+?)_", f).group(1)) for f in training_files]
        samples = (
            pd.read_csv(self.samples_file, delimiter=",", skipinitialspace=True)
            .squeeze("columns")
            .sort_values(by=identifier_name)
        )
        samples.index = samples[identifier_name]
        samples.index.name = None

        ids_df = pd.DataFrame(data=ids, columns=["id"])
        ids_df.index = ids_df[identifier_name]
        ids_df.index.name = None

        # It is possible that not all ensemble simulations succeeded and returned a value
        # so we much search for missing response values
        missing_ids = list(set(samples["id"]).difference(ids_df["id"]))
        if missing_ids:
            if self.verbose:
                print(f"The following simulations are missing:\n   {missing_ids}")
                print("  ... adjusting priors")
            # and remove the missing samples and responses
            samples_missing_removed = samples[~samples["id"].isin(missing_ids)]
            samples = samples_missing_removed

        samples = samples.drop(samples.columns[0], axis=1)
        m_samples, n_parameters = samples.shape
        self.X_keys = samples.keys()

        ds0 = xr.open_dataset(training_files[0])
        _, ny, nx = (
            ds0.variables["velsurf_mag"]
            .values[:, ::thinning_factor, ::thinning_factor]
            .shape
        )

        ds0.close()
        self.nx = nx
        self.ny = ny
        response = np.zeros((m_samples, len(self.sparse_idx_1d)))

        print("  Loading data sets...")
        training_files.sort(key=lambda x: int(re.search("id_(.+?)_", x).group(1)))

        for idx, m_file in tqdm(enumerate(training_files)):
            ds = xr.open_dataset(m_file)
            data = np.squeeze(
                np.nan_to_num(
                    ds.variables[training_var].values[
                        :, ::thinning_factor, ::thinning_factor
                    ],
                    nan=epsilon,
                )
            )
            response[idx, :] = data[self.sparse_idx_2d].flatten()
            ds.close()

        p = response.max(axis=1) < self.threshold
        if self.log_y:
            response = np.log10(response)
            response[np.isneginf(response)] = 0

        X = np.array(samples[p], dtype=np.float32)
        Y = np.array(response[p], dtype=np.float32)
        if not return_numpy:
            X = torch.from_numpy(X)
            Y = torch.from_numpy(Y)
        Y[Y < 0] = 0

        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        self.X_mean = X_mean
        self.X_std = X_std

        if self.normalize_x:
            X = (X - X_mean) / X_std

        self.X = X
        self.Y = Y

        n_parameters = X.shape[1]
        self.n_parameters = n_parameters
        n_samples, n_grid_points = Y.shape
        self.n_samples = n_samples
        self.n_grid_points = n_grid_points

        normed_area = np.ones(n_grid_points, dtype=np.float32)
        if not return_numpy:
            normed_area = torch.tensor(normed_area)
        normed_area /= normed_area.sum()
        self.normed_area = normed_area

    def return_original(self):
        if self.normalize_x:
            return self.X * self.X_std + self.X_mean
        else:
            return self.X
