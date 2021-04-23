import pytorch_lightning as pl
import torch
from netCDF4 import Dataset as ds
from glob import glob
from os.path import join
import pandas as pd
from tqdm import tqdm
import re
import numpy as np


class PISMDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir="path/to/dir", samples_file="path/to/file", thin=1, epsilon=1e-10):
        self.data_dir = data_dir
        self.samples_file = samples_file

        identifier_name = "id"
        training_files = glob(join(self.data_dir, "*.nc"))
        ids = [int(re.search("id_(.+?)_", f).group(1)) for f in training_files]
        samples = pd.read_csv(self.samples_file, delimiter=",", squeeze=True, skipinitialspace=True).sort_values(
            by=identifier_name
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
            print(f"The following simulations are missing:\n   {missing_ids}")
            print("  ... adjusting priors")
            # and remove the missing samples and responses
            samples_missing_removed = samples[~samples["id"].isin(missing_ids)]
            samples = samples_missing_removed

        samples = samples.drop(samples.columns[0], axis=1)
        m_samples, n_parameters = samples.shape

        ds0 = xr.open_dataset(training_files[0])
        _, my, mx = ds0.variables["velsurf_mag"][:].shape
        ds0.close()

        response = np.zeros((m_samples, my * mx))

        print("  Loading data sets...")
        for idx, m_file in tqdm(enumerate(training_files)):
            ds = xr.open_dataset(m_file)
            data = np.nan_to_num(ds.variables["velsurf_mag"].values[:, ::thin, ::thin].flatten(), epsilon)
            response[idx, :] = data
            ds.close()

        self.samples = samples
        self.response = response


class PISMDataModule(pl.LightningDataModule):
    def __init__(self, dataset, stride=1, batch_size: int = 32):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)
