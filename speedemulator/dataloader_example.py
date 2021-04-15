import pytorch_lightning as pl
import torch
from netCDF4 import Dataset as ds
from glob import glob
from os.path import join
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
import re
import numpy as np


class PISMDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir="path/to/dir", samples_file="path/to/file", transform=None):
        self.data_dir = data_dir
        self.samples_file = samples_file
        self.transform = transform

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

        nc0 = ds(training_files[0])
        _, my, mx = nc0.variables["velsurf_mag"][:].shape
        nc0.close()

        response = np.zeros((m_samples, my * mx))

        print("  Loading data sets...")
        for idx, m_file in tqdm(enumerate(training_files)):
            nc = ds(m_file)
            data = nc.variables["velsurf_mag"][0, :, :]
            response[idx, :] = data.filled(fill_value=0).reshape(1, -1)
            nc.close()

        self.samples = samples
        self.response = response
        self.mean, self.std = response.mean(), response.std()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
        samples = transform(samples)

        return {"samples": samples, "response": response}


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
