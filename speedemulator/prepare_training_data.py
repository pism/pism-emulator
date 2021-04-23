#!/bin/env python3

from glob import glob
import numpy as np
import pandas as pd
import re
import xarray as xr
from tqdm import tqdm

thin = 1
epsilon = 1e-10

s_files = glob("../data/speeds_v2/velsurf_mag_gris_g1800m_v4_id_*_0_50.nc")
nt = len(s_files)

# open first file to get the dimensions
vxr0 = xr.open_dataset(s_files[0])
speed = vxr0.variables["velsurf_mag"].values[:, ::thin, ::thin]
_, ny, nx = speed.shape
vxr0.close()

m_speeds = np.zeros((nt, ny * nx)) + epsilon

# Go through all files and don't forget to extract the experiment id
ids = []
for k, s_file in tqdm(enumerate(s_files)):
    print(f"{k}: Reading {s_file}")
    vxr = xr.open_dataset(s_file)

    m_id = re.search("id_(.+?)_0", s_file).group(1)
    ids.append(m_id)
    m_speeds[k, ::] = np.nan_to_num(vxr.variables["velsurf_mag"].values[:, ::thin, ::thin].flatten(), epsilon)
    vxr.close()


m_log_speeds = np.log10(m_speeds)
m_log_speeds[np.isneginf(m_log_speeds)] = 0
response_file = f"log_speeds_prune_{thin}"
print(f"Saving training data to {response_file}")
columns = [f"p_{p}" for p in range(m_log_speeds.shape[1])]
df = pd.DataFrame(data=m_log_speeds, columns=columns)
df.to_csv(f"{response_file}.csv.gz", index_label="id", compression="gzip")

# data_dir = "../data/speeds_v2"
# identifier_name = "id"
# training_files = glob(join(data_dir, "*.nc"))
# ids = [int(re.search("id_(.+?)_", f).group(1)) for f in training_files]
# samples = pd.read_csv(samples_file, delimiter=",", squeeze=True, skipinitialspace=True).sort_values(by=identifier_name)
# samples.index = samples[identifier_name]
# samples.index.name = None

# ids_df = pd.DataFrame(data=ids, columns=["id"])
# ids_df.index = ids_df[identifier_name]
# ids_df.index.name = None

# # It is possible that not all ensemble simulations succeeded and returned a value
# # so we much search for missing response values
# missing_ids = list(set(samples["id"]).difference(ids_df["id"]))
# if missing_ids:
#     print(f"The following simulations are missing:\n   {missing_ids}")
#     print("  ... adjusting priors")
#     # and remove the missing samples and responses
#     samples_missing_removed = samples[~samples["id"].isin(missing_ids)]
#     samples = samples_missing_removed

# samples = samples.drop(samples.columns[0], axis=1)
# m_samples, n_parameters = samples.shape

# ds0 = xr.open_dataset(training_files[0])
# _, my, mx = ds0.variables["velsurf_mag"][:].shape
# ds0.close()

# response = np.zeros((m_samples, my * mx))

# print("  Loading data sets...")
# for idx, m_file in tqdm(enumerate(training_files)):
#     ds = xr.open_dataset(m_file)
#     data = np.nan_to_num(ds.variables["velsurf_mag"].values[:, ::thin, ::thin].flatten(), epsilon)
#     response[idx, :] = data
#     ds.close()

# m_log_speeds = np.log10(response)
# m_log_speeds[np.isneginf(m_log_speeds)] = 0
# response_file = f"log_speeds_prune_{thin}.csv.gz"
# print(f"Saving training data to {response_file}")
# columns = [f"p_{p}" for p in range(m_log_speeds.shape[1])]
# df.to_csv(response_file, index_label="id", compression="gzip")
# re_file = f"log_speeds_prune_{thin}.csv.gz"
