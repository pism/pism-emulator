#!/bin/env python3

from glob import glob
import numpy as np
import pandas as pd
import re
import xarray as xr

thin = 5
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
for k, s_file in enumerate(s_files):
    print(f"{k}: Reading {s_file}")
    vxr = xr.open_dataset(s_file)

    m_id = re.search("id_(.+?)_0", s_file).group(1)
    ids.append(m_id)
    m_speeds[k, ::] = np.nan_to_num(vxr.variables["velsurf_mag"].values[:, ::thin, ::thin].flatten(), epsilon)
    vxr.close()


m_log_speeds = np.log10(m_speeds)
m_log_speeds[np.isneginf(m_log_speeds)] = 0
response_file = "log_speeds.csv.gz"
print(f"Saving training data to {response_file}")
columns = [f"p_{p}" for p in range(m_log_speeds.shape[1])]
df = pd.DataFrame(data=m_log_speeds, columns=columns)
# df.to_csv(response_file, index_label="id", compression="gzip")
