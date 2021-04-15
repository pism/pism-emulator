#!/usr/bin/env python3

from cf_units import Unit
from netCDF4 import Dataset as NC
import xarray as xr
import re
import numpy as np
import pandas as pd
import pylab as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from scipy.stats import dirichlet
from tqdm import tqdm
from time import time
import os
from pismemulator.utils import prepare_data

from dataloader_example import PISMDataset


thin = 5

o_file = "../data/validation/greenland_vel_mosaic250_v1_g1800m.nc"
o_xr = xr.open_dataset(o_file)

o_speed = o_xr.variables["velsurf_mag"].values[::thin, ::thin]
o_speed_sigma = o_xr.variables["velsurf_mag_error"].values[::thin, ::thin]

o_ny, o_nx = o_speed.shape
o_xr.close()

o_speeds = np.nan_to_num(o_speed, 0).reshape(-1, 1)
o_log_speeds = np.log10(o_speeds)
o_log_speeds[np.isneginf(o_log_speeds)] = 0

o_speeds_sigma = np.nan_to_num(o_speed_sigma, 0).reshape(-1, 1)

from glob import glob

s_files = glob("../data/speeds_v2/velsurf_mag_gris_g1800m_v4_id_*_0_50.nc")
nt = len(s_files)

# open first file to get the dimensions
vxr0 = xr.open_dataset(s_files[0])
speed = vxr0.variables["velsurf_mag"].values[:, ::thin, ::thin]
_, ny, nx = speed.shape
vxr0.close()

m_speeds = np.zeros((nt, ny * nx))

# Go through all files and don't forget to extract the experiment id
ids = []
for k, s_file in enumerate(s_files):
    print(f"Reading {s_file}")
    vxr = xr.open_dataset(s_file)
    ids.append(re.search("id_(.+?)_0", s_file).group(1))
    m_speeds[k, ::] = vxr.variables["velsurf_mag"].values[:, ::thin, ::thin].flatten()
    vxr.close()


m_log_speeds = np.log10(np.nan_to_num(m_speeds, 0))
m_log_speeds[np.isneginf(m_log_speeds)] = 0


samples_file = "../data/samples/velocity_calibration_samples_100.csv"
response_file = "log_speeds.csv"

df = pd.DataFrame(data=m_log_speeds, index=ids)
print(df.head())
df.to_csv(response_file, index_label="id")

samples, response = prepare_data(samples_file, response_file)

F = response.values
X = samples.values
