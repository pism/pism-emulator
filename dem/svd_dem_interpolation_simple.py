from argparse import ArgumentParser
from glob import glob

import lightning as pl
import numpy as np
import pylab as plt
import torch
import xarray as xr
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MeanSquaredError

training_files = glob("usurf_gris_g1800m_v5_RAGIS_id_0_1980-1-1_2020-1-1_YM.nc")
epsilon = 0
training_var = "usurf"
thinning_factor = 1
# Number of principal components
q = 10

all_data = []
for idx, m_file in enumerate(training_files):
    print(f"Loading {m_file}")
    with xr.open_dataset(m_file) as ds:
        data = ds.variables[training_var]
        data = np.squeeze(
            np.nan_to_num(
                data.values[::thinning_factor, ::thinning_factor, ::thinning_factor],
                nan=epsilon,
            )
        )

        nt, ny, nx = data.shape
        all_data.append(data.reshape(nt, -1))
        ds.close()
data = np.concatenate(all_data, axis=0)

with xr.open_dataset("aerodem_1978_1987_wgs84_g1800m.nc") as ds:
    obs = ds.variables["surface_altitude"]
    mask = obs.isnull()
    m_mask = np.ones_like(mask)
    m_mask[mask == True] = 0
    obs = obs[::thinning_factor, ::thinning_factor]
    m_mask = m_mask[::thinning_factor, ::thinning_factor]
    I = torch.from_numpy(m_mask.ravel())
    R = torch.from_numpy(np.nan_to_num(obs.values.ravel(), 0))
    n_row, n_col = obs.shape


D = torch.from_numpy(data)
D_mean = D.mean(axis=0)
D_scaled = D - D_mean
U, S, V = torch.svd_lowrank(D_scaled, q=q)


n = len(R[I])

inputDim = q  # takes variable 'x
outputDim = 1  # takes variable 'y'
learningRate = 0.01
epochs = 5000


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.w = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.zeros(inputSize, outputSize)),
            requires_grad=True,
        )
        self.b = torch.nn.Parameter(
            torch.zeros(outputSize),
            requires_grad=True,
        )

    def forward(self, x):
        out = x @ self.w - self.b
        return out


V_i = V[I]
R_i = R[I]

model = linearRegression(inputDim, outputDim)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

K_reg = 1

inputs = V_i * S
labels = R_i.reshape(-1, 1)
print(V.shape, S.shape, inputs.shape, labels.shape)

for epoch in range(epochs):
    # Converting inputs and labels to Variable

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(inputs)

    # get loss for the predicted output
    loss = criterion(outputs, labels) + K_reg * (model.w**2).sum()
    # get gradients w.r.t to parameters
    loss.backward()

    # update parameters
    optimizer.step()

    print("epoch {}, loss {}".format(epoch, loss.item()))


p = []
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
        p.append(param.data)
Sr = model.w

bM = (V * S) @ model.w
# M = ((U * Sr) @ V.T).T
Ms = M.detach().numpy().reshape(n_row, n_col) + D_mean.detach().numpy().reshape(
    n_row, n_col
)


mds = xr.Dataset(
    {"usurf": (("y", "x"), Ms)},
    coords={
        "x": ("x", ds.x.values),
        "y": ("y", ds.y.values),
    },
).to_netcdf("test_usurf.nc")
