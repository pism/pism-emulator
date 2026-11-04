# How to install

pism-emulator installs into a conda environment. Every dependency comes from
[conda-forge](https://conda-forge.org) except `cmglaciology` (colormaps), which
the environment files install with pip because it is not on conda-forge.

## CPU and Apple Silicon

```bash
git clone https://github.com/pism/pism-emulator.git
cd pism-emulator
conda env create -f environment.yml
conda activate pism-emulator
python -m pip install -e . --no-deps
```

Or using [Mamba](https://mamba.readthedocs.io/) instead, which resolves the
environment considerably faster:

```bash
git clone https://github.com/pism/pism-emulator.git
cd pism-emulator
mamba env create -f environment.yml
mamba activate pism-emulator
python -m pip install -e . --no-deps
```

`--no-deps` keeps pip from installing a second copy of packages conda already
provides. Python is pinned to `>=3.11,<3.14`, the versions all compiled
dependencies (h5py, netCDF4, PyTorch) have conda-forge builds for.

## NVIDIA GPUs

On Linux with an NVIDIA GPU use `environment-cuda.yml`, which installs
conda-forge's CUDA build of PyTorch (`pytorch-gpu`):

```bash
mamba env create -f environment-cuda.yml
mamba activate pism-emulator
python -m pip install -e . --no-deps
```

Check that PyTorch sees the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

If the solver picks a CUDA version newer than your driver supports, add
`cuda-version=12.6` (or whichever your driver supports) to
`environment-cuda.yml`.

## Without conda

`pip install .` installs the dependencies listed in `requirements.txt`. This
works where wheels exist for every compiled package, but conda-forge is the
tested route.

## Documentation

The documentation toolchain is separate:

```bash
python -m pip install -r docs/requirements.txt
```

See {doc}`../developer/documentation`.
