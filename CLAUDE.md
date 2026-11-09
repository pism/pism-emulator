# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pism-emulator implements a two-step Bayesian calibration framework for ice sheet models (primarily PISM - Parallel Ice Sheet Model). The workflow uses neural network emulators as surrogates for expensive ice flow simulations, enabling efficient parameter calibration against observations of surface speeds and mass loss.

**Key workflow:**
1. Train neural network emulators on high-fidelity ice flow model outputs
2. Use MALA (Metropolis-adjusted Langevin algorithm) MCMC to calibrate flow parameters against observed surface speeds
3. Perform importance sampling to condition ensemble members on observed mass change

## Development Environment

### Installation
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate pism-emulator

# Install package in development mode (dependencies come from conda-forge)
pip install -e . --no-deps
```

For GPU support on CUDA systems, use `environment-cuda.yml` instead. Both environment
files list every dependency explicitly (conda-forge, plus `cmglaciology` via pip because it is not on conda-forge; Python pinned `>=3.12,<3.14`);
`requirements.txt` is only used by `pyproject.toml` for pip-only installs, so keep the
three files in sync when adding a dependency.

### Testing
```bash
# Run all tests
pytest -v tests/

# Run specific test file
pytest -v tests/test_nnemulators.py

# Run with warnings suppressed
pytest -v -W ignore::UserWarning tests/
```

### Code Quality
The project uses pre-commit hooks for code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually on all files
pre-commit run --all-files

# Run specific tools
pre-commit run black --all-files
pre-commit run mypy --all-files
```

**Enforced standards:**
- **black**: Code formatting (line length: 88)
- **isort**: Import sorting (profile: black)
- **mypy**: Type checking (Python 3.12)
- **pylint**: Linting
- **numpydoc**: Docstring validation

### Linting Configuration
- Max line length: 120 characters (flake8, pylint) or 88 (black)
- Tool configurations are in `pyproject.toml`
- Many warnings are ignored in flake8 (E203, E501, W503, etc.) - see `[tool.flake8]`

## Documentation

Sphinx + MyST site in `docs/` (same layout as pism-terra): `pip install -r docs/requirements.txt`, then `cd docs && make html`. The `Docs` workflow builds it with warnings as errors; Read the Docs uses `.readthedocs.yaml` (CPU-only PyTorch, then `requirements.txt`). New public symbols go into the autosummary tables in `docs/source/reference/api.md`.

## CI and Releases

- GitHub Actions in `.github/workflows/`: `test.yml` (pytest on Python 3.12 and 3.13), `static-analysis.yml` (pre-commit, pylint, secrets scan), `changelog.yml` (every PR must edit `CHANGELOG.md` unless labelled `bumpless`), `labeled-pr.yml` (PRs into `release` need a `major`/`minor`/`patch`/`bumpless` label).
- Releases: merge a labelled PR from `main` into `release` -> `tag-version.yml` bumps the latest `vX.Y.Z` tag -> `release.yml` creates the GitHub release from the matching `## [X.Y.Z]` section of `CHANGELOG.md` and fast-forwards `main`. Both use the `CICD_PAT` repository secret.
- The package version comes from git tags via `setuptools_scm`.

## Core Architecture

### Package Structure

```
pism_emulator/
├── datasets.py         # Dataset classes for loading PISM outputs
├── glaciers.py         # Multi-glacier dataset, discovery and GlacierIndex
├── datamodules.py      # PyTorch Lightning DataModules
├── emulators/
│   ├── nnemulator.py   # Neural network emulator models
│   └── speed/
│       ├── train.py    # Training script for speed emulators
│       ├── evaluate.py # Evaluation script
│       └── evaluate_glaciers.py # Multi-glacier evaluation (per glacier stats)
├── mcmc/
│   ├── mala.py         # MALA sampler implementation
│   ├── sample_speed.py # Speed parameter sampling
│   ├── sample_pdd.py   # PDD parameter sampling
│   ├── sample_pdd_reparametrized.py  # Reparametrized PDD sampler
│   ├── sample_pdd_verify.py          # Verification variant
│   ├── sample_pdd_reparametrized_verify.py  # Reparametrized verify variant
│   ├── plot_posterior.py
│   └── writer.py       # Checkpoint/output writers
├── models/
│   └── pdd.py          # Positive Degree Day model
├── lhs/
│   └── draw.py         # Latin Hypercube Sampling
├── metrics.py          # Custom metrics (area-weighted errors)
├── plotting.py         # Visualization utilities
├── stats.py            # Statistical utilities
└── utils.py            # General utilities
```

### Key Components

#### 1. Neural Network Emulators (`emulators/nnemulator.py`)

Multiple architectures available:
- **NNEmulator**: Standard feed-forward emulator
- **NN5Emulator**: 5-layer variant
- **DNNEmulator**: Deep neural network emulator (default)
- **LegacyNNEmulator**: Backward-compatible implementation

All emulators:
- Inherit from `pl.LightningModule` (PyTorch Lightning)
- Use eigenglacier decomposition (SVD) to compress spatial fields
- Support custom learning rate schedules and optimizers
- Implement area-weighted loss functions

**Key method:** `forward(X)` - takes parameter vectors, returns predicted field in eigenspace

#### 2. Dataset Loading (`datasets.py`)

**PISMInterpolatedDataset** (primary dataset):
- Loads NetCDF files from PISM simulations
- Extracts input parameters (X) and output fields (F)
- Applies transformations (log10, robust scaling)
- Supports lazy loading and parallel file reading
- Uses ID extraction pattern: `ID_RE = re.compile(r"id_(?P<id>\d+)_")`

**Important:** Dataset expects PISM files with specific naming convention containing `id_XXX_` pattern.

#### 3. Data Modules (`datamodules.py`)

**PISMDataModule**:
- Manages train/val/test splits
- Computes eigenglacier basis (SVD decomposition) on-the-fly or from cache
- Handles weighted sampling (via `omegas` weights)
- Thread-safe eigenglacier computation with `EigCache`

**Key attributes:**
- `eig_cache.V`: Right singular vectors (eigenglaciers)
- `eig_cache.S`: Singular values
- Uses `svd_cache/` directory for caching eigenglaciers

#### 4. MALA Sampler (`mcmc/mala.py`)

**MALASamplerModule**:
- Gradient-based MCMC using autodifferentiation
- Runs multiple chains in parallel via PyTorch Lightning's Trainer
- Outputs samples to NetCDF via `DiskPredictionWriter`

**Key parameters:**
- `step_size`: MALA step size (typically 1e-3 to 1e-2)
- `alpha`: Target acceptance rate (default: 0.01)
- `burn`: Burn-in samples to discard
- `samples`: Number of posterior samples

#### 5. PDD Model (`models/pdd.py`)

Positive Degree Day model for surface mass balance:
- **ReferencePDDModel**: NumPy-based reference implementation (time is first axis)
- **PDD**: PyTorch Lightning module for gradient computation (time is last axis)
- Uses `freeze_it` decorator to prevent attribute modification after init
- Key parameters: degree-day factors, refreezing fractions, temperature thresholds

## Command-Line Tools

The package provides several entry points (defined in `pyproject.toml`):

### Training Emulators
```bash
pism-train-emulator \
    --data-dir path/to/pism/outputs \
    --emulator-dir path/to/save/emulators \
    --target-file data/observed_speeds/greenland_vel_mosaic250_v1_g1800m.nc \
    --emulator NN \
    --gpus 1 \
    --max-epochs 1000
```

**Available emulators:** `NN`, `NN5`, `DNN`, `LegacyNN`

### Training on Several Glaciers
```bash
pism-train-emulator --emulator DNN --emulator-dir emulators \
    --glacier-dir ~/base/pism-terra/2026_11_s4f_iceflow_calib \
    --target-template '~/base/pism-terra/glacier_s4f_input/{rgi_id}/input/obs_{rgi_id}.nc' \
    --target-var v --y-lim 1 100e3
pism-evaluate-emulator-glaciers --emulator-dir emulators
```
Members missing on any glacier are dropped everywhere; `glacier_index.nc` in the
emulator directory maps the stacked node vector back to each glacier's grid.
Ice-free cells are masked with the target's `landice` (`--target-mask-var`).

### Evaluating Emulators
```bash
pism-evaluate-emulator \
    --emulator-dir path/to/emulators \
    --model-index 0
```

### Sampling Posterior (Speed)
```bash
pism-sample-posterior-speed \
    --emulator-dir path/to/emulators \
    --target-file data/observed_speeds/greenland_vel_mosaic250_v1_g1800m.nc \
    --samples 10000 \
    --burn 1000 \
    --chains 4 \
    --accelerator cuda \
    MODEL_FILE.ckpt
```

### Sampling Posterior (PDD)
```bash
pism-sample-posterior-pdd \
    --samples 10000 \
    --burn 1000 \
    --chains 4 \
    --accelerator cuda \
    CLIMATEFILE.nc

# Reparametrized variant
pism-sample-posterior-pdd-reparametrized [same options]

# Verification variants (for debugging/testing)
pism-sample-posterior-pdd-verify [same options]
pism-sample-posterior-pdd-reparametrized-verify [same options]
```

### Plotting Posterior
```bash
pism-plot-posterior \
    --posterior-file path/to/posterior.nc
```

## Important Implementation Details

### Device Handling
- Code supports CPU, CUDA, and MPS (Apple Silicon) backends
- Set device via `--accelerator` argument: `auto`, `cpu`, `cuda`, or `mps`
- PyTorch Lightning handles device placement automatically
- For MPS, set `num_workers=0` in DataLoaders to avoid spawn overhead

### Eigenglacier Computation
- Uses SVD to decompose spatial fields into eigenbasis
- **Critical:** Eigenglacier basis must be computed with same dataset used for training
- Cached in `svd_cache/` directory with hash-based filenames
- Use `--use-eig` flag in PDD sampling to enable eigenspace sampling
- **New option:** `--use-linalg-eig` command line option enables `torch.linalg.eig` instead of SVD for eigenglacier computation

### Y-Transform Options
Three transformation modes for output fields (specified via `--y-transform`):
- `none`: No transformation
- `log10`: Log10 transform with clamping via `--y-lim`
- `robust`: Robust scaling (center + scale normalization)

Inverse transforms are automatically applied during prediction.

### Numerical Stability
- Uses `torch.use_deterministic_algorithms(True)` for reproducibility
- Kaiming uniform initialization for linear layers
- Layer normalization in MLP blocks
- Gradient clipping in MALA sampler

### File Naming Conventions
- Training data: Must contain `id_XXX_` pattern in filename
- Emulator checkpoints: `emulator_*.ckpt` or similar
- Posterior outputs: NetCDF format with ArviZ-compatible structure
- TensorBoard logs: `tb_logs/` or `lightning_logs/`

## Calibration Workflow

The complete two-step calibration process:

1. **Train Speed Emulators:**
   ```bash
   pism-train-emulator --data-dir data/training --emulator-dir emulators
   ```

2. **Sample Speed Posterior:**
   ```bash
   pism-sample-posterior-speed --emulator-dir emulators MODEL.ckpt
   ```

3. **Run High-Fidelity Model:** Use posterior samples to initialize new PISM ensemble

4. **Importance Sampling:**
   ```bash
   cd calibration
   python calibrate-as19.py  # Conditions on mass loss observations
   ```

## GPU Acceleration

### CUDA Setup
- Use `environment-cuda.yml` (conda-forge `pytorch-gpu`; add `cuda-version=12.x` to match older drivers)
- Set `--accelerator cuda` in command-line tools
- Code includes CUDA-specific directives: `torch.backends.cudnn.conv.fp32_precision = "tf32"`

### Performance Tips
- Use `num_workers > 0` for DataLoaders on Linux/CUDA
- Use `num_workers = 0` on macOS/MPS
- Batch sizes of 128-512 work well for most emulators
- Multi-GPU training: PyTorch Lightning handles automatically with `--devices N`

## Common Gotchas

1. **Missing ID in filenames:** Dataset expects `id_XXX_` pattern - files without this will be skipped
2. **Eigenglacier cache mismatch:** Changing dataset requires regenerating SVD cache
3. **Transform compatibility:** Emulator must use same `y_transform` and `y_lim` for training and inference
4. **Pre-commit failures:** Run `pre-commit run --all-files` before committing
5. **Import errors:** Some imports are in-function to avoid circular dependencies
6. **Legacy code warnings:** `calibration/calibrate-as19.py` disables many pylint checks - this is intentional
7. **PDD time axis:** ReferencePDDModel uses time as first axis; PyTorch PDD uses time as last axis

## Data Files

Expected data locations (relative to repo root):
- `data/observed_speeds/`: Observed surface velocity mosaics
- `data/climate/`: Climate forcing data
- `data/emulators/`: Saved emulator checkpoints
- `data/samples/`: LHS parameter samples
- `svd_cache/`: Eigenglacier basis cache
- `posterior/`: MCMC output directory

## References

This codebase implements methods from:

Aschwanden, A., & Brinkerhoff, D. J. (2022). Calibrated mass loss predictions for the Greenland Ice Sheet. *Geophysical Research Letters*, 49, e2022GL099058. https://doi.org/10.1029/2022GL099058
