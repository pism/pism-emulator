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

# Install package in development mode
pip install -e .
```

For GPU support on CUDA systems, use `environment-cuda.yml` instead.

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
- **mypy**: Type checking (Python 3.11)
- **pylint**: Linting
- **numpydoc**: Docstring validation

### Linting Configuration
- Max line length: 120 characters (flake8, pylint) or 88 (black)
- Tool configurations are in `pyproject.toml`
- Many warnings are ignored in flake8 (E203, E501, W503, etc.) - see `[tool.flake8]`

## Core Architecture

### Package Structure

```
pism_emulator/
├── datasets.py         # Dataset classes for loading PISM outputs
├── datamodules.py      # PyTorch Lightning DataModules
├── emulators/
│   ├── nnemulator.py   # Neural network emulator models
│   └── speed/
│       ├── train.py    # Training script for speed emulators
│       └── evaluate.py # Evaluation script
├── mcmc/
│   ├── mala.py         # MALA sampler implementation
│   ├── sample_speed.py # Speed parameter sampling
│   ├── sample_pdd.py   # PDD parameter sampling
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
- Implemented as PyTorch module for gradient computation
- Uses `freeze_it` decorator to prevent attribute modification after init
- Supports both NumPy and PyTorch backends

## Command-Line Tools

The package provides several entry points (defined in `pyproject.toml`):

### Training Emulators
```bash
train-emulator \
    --data-dir path/to/pism/outputs \
    --emulator-dir path/to/save/emulators \
    --target-file data/observed_speeds/greenland_vel_mosaic250_v1_g1800m.nc \
    --emulator NN \
    --gpus 1 \
    --max-epochs 1000
```

**Available emulators:** `NN`, `NN5`, `DNN`, `LegacyNN`

### Evaluating Emulators
```bash
evaluate-emulator \
    --emulator-dir path/to/emulators \
    --model-index 0
```

### Sampling Posterior (Speed)
```bash
sample-posterior-speed \
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
sample-posterior-pdd \
    --samples 10000 \
    --burn 1000 \
    --chains 4 \
    --accelerator cuda \
    CLIMATEFILE.nc
```

### Plotting Posterior
```bash
plot-posterior \
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
   train-emulator --data-dir data/training --emulator-dir emulators
   ```

2. **Sample Speed Posterior:**
   ```bash
   sample-posterior-speed --emulator-dir emulators MODEL.ckpt
   ```

3. **Run High-Fidelity Model:** Use posterior samples to initialize new PISM ensemble

4. **Importance Sampling:**
   ```bash
   cd calibration
   python calibrate-as19.py  # Conditions on mass loss observations
   ```

## GPU Acceleration

### CUDA Setup
- Install PyTorch with CUDA support
- Use `environment-cuda.yml` for conda environment
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
