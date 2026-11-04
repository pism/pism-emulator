# Tests and code quality

## Running the tests

```bash
pytest -v tests/
```

Long statistical tests are marked `slow`; skip them while iterating:

```bash
pytest -v -m "not slow" tests/
```

```{list-table}
:header-rows: 1
:widths: 35 65

* - File
  - What it covers
* - `test_mala.py`
  - The samplers on a two-parameter toy target (`mala_toy.py`) whose posterior
    moments are known exactly by quadrature: densities against SciPy and
    PyTorch references, the local geometry, the MAP, `lp`, the default
    acceptance targets, posterior moments of every proposal, and that
    `proposal="legacy"` reproduces golden chains of the old sampler
    (`test_data/mala_legacy_golden.pt`) draw for draw.
* - `test_eigenglaciers.py`
  - Eigenglacier bases (low-rank SVD against eigh, reconstruction of the
    weighted covariance, the cutoff, the cache key) and the data loaders.
* - `test_sample_cli.py`
  - `pism-sample-posterior-pdd-verify` end to end with three chains.
* - `test_nnemulators.py`
  - Emulator architectures and their command-line options.
* - `test_pdd.py`
  - The PyTorch PDD model against the NumPy reference.
* - `test_dataset.py`, `test_metrics.py`, `test_stats.py`
  - Datasets, the area-weighted error, and the statistics helpers.
```

When a change is meant to keep results unchanged, add a test that pins the
current output first, then make the change.

## Code quality

Pre-commit runs black, isort, numpydoc validation, mypy and pylint:

```bash
pre-commit install
pre-commit run --all-files
```

## Continuous integration

GitHub Actions run on every push and pull request:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Workflow
  - Checks
* - `test.yml`
  - pytest on Python 3.12 and 3.13 in the conda environment
* - `static-analysis.yml`
  - pre-commit, pylint on the package, and a secrets scan
* - `docs.yml`
  - the documentation builds without warnings
* - `changelog.yml`
  - the pull request edits `CHANGELOG.md` (unless labelled `bumpless`)
* - `labeled-pr.yml`
  - pull requests into `release` carry a `major`, `minor`, `patch` or
    `bumpless` label
```
