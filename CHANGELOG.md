# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.42.0]

### Changed
- `environment.yml` and `environment-cuda.yml` list every dependency from conda-forge and pin Python to `>=3.11,<3.14`; the `pip install -r requirements.txt` step is gone, which fixes h5py failing to build from source on Python 3.15. Install the package with `pip install -e . --no-deps`.
- `environment-cuda.yml` uses conda-forge's `pytorch-gpu` instead of the `pytorch`/`nvidia` channels, so pip no longer replaces the CUDA build of PyTorch.
- The version is now taken from git tags (`setuptools_scm`).
- CircleCI is replaced by GitHub Actions: tests, static analysis, changelog and label checks, and automatic tagging and GitHub releases from the `release` branch. Dependabot keeps the actions and `requirements.txt` up to date.

### Fixed
- `torchmetrics`, which the code imports, is now a declared dependency.

## [0.41.2]

- Thorough linting.
