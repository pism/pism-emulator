# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.43.0]

### Changed
- :warning: The MALA samplers (`sample-posterior-speed`, `sample-posterior-pdd*`) take `--proposal {manifold,fixed,legacy}` (`MALASamplerModule(proposal=...)`), replacing `metric_mode`. The default, `manifold`, is simplified manifold MALA: the proposal has a Langevin drift, and the Metropolis-Hastings ratio uses the geometry at the proposal for the reverse move (one Hessian per step). The old sampler reused the metric of the current state in both directions although the metric is recomputed at every accepted state, so it was not reversible and its posteriors were slightly biased (on a 2-D test target the mean was off by about 6 Monte Carlo standard errors). Posterior samples therefore differ from earlier runs. `legacy` restores the old sampler and reproduces earlier chains draw for draw; `delayed_accept` and `hess_refresh > 1` are only available with it.
- `--proposal fixed` computes the Hessian metric once per chain, at its starting point (the MAP), and keeps it: MALA with a constant preconditioner, which is exact and needs one gradient instead of one Hessian per step. This is the cheap variant of the simplification the original sampler intended (a reused Hessian makes the proposal symmetric); the original recomputed the Hessian at every accepted state, which is what made it biased.
- The `sample-posterior-*` tools take `--h-max`, the largest step size the burn-in adaptation may reach (default 1.0, as before), and `--acc-target`, the acceptance rate the adaptation aims for. It defaults to 0.57 with `--proposal fixed` (on the 1800 m Greenland speed calibration this raised the effective sample size up to 1.7x for most parameters) and stays 0.25 otherwise: at 0.57 the `manifold` sampler had to shrink its step size to 0.03 and mixed far worse.
- The sampler metric is the exact `|H|` from an eigendecomposition. The 6-iteration Newton-Schulz square root of `H @ H` it replaces missed `Hpos @ Hinv = I` by 30 % (condition number 10) to 175 % (condition number 1e4).
- Eigenglacier caches are named and checked by a hash of the responses, the weights, `q`, `cutoff` and `svd_lowrank` (`svd_cache/eig_<hash>.pt`, `PISMDataModule.eig_cache_key`).
- `DNNEmulator` no longer takes `--norm`, which was ignored, and its `depth` fallback is 4 as on the command line. The emulators log their validation losses once per epoch.

### Fixed
- `ReparametrizedMALASamplerModule` (`sample-posterior-pdd-reparametrized*`) evaluated the likelihood at `phi_to_X(phi_to_X(phi))` instead of `phi_to_X(phi)`, so it sampled close to the prior. Results of earlier reparametrized runs are not valid and cannot be reproduced with `--proposal legacy`.
- `find_MAP` (LBFGS) accumulated gradients across closure calls, so MAP starting points could be wrong.
- The `lp` of every accepted draw was NaN; it is now the log posterior of each kept state.
- `train-emulator` reused one eigenglacier cache (`svd_cache/svd.h5`) for every ensemble member, so members 1..N trained on member 0's Dirichlet weights; it also reused stale bases after changing `q`, `cutoff` or the data.
- The eigenglacier `cutoff` kept one mode fewer than needed and could keep none.
- The eigh eigenglacier path ignored `q` and could produce NaN from round-off negative eigenvalues; `LegacyPISMDataModule`'s eigh path raised `IndexError`.
- `sample-posterior-speed` crashed after sampling whenever `--chains` > 1, while writing the ArviZ file (the prior was split across chains but given the posterior's draw coordinate), so the chains were lost. The `sample-posterior-pdd*` tools failed an assertion when the chain count did not divide their 10,000 prior draws.
- `PISMDataModule` loaders crashed with `num_workers=0` (`persistent_workers` needs workers); `pin_memory` is used only with CUDA.

## [0.42.0]

### Changed
- `environment.yml` and `environment-cuda.yml` list every dependency from conda-forge and pin Python to `>=3.11,<3.14`; the `pip install -r requirements.txt` step is gone, which fixes h5py failing to build from source on Python 3.15. Install the package with `pip install -e . --no-deps`.
- `environment-cuda.yml` uses conda-forge's `pytorch-gpu` instead of the `pytorch`/`nvidia` channels, so pip no longer replaces the CUDA build of PyTorch.
- The version is now taken from git tags (`setuptools_scm`).
- CircleCI is replaced by GitHub Actions: tests, static analysis, changelog and label checks, and automatic tagging and GitHub releases from the `release` branch. Dependabot keeps the actions and `requirements.txt` up to date.

### Fixed
- `torchmetrics`, which the code imports, is now a declared dependency.
- The release tagging job no longer uses bump2version, which fails on Python 3.15 (the runners' `3.x`); it bumps the version with a shell script (`.github/scripts/next_version.sh`).

## [0.41.2]

- Thorough linting.
