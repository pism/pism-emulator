# Command-line interface

Every command is installed as a console script by `pip install -e . --no-deps`.
The full list lives in `pyproject.toml`'s `[project.scripts]`.

## Emulators

```{list-table}
:header-rows: 1
:widths: 35 65

* - Command
  - Purpose
* - `pism-train-emulator`
  - Train one emulator on a PISM ensemble (`TRAINING_FILES`, `--samples-file`,
    `--target-file`). `--emulator {DNN,NN,NN5,LegacyNN}`, `--model-index N`
    (bootstrap member and seed), `-q`/`--cutoff` (eigenglaciers),
    `--y-transform`/`--y-lim`, `--max-epochs`, `--batch-size`,
    `--accelerator`/`--devices`/`--strategy`, plus the architecture's own
    options (`--width`, `--depth`, `--dropout`, `--activation`,
    `--learning-rate`, …). Writes `<emulator-dir>/emulator/emulator_N.ckpt`.
* - `pism-evaluate-emulator`
  - Compare trained emulators (`EMULATOR_FILES`) with PISM on a random subset
    of the ensemble (`--sample-size`): maps and error statistics.
```

## Posterior sampling

```{list-table}
:header-rows: 1
:widths: 35 65

* - Command
  - Purpose
* - `pism-sample-posterior-speed`
  - Sample the flow-parameter posterior given observed speeds with a trained
    emulator (`TRAINING_FILES MODEL_FILE`). Writes
    `<emulator-dir>/posterior/X_posterior_model_N.nc` and a trace plot.
* - `pism-sample-posterior-pdd`
  - Calibrate the PDD model against a regional climate model (`CLIMATEFILE`,
    `--years`, `--thin`, `--normalize`).
* - `pism-sample-posterior-pdd-reparametrized`
  - The same in unbounded logit coordinates.
* - `pism-sample-posterior-pdd-verify`, `pism-sample-posterior-pdd-reparametrized-verify`
  - The PDD calibration on a synthetic climate with known parameters.
* - `pism-plot-posterior`
  - Plot the marginals of one or more posterior files (`POSTERIOR_FILES`,
    `--fraction`).
```

All `pism-sample-posterior-*` commands share the sampler options:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Option
  - Meaning
* - `--proposal {manifold,fixed,legacy}`
  - Proposal (default `manifold`); see {doc}`../features/speed_calibration`
* - `--chains`, `--burn`, `--samples`
  - Number of chains, burn-in steps, kept draws per chain
* - `--acc-target`
  - Acceptance rate the burn-in adaptation aims for (default 0.57 with
    `--proposal fixed`, else 0.25)
* - `--h-max`
  - Largest step size the adaptation may reach (default 1.0)
* - `--alpha`
  - Weight of the likelihood against the prior (default 0.01)
* - `--accelerator {auto,cpu,cuda,mps}`
  - Device; with `cpu` and several chains, one process per chain
```

## Mass-loss calibration

```{list-table}
:header-rows: 1
:widths: 35 65

* - Script
  - Purpose
* - `calibration/calibrate-as19.py`
  - Importance-sample the AS19 ensemble against IMBIE mass loss and make the
    figures of the paper; see {doc}`../features/importance_sampling`.
```
