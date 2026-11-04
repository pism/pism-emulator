# Sampler comparison on real data

The old sampler (v0.42) and the `manifold` and `fixed` proposals of v0.43
({doc}`../features/speed_calibration`), compared on the 1800 m Greenland speed
calibration. The scripts and plots are in `sampler_comparison/` in the
repository.

## Setup

- Emulator: `legacy/emulators_dnn/emulator/emulator_0.ckpt` (DNN with ReLU,
  8 parameters, 100 eigenglaciers, trained for only 20 epochs, so the
  posteriors are not a scientific result; all samplers see the same target).
- 4 chains × (1000 burn-in + 3000 draws) on a CPU, `--alpha 0.01`.

## Results

ESS is the effective sample size out of 12,000 draws. Posterior means of all
runs agree within 0.2 posterior standard deviations.

```{list-table}
:header-rows: 1
:stub-columns: 1

* -
  - old (v0.42)
  - manifold, 0.25
  - manifold, 0.57
  - fixed, 0.25
  - fixed, 0.57
* - Sampling time
  - 26.5 min
  - 26.8 min
  - 26.0 min
  - 2.9 min
  - 2.8 min
* - Acceptance rate
  - 0.15
  - 0.23
  - 0.60
  - 0.63 (step at `h_max`)
  - 0.47
* - `lp` NaN
  - 14.7 %
  - 0
  - 0
  - 0
  - 0
* - Worst R-hat
  - 1.05
  - 1.02
  - 1.11
  - 1.06
  - 1.03
* - ESS, SIAE
  - 481
  - 520
  - 70
  - 62
  - 74
* - ESS, other 7
  - 100–264
  - 306–436
  - 30–92
  - 253–1324
  - 316–1405
```

- The old sampler's bias shows as worse mixing and 5–12 % wider standard
  deviations for several parameters (e.g. PHIMIN 3.32 vs 2.96), not as shifted
  means, for this broad posterior. Its `lp` is NaN for every accepted draw.
- `manifold` mixes evenly at the 0.25 target but collapses at 0.57: its step
  size falls to 0.03. The likely cause is the ReLU emulator, whose curvature
  changes abruptly. It keeps 0.25 as its default.
- `fixed` is nine times faster and, at the 0.57 target (now its default), mixes
  better than `manifold` for seven of eight parameters; SIAE mixes poorly with
  any fixed metric.
- Raising `--h-max` to 10 or 100 for `fixed` at 0.25 did not help: the step
  settled at 2.3–4.7 with 19 % acceptance.

```{figure} ../../../sampler_comparison/acceptance_targets.png
:alt: Posterior marginals of all five runs
:width: 100%

Posterior marginals of all five runs: `new25`/`new57` are `manifold` at
targets 0.25 and 0.57, `fix25`/`fix57` are `fixed`, `old` is v0.42.
```

## Reproducing

```bash
PYTHON=$(which python) sampler_comparison/run_sample.sh . runs/manifold 1000 3000 4
PYTHON=$(which python) sampler_comparison/run_sample.sh . runs/fixed 1000 3000 4 --proposal fixed
python sampler_comparison/compare.py runs/manifold out.png manifold=runs/manifold fixed=runs/fixed
```

The inputs (`data/speeds_v2/`, `legacy/…`) are local and not in the
repository.
