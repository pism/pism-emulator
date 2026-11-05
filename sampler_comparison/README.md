# MALA sampler comparison on real data

Old (v0.42, `main`) vs new samplers on the 1800 m Greenland speed calibration,
run while reviewing the sampler for v0.43.

## Setup

- Emulator: `legacy/emulators_dnn/emulator/emulator_0.ckpt` (DNN, ReLU, 8 parameters,
  100 eigenglaciers, trained for only 20 epochs, so the posteriors are not a
  scientific result; all samplers see the same target).
- Inputs (local, not in the repository): `data/speeds_v2/velsurf_mag_gris_g1800m_v4_id_*.nc`
  (996 files), `data/samples/velocity_calibration_samples_100.csv`,
  `legacy/observed_speeds/greenland_vel_mosaic250_v1_g1800m.nc`.
- 4 chains x (1000 burn-in + 3000 draws), CPU (`ddp_spawn`), `alpha=0.01`.

Reproduce (each run keeps its chains in `OUT_DIR/_preds`):

```bash
PYTHON=$(which python) sampler_comparison/run_sample.sh . runs/manifold 1000 3000 4
PYTHON=$(which python) sampler_comparison/run_sample.sh . runs/fixed 1000 3000 4 --proposal fixed
python sampler_comparison/compare.py runs/manifold out.png manifold=runs/manifold fixed=runs/fixed
```

`compare.py` reads the `_preds` chain files, so it also works for runs of the old code,
which crashed while writing the ArviZ file with more than one chain. Its first argument
is a finished run whose posterior NetCDF gives the parameter names and normalization.

## Results

ESS is out of 12,000 draws. Posterior means of all runs agree within 0.2 posterior sd.

| | old (v0.42) | manifold, 0.25 | manifold, 0.57 | fixed, 0.25 | fixed, 0.57 |
|---|---|---|---|---|---|
| Sampling time | 26.5 min | 26.8 min | 26.0 min | 2.9 min | 2.8 min |
| Acceptance | 0.15 | 0.23 | 0.60 | 0.63 (h at h_max=1) | 0.47 |
| `lp` NaN | 14.7 % | 0 | 0 | 0 | 0 |
| Worst R-hat | 1.05 | 1.02 | 1.11 | 1.06 | 1.03 |
| ESS, SIAE | 481 | 520 | 70 | 62 | 74 |
| ESS, other 7 | 100-264 | 306-436 | 30-92 | 253-1324 | 316-1405 |

- The old sampler's bias shows as worse mixing and 5-12 % wider sds for several
  parameters (e.g. PHIMIN 3.32 vs 2.96), not as shifted means, for this broad posterior.
- `manifold` mixes evenly at 0.25 but collapses at 0.57 (step size 0.03), probably because
  the ReLU emulator's curvature changes abruptly. It keeps the 0.25 default.
- `fixed` is 9x faster and mixes better than `manifold` for 7 of 8 parameters at 0.57,
  which is now its default; SIAE mixes poorly with any fixed metric.
- Raising `h_max` to 10 or 100 for `fixed` at 0.25 did not help (step settles at 2.3-4.7,
  acceptance 0.19).

`old_new_fixed.png`: old, manifold (0.25) and fixed (0.25).
`acceptance_targets.png`: all five runs.
