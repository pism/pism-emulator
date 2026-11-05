# Quick start

This walks through the flow calibration for Greenland at 1800 m: train an
emulator, check it, and sample the posterior of the eight flow parameters.

## Inputs

```{list-table}
:header-rows: 1
:widths: 35 65

* - Input
  - Example
* - Training ensemble: PISM surface speeds, one file per member, with
    `id_<N>_` in the file name
  - `data/speeds_v2/velsurf_mag_gris_g1800m_v4_id_*.nc`
* - Parameter values of the members, one row per `id`
  - `data/samples/velocity_calibration_samples_100.csv` (columns `id`, `SIAE`,
    `SSAN`, `PPQ`, `TEFO`, `PHIMIN`, `PHIMAX`, `ZMIN`, `ZMAX`)
* - Observed speeds and their errors on the same grid
  - `data/observed_speeds/greenland_vel_mosaic250_v1_g1800m.nc` (variables
    `velsurf_mag` and `velsurf_mag_error`)
```

The training ensemble is available from [arcticdata.io](https://arcticdata.io);
the observed speeds are the MEaSUREs mosaic
[NSIDC-0670](https://nsidc.org/data/NSIDC-0670/versions/1) regridded to the
model grid.

## 1. Train an emulator

```bash
pism-train-emulator \
    --emulator DNN \
    --emulator-dir emulators \
    --model-index 0 \
    --samples-file data/samples/velocity_calibration_samples_100.csv \
    --target-file data/observed_speeds/greenland_vel_mosaic250_v1_g1800m.nc \
    --y-transform log10 --y-lim 1 1e5 \
    --max-epochs 1000 \
    data/speeds_v2/velsurf_mag_gris_g1800m_v4_id_*.nc
```

The checkpoint is written to `emulators/emulator/emulator_0.ckpt`, TensorBoard
logs to `emulators/tb_logs/`, and the eigenglacier basis is cached in
`svd_cache/`. Repeat with `--model-index 1`, `2`, … for a bootstrap ensemble:
each member is trained with different Dirichlet weights on the training runs
(see {doc}`../features/emulators`).

## 2. Evaluate it

```bash
pism-evaluate-emulator \
    --emulator-dir emulators \
    --samples-file data/samples/velocity_calibration_samples_100.csv \
    --target-file data/observed_speeds/greenland_vel_mosaic250_v1_g1800m.nc \
    --y-transform log10 --y-lim 1 1e5 \
    --training-files data/speeds_v2/velsurf_mag_gris_g1800m_v4_id_*.nc \
    -- emulators/emulator/emulator_*.ckpt
```

```{image} ../../../images/speed_emulator_train.png
:alt: PISM vs emulator speeds
:width: 600px
```

## 3. Sample the posterior

```bash
pism-sample-posterior-speed \
    --emulator-dir emulators \
    --model-index 0 \
    --samples-file data/samples/velocity_calibration_samples_100.csv \
    --target-file data/observed_speeds/greenland_vel_mosaic250_v1_g1800m.nc \
    --y-transform log10 --y-lim 1 1e5 \
    --chains 4 --burn 1000 --samples 3000 \
    data/speeds_v2/velsurf_mag_gris_g1800m_v4_id_*.nc \
    emulators/emulator/emulator_0.ckpt
```

Each chain starts at the maximum a posteriori (MAP) point. The posterior,
prior and per-draw statistics are written to
`emulators/posterior/X_posterior_model_0.nc` (ArviZ format) together with a
trace plot. Add `--proposal fixed` for a sampler that is about nine times
faster; see {doc}`../features/speed_calibration` for the trade-off.

## 4. Plot

```bash
pism-plot-posterior emulators/posterior/X_posterior_model_*.nc
```

## Next

The posterior draws initialise the calibrated PISM projections, which are then
weighted against observed mass loss: {doc}`../features/importance_sampling`.
