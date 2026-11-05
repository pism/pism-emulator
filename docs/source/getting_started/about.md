# About pism-emulator

Projections of the Greenland ice sheet's contribution to sea level depend on
poorly constrained ice flow parameters: the flow enhancement of the shallow ice
approximation, the exponent of the sliding law, the till friction angle and how
it varies with bed elevation, and so on. pism-emulator constrains these
parameters with observations in two steps {cite:p}`Aschwanden2022`:

1. **Flow calibration.** A neural network *emulator* learns the map from flow
   parameters to the modelled surface speed field from an ensemble of PISM
   simulations. Because the emulator is cheap and differentiable, a
   gradient-based Markov chain Monte Carlo sampler (MALA) can draw from the
   posterior distribution of the flow parameters given observed surface speeds
   {cite:p}`Joughin2018`. See {doc}`../features/emulators` and
   {doc}`../features/speed_calibration`.
2. **Mass-loss calibration.** Parameter sets drawn from that posterior
   initialise a new, high-fidelity PISM ensemble of projections, which is then
   weighted against observed cumulative mass change {cite:p}`IMBIE2020` by
   importance sampling. See {doc}`../features/importance_sampling`.

The same machinery calibrates a positive degree-day (PDD) surface mass balance
model against climate data (see {doc}`../features/pdd`).

## How it fits together

```{list-table}
:header-rows: 1
:widths: 30 70

* - Step
  - Tool
* - Draw the training ensemble's parameters
  - {func}`pism_emulator.lhs.draw.draw_samples` (Latin hypercube) or your own
    sample file
* - Run PISM for every parameter set
  - PISM (outside this package), one NetCDF file per member with `id_<N>_` in
    its name
* - Train the emulator ensemble
  - `pism-train-emulator` (one call per bootstrap member, `--model-index N`)
* - Check the emulators
  - `pism-evaluate-emulator`
* - Sample the flow-parameter posterior
  - `pism-sample-posterior-speed`, `pism-plot-posterior`
* - Run calibrated PISM projections
  - PISM, initialised from posterior draws
* - Condition on mass loss
  - `calibration/calibrate-as19.py`
```

Everything is built on [PyTorch](https://pytorch.org) and
[PyTorch Lightning](https://lightning.ai), so emulator training and MCMC chains
run on CPUs, CUDA GPUs and Apple Silicon (MPS) alike.
