# Citing and method overview

If pism-emulator contributes to a publication, please cite the method paper,
the software, and the underlying ice sheet model.

## Citing the method

{cite:t}`Aschwanden2022`:

```bibtex
@article{Aschwanden2022,
  author  = {Aschwanden, Andy and Brinkerhoff, Douglas J.},
  title   = {Calibrated Mass Loss Predictions for the {Greenland} Ice Sheet},
  journal = {Geophysical Research Letters},
  volume  = {49},
  pages   = {e2022GL099058},
  year    = {2022},
  doi     = {10.1029/2022GL099058},
}
```

## Citing the software

pism-emulator is archived on Zenodo:
[10.5281/zenodo.6257002](https://doi.org/10.5281/zenodo.6257002). Every GitHub
release gets its own version DOI there; cite the version you used.

## Citing PISM

PISM should be cited following the guidance on
[the PISM project website](https://www.pism.io/docs/index.html#citing-pism).

## Method overview

- **Emulation in a reduced basis.** Speed fields are projected onto
  *eigenglaciers*, the leading principal components of the weighted training
  ensemble; the neural network predicts their coefficients
  ({doc}`../features/emulators`).
- **Bootstrap ensembles.** Each emulator is trained with Dirichlet-distributed
  weights on the training runs, so an ensemble of emulators carries the
  emulator's own uncertainty.
- **Gradient-based posterior sampling.** The posterior of the flow parameters
  combines a Student-t likelihood of observed speeds with Beta priors and is
  sampled with manifold MALA {cite:p}`Girolami2011`
  ({doc}`../features/speed_calibration`).
- **Importance sampling on mass loss.** The calibrated PISM projections are
  weighted by their fit to observed cumulative mass change
  ({doc}`../features/importance_sampling`).
