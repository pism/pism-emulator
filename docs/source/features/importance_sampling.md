# Mass-loss calibration

The second step of {cite:t}`Aschwanden2022` conditions projections on observed
mass loss. Parameter sets drawn from the flow posterior
({doc}`speed_calibration`) initialise a new PISM ensemble, which is run over
the observational period and into the future. Each member is then weighted by
how well its cumulative mass change matches the observations, here the IMBIE
Greenland record {cite:p}`IMBIE2020` (`calibration/imbie_greenland_2021_Gt.csv`,
read with {func}`~pism_emulator.utils.load_imbie_csv`).

## Weights

For member $m$ with simulated mass change $M_m(t)$ and observations
$M^\text{obs}(t) \pm \sigma(t)$ over the calibration period (2010–2020 by
default), the log-weight is the Gaussian log-likelihood

$$
\log w_m = -\frac{1}{2} \sum_t
\frac{\bigl(M_m(t) - M^\text{obs}(t)\bigr)^2}{\bigl(f\,\sigma(t)\bigr)^2},
$$

with a fudge factor $f$ (default 3) that inflates the observational uncertainty
for model error. The ensemble is resampled with probabilities proportional to
$w_m$, separately for each emissions scenario.

## Reproducing AS19

`calibration/calibrate-as19.py` applies this to the large ensemble of
{cite:t}`Aschwanden2019` and makes the probability plots of
{cite:t}`Aschwanden2022`. It needs no other data and no installation of the
package:

```bash
cd calibration
python calibrate-as19.py
```

`--as19_results_file`, `--as19_samples_file`, `--calibrated_results_file` and
`--calibrated_samples_file` point it at other ensembles. The notebooks in
`calibration/` apply the same procedure to an ISMIP6 ensemble.

```{admonition} TODO
- Move `resample_ensemble_by_data` from the script into the package so it can
  be used on any ensemble.
```
