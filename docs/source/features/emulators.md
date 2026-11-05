# Emulators

An emulator replaces PISM in the calibration: it maps a vector of $d$ flow
parameters $X$ to the modelled surface speed field on $n$ grid cells, in
milliseconds instead of hours, and it is differentiable, so the sampler can use
its gradient and Hessian.

## Training data

{class}`~pism_emulator.datasets.PISMInterpolatedDataset` reads the training
ensemble:

- one NetCDF file per PISM run, with `id_<N>_` in the file name
  (files without that pattern are skipped);
- a CSV of parameter values with an `id` column matching those `<N>`;
- the observed field (and its error) on the same grid, which also defines the
  mask of cells used.

The parameters are standardised (`X_mean`, `X_std`); the responses are
transformed with `--y-transform` and clamped to `--y-lim`:

```{list-table}
:header-rows: 1
:widths: 20 80

* - `--y-transform`
  - Meaning
* - `log10` (default)
  - $\log_{10}$ of the speed, after clamping to `--y-lim` (default `1 1e4`)
* - `robust`
  - centred and scaled by robust statistics
* - `none`
  - no transform
```

The same `--y-transform` and `--y-lim` must be used for training, evaluation
and sampling.

## Eigenglaciers

Speed fields are strongly correlated in space, so the emulator predicts them in
a reduced basis. With bootstrap weights $\omega_i$ on the $N$ training runs
$F_i$ {cite:p}`Aschwanden2022`,

$$
\bar F = \sum_i \omega_i F_i, \qquad
\tilde F = F - \bar F, \qquad
C = \tilde F^\top \operatorname{diag}(\omega)\, \tilde F = V \Lambda V^\top ,
$$

and the *eigenglaciers* are the columns of $\hat V = V_{:q} \Lambda_{:q}^{1/2}$,
the $q$ leading eigenvectors scaled by the square root of their eigenvalues.
{meth}`~pism_emulator.datamodules.PISMDataModule.get_eigenglaciers` computes
them with a low-rank SVD of $\operatorname{diag}(\sqrt{\omega n})\,\tilde F$
(the default) or an explicit eigendecomposition. `-q` (default 100) sets the
number of modes; `--cutoff 0.95` instead keeps the fewest modes that explain
95 % of the weighted variance.

```{image} ../../../images/eigenglaciers.png
:alt: The first six eigenglaciers
:width: 600px
```

The basis is cached in `svd_cache/eig_<hash>.pt`. The hash covers the
responses, the weights, `q`, `cutoff` and the decomposition, so every ensemble
member and every setting gets its own file and a stale basis is never reused.

## Architectures

Every emulator is a {class}`lightning.pytorch.LightningModule` whose
`forward(X, add_mean=False)` returns coefficients times $\hat V^\top$ (plus
$\bar F$ when `add_mean=True`):

```{list-table}
:header-rows: 1
:widths: 20 80

* - `--emulator`
  - Network
* - `DNN` (default)
  - {class}`~pism_emulator.emulators.nnemulator.DNNEmulator`: residual MLP
    blocks of `Linear → LayerNorm → Dropout → activation`; `--width` (128),
    `--depth` (4), `--dropout` (0.5), `--activation` (`relu`, `silu`, `gelu`),
    `--learning-rate` (0.01), optional `--compile` for {func}`torch.compile`
* - `NN`
  - {class}`~pism_emulator.emulators.nnemulator.NNEmulator`: four hidden
    layers of `--n-hidden` units with residual connections
* - `NN5`
  - {class}`~pism_emulator.emulators.nnemulator.NN5Emulator`: five hidden
    layers
* - `LegacyNN`
  - {class}`~pism_emulator.emulators.nnemulator.LegacyNNEmulator`: the
    network of {cite:t}`Aschwanden2022`, kept to load its checkpoints
```

All use Adam with an exponential learning-rate decay (factor 0.9975 per epoch)
and Kaiming-uniform initialisation.

```{admonition} Smooth activations and the sampler
:class: tip
With ReLU the emulator's curvature with respect to the parameters changes
abruptly from one region of parameter space to the next. The manifold sampler,
which uses the local Hessian, then has to take small steps. `--activation gelu`
or `silu` gives a smooth Hessian; see {doc}`../resources/sampler_comparison`.
```

## Loss and bootstrap ensemble

The loss is the area-weighted squared error

$$
\mathcal L = \sum_i \omega_i \sum_j a_j \bigl(\hat F_{ij} - F_{ij}\bigr)^2 ,
$$

with normalised cell areas $a_j$
({func}`~pism_emulator.metrics.area_weighted_error`). For `--model-index k` the
weights $\omega$ are drawn from a flat Dirichlet distribution with seed `k`, so
training `k = 0, 1, …, K-1` yields a bootstrap ensemble of emulators. During
validation, `train_loss` uses these weights and `test_loss` uniform weights.

## Evaluation

`pism-evaluate-emulator` compares the emulators with PISM on a random selection
of runs (`--sample-size`, default 80): it plots observed, emulated and
difference maps for four of them (speeds with the `cmg.speed` colormap from
[cmglaciology](https://pypi.org/project/cmglaciology/)) and reports the mean
absolute error, mean bias, Pearson $r$ and $R^2$.

```{image} ../../../images/speed_emulator_train.png
:alt: PISM vs emulator speeds
:width: 600px
```
