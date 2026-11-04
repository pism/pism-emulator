# Flow calibration with MALA

`pism-sample-posterior-speed` draws from the posterior of the flow parameters
given observed surface speeds, using a trained emulator as the forward model
({class}`~pism_emulator.mcmc.mala.MALASamplerModule`).

## Posterior

With $Y$ the observed speeds, $\hat Y(X)$ the emulated ones and
$\bar X = (X - X_\text{min}) / (X_\text{max} - X_\text{min})$ the parameters
scaled to the range of the training ensemble, the sampler targets
$\pi(X) \propto e^{-U(X)}$ with

$$
U(X) = -\alpha \sum_j \log t_\nu\!\left(Y_j - \hat Y_j(X);\ \hat\sigma_j\right)
       - \sum_k \log \mathrm{Beta}\!\left(\bar X_k;\ 3, 3\right).
$$

- $t_\nu(\cdot; \hat\sigma)$ is a Student-t density with $\nu$ degrees of
  freedom (`nu`, default 1) and scale $\hat\sigma$. In
  `pism-sample-posterior-speed`, $\hat\sigma_j$ is the observational error
  inflated by $(10\,\text{km} / \Delta x)^2$, which accounts for errors that
  are correlated over about 10 km.
- $\alpha$ (`--alpha`, default 0.01) weights the likelihood against the prior.
- The Beta(3, 3) priors keep the parameters within the range the emulator was
  trained on.

Chains start at the maximum a posteriori point, found with L-BFGS
({meth}`~pism_emulator.mcmc.mala.MALASamplerModule.find_MAP`).

## Proposals

`--proposal` selects how proposals are made. With $G(x) = |H(x)|$, the
absolute value of the Hessian of $U$ (computed by eigendecomposition) and step
size $h$, the `manifold` and `fixed` proposals are

$$
x' \sim \mathcal N\!\left(x - h\, G^{-1} \nabla U(x),\ 2h\, G^{-1}\right),
$$

accepted with the Metropolis–Hastings ratio
$\exp\bigl(U(x) - U(x')\bigr)\, q(x \mid x') / q(x' \mid x)$.

```{list-table}
:header-rows: 1
:widths: 13 47 20 20

* - `--proposal`
  - What it does
  - Cost per step
  - Default `--acc-target`
* - `manifold` (default)
  - Simplified manifold MALA {cite:p}`Girolami2011`: $G$ recomputed at every
    state, and the reverse density uses $G(x')$, so the chain is exact although
    the proposal changes with position.
  - one Hessian
  - 0.25
* - `fixed`
  - MALA preconditioned with $G$ computed once per chain at its start (the MAP)
    and kept. Exact and much cheaper; adapts less to posteriors whose curvature
    changes strongly.
  - one gradient
  - 0.57 {cite:p}`Roberts1998`
* - `legacy`
  - The sampler before v0.43: no drift, and the reverse density reuses $G(x)$
    although $G$ is recomputed at every accepted state. Not reversible, so its
    samples are slightly biased; use it only to reproduce earlier runs (it does,
    draw for draw). `delayed_accept` and `hess_refresh > 1` exist only here.
  - one Hessian per accepted step
  - 0.25
```

{doc}`../resources/sampler_comparison` compares the three on the Greenland
calibration.

## Step-size adaptation

During burn-in (`--burn`) the step size starts at 0.1 and is adapted towards
the acceptance target (`--acc-target`), within `[1e-3, --h-max]` (default
`--h-max 1.0`), by an exponential moving average of the acceptance rate. Dual
averaging {cite:p}`Hoffman2014` is available through
`adapt_method="dual"` in the Python API. After burn-in the step size is fixed,
and `--samples` draws per chain are kept.

## Chains and devices

`--chains N` runs `N` independent chains from the same MAP. With
`--accelerator cpu` and `N > 1` they run in parallel, one process per chain
(`ddp_spawn`); each process writes its chain to `./_preds/` before the driver
assembles them. On a GPU or MPS device the chains run one after another.

## Output

`<emulator-dir>/posterior/X_posterior_model_<model-index>.nc` is an
[ArviZ](https://python.arviz.org) data tree with three groups:

- `posterior`: the draws in physical units, `(chain, draw)` per parameter;
- `prior`: Beta prior draws in the same shape;
- `sample_stats`: `lp` (log posterior of each kept state), `step` (step size)
  and `accept` (whether the draw was an accepted move).

A trace plot is written next to it. `pism-plot-posterior` stacks the posteriors
of several emulators and plots their marginals. Check R-hat and the effective
sample size before using the draws, e.g. with `arviz.summary`.
