```{image} _static/pism_logo_transp.png
:alt: pism-emulator
:width: 240px
:align: left
:class: only-light
```

**pism-emulator** implements the two-step Bayesian calibration of
{cite:t}`Aschwanden2022` for ice sheet models such as
[PISM](https://www.pism.io) (the Parallel Ice Sheet Model). A neural network
emulator trained on an ensemble of PISM simulations stands in for the ice flow
model, a Metropolis-adjusted Langevin (MALA) sampler conditions the flow
parameters on observed surface speeds, and importance sampling conditions the
resulting projections on observed mass loss. The method is model-agnostic:
other calibration targets and other ice sheet models work the same way.

The structure of this documentation follows
[pism-terra](https://github.com/pism/pism-terra), which was in turn inspired by
Romain Hugonnet's [xDEM](https://xdem.readthedocs.io).

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Quick start
:link: getting_started/quick_start
:link-type: doc

Train an emulator and sample the speed posterior.
:::

:::{grid-item-card} {octicon}`book` Features
:link: features/emulators
:link-type: doc

Emulators, MALA sampling, the PDD model and importance sampling.
:::

:::{grid-item-card} {octicon}`code-square` API reference
:link: reference/api
:link-type: doc

Every public class and function, grouped by subpackage.
:::
::::

```{toctree}
:caption: Getting started
:hidden:

getting_started/about
getting_started/installation
getting_started/quick_start
getting_started/citing
```

```{toctree}
:caption: Features
:hidden:

features/emulators
features/speed_calibration
features/pdd
features/importance_sampling
```

```{toctree}
:caption: Resources
:hidden:

resources/sampler_comparison
```

```{toctree}
:caption: Developer's Corner
:hidden:

developer/documentation
developer/testing
developer/releases
```

```{toctree}
:caption: Reference
:hidden:

reference/api
reference/cli
reference/release_notes
```

```{toctree}
:caption: Project information
:hidden:

project/publications
project/credits
references
```
