# API reference

Hand-grouped by subpackage. Each table is built with `autosummary` and
generates one page per symbol under `generated/`.

## Emulators

```{eval-rst}
.. currentmodule:: pism_emulator.emulators.nnemulator

.. autosummary::
    :toctree: generated/

    DNNEmulator
    NNEmulator
    NN5Emulator
    LegacyNNEmulator
    MLPBlock
    project_basis
```

## Data

```{eval-rst}
.. currentmodule:: pism_emulator.datasets

.. autosummary::
    :toctree: generated/

    PISMInterpolatedDataset
    PISMDataset
    LegacyPISMDataset
    DatasetConfig
    SamplesData
    TargetData
    inverse_y_transform_np
    parse_id_from_path
```

```{eval-rst}
.. currentmodule:: pism_emulator.datamodules

.. autosummary::
    :toctree: generated/

    PISMDataModule
    LegacyPISMDataModule
    PDDDataModule
    DataConfig
    PDDConfig
    EigCache
```

## MCMC

```{eval-rst}
.. currentmodule:: pism_emulator.mcmc.mala

.. autosummary::
    :toctree: generated/

    MALASamplerModule
    ReparametrizedMALASamplerModule
    ChainInitDataset
    run_sampling
    make_trainer_for_chains
```

```{eval-rst}
.. currentmodule:: pism_emulator.mcmc.writer

.. autosummary::
    :toctree: generated/

    DiskPredictionWriter
    load_pred_dir
    load_pred_dir_with_stats
```

```{eval-rst}
.. currentmodule:: pism_emulator.mcmc.plot_posterior

.. autosummary::
    :toctree: generated/

    load_and_stack_idatas
```

## Surface mass balance

```{eval-rst}
.. currentmodule:: pism_emulator.models.pdd

.. autosummary::
    :toctree: generated/

    PDD
    ReferencePDDModel
    PDDResult
    make_fake_climate
    make_fake_climate_2d
```

## Sampling, metrics and statistics

```{eval-rst}
.. currentmodule:: pism_emulator.lhs.draw

.. autosummary::
    :toctree: generated/

    draw_samples
```

```{eval-rst}
.. currentmodule:: pism_emulator.metrics

.. autosummary::
    :toctree: generated/

    area_weighted_error
    AreaWeightedError
```

```{eval-rst}
.. currentmodule:: pism_emulator.stats

.. autosummary::
    :toctree: generated/

    gelman_rubin
    kl_divergence
    calc_bic
    stepwise_bic
```

## Plotting and utilities

```{eval-rst}
.. currentmodule:: pism_emulator.plotting

.. autosummary::
    :toctree: generated/

    plot_compare
    plot_eigenglaciers
    plot_legacy_eigenglaciers
```

```{eval-rst}
.. currentmodule:: pism_emulator.utils

.. autosummary::
    :toctree: generated/

    load_imbie_csv
    distributions_as19
```
