# pism-emulator

[![License: GPL-3.0](https://img.shields.io:/github/license/pism/pism-emulator)](https://opensource.org/licenses/GPL-3.0)

## Introduction

pism-emulator is the codebase for the two-step Bayesian calibration process proposed by Aschwanden and Brinkerhoff [1]. The goal is to condition ensemble predictions of Greenland's contribution to future sea-level on contemporary [surface speeds](https://nsidc.org/data/NSIDC-0670/versions/1) and [cumulative mass loss](http://imbie.org).

## Methods

1. We first calibrate ice dynamics parameters using artificial neural network to act as a surrogate for the ice flow model, which provides a mapping from ice flow parameters to surface speeds at a fraction of PISM's computational cost. This method also provides us the means to employ efficient statistical methods that rely on the gradients of model outputs with respect to its input. We then find the joint marginal distributions given surface speed observations [Metropolis-adjusted Langevin algorithm](https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm) (MALA), a Markov-chain Monte Carlo Method. Both the surrogate model and the MALA sampler are implemented in [PyTorch](https://pytorch.org) and [PyTorch-Lightning](https://www.pytorchlightning.ai) and can be run on GPUs.

2. In the second step, we use Bayesian calibration, also known as [Importance Sampling](https://en.wikipedia.org/wiki/Importance_sampling) to condition ensemble members on observations of cumulative mass change.

## Basic usage: Bayesian calibration

To perform the Bayesian calibration and reporduce the probability plots from the manuscript, run `python calibrate-as19.py` in `calibration`. No need to download additional data.

## Installation

1. Download the repository with `git clone https://github.com/pism/pism-emulator`.

2. Install repository and dependencies with `python setup.py install --user`

3. Download observations of [surface speeds](https://nsidc.org/data/NSIDC-0670/versions/1) to `data/observed_speeds` by running `01_download_nsidc_0670.py` and `02_convert.sh`.

## Procedure

![The first six eigen-glaciers](https://github.com/pism/pism-emulator/blob/master/images/eigenglaciers.png)

![PISM vs Emulator](https://github.com/pism/pism-emulator/blob/master/images/speed_emulator_train.png)


The two-step Bayesian calibration requires requires running the high-fidelity ice sheet model twice, first to create the training data, and second to perform the calibrated projections. Below we lay out the steps required to reproduce the results.

1. Generate training data for flow calibration. The current implementation uses training data generated by the [Parallel Ice Sheet Model](https://pism.io) although it can be adapted to work with other ice sheet models as well. Generating training data may require access to an HPC system as running the high-fidelity model is computationally. Here you can download the training data from [arcticdata.io](https://arcticdata.io).

2. In `speedemulator`, run `python train_emulator.py`. `python train_emulator.py -h` lists options available. `python train_emulator.py --gpus 1 --data_dir dir-with-training-data --emulator_dir dir-where-emulator-goes --target_file ../data/observed_speeds/greenland_vel_mosaic250_v1_g1800m.nc` trains model 0 on 1 GPU, with training data in `dir-with-training-data` and writes the trained emulator to `dir-where-emulator-goes`.

3. In `speedemulator`, run `sample_posterior.py --device cuda --emulator_dir dir-where-emulator-goes --target_file ../data/observed_speeds/greenland_vel_mosaic250_v1_g1800m.nc` to run the sampler on a GPU.

4. (optional) Plot the posterior distributions with `plot_posterior.py`.

5. Sample from posterior to create new ensemble for the high fidelilty model.

6. Run high-fidelity model on HPC system, generate csv file with timeseries of cumulative mass, discharge and smb rates.

7. In `calibration` run `python calibrate-as19.py` to perform the importance sampling.

## Minimal Example

No need to download large data sets. A low-resolution working example is provided as a notebook [`minimal_example.ipynb`](https://github.com/pism/pism-emulator/blob/master/notebooks/minimal_example.ipyng)



## References

Aschwanden, A. and D. J. Brinkerhoff: Calibrated mass loss projections from the Greenland Ice Sheet (in prep.)
