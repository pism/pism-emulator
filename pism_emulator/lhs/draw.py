#!/bin/env python3
# Copyright (C) 2021-25 Andy Aschwanden, Douglas C Brinkerhoff
#
# This file is part of pism-emulator.
#
# PISM-EMULATOR is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# PISM-EMULATOR is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with PISM; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

# pylint: disable=too-many-statements,too-many-branches,redefined-builtin
"""
Latin-Hypercube Sampling.
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
from scipy.stats.distributions import uniform


def draw_samples(n_samples: int = 10_000, random_seed: int = 2) -> pd.DataFrame:
    """
    Draw Latin-hypercube samples for PDD model parameters.

    Samples are generated in the unit hypercube using Latin Hypercube Sampling
    (LHS) and then transformed to user-specified parameter distributions using
    each distribution's percent-point function (PPF; inverse CDF).

    Parameters
    ----------
    n_samples : int, optional
        Number of parameter sets to draw. Default is 10_000.
    random_seed : int, optional
        Seed for the random number generator used by the sampling routine.
        Default is 2.

    Returns
    -------
    pandas.DataFrame
        DataFrame with shape ``(n_samples, 6)`` containing the sampled parameters.
        Columns (in order) are:

        - ``pdd_factor_snow`` : snow degree-day factor (uniform in [1, 6])
        - ``pdd_factor_ice``  : ice degree-day factor (uniform in [3, 15])
        - ``refreeze_snow``   : snow refreezing fraction (uniform in [0, 0.8])
        - ``refreeze_ice``    : ice refreezing fraction (uniform in [0, 0.8])
        - ``temp_snow``       : snow/rain transition lower bound in °C (uniform in [-2, 0])
        - ``temp_rain``       : snow/rain transition upper bound in °C (uniform in [0, 4])

    Notes
    -----
    * LHS produces stratified samples over each dimension of the unit hypercube.
    * Uses SciPy's ``scipy.stats.qmc.LatinHypercube`` with a seeded RNG for
      deterministic output.
    """
    distributions = {
        "pdd_factor_snow": uniform(loc=1.0, scale=5.0),   # [1, 6]
        "pdd_factor_ice": uniform(loc=3.0, scale=12.0),   # [3, 15]
        "refreeze_snow": uniform(loc=0.0, scale=0.8),     # [0, 0.8]
        "refreeze_ice": uniform(loc=0.0, scale=0.8),      # [0, 0.8]
        "temp_snow": uniform(loc=-2.0, scale=2.0),        # [-2, 0]
        "temp_rain": uniform(loc=0.0, scale=4.0),         # [0, 4]
    }
    keys = list(distributions.keys())
    d = len(keys)

    # SciPy QMC uses a NumPy Generator for reproducibility
    rng = np.random.default_rng(random_seed)
    sampler = qmc.LatinHypercube(d=d, seed=rng)  # you can also pass seed=random_seed
    unif_sample = sampler.random(n=n_samples)    # shape (n_samples, d)

    dist_sample = np.empty_like(unif_sample)
    for i, key in enumerate(keys):
        dist_sample[:, i] = distributions[key].ppf(unif_sample[:, i])

    return pd.DataFrame(dist_sample, columns=keys)


