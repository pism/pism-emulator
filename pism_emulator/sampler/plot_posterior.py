#!/bin/env python3
# Copyright (C) 2021-22 Andy Aschwanden, Douglas C Brinkerhoff
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

from argparse import ArgumentParser
from os.path import join
from pathlib import Path
from typing import Sequence

import arviz as az
import matplotlib as mpl
import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import NullFormatter
from scipy.stats import beta, gaussian_kde

from pism_emulator.utils import param_keys_dict as keys_dict

fontsize = 6
lw = 1.0
aspect_ratio = 1
markersize = 2

rcparams = {
    "backend": "ps",
    "axes.linewidth": 0.25,
    "lines.linewidth": lw,
    "axes.labelsize": fontsize,
    "font.size": fontsize,
    "xtick.direction": "in",
    "xtick.labelsize": fontsize,
    "xtick.major.size": 2.5,
    "xtick.major.width": 0.25,
    "ytick.direction": "in",
    "ytick.labelsize": fontsize,
    "ytick.major.size": 2.5,
    "ytick.major.width": 0.25,
    "legend.fontsize": fontsize,
    "lines.markersize": markersize,
    "font.size": fontsize,
}


def _group_to_df(ds, label, cols):
    df = ds[cols].to_dataframe().reset_index()  # gives columns: chain, draw, <vars>
    df["ensemble"] = label
    return df


def load_and_stack_idatas(
    paths: Sequence[str | Path],
    labels: Sequence[str] | None = None,
    dim_name: str = "model",
) -> az.InferenceData:
    """
    Load multiple InferenceData .nc files and stack them along a new dimension (e.g. 'model').
    Concatenates all present groups (posterior, prior, sample_stats, …) across files.
    """
    paths = [Path(p) for p in paths]
    if labels is None:
        labels = [str(i) for i in range(len(paths))]
    if len(labels) != len(paths):
        raise ValueError("labels and paths must have the same length")

    # Expand each idata by adding a new length-1 dim (dim_name)
    def _expanded(idata: az.InferenceData, label: str) -> az.InferenceData:
        group_ds: dict[str, xr.Dataset] = {}
        for group in idata._groups_all:
            ds = getattr(idata, group, None)
            if ds is None:
                continue
            ds2 = ds.expand_dims({dim_name: [label]})
            group_ds[group] = ds2
        return az.InferenceData(**group_ds)

    expanded = [_expanded(az.from_netcdf(p), lab) for p, lab in zip(paths, labels)]

    # Concatenate per group along the new dim
    groups = set().union(*(set(e._groups_all) for e in expanded))
    concatenated: dict[str, xr.Dataset] = {}
    for g in groups:
        dses = [getattr(e, g, None) for e in expanded]
        dses = [ds for ds in dses if ds is not None]
        if dses:
            concatenated[g] = xr.concat(dses, dim=dim_name)

    return az.InferenceData(**concatenated)


params = [
    "basal_resistance.pseudo_plastic.q",
    "basal_yield_stress.mohr_coulomb.till_effective_fraction_overburden",
    "basal_resistance.pseudo_plastic.u_threshold",
    "basal_yield_stress.mohr_coulomb.till_phi_default",
    "stress_balance.blatter.enhancement_factor",
    "stress_balance.blatter.Glen_exponent",
]

plt.rcParams.update(rcparams)

if __name__ == "__main__":
    __spec__ = None  # type: ignore

    parser = ArgumentParser()
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("POSTERIOR_FILES", nargs="*", help="Posterior samples")

    args = parser.parse_args()

    frac = args.fraction
    posterior_files = args.POSTERIOR_FILES

    idata = load_and_stack_idatas(posterior_files, dim_name="model")

    # keep only vars present in both groups (avoids KeyError)
    vars_common = [
        v
        for v in idata.posterior.data_vars
        if (v in idata.posterior.data_vars) and (v in idata.prior.data_vars)
    ]

    df_post = _group_to_df(idata.posterior, "Posterior", vars_common)
    df_prior = _group_to_df(idata.prior, "Prior", vars_common)

    df = pd.concat([df_prior, df_post], ignore_index=True).sample(frac=frac)
    df = (
        df[["chain", "draw", "model"] + vars_common + ["ensemble"]]
        .set_index(["chain", "draw"])
        .reset_index(drop=True)
    ).rename(columns={k: keys_dict[k] for k in params})

    with mpl.rc_context(rcparams):
        g = sns.pairplot(
            df[df["ensemble"] == "Posterior"], hue="model", palette="crest"
        )
        g.fig.set_size_inches(6.4, 6.4)  # (width, height) in inches
        g.fig.tight_layout()
        g.fig.savefig("test.png", dpi=300)
        g = sns.pairplot(df, hue="ensemble", palette="crest")
        g.fig.set_size_inches(6.4, 6.4)  # (width, height) in inches
        g.fig.tight_layout()
        g.fig.savefig("test_prior_posterior.png", dpi=300)
