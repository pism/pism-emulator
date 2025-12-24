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

"""
Plot posteriors.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Sequence

import arviz as az
import matplotlib as mpl
import pandas as pd
import pylab as plt
import seaborn as sns
import xarray as xr

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


def _group_to_df(ds: xr.Dataset, label: str, cols: Sequence[str]) -> pd.DataFrame:
    """
    Convert an ArviZ group dataset to a tidy DataFrame and annotate its ensemble label.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset corresponding to a single ArviZ group (e.g., ``idata.posterior``).
        Expected to include sampling dimensions such as ``chain`` and ``draw``.
    label : str
        Label identifying the ensemble/model this dataset came from.
    cols : Sequence[str]
        Variable names in ``ds`` to include in the output DataFrame.

    Returns
    -------
    pandas.DataFrame
        Tidy DataFrame with sampling coordinates (e.g., ``chain``, ``draw``) and
        the requested variables in columns, plus an ``ensemble`` column set to
        ``label``.
    """
    df = ds[cols].to_dataframe().reset_index()  # gives columns: chain, draw, <vars>
    df["ensemble"] = label
    return df


def load_and_stack_idatas(
    paths: Sequence[str | Path],
    labels: Sequence[str] | None = None,
    dim_name: str = "model",
) -> az.InferenceData:
    """
    Load multiple ArviZ InferenceData NetCDF files and stack them along a new dimension.

    Each input file is loaded with :func:`arviz.from_netcdf`. For every group present
    in each :class:`arviz.InferenceData` (e.g., ``posterior``, ``prior``,
    ``sample_stats``), this function inserts a new length-1 dimension named
    ``dim_name`` (default: ``"model"``) with coordinate value given by the
    corresponding entry in ``labels``. Groups are then concatenated across inputs
    along ``dim_name``.

    Parameters
    ----------
    paths : Sequence[str or pathlib.Path]
        Paths to ArviZ InferenceData NetCDF files (``.nc``).
    labels : Sequence[str], optional
        Labels corresponding to each path. These become the coordinate values for
        the new dimension ``dim_name``. If None (default), labels are generated as
        ``"0"``, ``"1"``, ..., ``str(len(paths)-1)``.
    dim_name : str, optional
        Name of the new dimension used to stack/concatenate the InferenceData
        objects. Default is ``"model"``.

    Returns
    -------
    arviz.InferenceData
        A single InferenceData object whose groups have been concatenated along
        ``dim_name``. Only groups present in at least one input are included.

    Raises
    ------
    ValueError
        If ``labels`` is provided and ``len(labels) != len(paths)``.

    Notes
    -----
    * This function iterates over the groups reported by the private attribute
      ``idata._groups_all``. If ArviZ changes its internal API, this may need to
      be updated. A more future-proof approach is to use the public group names
      available via ``idata.groups()`` when appropriate.
    * If different files contain different sets of groups, groups are concatenated
      only where they exist; missing groups are simply omitted for that file.
    """
    paths = [Path(p) for p in paths]
    if labels is None:
        labels = [str(i) for i in range(len(paths))]
    if len(labels) != len(paths):
        raise ValueError("labels and paths must have the same length")

    def _expanded(idata: az.InferenceData, label: str) -> az.InferenceData:
        """
        Add a new length-1 stacking dimension to every group dataset in an InferenceData.

        Parameters
        ----------
        idata : arviz.InferenceData
            Input inference data.
        label : str
            Coordinate value to use for the new dimension ``dim_name``.

        Returns
        -------
        arviz.InferenceData
            New InferenceData with group datasets expanded along ``dim_name``.
        """
        group_ds: dict[str, xr.Dataset] = {}
        for group in idata._groups_all:
            ds = getattr(idata, group, None)
            if ds is None:
                continue
            ds2 = ds.expand_dims({dim_name: [label]})
            group_ds[group] = ds2
        return az.InferenceData(**group_ds)

    expanded = [_expanded(az.from_netcdf(p), lab) for p, lab in zip(paths, labels)]

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

    with mpl.rc_context(rc=rcparams):
        # variance across chain & draw

        axes = az.plot_trace(
            idata.rename_vars({k: keys_dict[k] for k in params}),
            hist_kwargs={"bins": 50},
            figsize=(6.4, 8.4),
        )  # <-- key fix: kind/hist_kwargs at top level

        fig = axes.flatten()[0].get_figure()

        fig.suptitle("Posterior Traces")
        fig.savefig("traces.png", dpi=300)

    with mpl.rc_context(rcparams):
        g = sns.pairplot(
            df[df["ensemble"] == "Posterior"],
            hue="model",
            palette="crest",
        )
        g.fig.set_size_inches(6.4, 6.4)  # (width, height) in inches
        g.fig.tight_layout()
        g.fig.savefig("test.png", dpi=300)
        g = sns.pairplot(
            df,
            hue="ensemble",
            hue_order=["Prior", "Posterior"],
            palette=["#97a6c4", "#384860"],
        )
        g.fig.set_size_inches(6.4, 6.4)  # (width, height) in inches
        g.fig.tight_layout()
        g.fig.savefig("test_prior_posterior.png", dpi=300)
