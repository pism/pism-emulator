#!/usr/bin/env python

# Copyright (C) 2019 Rachel Chen, Andy Aschwanden
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

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from functools import reduce
from glob import glob
import GPy as gp
from math import sqrt
import numpy as np
import os
import re
import pandas as pd
import pylab as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error
import sys

from pismemulator.utils import golden_ratio
from pismemulator.utils import kl_divergence
from pismemulator.utils import prepare_data
from pismemulator.utils import rmsd
from pismemulator.utils import set_size

from sklearn.metrics import mean_squared_error

default_les_directory = "emulator_results"
default_loo_directory = "loo_results_dp16"


def distance_f(df):

    return np.sqrt(np.sum(df["distance"]))


def rmsd_f(df):

    return rmsd(df["Y_mean"], df["Y_true"])


def load_data_frames(csv_files):
    """
    Load results and return a pandas.DataFrame
    """
    dfs = []
    for k, response_file in enumerate(csv_files):
        df = pd.read_csv(response_file, delimiter=",", squeeze=True, skipinitialspace=True).sort_values(by="id")
        dfs.append(df)

    m_df = pd.concat(dfs, sort=False)
    m_df.reset_index(inplace=True, drop=True)

    return m_df


if __name__ == "__main__":

    __spec__ = None

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Validate Regression Methods."
    parser.add_argument(
        "--loo_dir",
        dest="loodir",
        help=f"Directory where the LOO files are. Default = {default_loo_directory}.",
        default=default_loo_directory,
    )

    options = parser.parse_args()
    loodir = options.loodir

    # Load the "True" data.
    dp16_df = pd.read_csv("../data/dp16/dp16.csv")
    samples = dp16_df[["OCFAC", "CREVLIQ", "VCLIF", "BIAS"]]
    # response = dp16_df[[simulation]]

    loo_files = glob(f"{loodir}/loo_*.csv")
    loo_df = load_data_frames(loo_files)

    simulations = ["LIG", "PLIO", "RCP45_pres", "RCP26_2100", "RCP45_2100", "RCP85_2100"]
    dfs = []
    for simulation in simulations:
        df = dp16_df[["id", simulation]]
        df["id"] = df.index
        df["simulation"] = simulation
        df = df.rename(columns={simulation: "Y_true"})
        dfs.append(df)
    r_true = pd.concat(dfs)

    m_df = pd.merge(r_true, loo_df, left_on=["id", "simulation"], right_on=["loo_id", "simulation"])
    m_df = m_df.drop(columns=["id_x", "id_y", "loo_id"])

    kendall_tau_df = m_df.groupby(["method", "simulation"])["Y_true"].corr(m_df["Y_mean"], method="kendall")
    kendall_tau_df = kendall_tau_df.to_frame(name="tau").reset_index()

    # Calculate the RMSD between Y_true and the emulator
    rmsd_df = m_df.groupby(["method", "simulation"]).apply(rmsd_f)
    rmsd_df = rmsd_df.to_frame(name="rmsd").reset_index()

    distance_df = m_df.groupby(["method", "simulation"]).apply(distance_f)
    distance_df = distance_df.to_frame(name="distance").reset_index()

    for simulation in simulations:
        print(simulation)
        df = distance_df[distance_df["simulation"] == simulation]
        print(df.sort_values(["distance"]))
        df = kendall_tau_df[kendall_tau_df["simulation"] == simulation]
        print(df.sort_values(["tau"]))
        df = rmsd_df[rmsd_df["simulation"] == simulation]
        print(df.sort_values(["rmsd"]))
