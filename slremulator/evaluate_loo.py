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

from sklearn.metrics import mean_squared_error, r2_score

default_les_directory = "emulator_results"
default_loo_directory = "loo_results"


def r2_f(df):

    return r2_score(df["Y_mean"], df["Y_true"])


def s_res(df):
    """
    Standardized residual
    """

    df["sres"] = (df["Y_mean"] - df["Y_true"]) / df["Y_var"]
    return df["sres"]


def distance_f(df):

    return np.sqrt(np.sum(df["distance"]))


def rmsd_f(df):

    return rmsd(df["Y_mean"], df["Y_true"])


def nrmsd_f(df):

    Q1 = df["Y_true"].quantile(0.25)
    Q3 = df["Y_true"].quantile(0.75)
    IQR = Q3 - Q1

    return rmsd(df["Y_mean"], df["Y_true"]) / IQR


def mse_f(df):

    return np.sum(np.sqrt(df["Y_var"])) / len(df)


def kldiv_f(df):
    """
    Calculate the histogram Q of df["Y_mean"] and then
    return the KL-divergence of (P, Q).
    """
    Q = np.histogram(df["Y_mean"], bins=bins, density=True)[0]
    return kl_divergence(P, Q)


def hist_f(df):
    """
    Calculate the histogram Q of df["Y_mean"].
    """
    Q = np.histogram(df["Y_mean"], bins=bins, density=True)[0]
    return Q


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


def evaluate_loo_distance(df):
    """
    Evaluate LOO using the distance
    """

    print("\nLeave-one-out validation")
    print("-----------------------------------\n\n")

    for n in range(100, 600, 100):
        print(f"Using {n} LHS samples:")
        d_df = df[df.n_lhs == n].reset_index(drop=True)
        d_df = d_df.groupby(["method", "n_lhs"]).apply(distance_f)
        d_df = d_df.to_frame(name="distance").reset_index()
        # Make sure n_lhs is int
        d_df = d_df.astype({"n_lhs": "int"})
        print(d_df.sort_values(["distance"]))
        print("\n")


def evaluate_kldiv(df):
    """
    Evaluate the KL divergence for AS19
    """

    df = df[df.n_lhs == 500].reset_index(drop=True)
    d_df = df.groupby(["method", "n_lhs"]).apply(kldiv_f)
    d_df = d_df.to_frame(name="kldiv").reset_index()
    # Make sure n_lhs is int
    d_df = d_df.astype({"n_lhs": "int"})
    print("\nKL-div validation")
    print("-----------------------------------")
    print(d_df.sort_values(["kldiv"]))


def evaluate_loo_nrmsd(df):
    """
    Evaluate LOO using the normalized RMSD
    """

    d_df = df.groupby(["method", "n_lhs"]).apply(nrmsd_f)
    d_df = d_df.to_frame(name="rmsd").reset_index()
    # Make sure n_lhs is int
    d_df = d_df.astype({"n_lhs": "int"})
    print("\nNRMSD validation")
    print("-----------------------------------")
    print(d_df.sort_values(["rmsd"]))


def evaluate_rmsd(df):
    """
    Evaluate RMSD
    """

    d_df = df.groupby(["method", "n_lhs"]).apply(nrmsd_f)
    d_df = d_df.to_frame(name="rmsd").reset_index()
    # Make sure n_lhs is int
    d_df = d_df.astype({"n_lhs": "int"})
    print("\nNRMSD validation")
    print("-----------------------------------")
    print(d_df.sort_values(["rmsd"]))


if __name__ == "__main__":

    __spec__ = None

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Validate Regression Methods."
    parser.add_argument(
        "-b", "--bin_width", dest="bin_width", help="Width of histogram bins. Default=1cm", default=1.0
    )
    parser.add_argument(
        "-n",
        dest="n_samples_validation",
        choices=[20, 170, 240, 320, 400],
        type=int,
        help="Number of validation samples.",
        default=400,
    )
    parser.add_argument(
        "--les_dir",
        dest="lesdir",
        help=f"Directory where the LES files are. Default = {default_les_directory}.",
        default=default_les_directory,
    )
    parser.add_argument(
        "--loo_dir",
        dest="loodir",
        help=f"Directory where the LOO files are. Default = {default_loo_directory}.",
        default=default_loo_directory,
    )

    options = parser.parse_args()
    bin_width = options.bin_width
    n_samples_validation = options.n_samples_validation
    lesdir = options.lesdir
    loodir = options.loodir
    rcp = 45
    year = 2100
    n_lhs_samples = 500

    loo_files = glob(f"{loodir}/loo_*.csv")
    loo_df = load_data_frames(loo_files)

    evaluate_loo_distance(loo_df)
