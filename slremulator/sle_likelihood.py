#!/usr/bin/env python

# Copyright (C) 2019-2020 Andy Aschwanden
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

from glob import glob
import os
import pandas as pd
import pylab as plt
import seaborn as sns

files = glob("../data/response/dgmsl_rcp_*.csv")
dfs = []
for file in files:
    _, _, rcp, _, year = os.path.split(file)[-1].split(".csv")[0].split("_")
    df = pd.read_csv(file, dtype=float)
    df["RCP"] = rcp
    df["Year"] = int(year)
    dfs.append(df)
sle = pd.concat(dfs)

thresholds = ["10", "25", "50", "100"]

m_sles = []
for threshold in thresholds:
    df = sle[(sle["limnsw(cm)"] < int(threshold) + 0.5) & (sle["limnsw(cm)"] > int(threshold) - 0.5)]
    df["Threshold (cm SLE)"] = threshold
    m_sles.append(df)
m_sle = pd.concat(m_sles).reset_index(drop=True)

rcp_col_dict = {"CTRL": "k", "85": "#990002", "45": "#5492CD", "26": "#003466"}
rcp_shade_col_dict = {"CTRL": "k", "85": "#F4A582", "45": "#92C5DE", "26": "#4393C3"}
colors = list(rcp_col_dict.values())[1::]
fig = plt.figure()
ax = fig.add_subplot(111)

sns.boxenplot(
    data=m_sle,
    x="Year",
    y="Threshold (cm SLE)",
    hue="RCP",
    hue_order=["85", "45", "26"],
    order=thresholds[::-1],
    linewidth=0.15,
    showfliers=False,
    dodge=True,
    palette=colors,
    ax=ax,
)
fig.savefig("sle_likelihood.pdf", bbox_inches="tight")
