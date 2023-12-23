# Copyright (C) 2019-21 Andy Aschwanden
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

# utils.py contains generic functions to read data or perform statistical analyses.

import sys
from math import sqrt
from os import mkdir
from os.path import isdir, join
from typing import Union

import numpy as np
import pandas as pd
import pylab as plt
import xarray as xr
from matplotlib.colors import LogNorm
from pyDOE2 import lhs
from SALib.sample import saltelli
from scipy.stats.distributions import gamma, randint, truncnorm, uniform
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(0)


param_keys_dict = {
    "GCM": "GCM (1)",
    "FICE": "$f_i$ (mm K$^{-1}$ day$^{-1}$)",
    "FSNOW": "$f_s$ (mm K$^{-1}$ day$^{-1}$)",
    "RFR": "$\psi (1)$",
    "PRS": "$\omega$ (% K$^{-1}$)",
    "OCM": "$m_{t}$ (1)",
    "OCS": "$m_{x}$ (1)",
    "TCT": "$h_{\mathrm{min}}$ (1)",
    "VCM": "$\sigma_{\mathrm{max}}$ (MPa)",
    "SIAE": "$E_{\mathrm{SIA}}$ (1)",
    "SSAN": "$n_{\mathrm{SSA}}$ (1)",
    "TEFO": "$\delta$ (1)",
    "PPQ": "$q$ (1)",
    "PHIMIN": "$\phi_{\mathrm{min}}$ ($^{\circ}$)",
    "PHIMAX": "$\phi_{\mathrm{max}}$ ($^{\circ}$)",
    "ZMIN": "$z_{\mathrm{min}}$ (m)",
    "ZMAX": "$z_{\mathrm{max}}$ (m)",
    "a_glen": "A (Pa^{-n} s^{-1})",
    "sia_e": "$E_{\mathrm{SIA}}$ (1)",
    "ssa_e": "$E_{\mathrm{SSA}}$ (1)",
    "ssa_n": "$n_{\mathrm{SSA}}$ (1)",
    "ppq": "$q$ (1)",
    "tefo": "$\delta$ (1)",
    "till_effective_fraction_overburden": "$\delta$ (1)",
    "pseudo_plastic_uthershold": "u_{\mathrm{thr} (m yr^{-1})}",
    "phi_min": "$\phi_{\mathrm{min}}$ ($^{\circ}$)",
    "z_min": "$z_{\mathrm{min}}$ (m)",
    "z_max": "$z_{\mathrm{max}}$ (m)",
    "pseudo_plastic_uthreshold": "$u_{\mathrm{th}}$ (m yr$^{-1}$)",
    "SIAe": "$E_{\mathrm{SIA}}$ (1)",
    "SSAe": "$E_{\mathrm{SSA}}$ (1)",
    "topg_to_phi_base": "$b_{\mathrm{base}}$ (m)",
    "topg_to_phi_range": "$b_{\mathrm{range}}$ (m)",
}


def load_hirham_climate(file="DMI-HIRHAM5_1980_MM.nc", thinning_factor=1):
    """
    Read and return Obs
    """

    with xr.open_dataset(file) as Obs:
        stacked = Obs.stack(z=("rlat", "rlon"))
        ncl_stacked = Obs.stack(z=("ncl4", "ncl5"))

        temp = stacked.tas.dropna(dim="z").values - 273.15
        rainfall = stacked.rainfall.dropna(dim="z").values * 365.242198781 / 1000
        snowfall = stacked.snfall.dropna(dim="z").values * 365.242198781 / 1000
        smb = stacked.gld.dropna(dim="z").values * 365.242198781 / 1000 / 12
        refreeze = ncl_stacked.rfrz.dropna(dim="z").values * 365.242198781 / 1000 / 12
        melt = stacked.snmel.dropna(dim="z").values * 365.242198781 / 1000 / 12
        runoff = stacked.rogl.dropna(dim="z").values * 365.242198781 / 1000 / 12
        precip = rainfall + snowfall

    return (
        temp[..., ::thinning_factor],
        precip[..., ::thinning_factor],
        snowfall.sum(axis=0)[::thinning_factor],
        melt.sum(axis=0)[::thinning_factor],
        runoff.sum(axis=0)[::thinning_factor],
        refreeze.sum(axis=0)[::thinning_factor],
        smb.sum(axis=0)[::thinning_factor],
    )


def load_hirham_climate_w_std_dev(
    file="DMI-HIRHAM5_1980_2020_MMS.nc", thinning_factor=1
):
    """
    Read and return HIRHAM5 data grouped by year

    n: monthly forcing (12

    Returns

    temp (n, m) array
    precip (n, m) array
    std_dev (n, m) array
    a (1, m) array
    m (1, m) array
    r (1, m) array
    f (1, m) array
    b (1, m) array

    """

    with xr.open_dataset(file) as Obs:
        stacked = Obs.stack(z=("rlat", "rlon"))
        ncl_stacked = Obs.stack(z=("ncl4", "ncl5"))

        temp = (
            np.hstack(
                [d.dropna(dim="z").values for _, d in stacked.tas.groupby("time.year")]
            )
            - 273.15
        )
        temp_std_dev = np.hstack(
            [
                d.dropna(dim="z").values
                for _, d in stacked.tas_std_dev.groupby("time.year")
            ]
        )
        rainfall = (
            np.hstack(
                [
                    d.dropna(dim="z").values
                    for _, d in stacked.rainfall.groupby("time.year")
                ]
            )
            * 365.242198781
            / 1000
        )
        snowfall = (
            np.hstack(
                [
                    d.dropna(dim="z").values
                    for _, d in stacked.snfall.groupby("time.year")
                ]
            )
            * 365.242198781
            / 1000
        )
        smb = (
            np.hstack(
                [d.dropna(dim="z").values for _, d in stacked.gld.groupby("time.year")]
            )
            * 365.242198781
            / 1000
            / 12
        )
        refreeze = (
            np.hstack(
                [
                    d.dropna(dim="z").values
                    for _, d in ncl_stacked.rfrz.groupby("time.year")
                ]
            )
            * 365.242198781
            / 1000
            / 12
        )
        snowmelt = (
            np.hstack(
                [
                    d.dropna(dim="z").values
                    for _, d in stacked.snmel.groupby("time.year")
                ]
            )
            * 365.242198781
            / 1000
            / 12
        )
        snowdepth = np.hstack(
            [d.dropna(dim="z").values for _, d in stacked.sn.groupby("time.year")]
        )
        runoff = (
            np.hstack(
                [d.dropna(dim="z").values for _, d in stacked.rogl.groupby("time.year")]
            )
            * 365.242198781
            / 1000
            / 12
        )
        precip = rainfall + snowfall

        obs = {
            "snow_depth": snowdepth[..., ::thinning_factor]
            - snowdepth[0, ::thinning_factor],
            "accumulation": snowfall.sum(axis=0)[::thinning_factor],
            "melt": snowmelt.sum(axis=0)[::thinning_factor],
            "runoff": runoff.sum(axis=0)[::thinning_factor],
            "refreeze": refreeze.sum(axis=0)[::thinning_factor],
            "smb": smb.sum(axis=0)[::thinning_factor],
        }

    return (
        temp[..., ::thinning_factor],
        precip[..., ::thinning_factor],
        temp_std_dev[..., ::thinning_factor],
        obs,
    )


def load_hirham_climate_simple(file="DMI-HIRHAM5_1980_MM.nc", thinning_factor=1):
    """
    Read and return Obs
    """

    with xr.open_dataset(file) as Obs:
        stacked = Obs.stack(z=("rlat", "rlon"))
        ncl_stacked = Obs.stack(z=("ncl4", "ncl5"))

        temp = stacked.tas.dropna(dim="z").values
        rainfall = stacked.rainfall.dropna(dim="z").values
        snowfall = stacked.snfall.dropna(dim="z").values
        smb = stacked.gld.dropna(dim="z").values
        refreeze = ncl_stacked.rfrz.dropna(dim="z").values
        melt = stacked.snmel.dropna(dim="z").values
        precip = rainfall + snowfall
        runoff = stacked.rogl.dropna(dim="z").values

    return (
        (temp[::thinning_factor] - 273.15).reshape(1, -1),
        precip[::thinning_factor].reshape(1, -1),
        snowfall[::thinning_factor].reshape(1, -1),
        melt[::thinning_factor].reshape(1, -1),
        runoff[::thinning_factor].reshape(1, -1),
        smb[::thinning_factor].reshape(1, -1),
        refreeze[::thinning_factor].reshape(1, -1),
    )


def load_imbie_csv(proj_start=2008):
    df = pd.read_csv("imbie_greenland_2021_Gt.csv")

    df = df.rename(
        columns={
            "Mass balance (Gt/yr)": "Mass change (Gt/yr)",
            "Mass balance uncertainty (Gt/yr)": "Mass change uncertainty (Gt/yr)",
            "Cumulative mass balance (Gt)": "Mass (Gt)",
            "Cumulative mass balance uncertainty (Gt)": "Mass uncertainty (Gt)",
        }
    )
    for v in [
        "Mass (Gt)",
    ]:
        df[v] -= df[df["Year"] == proj_start][v].values

    cmSLE = 1.0 / 362.5 / 10.0
    # df["Mass change uncertainty squared (Gt/yr)"] = (
    #     df["Mass change uncertainty (Gt/yr)"] ** 2
    # )
    # df["Mass uncertainty (Gt)"] = np.sqrt(df["Mass change uncertainty squared (Gt/yr)"])
    # df = df.drop(columns=["Mass change uncertainty squared (Gt/yr)"])
    df["SLE (cm)"] = -df["Mass (Gt)"] * cmSLE
    df["SLE uncertainty (cm)"] = df["Mass uncertainty (Gt)"] * cmSLE
    df["SLE change uncertainty (cm/yr)"] = df["Mass change uncertainty (Gt/yr)"] * cmSLE
    return df


def load_imbie(proj_start=2008):
    """
    Loading the IMBIE Greenland data set downloaded from
    http://imbie.org/wp-content/uploads/2012/11/imbie_dataset_greenland_dynamics-2020_02_28.xlsx

    """
    try:
        df_df = pd.read_excel(
            "http://imbie.org/wp-content/uploads/2012/11/imbie_dataset_greenland_dynamics-2020_02_28.xlsx",
            sheet_name="Greenland Ice Mass",
            engine="openpyxl",
        )
    except:
        df_df = pd.read_excel(
            "imbie_dataset_greenland_dynamics-2020_02_28.xlsx",
            sheet_name="Greenland Ice Mass",
            engine="openpyxl",
        )
    df = df_df[
        [
            "Year",
            "Cumulative ice sheet mass change (Gt)",
            "Cumulative ice sheet mass change uncertainty (Gt)",
            "Cumulative surface mass balance anomaly (Gt)",
            "Cumulative surface mass balance anomaly uncertainty (Gt)",
            "Cumulative ice dynamics anomaly (Gt)",
            "Cumulative ice dynamics anomaly uncertainty (Gt)",
            "Rate of mass balance anomaly (Gt/yr)",
            "Rate of ice dynamics anomaly (Gt/yr)",
            "Rate of mass balance anomaly uncertainty (Gt/yr)",
            "Rate of ice dyanamics anomaly uncertainty (Gt/yr)",
        ]
    ].rename(
        columns={
            "Cumulative ice sheet mass change (Gt)": "Mass (Gt)",
            "Cumulative ice sheet mass change uncertainty (Gt)": "Mass uncertainty (Gt)",
            "Rate of mass balance anomaly (Gt/yr)": "SMB (Gt/yr)",
            "Rate of ice dynamics anomaly (Gt/yr)": "D (Gt/yr)",
            "Rate of mass balance anomaly uncertainty (Gt/yr)": "SMB uncertainty (Gt/yr)",
            "Rate of ice dyanamics anomaly uncertainty (Gt/yr)": "D uncertainty (Gt/yr)",
        }
    )

    for v in [
        "Mass (Gt)",
        "Cumulative ice dynamics anomaly (Gt)",
        "Cumulative surface mass balance anomaly (Gt)",
    ]:
        df[v] -= df[df["Year"] == proj_start][v].values

    s = df[(df["Year"] >= 1980) & (df["Year"] < 1990)]
    mass_mean = s["Mass (Gt)"].mean() / (1990 - 1980)
    smb_mean = s["Cumulative surface mass balance anomaly (Gt)"].mean() / (1990 - 1980)
    df["SMB (Gt/yr)"] += 2 * 1964 / 10
    df["D (Gt/yr)"] -= 2 * 1964 / 10
    cmSLE = 1.0 / 362.5 / 10.0
    df["SLE (cm)"] = -df["Mass (Gt)"] * cmSLE
    df["SLE uncertainty (cm)"] = df["Mass uncertainty (Gt)"] * cmSLE

    return df


def plot_compare(
    F_p,
    F_v,
    dataset,
    X_val_unscaled,
    validation=False,
    return_fig=False,
):
    """
    Plot target (PISM) and predicted (Emulator) speeds for validation
    """
    cmap = "viridis"
    fig, axs = plt.subplots(
        nrows=3, ncols=1, sharex="col", sharey="row", figsize=(2.5, 8)
    )

    rmse = np.sqrt(((F_p - F_v) ** 2).mean())
    corr = np.corrcoef(F_v.flatten(), F_p.flatten())[0, 1]
    c1 = axs[0].imshow(F_v, origin="lower", cmap=cmap, norm=LogNorm(vmin=1, vmax=3e3))
    axs[1].imshow(F_p, origin="lower", cmap=cmap, norm=LogNorm(vmin=1, vmax=3e3))
    c2 = axs[2].imshow(F_p - F_v, origin="lower", vmin=-50, vmax=50, cmap="coolwarm")
    axs[1].text(
        0.01,
        0.00,
        f"r={corr:.3f}",
        c="k",
        size=7,
        transform=axs[1].transAxes,
    )
    axs[-1].text(
        0.01,
        -0.51,
        "\n".join([f"{i}: {j:.3f}" for i, j in zip(dataset.X_keys, X_val_unscaled)]),
        c="k",
        size=7,
        transform=axs[-1].transAxes,
    )

    axs[2].text(
        0.01,
        0.00,
        f"RMSE: {rmse:.0f} m/yr",
        c="k",
        size=7,
        transform=axs[2].transAxes,
    )

    axs[0].set_axis_off()
    axs[1].set_axis_off()
    axs[2].set_axis_off()
    axs[0].text(
        0.01,
        0.98,
        "PISM",
        c="k",
        size=7,
        weight="bold",
        transform=axs[0, 0].transAxes,
    )
    axs[1].text(
        0.01,
        0.98,
        "Emulator",
        c="k",
        size=7,
        weight="bold",
        transform=axs[1, 0].transAxes,
    )
    axs[2].text(
        0.01,
        0.98,
        "PISM-Emulator",
        c="k",
        size=7,
        weight="bold",
        transform=axs[2, 0].transAxes,
    )

    cb_ax = fig.add_axes([0.88, 0.525, 0.025, 0.15])
    plt.colorbar(
        c1,
        cax=cb_ax,
        shrink=0.9,
        label="speed (m/yr)",
        orientation="vertical",
        extend="both",
    )
    cb_ax2 = fig.add_axes([0.88, 0.15, 0.025, 0.15])
    plt.colorbar(
        c2,
        cax=cb_ax2,
        shrink=0.9,
        label="diff. (m/yr)",
        orientation="vertical",
        extend="both",
    )
    cb_ax.tick_params(labelsize=7)
    cb_ax.set_yticklabels([1, 10, 100, 1000])
    cb_ax2.tick_params(labelsize=7)
    fig.subplots_adjust(wspace=0.05, hspace=0.15)
    if validation:
        mode = "val"
    else:
        mode = "train"

    fig.savefig("test_comp.pdf")

    if return_fig:
        return fig


def plot_eigenglaciers(
    dataset,
    data_loader,
    model_index,
    emulator_dir,
    nrows=2,
    ncols=3,
    figsize=(3.2, 3.6),
    q: int = 6,
):
    V_hat, _, _, lamda = data_loader.get_eigenglaciers(eigenvalues=True, q=q)

    lamda_scaled = lamda / lamda.sum() * 100
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, sharex="col", sharey="row", figsize=figsize
    )
    for k, ax in enumerate(axs.ravel()):
        V = V_hat[:, k]
        data = np.zeros((dataset.ny, dataset.nx))
        data.put(dataset.sparse_idx_1d, V)
        eigen_glacier = np.ma.array(data=data, mask=dataset.mask_2d)
        c = ax.imshow(
            eigen_glacier, origin="lower", cmap="twilight_shifted", vmin=-0.3, vmax=0.3
        )

        ax.text(
            0.05,
            -0.025,
            f"$\Lambda_{k}$={lamda_scaled[k]:.1f}%",
            transform=ax.transAxes,
        )
        ax.axis("off")
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.tight_layout()
    fig_dir = f"{emulator_dir}/eigenglaciers"
    if not isdir(fig_dir):
        mkdir(fig_dir)

    fig.savefig(join(fig_dir, f"eigenglaciers_{model_index}.pdf"))


def calc_bic(
    X: Union[np.ndarray, pd.core.frame.DataFrame],
    Y: Union[np.ndarray, pd.core.frame.DataFrame],
) -> float:
    """
    Bayesian Information Criterion

    Calculates the Bayesian Information Criterion (BIC) under the assumption that the model errors or disturbances are independent and identically distributed according to a normal distribution and that the boundary condition that the derivative of the log likelihood with respect to the true variance is zero, this becomes (up to an additive constant, which depends only on n and not on the model). BIC is given by:

    BIC = n*ln(RSS/n) + ln(n)*k,

    where

    n = the number of data points in x, the number of observations, or equivalently, the sample size
    k = the number of parameters estimated by the model. For example, in multiple linear regression, the estimated parameters are the intercept, the q slope parameters, and the constant variance of the errors; thus, k = q + 2
    RSS = residual sums of squares (FIXME)

    Parameters
    ----------
    X : ndarray
        input observations
    Y : ndarray
        output observations

    Returns
    -------
    BIC : scalar
          The Bayesian Information Criterion (BIC)
    """

    lm = LinearRegression()
    lm.fit(X, Y)
    Y_hat = lm.predict(X)
    res = Y - Y_hat
    RSS = np.sum(np.power(res, 2))
    n, q = X.shape
    # k = q + 2
    k = q
    BIC = n * np.log(RSS / n) + k * np.log(n)

    return BIC


def stepwise_bic(
    X: np.array, Y: np.array, varnames: list = [], interactions: bool = True, **kwargs
):
    """
    Stepwise model selection using the Bayesian Information Criterion (BIC)

    General function modeled after R's stepAIC function.

    Starts with full least squares model as in backward selection. If interactions=True, performs
    bidirectional stepwise selection. Otherwise only performs backwards selection.

    User may supply a list of variable names where the length of the list
    must equal n=X.shape[1]. Otherwise, a list of phony variables
    ["X1", "X2", ...,"Xn"] is generated.

    Then params_dict is generated from the names list.

    Parameters
    ----------
    X : array-like (n-d shaped)
        input model params
    Y : array_like (d shaped)
        input observations

    Returns
    -------
    V : array_like
        List of select model parameters including interactions, e.g.
        V = ["X1", "X2", "X1*X2"]
    """

    n = X.shape[1]
    names = ["X{}".format(x) for x in range(n)]
    if varnames is not []:
        names = varnames

    assert n == len(names)
    # Need assertion error here

    params_dict = {k: v for v, k in enumerate(names)}
    params_to_check = list(names)
    # Variables in model, initially start with model Y = X1 + X2 + X3 + ... + Xn (no interactions)
    # Any changes in varlist are reflected in the variables in X
    varlist = list(names)
    newnames = []
    if interactions:
        # List of first-order interactions between all variables
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                newnames.append(names[i] + "*" + names[j])
    # Variables to test
    names = list(np.append(names, newnames))
    names_dup = []

    whole_lm_bic = calc_bic(X, Y)

    in_progress = True
    step = 1
    while in_progress:
        print("\nStep {}".format(step))
        print("  Baseline model BIC: {:2.2f}".format(whole_lm_bic))
        bic_dict = {}
        for m_str in names:
            if "*" in m_str:
                subnames = m_str.split("*")
                if len(subnames) != 2:
                    sys.exit("Interaction unexpected")
                # Temporary X that contains the interaction term
                tempX = np.column_stack(
                    (X, X[:, params_dict[subnames[0]]] * X[:, params_dict[subnames[1]]])
                )
                # BIC for baseline model + interaction term
                lm_bic = calc_bic(tempX, Y)
            else:
                # Temporary X that drops main effect
                tempX = np.delete(X, params_dict[m_str], axis=1)  # type: ignore
                # BIC for baseline model without main effect
                lm_bic = calc_bic(tempX, Y)
            # Number of entries in bic_dict should equal the number of variables left to test
            bic_dict[m_str] = lm_bic

        # Lowest BIC and variable associated from variables left to test
        min_key = min(bic_dict.keys(), key=(lambda k: bic_dict[k]))
        min_bic = bic_dict[min_key]
        if "*" in min_key:
            print(
                "  Minimum BIC = {:2.2f} when adding {} to model".format(
                    min_bic, min_key
                )
            )
        else:
            print(
                "  Minimum BIC = {:2.2f} when removing {} from model".format(
                    min_bic, min_key
                )
            )

        # Compare lowest BIC to baseline model BIC
        if min_bic < whole_lm_bic:
            if "*" in min_key:
                names_dup = names
                # Add interaction term to list of variables in model to become new baseline model
                varlist.append(min_key)
                print("  Added {}, BIC = {}".format(min_key, min_bic))
                print("  Updated variable list: {}".format(varlist))
                names_dup.remove(min_key)
                print("  Removed {} from model-eligible variables".format(min_key))

                subnames = min_key.split("*")
                if len(subnames) != 2:
                    sys.exit("Interaction unexpected")

                for s in subnames:
                    if s in names_dup:
                        names_dup.remove(s)
                        print("  Removed {} from model-eligible variables".format(s))

                # Update X and BIC to reflect new baseline model
                X = np.column_stack(
                    (X, X[:, params_dict[subnames[0]]] * X[:, params_dict[subnames[1]]])
                )
                whole_lm_bic = calc_bic(X, Y)

            else:
                names_dup = names
                # Remove main effect term from list of variables in model to update baseline model
                varlist.remove(min_key)
                print("  Removed {}, BIC = {:2.2f}".format(min_key, min_bic))
                print("  Updated variable list: {}".format(varlist))
                names_dup.remove(min_key)
                print("  Removed {} from model-eligible variables".format(min_key))

                # Remove interaction terms that contain main effect
                rem = []
                for var in names_dup:
                    if min_key in var:
                        rem.append(var)
                names_dup = [x for x in names_dup if x not in rem]
                print("  Removed {} from model-eligible variables".format(rem))

                # Update X and BIC to reflect new baseline model
                X = np.delete(X, params_dict[min_key], axis=1)
                whole_lm_bic = calc_bic(X, Y)

                # Update variable position in X through params_dict; removing a variable changes the
                # positions of all following variables - update to avoid skipping positions
                for p in params_to_check:
                    if params_dict[min_key] <= params_dict[p]:
                        params_dict[p] -= 1
                params_to_check.remove(min_key)

            names = names_dup
            print("  Remaining variables to test: {}".format(names))
            step += 1
        else:
            # No lower BIC can be obtained by modifying model variables, so baseline model is final
            in_progress = False
            print("\n\nMinimum model BIC reached - completed stepwise regression")
            print("Final model:")
            print("  Y ~ {}".format(" + ".join(varlist)))
            print("Total number of steps: {}\n".format(step))
            return varlist


def prepare_data(
    samples_file,
    response_file,
    identifier_name="id",
    skipinitialcolumn=True,
    return_missing=False,
    return_numpy=False,
):
    """
    Reads samples_file and response_file as a pandas.DataFrame. Removes samples that do
    not have a response by differencing the DataFrames based on "id", i.e.
    both samples_file and response_file must contain "id".

    Parameters
    ----------
    samples_file  : str
    response_file : str

    Returns
    -------
    X : pandas.DataFrame (ndarray if return_numpy=True)
    Y : pandas.DataFrame (ndarray if return_numpy=True)
    """

    print("\nPreparing sample {} and response {}".format(samples_file, response_file))

    # Load Samples file as Pandas DataFrame
    samples = (
        pd.read_csv(samples_file, delimiter=",", skipinitialspace=True)
        .squeeze("columns")
        .sort_values(by=identifier_name)
    )
    samples.index = samples[identifier_name]
    samples.index.name = None

    # Load Response file as Pandas DataFrame
    response = (
        pd.read_csv(response_file, delimiter=",", skipinitialspace=True)
        .squeeze("columns")
        .sort_values(by=identifier_name)
    )
    response.index = response[identifier_name]
    response.index.name = None

    # It is possible that not all ensemble simulations succeeded and returned a value
    # so we much search for missing response values
    missing_ids = list(set(samples["id"]).difference(response["id"]))
    if missing_ids:
        print("The following simulation ids are missing:\n   {}".format(missing_ids))
        # and remove the missing samples and responses
        samples_missing_removed = samples[~samples["id"].isin(missing_ids)]
        samples = samples_missing_removed
        response_missing_removed = response[~response["id"].isin(missing_ids)]
        response = response_missing_removed

    if skipinitialcolumn:
        samples = samples.drop(samples.columns[0], axis=1)
        response = response.drop(response.columns[0], axis=1)

    s = samples
    r = response

    if return_numpy:
        s = samples.values
        r = response.values

    if return_missing:
        return s, r, missing_ids
    else:
        return s, r


def draw_samples(distributions, n_samples=100000, method="lhs"):
    """
    Draw n_samples Sobol sequences using the Saltelli method
    or using Latin Hypercube Sampling (LHS)

    Provide a dictionary with distributions of the form

    distributions = {
        "X1": scipy.stats.distributions.randint(0, 2),
        "X2": scipy.stats.distributions.truncnorm(-4 / 4.0, 4.0 / 4, loc=8, scale=4),
        "X3": scipy.stats.distributions.uniform(loc=5, scale=2),
    }

    Parameters
    ----------
    distributions : dictionary of scipy.stats.distributions.
    n_samples : number of samples to draw.
    method: drawing method. Either Latin Hypercube ("lhs")
            or Saltelli ("saltelli").

    Returns
    -------
    d : pandas.DataFrame
    """

    # Names of all the variables
    keys = [x for x in distributions.keys()]

    # Describe the Problem
    problem = {"num_vars": len(keys), "names": keys, "bounds": [[0, 1]] * len(keys)}

    # Generate uniform samples (i.e. one unit hypercube)
    if method == "lhs":
        unif_sample = lhs(len(keys), n_samples)
    elif method == "saltelli":
        unif_sample = saltelli.sample(problem, n_samples, calc_second_order=False)
    else:
        import sys

        sys.exit("How did I get here? Invalid sampling method")

    # To hold the transformed variables
    dist_sample = np.zeros_like(unif_sample)

    # Now transform the unit hypercube to the prescribed distributions
    # For each variable, transform with the inverse of the CDF (inv(CDF)=ppf)
    for i, key in enumerate(keys):
        dist_sample[:, i] = distributions[key].ppf(unif_sample[:, i])

    # Save to CSV file using Pandas DataFrame and to_csv method
    # Convert to Pandas dataframe, append column headers, output as csv
    header = keys
    return pd.DataFrame(data=dist_sample, columns=header)


def kl_divergence(p, q):
    """
    Kullback-Leibler divergence

    From https://en.wikipedia.org/wiki/Kullback–Leibler_divergence:

    In the context of machine learning, {isplaystyle D_{\text{KL}}(P\parallel Q)} is often called the information gain achieved if Q is used instead of P. By analogy with information theory, it is also called the relative entropy of P with respect to Q. In the context of coding theory, {\displaystyle D_{\text{KL}}(P\parallel Q)} can be constructed by measuring the expected number of extra bits required to code samples from P using a code optimized for Q rather than the code optimized for.

    Expressed in the language of Bayesian inference, {\displaystyle D_{\text{KL}}(P\parallel Q)} is a measure of the information gained when one revises one's beliefs from the prior probability distribution Q to the posterior probability distribution P. In other words, it is the amount of information lost when Q is used to approximate P. In applications, P typically represents the "true" distribution of data, observations, or a precisely calculated theoretical distribution, while Q typically represents a theory, model, description, or approximation of P. In order to find a distribution Q that is closest to P, we can minimize KL divergence and compute an information projection.


    """
    return np.sum(
        np.where(
            np.logical_and(np.logical_and(p != 0, q != 0), np.isfinite(p / q)),
            p * np.log(p / q),
            0,
        )
    )


def distributions_as19():
    """

    Returns the distributions used by Aschwanden et al (2019):

    @article{Aschwanden2019,
    author = {Aschwanden, Andy and Fahnestock, Mark A. and Truffer, Martin and Brinkerhoff, Douglas J. and Hock, Regine and Khroulev, Constantine and Mottram, Ruth and Khan, S. Abbas},
    doi = {10.1126/sciadv.aav9396},
    issn = {2375-2548},
    journal = {Science Advances},
    month = {jun},
    number = {6},
    pages = {eaav9396},
    title = {{Contribution of the Greenland Ice Sheet to sea level over the next millennium}},
    url = {http://advances.sciencemag.org/lookup/doi/10.1126/sciadv.aav9396},
    volume = {5},
    year = {2019}
    }

    """

    return {
        "GCM": randint(0, 4),
        "FICE": truncnorm(-4 / 4.0, 4.0 / 4, loc=8, scale=4),
        "FSNOW": truncnorm(-4.1 / 3, 4.1 / 3, loc=4.1, scale=1.5),
        "PRS": uniform(loc=5, scale=2),
        "RFR": truncnorm(-0.4 / 0.3, 0.4 / 0.3, loc=0.5, scale=0.2),
        "OCM": randint(-1, 2),
        "OCS": randint(-1, 2),
        "TCT": randint(-1, 2),
        "VCM": truncnorm(-0.35 / 0.2, 0.35 / 0.2, loc=0.4, scale=0.2),
        "PPQ": truncnorm(-0.35 / 0.2, 0.35 / 0.2, loc=0.6, scale=0.2),
        "SIAE": gamma(1.5, scale=0.8, loc=1),
    }


def rmsd(a, b):
    """
    Root mean square difference between a and b

    a, b: array-like
    """
    return sqrt(mean_squared_error(a, b))


def set_size(w, h, ax=None):
    """
    w, h: width, height in inches
    """

    if not ax:
        ax = plt.gca()
    left = ax.figure.subplotpars.left
    right = ax.figure.subplotpars.right
    top = ax.figure.subplotpars.top
    bottom = ax.figure.subplotpars.bottom
    figw = float(w) / (right - left)
    figh = float(h) / (top - bottom)
    ax.figure.set_size_inches(figw, figh)


def gelman_rubin(p: np.ndarray, q: np.ndarray):
    """
    Returns estimate of R for a set of two traces.
    The Gelman-Rubin diagnostic tests for lack of convergence by comparing
    the variance between multiple chains to the variance within each chain.
    If convergence has been achieved, the between-chain and within-chain
    variances should be identical. To be most effective in detecting evidence
    for nonconvergence, each chain should have been initialized to starting
    values that are dispersed relative to the target distribution.

    Parameters
    ----------
    p, q : ndarray
           Arrays containing the 2 traces of a stochastic parameter. That is, an array of dimension m x 2, where m is the number of traces, n the number of samples.

    Returns
    -------
    Rhat : float
           Return the potential scale reduction factor, :math:`\hat{R}`.

    Notes
    -----
    The diagnostic is computed by:
      .. math:: \hat{R} = \sqrt{\frac{\hat{V}}{W}}
    where :math:`W` is the within-chain variance and :math:`\hat{V}` is
    the posterior variance estimate for the pooled traces. This is the
    potential scale reduction factor, which converges to unity when each
    of the traces is a sample from the target posterior. Values greater
    than one indicate that one or more chains have not yet converged.

    References
    ----------
    Brooks and Gelman (1998)
    Gelman and Rubin (1992)
    """

    n = len(p)
    W = p.std() ** 2 + q.std() ** 2
    P_mean = p.mean()
    Q_mean = q.mean()
    mean = (P_mean + Q_mean) / 2
    B = n * ((P_mean - mean) ** 2 + (Q_mean - mean) ** 2)
    V = (1 - 1 / n) * W + 1 / n * B
    R = V / W
    return np.sqrt(R)


# Define constants

golden_ratio = (1 + np.sqrt(5)) / 2