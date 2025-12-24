# Copyright (C) 2019-25 Andy Aschwanden
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
Utils.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats.distributions import gamma, randint, truncnorm, uniform

np.random.seed(0)


param_keys_dict = {
    "GCM": "GCM (1)",
    "FICE": "$f_i$ (mm K$^{-1}$ day$^{-1}$)",
    "FSNOW": "$f_s$ (mm K$^{-1}$ day$^{-1}$)",
    "RFR": r"$\psi (1)$",
    "PRS": r"$\omega$ (% K$^{-1}$)",
    "OCM": "$m_{t}$ (1)",
    "OCS": "$m_{x}$ (1)",
    "TCT": r"$h_{\mathrm{min}}$ (1)",
    "VCM": r"$\sigma_{\mathrm{max}}$ (MPa)",
    "SIAE": r"$E_{\mathrm{SIA}}$ (1)",
    "SSAN": r"$n_{\mathrm{SSA}}$ (1)",
    "TEFO": r"$\delta$ (1)",
    "PPQ": "$q$ (1)",
    "PHIMIN": r"$\phi_{\mathrm{min}}$ ($^{\circ}$)",
    "PHIMAX": r"$\phi_{\mathrm{max}}$ ($^{\circ}$)",
    "ZMIN": r"$z_{\mathrm{min}}$ (m)",
    "ZMAX": r"$z_{\mathrm{max}}$ (m)",
    "a_glen": "A (Pa^{-n} s^{-1})",
    "sia_e": r"$E_{\mathrm{SIA}}$ (1)",
    "ssa_e": r"$E_{\mathrm{SSA}}$ (1)",
    "ssa_n": r"$n_{\mathrm{SSA}}$ (1)",
    "ppq": "$q$ (1)",
    "tefo": r"$\delta$ (1)",
    "till_effective_fraction_overburden": r"$\delta$ (1)",
    "pseudo_plastic_uthershold": r"u_{\mathrm{thr} (m yr^{-1})}",
    "phi_min": r"$\phi_{\mathrm{min}}$ ($^{\circ}$)",
    "z_min": r"$z_{\mathrm{min}}$ (m)",
    "z_max": r"$z_{\mathrm{max}}$ (m)",
    "pseudo_plastic_uthreshold": r"$u_{\mathrm{th}}$ (m yr$^{-1}$)",
    "SIAe": r"$E_{\mathrm{SIA}}$ (1)",
    "SSAe": r"$E_{\mathrm{SSA}}$ (1)",
    "topg_to_phi_base": r"$b_{\mathrm{base}}$ (m)",
    "topg_to_phi_range": r"$b_{\mathrm{range}}$ (m)",
    "calving.vonmises_calving.sigma_max": r"$\sigma_{\mathrm{max}}$ (kPa)",
    "geometry.front_retreat.prescribed.file": "Retreat Method",
    "ocean.th.gamma_T": r"$\gamma_{S}$ (10$^{-4}$ 1)",
    "surface.given.file": "Climate Forcing",
    "ocean.th.file": "Ocean Forcing",
    "frontal_melt.routing.parameter_a": r"$a$ (10$^{-4}$ m$^{-\alpha}$ day$^{\alpha-1}$ Celsius$^{-\beta}$)",
    "frontal_melt.routing.parameter_b": r"$b$ (day$^{\alpha-1)}$ Celsius$^{-\beta}$)",
    "frontal_melt.routing.power_alpha": r"$\alpha$ (1)",
    "frontal_melt.routing.power_beta": r"$\beta$ (1)",
    "stress_balance.sia.enhancement_factor": r"$E_{\mathrm{SIA}}$ (1)",
    "stress_balance.ssa.enhancement_factor": r"$E_{\mathrm{SSA}}$ (1)",
    "stress_balance.blatter.enhancement_factor": r"$E_{\mathrm{HO}}$ (1)",
    "stress_balance.sis.Glen_exponent": r"$n_{\mathrm{SIA}}$ (1)",
    "stress_balance.ssa.Glen_exponent": r"$n_{\mathrm{SSA}}$ (1)",
    "stress_balance.blatter.Glen_exponent": r"$n_{\mathrm{HO}}$ (1)",
    "basal_resistance.pseudo_plastic.q": r"$q$ (1)",
    "basal_yield_stress.mohr_coulomb.till_effective_fraction_overburden": r"$\delta$ (1)",
    "basal_yield_stress.mohr_coulomb.topg_to_phi.phi_min": r"$\phi_{\mathrm{min}} (^{\circ{}})$",
    "basal_yield_stress.mohr_coulomb.topg_to_phi.phi_max": r"$\phi_{\mathrm{max}} (^{\circ{}})$",
    "basal_yield_stress.mohr_coulomb.topg_to_phi.topg_min": r"$z_{\mathrm{min}}$ (m)",
    "basal_yield_stress.mohr_coulomb.topg_to_phi.topg_max": r"$z_{\mathrm{max}}$ (m)",
    "basal_resistance.pseudo_plastic.u_threshold": r"$u_{\mathrm{th}}$",
    "basal_yield_stress.mohr_coulomb.till_phi_default": r"$\phi$ ($^{\circ}$)",
}


def load_imbie_csv(
    proj_start: int = 2008,
    file: str | Path = "imbie_greenland_2021_Gt.csv",
) -> pd.DataFrame:
    """
    Load IMBIE Greenland mass balance CSV and derive sea-level equivalents (SLE).

    The CSV is expected to contain IMBIE columns for mass balance and cumulative
    mass balance (with uncertainties). Columns are renamed to a consistent schema,
    cumulative mass is re-referenced to ``proj_start``, and SLE quantities are
    computed assuming ``1 cm SLE = 362.5 Gt``.

    Parameters
    ----------
    proj_start : int, default=2008
        Reference year. The cumulative mass time series ``"Mass (Gt)"`` is
        shifted so that its value in this year becomes zero.
    file : str or pathlib.Path, default="imbie_greenland_2021_Gt.csv"
        Path to the IMBIE CSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with at least the following columns (after renaming and
        augmentation):
        - ``"Year"``
        - ``"Mass change (Gt/yr)"``
        - ``"Mass change uncertainty (Gt/yr)"``
        - ``"Mass (Gt)"``  (re-referenced so value at ``proj_start`` is 0)
        - ``"Mass uncertainty (Gt)"``
        - ``"SLE (cm)"``  (computed as ``-Mass(Gt) / 362.5 / 10``)
        - ``"SLE uncertainty (cm)"``  (propagated linearly from mass uncertainty)
        - ``"SLE change uncertainty (cm/yr)"``  (from mass-change uncertainty)

    Raises
    ------
    KeyError
        If expected IMBIE columns are missing.
    ValueError
        If ``proj_start`` is not present in the ``"Year"`` column.

    Notes
    -----
    - The conversion used is ``cmSLE = 1 / 362.5 / 10`` (i.e., 362.5 Gt per cm SLE).
    - SLE is defined with opposite sign to mass: increasing mass lowers sea level,
      hence ``SLE = -Mass * cmSLE``.
    - The function assumes the CSV includes IMBIE-standard columns:
      ``"Mass balance (Gt/yr)"``, ``"Mass balance uncertainty (Gt/yr)"``,
      ``"Cumulative mass balance (Gt)"``, and
      ``"Cumulative mass balance uncertainty (Gt)"``. These are renamed to
      ``"Mass change (Gt/yr)"``, ``"Mass change uncertainty (Gt/yr)"``,
      ``"Mass (Gt)"``, and ``"Mass uncertainty (Gt)"``, respectively.
    """
    df = pd.read_csv(file)

    # Rename IMBIE columns to a consistent internal schema
    rename_map = {
        "Mass balance (Gt/yr)": "Mass change (Gt/yr)",
        "Mass balance uncertainty (Gt/yr)": "Mass change uncertainty (Gt/yr)",
        "Cumulative mass balance (Gt)": "Mass (Gt)",
        "Cumulative mass balance uncertainty (Gt)": "Mass uncertainty (Gt)",
    }
    missing = [k for k in rename_map if k not in df.columns]
    if missing:
        raise KeyError(f"Missing expected IMBIE columns: {missing}")
    df = df.rename(columns=rename_map)

    # Re-reference cumulative mass to the project start year
    if not (df["Year"] == proj_start).any():
        raise ValueError(
            f"'proj_start' year {proj_start} not found in CSV 'Year' column."
        )
    ref_val = df.loc[df["Year"] == proj_start, "Mass (Gt)"].values
    # subtract the scalar reference from the mass time series
    df["Mass (Gt)"] = df["Mass (Gt)"] - ref_val

    # Sea-level equivalent conversions (362.5 Gt per cm SLE)
    cmSLE = 1.0 / 362.5 / 10.0
    df["SLE (cm)"] = -df["Mass (Gt)"] * cmSLE
    df["SLE uncertainty (cm)"] = df["Mass uncertainty (Gt)"] * cmSLE
    df["SLE change uncertainty (cm/yr)"] = df["Mass change uncertainty (Gt/yr)"] * cmSLE

    return df


def distributions_as19() -> dict[str, object]:
    r"""
    Return prior distributions from Aschwanden et al. (2019).

    This helper returns a dictionary mapping parameter names to SciPy
    random-variable objects (``scipy.stats`` distributions). These priors were
    used in Aschwanden et al. (2019) for Greenland Ice Sheet projections.

    Returns
    -------
    dict[str, object]
        Mapping from parameter name to a SciPy distribution object (e.g.,
        ``scipy.stats.randint``, ``scipy.stats.truncnorm``, ``scipy.stats.uniform``,
        ``scipy.stats.gamma``).

    Notes
    -----
    Reference (BibTeX)::

        @article{Aschwanden2019,
        author = {Aschwanden, Andy and Fahnestock, Mark A. and Truffer, Martin and
                  Brinkerhoff, Douglas J. and Hock, Regine and Khroulev, Constantine
                  and Mottram, Ruth and Khan, S. Abbas},
        doi = {10.1126/sciadv.aav9396},
        issn = {2375-2548},
        journal = {Science Advances},
        month = {jun},
        number = {6},
        pages = {eaav9396},
        title = {{Contribution of the Greenland Ice Sheet to sea level over the next millennium}},
        volume = {5},
        year = {2019}
        }

    Examples
    --------
    >>> dists = distributions_as19()
    >>> dists["FICE"].mean()
    8.0
    >>> dists["GCM"].rvs(size=5, random_state=0)  # doctest: +SKIP
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
