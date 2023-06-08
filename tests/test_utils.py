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

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal

from pismemulator.utils import (
    calc_bic,
    gelman_rubin,
    kl_divergence,
    prepare_data,
    rmsd,
    stepwise_bic,
)


def test_calc_bic():
    X = np.array([[1.0, 4.0, 9.0], [2.0, 0.0, -1.0]])
    Y = X**2 - 1
    bic = calc_bic(X, Y)
    assert_array_almost_equal(bic, -125.68336950813814, decimal=2), "foo"


def test_kl_divergence(pq):
    p, q = pq

    assert_array_almost_equal(kl_divergence(p, q), 0.08529960)
    assert_array_almost_equal(kl_divergence(q, p), 0.09745500)


def test_gelman_rubin(pq):
    p, q = pq

    assert_array_almost_equal(gelman_rubin(p, q), 0.816496580927726)
    assert_array_almost_equal(gelman_rubin(q, p), 0.816496580927726)


def test_prepare_data(saltsamples, saltresponse, tmpdir):
    dfs = tmpdir.mkdir("tmpdir_samples").join("samples.csv")
    dfr = tmpdir.mkdir("tmpdir_response").join("response.csv")
    dfr_m = tmpdir.mkdir("tmpdir_response_missing").join("response_missing.csv")
    saltsamples.to_csv(dfs)
    saltresponse.to_csv(dfr)

    # check if returning DataFrame
    s, r = prepare_data(dfs, dfr)
    assert isinstance(s, pd.core.frame.DataFrame)
    assert isinstance(r, pd.core.frame.DataFrame)

    # check if ndarray
    s, r = prepare_data(dfs, dfr, return_numpy=True)
    assert isinstance(s, np.ndarray)
    assert isinstance(r, np.ndarray)

    # check return_missing function
    saltresponse.drop([2, 6]).to_csv(dfr_m)
    s, r, m = prepare_data(dfs, dfr_m, return_missing=True)
    assert m == [2, 6]


def test_rmsd():
    a = np.array([1.0, 2.0, 3.0])
    b = a + 1
    assert_array_almost_equal(rmsd(a, b), 1.0)


def test_stepwise_bic(dp16data):
    """
    Test with and without first-order interactions

    Replicates Edwards et al (2019) model

    """

    X = dp16data[["OCFAC", "CREVLIQ", "VCLIF", "BIAS"]]
    Y = dp16data[["RCP85_2100"]]

    # From Edwards et al (2019)
    dp16_no_interactions = ["OCFAC", "CREVLIQ", "VCLIF", "BIAS"]
    dp16_with_interactions = [
        "OCFAC",
        "CREVLIQ",
        "VCLIF",
        "BIAS",
        "CREVLIQ*VCLIF",
        "OCFAC*BIAS",
        "OCFAC*VCLIF",
    ]
    dp16_no_varnames = ["X0", "X1", "X2", "X3", "X1*X2", "X0*X3", "X0*X2"]

    # Write Assertion exceptions and useful error messages

    test_vars = stepwise_bic(X.values, Y.values, varnames=X.columns, interactions=False)
    assert test_vars == dp16_no_interactions
    test_vars = stepwise_bic(X.values, Y.values, varnames=X.columns, interactions=True)
    assert test_vars == dp16_with_interactions
    # Check default for interactions
    test_vars = stepwise_bic(X.values, Y.values, varnames=X.columns)
    assert test_vars == dp16_with_interactions
    # Check if varnames are not provided
    test_vars = stepwise_bic(X.values, Y.values)
    assert test_vars == dp16_no_varnames
