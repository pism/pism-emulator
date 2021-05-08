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

import GPy as gp

import numpy as np
from numpy.testing import assert_array_almost_equal

from pismemulator.gpemulator import emulate_gp
from pismemulator.gpemulator import emulate_sklearn
from pismemulator.gpemulator import generate_kernel


def test_generate_kernel():

    """
    Any ideas on how to test that function???
    """

    varlist = ["X1", "X2", "X1*X2"]
    kernel = gp.kern.Exponential
    varnames = ["X1", "X2"]
    k = generate_kernel(varlist, kernel, varnames)
    # Not sure these are useful assertions
    assert k.input_dim == 2
    assert k.size == 7


def test_emulate_gp(dp16data):

    X = dp16data[["OCFAC", "CREVLIQ", "VCLIF", "BIAS"]]
    Y = dp16data[["RCP85_2100"]]

    X_new = np.array(X.values[100:104, :])

    p_true = np.array([[0.089], [0.39199997], [0.99299997], [1.65899989]])
    var_true = np.array([[9.99999172e-09], [9.99999217e-09], [9.99999239e-09], [9.99999306e-09]])
    p, status = emulate_gp(X, Y, X_new, gp.kern.Exponential, stepwise=False)
    assert status.lower() == "converged".lower()
    assert_array_almost_equal(p[0], p_true, decimal=4), "Predicted values != true values"
    assert_array_almost_equal(p[1], var_true, decimal=4), "Predicted variances != true variances"


def test_emulate_sklearn():

    pass
