# Copyright (C) 2022 Andy Aschwanden
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

import torch
from numpy.testing import assert_almost_equal

from pism_emulator.metrics import AbsoluteError, AreaAbsoluteError, L2MeanSquaredError


def test_AreaAbsoluteError():
    x = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]).T
    y = torch.tensor([[0, 1, 2, 1], [2, 3, 4, 4]]).T
    o = torch.tensor([0.25, 0.25, 0.3, 0.2])
    a = torch.tensor([0.25, 0.25])
    aae = AreaAbsoluteError()
    ae = aae(x, y, o, a)
    assert_almost_equal(ae, 0.4000, decimal=4)


def test_AbsoluteError():
    x = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]).T
    y = torch.tensor([[0, 1, 2, 1], [2, 3, 4, 4]]).T
    o = torch.tensor([0.25, 0.25, 0.3, 0.2])
    aae = AbsoluteError()
    ae = aae(x, y, o)
    assert_almost_equal(ae, 1.6000, decimal=4)


def test_L2MeanSquaredError():
    target = torch.tensor([2.5, 5.0, 4.0, 8.0])
    preds = torch.tensor([3.0, 5.0, 2.5, 7.0])
    weight = torch.tensor([0.1, 0.2, 0.5, 0.2])
    k = 1e-1
    l2_mean_squared_error = L2MeanSquaredError()
    l2mse = l2_mean_squared_error(preds, target, weight, k)
    assert_almost_equal(l2mse, torch.tensor(0.8835))
