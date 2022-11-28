# Copyright (C) 2021 Andy Aschwanden, Douglas C Brinkerhoff
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
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.checks import _check_same_shape


def _area_absolute_error_update(
    preds: Tensor, target: Tensor, omegas: Tensor, area: Tensor
) -> Tensor:
    _check_same_shape(preds, target)
    diff = torch.abs(preds - target)
    sum_abs_error = torch.sum(diff * diff * area, axis=1)
    absolute_error = torch.sum(sum_abs_error * omegas.squeeze())
    return absolute_error


def _area_absolute_error_compute(absolute_error) -> Tensor:
    return absolute_error


def area_absolute_error(
    preds: Tensor, target: Tensor, omegas: Tensor, area: Tensor
) -> Tensor:
    """
    Computes squared absolute error
    Args:
        preds: estimated labels
        target: ground truth labels
        omegas: weights
        area: area of each cell
    Return:
        Tensor with absolute error
    Example:
        >>> x = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]).T
        >>> y = torch.tensor([[0, 1, 2, 1], [2, 3, 4, 4]]).T
        >>> o = torch.tensor([0.25, 0.25, 0.3, 0.2])
        >>> a = torch.tensor([0.25, 0.25])
        >>> absolute_error(x, y, o, a)
        tensor(0.4000)
    """
    sum_abs_error = _area_absolute_error_update(preds, target, omegas, area)
    return _area_absolute_error_compute(sum_abs_error)


class AreaAbsoluteError(Metric):

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    # Use:
    # x = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]).T
    # y = torch.tensor([[0, 1, 2, 1], [2, 3, 4, 4]]).T
    # o = torch.tensor([0.25, 0.25, 0.3, 0.2])
    # a = torch.tensor([0.25, 0.25])
    # torchmetrics.utilities.check_forward_full_state_property(AreaAbsoluteError, input_args={"preds": x, "target": y, "omegas": o, "area": a})

    full_state_update: bool = False

    def __init__(self, compute_on_step: bool = True, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(
            compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step
        )

        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, omegas: Tensor, area: Tensor):
        """
        Update state with predictions and targets, and area.
        Args:
            preds: Predictions from model
            target: Ground truth values
            omegas: Weights
            area: Area of each cell
        """
        sum_abs_error = _area_absolute_error_update(preds, target, omegas, area)
        self.sum_abs_error += sum_abs_error

    def compute(self):
        """
        Computes absolute error over state.
        """
        return _area_absolute_error_compute(self.sum_abs_error)

    @property
    def is_differentiable(self):
        return True


def _absolute_error_update(preds: Tensor, target: Tensor, omegas: Tensor) -> Tensor:
    _check_same_shape(preds, target)
    diff = torch.abs(preds - target)
    sum_abs_error = torch.sum(diff * diff, axis=1)
    absolute_error = torch.sum(sum_abs_error * omegas.squeeze())
    return absolute_error


def _absolute_error_compute(absolute_error) -> Tensor:
    return absolute_error


def absolute_error(preds: Tensor, target: Tensor, omegas: Tensor) -> Tensor:
    """
    Computes squared absolute error
    Args:
        preds: estimated labels
        target: ground truth labels
        omegas: weights
    Return:
        Tensor with absolute error
    Example:
        >>> x = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]).T
        >>> y = torch.tensor([[0, 1, 2, 1], [2, 3, 4, 4]]).T
        >>> o = torch.tensor([0.25, 0.25, 0.3, 0.2])
        >>> absolute_error(x, y, o)
        tensor(0.4000)
    """
    sum_abs_error = _absolute_error_update(preds, target, omegas)
    return _absolute_error_compute(sum_abs_error)


class AbsoluteError(Metric):

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    # Use:
    # x = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]).T
    # y = torch.tensor([[0, 1, 2, 1], [2, 3, 4, 4]]).T
    # o = torch.tensor([0.25, 0.25, 0.3, 0.2])
    # torchmetrics.utilities.check_forward_full_state_property(AreaAbsoluteError, input_args={"preds": x, "target": y, "omegas": o})

    full_state_update: bool = False

    def __init__(self, compute_on_step: bool = True, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(
            compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step
        )

        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, omegas: Tensor):
        """
        Update state with predictions and targets, and area.
        Args:
            preds: Predictions from model
            target: Ground truth values
            omegas: Weights
        """
        sum_abs_error = _absolute_error_update(preds, target, omegas)
        self.sum_abs_error += sum_abs_error

    def compute(self):
        """
        Computes absolute error over state.
        """
        return _absolute_error_compute(self.sum_abs_error)

    @property
    def is_differentiable(self):
        return True
