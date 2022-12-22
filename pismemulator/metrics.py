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

from typing import Any, Tuple

import torch
from torch import Tensor, tensor
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


class L2MeanSquaredError(Metric):
    r"""Computes an L2 regularized `mean squared error`_ (MSE):
    .. math:: \text{MSE} = \frac{1}{N}\sum_i^N(y_i - \hat{y_i})^2 + K\frac{1}{N}\sum_i^N(w_i)^2
    Where :math:`y` is a tensor of target values, :math:`\hat{y}` is a tensor of predictions, :math: `w` is a tensor of weights, and :math:`K` is a regularization constant. Equivalent to Mean Squared Error for :math:`K=0`.
    Args:
        squared: If True returns MSE value, if False returns RMSE value.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.
    Example:
        >>> from pismemulator.metrics import L2MeanSquaredError
        >>> target = torch.tensor([2.5, 5.0, 4.0, 8.0])
        >>> preds = torch.tensor([3.0, 5.0, 2.5, 7.0])
        >>> weight = torch.tensor([0.1, 0.2, 0.5, 0.2])
        >>> k = 1e-1
        >>> l2_mean_squared_error = L2MeanSquaredError()
        >>> l2_mean_squared_error(preds, target, weight, k)
        tensor(0.8835)
    """
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    sum_squared_error: Tensor
    total: int

    def __init__(
        self,
        squared: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_squared_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")
        self.squared = squared

    def update(self, preds: Tensor, target: Tensor, weight: Tensor, K: float) -> None:  # type: ignore
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model
            target: Ground truth values
            weight: linear weights
            k: regularization constant
        """
        sum_squared_error, n_obs = _l2_mean_squared_error_update(
            preds, target, weight, K
        )

        self.sum_squared_error += sum_squared_error
        self.total += n_obs

    def compute(self) -> Tensor:
        """Computes mean squared error over state."""
        return _l2_mean_squared_error_compute(
            self.sum_squared_error, self.total, squared=self.squared
        )


def _l2_mean_squared_error_update(
    preds: Tensor, target: Tensor, weight: Tensor, K: float
) -> Tuple[Tensor, int]:
    """Updates and returns variables required to compute Mean Squared Error.
    Checks for same shape of input tensors.
    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        weight: linear weights
        k: regularization constant
    """
    _check_same_shape(preds, target)
    diff = preds - target
    n_obs = target.numel()
    sum_squared_error = torch.sum(diff * diff) + K * torch.sum(weight * weight)
    return sum_squared_error, n_obs


def _l2_mean_squared_error_compute(
    sum_squared_error: Tensor, n_obs: int, squared: bool = True
) -> Tensor:
    """Computes Mean Squared Error with L2 regularization.
    Args:
        sum_squared_error: Sum of square of errors over all observations
        n_obs: Number of predictions or observations
        squared: Returns RMSE value if set to False.
    Example:
        >>> preds = torch.tensor([0., 1, 2, 3])
        >>> target = torch.tensor([0., 1, 2, 2])
        >>> weigth = torch.tensor([0.25, 0.5, 0.25, 0.25])
        >>> k = 1e-1
        >>> sum_squared_error, n_obs = _l2__mean_squared_error_update(preds, target, weight, k)
        >>> _l2_mean_squared_error_compute(sum_squared_error, n_obs)
        tensor(0.2609)
    """
    return (
        sum_squared_error / n_obs if squared else torch.sqrt(sum_squared_error / n_obs)
    )


def l2_mean_squared_error(
    preds: Tensor, target: Tensor, weight: Tensor, K: float, squared: bool = True
) -> Tensor:
    """Computes mean squared error with L2 regularization.
    Args:
        preds: estimated labels
        target: ground truth labels
        squared: returns RMSE value if set to False
    Return:
        Tensor with MSE
    Example:
        >>> from torchmetrics.functional import mean_squared_error
        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 2])
        >>> w = torch.tensor([0.25, 0.5, 0.25, 0.25])
        >>> k = 1e-1
        >>> mean_squared_error(x, y, w, k)
        tensor(0.2609)
    """
    sum_squared_error, n_obs = _l2_mean_squared_error_update(preds, target, weight, K)
    return _l2_mean_squared_error_compute(sum_squared_error, n_obs, squared=squared)
