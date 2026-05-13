# Copyright (C) 2021-25 Andy Aschwanden, Douglas C Brinkerhoff
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
# pylint: disable=arguments-differ
"""
Metrics Module.
"""

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.checks import _check_same_shape


def _area_weighted_error_update(
    preds: Tensor, target: Tensor, omegas: Tensor, area: Tensor
) -> Tensor:
    """
    Accumulate an area-weighted error over a batch.

    The function computes the per-sample spatial reduction
    ``sum_i (|preds - target|^2 * area[i])`` and then aggregates across the
    batch using the sample weights ``omegas``:

    .. math::

        E = \\sum_{b} \\omega_b \\sum_{i} (|p_{b,i} - t_{b,i}|^2)\\,A_i

    Parameters
    ----------
    preds : torch.Tensor
        Predicted field values with shape ``(B, N)`` where ``B`` is batch size
        and ``N`` is the number of spatial nodes.
    target : torch.Tensor
        Target/observed field values with shape ``(B, N)``.
    omegas : torch.Tensor
        Per-sample weights with shape ``(B,)`` or ``(B, 1)``. Will be squeezed.
    area : torch.Tensor
        Per-node area weights with shape ``(N,)`` or broadcastable to ``(B, N)``.

    Returns
    -------
    torch.Tensor
        Scalar tensor containing the accumulated area-weighted error over the batch.

    Raises
    ------
    ValueError
        If ``preds`` and ``target`` do not have the same shape.
    """
    _check_same_shape(preds, target)
    diff = preds - target
    sum_weighted_error = torch.sum(diff * diff * area, dim=1)  # spatial reduction
    _weighted_error = torch.sum(
        sum_weighted_error * omegas.squeeze()
    )  # batch reduction
    return _weighted_error


def _area_weighted_error_compute(_weighted_error: Tensor) -> Tensor:
    """
    Finalize the area-weighted error reduction.

    Parameters
    ----------
    _weighted_error : torch.Tensor
        Scalar tensor produced by :func:`_area_weighted_error_update`.

    Returns
    -------
    torch.Tensor
        The same scalar tensor (Lightning metric compatibility shim).

    Notes
    -----
    This function exists to mirror the ``update/compute`` pattern used by
    PyTorch-Lightning metrics. If you switch to a stateful Metric class, this
    would return the aggregated state instead.
    """
    return _weighted_error


def area_weighted_error(
    preds: Tensor, target: Tensor, omegas: Tensor, area: Tensor
) -> Tensor:
    """
    Compute the area weighted error between the predicted and target tensors.

    Parameters
    ----------
    preds : Tensor
        The predicted values as a PyTorch Tensor.
    target : Tensor
        The target values as a PyTorch Tensor.
    omegas : Tensor
        The omegas values as a PyTorch Tensor.
    area : Tensor
        The area values as a PyTorch Tensor.

    Returns
    -------
    Tensor
        The area weighted error as a PyTorch Tensor.

    Notes
    -----
    This function uses the '_area_weighted_error_update' and '_area_weighted_error_compute' functions
    to calculate the area weighted error.
    """
    sum_weighted_error = _area_weighted_error_update(preds, target, omegas, area)
    return _area_weighted_error_compute(sum_weighted_error)


class AreaWeightedError(Metric):
    """
    Area-weighted error aggregated over a batch (TorchMetrics-compatible).

    This metric accumulates an area-weighted discrepancy between ``preds`` and
    ``target`` over updates, and returns the aggregated value in :meth:`compute`.

    The actual per-batch discrepancy is delegated to
    :func:`_area_weighted_error_update`. Depending on that helper's definition,
    this metric behaves as an area-weighted L1 or L2-style error.

    Parameters
    ----------
    dist_sync_on_step : bool, optional
        If True, synchronizes internal state across distributed processes at each
        step. Default is False.

    Attributes
    ----------
    sum_weighted_error : torch.Tensor
        Running total of the area-weighted error. Reduced across DDP processes
        using ``sum``.
    full_state_update : bool
        TorchMetrics hint that per-batch updates are independent. Set to False.
    """

    full_state_update: bool = False
    sum_weighted_error: Tensor

    def __init__(self, dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            "sum_weighted_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(
        self, preds: Tensor, target: Tensor, omegas: Tensor, area: Tensor
    ) -> None:
        """
        Update metric state with a new batch.

        Parameters
        ----------
        preds : torch.Tensor
            Model predictions with shape ``(batch, n_nodes)`` (or broadcast-compatible
            with ``target``).
        target : torch.Tensor
            Reference/ground-truth values with the same shape as ``preds``.
        omegas : torch.Tensor
            Per-sample weights with shape ``(B,)`` or ``(B, 1)``. Will be squeezed.
        area : torch.Tensor
            Per-node area weights, shape ``(n_nodes,)`` (or broadcast-compatible).

        Returns
        -------
        None
            Updates internal state in place.
        """
        self.sum_weighted_error = self.sum_weighted_error + _area_weighted_error_update(
            preds, target, omegas, area
        )

    def compute(self) -> Tensor:
        """
        Compute the aggregated metric value.

        Returns
        -------
        torch.Tensor
            Aggregated area-weighted error as computed by
            :func:`_area_weighted_error_compute`.
        """
        return _area_weighted_error_compute(self.sum_weighted_error)

    @property
    def is_differentiable(self) -> bool:
        """
        Indicate whether the metric is differentiable.

        Returns
        -------
        bool
            True. (Note: TorchMetrics may still treat some metrics as non-differentiable
            depending on internal operations).
        """
        return True
