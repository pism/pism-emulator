#!/bin/env python3
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
"""
Write predictions to disk
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import lightning as pl
import torch
from lightning.pytorch.callbacks import BasePredictionWriter, Timer
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch import Tensor
from torch.utils.data import DataLoader


class DiskPredictionWriter(BasePredictionWriter):
    """Write each chain's predictions (and optional stats) to disk during predict.

    This callback is compatible with multi-process CPU runs (e.g. DDP spawn).
    Each process writes its own ``rankXX_chainYYYYYY.pt`` file containing a dict
    with at least ``{"chain": int, "rank": int, "samples": Tensor}`` and,
    if available, per-step statistics ``"lp"``, ``"step_size"``, and ``"accept"``.

    Parameters
    ----------
    out_dir : str
        Output directory where per-chain ``.pt`` files will be written.
    write_interval : {"batch", "epoch"}, optional
        Interval at which Lightning calls the writer. For prediction, ``"batch"``
        is typically used so each batch/chain is written as soon as it finishes.
        Default is ``"batch"``.
    """

    def __init__(self, out_dir: str, write_interval: str = "batch") -> None:
        super().__init__(write_interval=write_interval)
        self.out_dir = Path(out_dir)

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Any,
        batch_indices: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Handle predictions at the end of each batch.

        Lightning passes through whatever the model's ``predict_step`` returned.
        We support either a single dict or an iterable of dicts. Each dict must
        contain at least ``"chain"`` (int) and ``"samples"`` (Tensor of shape
        ``(S, D)`` on CPU). Optional keys ``"lp"``, ``"step_size"``, and
        ``"accept"`` (each ``(S,)`` CPU tensors) are included if present.

        Parameters
        ----------
        trainer : pl.Trainer
            The active trainer instance (used to read ``global_rank``).
        pl_module : pl.LightningModule
            The LightningModule doing prediction (unused here).
        prediction : Any
            The return value(s) from ``predict_step`` (dict or list[dict]).
        batch_indices : Any
            Indices of the current batch (unused).
        batch : Any
            The input batch (unused).
        batch_idx : int
            Batch index within the dataloader.
        dataloader_idx : int, optional
            Dataloader index when multiple predict dataloaders are used. Default is 0.

        Notes
        -----
        Files are written to ``out_dir / f"rank{rank:02d}_chain{chain:06d}.pt"``.
        """
        if prediction is None:
            return

        preds: Iterable[Dict[str, Any]]
        preds = prediction if isinstance(prediction, (list, tuple)) else [prediction]
        rank = int(getattr(trainer, "global_rank", 0))

        self.out_dir.mkdir(parents=True, exist_ok=True)

        for p in preds:
            chain = int(p["chain"])
            samples: Tensor = p["samples"]  # (S, D) CPU tensor

            rec: Dict[str, Any] = {"chain": chain, "rank": rank, "samples": samples}
            for key in ("lp", "step_size", "accept"):
                if key in p and p[key] is not None:
                    rec[key] = p[key]  # (S,) CPU tensors

            path = self.out_dir / f"rank{rank:02d}_chain{chain:06d}.pt"
            torch.save(rec, path)


def load_pred_dir(pred_dir: str | Path, expected_chains: int | None = None) -> Tensor:
    """Load per-chain prediction files and stack samples into a tensor.

    Parameters
    ----------
    pred_dir : str or pathlib.Path
        Directory containing the ``rank*_chain*.pt`` files written by
        :class:`DiskPredictionWriter`.
    expected_chains : int, optional
        If provided, verifies that the number of loaded chains matches this value.

    Returns
    -------
    torch.Tensor
        Stacked samples of shape ``(C, S, D)`` where ``C`` is number of chains,
        ``S`` samples per chain, and ``D`` parameter dimension.

    Raises
    ------
    RuntimeError
        If no prediction files are found or the number of chains mismatches.
    """
    pred_dir = Path(pred_dir)
    files = sorted(pred_dir.glob("rank*_chain*.pt"))
    records = [torch.load(f) for f in files]
    if not records:
        raise RuntimeError(f"No prediction files found in {pred_dir}")
    records.sort(key=lambda r: r["chain"])
    if expected_chains is not None and len(records) != expected_chains:
        raise RuntimeError(f"Expected {expected_chains} chains, found {len(records)}.")
    return torch.stack([r["samples"] for r in records])  # (C, S, D)


def load_pred_dir_with_stats(
    pred_dir: str | Path, expected_chains: int | None = None
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """Load per-chain predictions and optional statistics from disk.

    Parameters
    ----------
    pred_dir : str or pathlib.Path
        Directory with files produced by :class:`DiskPredictionWriter`.
    expected_chains : int, optional
        If provided, verifies that the number of loaded chains matches this value.

    Returns
    -------
    samples : torch.Tensor
        Array of shape ``(C, S, D)`` with post-burn samples.
    lp : torch.Tensor or None
        Log-probability per draw, shape ``(C, S)``, if present in files.
    step_size : torch.Tensor or None
        Per-draw step size, shape ``(C, S)``, if present.
    accept : torch.Tensor or None
        Boolean acceptance mask, shape ``(C, S)``, if present.

    Raises
    ------
    RuntimeError
        If no prediction files are found or the number of chains mismatches.

    Notes
    -----
    If any of the optional keys are missing from *any* chain file, that
    output will be returned as ``None``.
    """
    pred_dir = Path(pred_dir)
    files = sorted(pred_dir.glob("rank*_chain*.pt"))
    records = [torch.load(f) for f in files]
    if not records:
        raise RuntimeError(f"No prediction files found in {pred_dir}")
    records.sort(key=lambda r: r["chain"])
    if expected_chains is not None and len(records) != expected_chains:
        raise RuntimeError(f"Expected {expected_chains} chains, found {len(records)}.")

    samples = torch.stack([r["samples"] for r in records])  # (C, S, D)

    def _maybe_stack(key: str) -> Optional[Tensor]:
        if all((key in r) for r in records):
            return torch.stack([r[key] for r in records])  # (C, S)
        return None

    lp = _maybe_stack("lp")
    h = _maybe_stack("step_size")
    acc = _maybe_stack("accept")

    return samples, lp, h, acc
