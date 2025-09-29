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
Neural Network Emulators
"""
import math
from argparse import ArgumentParser
from collections import OrderedDict

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, _LRScheduler

from pism_emulator.metrics import AreaAbsoluteError, area_absolute_error


def _kaiming_init(module: nn.Module) -> None:
    """Kaiming-uniform init for Linear layers; zero-init biases."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class MLPBlock(nn.Module):
    """A single MLP block: Linear -> Norm -> Dropout -> Activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        p_dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=True)
        self.norm = nn.LayerNorm(out_features)

        if activation.lower() == "relu":
            self.act: nn.Module = nn.ReLU()
        elif activation.lower() == "silu":
            self.act = nn.SiLU()
        elif activation.lower() == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation '{activation}'")

        self.drop = nn.Dropout(p_dropout)

    # mirror old order: Linear -> Norm -> Dropout -> ReLU
    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        """Forward pass."""
        return self.act(self.drop(self.norm(self.lin(x))))


class DNNEmulator(pl.LightningModule):
    """
    A deeper residual MLP emulator that maps parameter vectors ``x``
    to glacier field coefficients then back to physical space via
    a fixed basis ``V_hat`` and optional mean ``F_mean``.

    Notes
    -----
    - Uses residual skip inside each block (pre-activation style).
    - ``V_hat`` and ``F_mean`` are stored as non-trainable buffers so they land in state_dict but are not optimized.
    """

    def __init__(
        self,
        n_parameters,
        V_hat,
        F_mean: Tensor,
        area,
        n_eigenglaciers: int | None = None,
        **hparams,
    ) -> None:
        super().__init__()
        flat = vars(hparams) if hasattr(hparams, "__dict__") else dict(hparams)

        # infer n_eigenglaciers if not provided
        if n_eigenglaciers is None:
            if V_hat is None:
                raise ValueError(
                    "n_eigenglaciers is None and V_hat is None; cannot infer output size."
                )
            n_eigenglaciers = int(V_hat.shape[1])
        self.save_hyperparameters(
            {
                **flat,
                "n_parameters": int(n_parameters),
                "n_eigenglaciers": int(n_eigenglaciers),
                "V_hat": V_hat.detach().cpu(),
                "F_mean": F_mean.detach().cpu(),
                "area": area.detach().cpu(),
            }
        )

        width: int = int(self.hparams.get("width", 128))
        depth: int = int(self.hparams.get("depth", 4))
        p_drop: float = float(self.hparams.get("dropout", 0.5))
        activation: str = str(self.hparams.get("activation", "relu"))

        # input projection
        self.inp = MLPBlock(n_parameters, width, 0.0, activation=activation)

        # residual stack
        self.blocks = nn.ModuleList(
            [
                MLPBlock(width, width, p_drop, activation=activation)
                for _ in range(depth)
            ]
        )

        # output head to coefficient space, then project back with V_hat
        self.head = nn.Linear(width, n_eigenglaciers, bias=False)

        # buffers (non-trainable, but saved with state)
        self.register_buffer("V_hat", V_hat, persistent=True)
        self.register_buffer("F_mean", F_mean, persistent=True)
        self.register_buffer("area", area, persistent=True)

        # metrics
        self.train_ae = AreaAbsoluteError()
        self.test_ae = AreaAbsoluteError()

        # init
        self.apply(_kaiming_init)

        # optional torch.compile for speed on PyTorch ≥ 2.0
        self._compiled = False
        if bool(self.hparams.get("compile", False)) and hasattr(torch, "compile"):
            try:
                self.forward = torch.compile(self.forward, dynamic=True)  # type: ignore[method-assign]
                self._compiled = True
            except Exception:
                # Fall back silently if not supported on the current platform
                pass

    def forward(self, x: Tensor, add_mean: bool = False) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shape (batch, n_parameters).
        add_mean : bool
            If True, add ``F_mean`` back to the reconstruction.

        Returns
        -------
        Tensor
            Reconstructed fields ``F_pred`` of shape (batch, n_nodes).
        """
        z = self.inp(x)
        for block in self.blocks:
            z = block(z) + z  # residual
        coeffs = self.head(z)  # (batch, n_eigenglaciers)
        F_pred = coeffs @ self.V_hat.T  # (batch, n_nodes)
        if add_mean:
            F_pred = F_pred + self.F_mean
        return F_pred

    # ----- Lightning plumbing -----

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("DNNEmulator")
        parser.add_argument("--width", type=int, default=128)
        parser.add_argument("--depth", type=int, default=4)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument(
            "--norm", type=str, default="batch", choices=["batch", "layer", "none"]
        )
        parser.add_argument(
            "--activation", type=str, default="relu", choices=["relu", "silu", "gelu"]
        )
        parser.add_argument("--learning_rate", type=float, default=1e-2)
        parser.add_argument(
            "--compile", action="store_true", help="Use torch.compile if available"
        )
        return parent_parser

    def configure_optimizers(
        self,
    ) -> tuple[list[Optimizer], list[dict[str, _LRScheduler]]]:
        opt = torch.optim.Adam(
            self.parameters(), lr=float(self.hparams.learning_rate), weight_decay=0.0
        )
        sch = {"scheduler": ExponentialLR(opt, gamma=0.9975)}
        return [opt], [sch]

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        x, f, o, _ = batch
        f_pred = self.forward(x)
        area = self.area
        loss = area_absolute_error(f_pred, f, o, area)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:
        x, f, o, o_0 = batch
        f_pred = self.forward(x)

        self.log(
            "train_loss",
            self.train_ae(f_pred, f, o, self.area),
            sync_dist=True,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_loss",
            self.test_ae(f_pred, f, o_0, self.area),
            sync_dist=True,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return {"x": x, "f": f, "f_pred": f_pred, "o": o, "o_0": o_0}

    def on_validation_epoch_end(self):
        self.log(
            "train_loss",
            self.train_ae,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test_loss",
            self.test_ae,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )


class NNEmulator(pl.LightningModule):
    def __init__(
        self,
        n_parameters,
        V_hat,
        F_mean,
        area,
        n_eigenglaciers: int | None = None,
        **hparams,
    ):
        super().__init__()
        flat = vars(hparams) if hasattr(hparams, "__dict__") else dict(hparams)
        # infer n_eigenglaciers if not provided
        if n_eigenglaciers is None:
            if V_hat is None:
                raise ValueError(
                    "n_eigenglaciers is None and V_hat is None; cannot infer output size."
                )
            n_eigenglaciers = int(V_hat.shape[1])
        self.save_hyperparameters(
            {
                **flat,
                "n_parameters": int(n_parameters),
                "n_eigenglaciers": int(n_eigenglaciers),
                "V_hat": V_hat.detach().cpu(),
                "F_mean": F_mean.detach().cpu(),
                "area": area.detach().cpu(),
            }
        )
        n_hidden = self.hparams.get("n_hidden", 128)

        # Inputs to hidden layer linear transformation
        self.l_1 = nn.Linear(n_parameters, n_hidden)
        self.norm_1 = nn.LayerNorm(n_hidden)
        self.dropout_1 = nn.Dropout(p=0.0)
        self.l_2 = nn.Linear(n_hidden, n_hidden)
        self.norm_2 = nn.LayerNorm(n_hidden)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.l_3 = nn.Linear(n_hidden, n_hidden)
        self.norm_3 = nn.LayerNorm(n_hidden)
        self.dropout_3 = nn.Dropout(p=0.5)
        self.l_4 = nn.Linear(n_hidden, n_hidden)
        self.norm_4 = nn.LayerNorm(n_hidden)
        self.dropout_4 = nn.Dropout(p=0.5)
        self.l_5 = nn.Linear(n_hidden, n_eigenglaciers)

        self.register_buffer("V_hat", V_hat, persistent=True)
        self.register_buffer("F_mean", F_mean, persistent=True)
        self.register_buffer("area", area, persistent=True)

        self.train_ae = AreaAbsoluteError()
        self.test_ae = AreaAbsoluteError()

    def forward(self, x, add_mean=False):
        # Pass the input tensor through each of our operations

        a_1 = self.l_1(x)
        a_1 = self.norm_1(a_1)
        a_1 = self.dropout_1(a_1)
        z_1 = torch.relu(a_1)

        a_2 = self.l_2(z_1)
        a_2 = self.norm_2(a_2)
        a_2 = self.dropout_2(a_2)
        z_2 = torch.relu(a_2) + z_1

        a_3 = self.l_3(z_2)
        a_3 = self.norm_3(a_3)
        a_3 = self.dropout_3(a_3)
        z_3 = torch.relu(a_3) + z_2

        a_4 = self.l_4(z_3)
        a_4 = self.norm_4(a_4)
        a_4 = self.dropout_4(a_4)
        z_4 = torch.relu(a_4) + z_3

        z_5 = self.l_5(z_4)

        if add_mean:
            F_pred = z_5 @ self.V_hat.T + self.F_mean
        else:
            F_pred = z_5 @ self.V_hat.T
        return F_pred

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("NNEmulator")
        parser.add_argument("--n_hidden", type=int, default=128)
        parser.add_argument("--learning_rate", type=float, default=0.1)

        return parent_parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), self.hparams.learning_rate, weight_decay=0.0
        )
        # This is an approximation to Doug's version:
        scheduler = {
            "scheduler": ExponentialLR(optimizer, 0.9975),
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, f, o, _ = batch
        f_pred = self.forward(x)
        area = self.area
        loss = area_absolute_error(f_pred, f, o, area)

        return loss

    def validation_step(self, batch, batch_idx):
        x, f, o, o_0 = batch
        f_pred = self.forward(x)
        area = self.area

        self.log(
            "train_loss",
            self.train_ae(f_pred, f, o, area),
            sync_dist=True,
        )
        self.log(
            "test_loss",
            self.test_ae(f_pred, f, o_0, area),
            sync_dist=True,
        )

        return {"x": x, "f": f, "f_pred": f_pred, "o": o, "o_0": o_0}

    def on_validation_epoch_end(self):
        self.log(
            "train_loss",
            self.train_ae,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test_loss",
            self.test_ae,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_after_backward(self):
        if self.global_rank == 0:
            unused = [
                n
                for n, p in self.named_parameters()
                if p.requires_grad and p.grad is None
            ]
            if unused:
                print(
                    "UNUSED PARAMS THIS STEP:",
                    unused[:10],
                    "   " if len(unused) > 10 else "",
                )


class NN5Emulator(pl.LightningModule):
    def __init__(
        self,
        n_parameters,
        V_hat,
        F_mean,
        area,
        n_eigenglaciers: int | None = None,
        **hparams,
    ):
        super().__init__()
        flat = vars(hparams) if hasattr(hparams, "__dict__") else dict(hparams)
        # infer n_eigenglaciers if not provided
        if n_eigenglaciers is None:
            if V_hat is None:
                raise ValueError(
                    "n_eigenglaciers is None and V_hat is None; cannot infer output size."
                )
            n_eigenglaciers = int(V_hat.shape[1])
        self.save_hyperparameters(
            {
                **flat,
                "n_parameters": int(n_parameters),
                "n_eigenglaciers": int(n_eigenglaciers),
                "V_hat": V_hat.detach().cpu(),
                "F_mean": F_mean.detach().cpu(),
                "area": area.detach().cpu(),
            }
        )
        n_hidden = self.hparams.get("n_hidden", 128)

        # Inputs to hidden layer linear transformation
        self.l_1 = nn.Linear(n_parameters, n_hidden)
        self.norm_1 = nn.LayerNorm(n_hidden)
        self.dropout_1 = nn.Dropout(p=0.0)
        self.l_2 = nn.Linear(n_hidden, n_hidden)
        self.norm_2 = nn.LayerNorm(n_hidden)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.l_3 = nn.Linear(n_hidden, n_hidden)
        self.norm_3 = nn.LayerNorm(n_hidden)
        self.dropout_3 = nn.Dropout(p=0.5)
        self.l_4 = nn.Linear(n_hidden, n_hidden)
        self.norm_4 = nn.LayerNorm(n_hidden)
        self.dropout_4 = nn.Dropout(p=0.5)
        self.l_5 = nn.Linear(n_hidden, n_hidden)
        self.norm_5 = nn.LayerNorm(n_hidden)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.l_6 = nn.Linear(n_hidden, n_eigenglaciers)

        self.register_buffer("V_hat", V_hat, persistent=True)
        self.register_buffer("F_mean", F_mean, persistent=True)
        self.register_buffer("area", area, persistent=True)

        self.train_ae = AreaAbsoluteError()
        self.test_ae = AreaAbsoluteError()

    def forward(self, x, add_mean=False):
        # Pass the input tensor through each of our operations

        a_1 = self.l_1(x)
        a_1 = self.norm_1(a_1)
        a_1 = self.dropout_1(a_1)
        z_1 = torch.relu(a_1)

        a_2 = self.l_2(z_1)
        a_2 = self.norm_2(a_2)
        a_2 = self.dropout_2(a_2)
        z_2 = torch.relu(a_2) + z_1

        a_3 = self.l_3(z_2)
        a_3 = self.norm_3(a_3)
        a_3 = self.dropout_3(a_3)
        z_3 = torch.relu(a_3) + z_2

        a_4 = self.l_4(z_3)
        a_4 = self.norm_4(a_4)
        a_4 = self.dropout_4(a_4)
        z_4 = torch.relu(a_4) + z_3

        a_5 = self.l_5(z_4)
        a_5 = self.norm_5(a_5)
        a_5 = self.dropout_5(a_5)
        z_5 = torch.relu(a_5) + z_4

        z_6 = self.l_6(z_5)

        if add_mean:
            F_pred = z_6 @ self.V_hat.T + self.F_mean
        else:
            F_pred = z_6 @ self.V_hat.T
        return F_pred

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("NNEmulator")
        parser.add_argument("--n_hidden", type=int, default=128)
        parser.add_argument("--learning_rate", type=float, default=0.1)

        return parent_parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), self.hparams.learning_rate, weight_decay=0.0
        )
        # This is an approximation to Doug's version:
        scheduler = {
            "scheduler": ExponentialLR(optimizer, 0.9975),
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, f, o, _ = batch
        f_pred = self.forward(x)
        area = self.area
        loss = area_absolute_error(f_pred, f, o, area)

        return loss

    def validation_step(self, batch, batch_idx):
        x, f, o, o_0 = batch
        f_pred = self.forward(x)
        area = self.area

        self.log(
            "train_loss",
            self.train_ae(f_pred, f, o, area),
            sync_dist=True,
        )
        self.log(
            "test_loss",
            self.test_ae(f_pred, f, o_0, area),
            sync_dist=True,
        )

        return {"x": x, "f": f, "f_pred": f_pred, "o": o, "o_0": o_0}

    def on_validation_epoch_end(self):
        self.log(
            "train_loss",
            self.train_ae,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test_loss",
            self.test_ae,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_after_backward(self):
        if self.global_rank == 0:
            unused = [
                n
                for n, p in self.named_parameters()
                if p.requires_grad and p.grad is None
            ]
            if unused:
                print(
                    "UNUSED PARAMS THIS STEP:",
                    unused[:10],
                    "   " if len(unused) > 10 else "",
                )


class LegacyNNEmulator(pl.LightningModule):
    def __init__(
        self,
        n_parameters,
        n_eigenglaciers,
        V_hat,
        F_mean,
        area,
        hparams,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        n_hidden_1 = self.hparams.n_hidden_1
        n_hidden_2 = self.hparams.n_hidden_2
        n_hidden_3 = self.hparams.n_hidden_3
        n_hidden_4 = self.hparams.n_hidden_4

        # Inputs to hidden layer linear transformation
        self.l_1 = nn.Linear(n_parameters, n_hidden_1)
        self.norm_1 = nn.LayerNorm(n_hidden_1)
        self.dropout_1 = nn.Dropout(p=0.0)
        self.l_2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.norm_2 = nn.LayerNorm(n_hidden_2)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.l_3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.norm_3 = nn.LayerNorm(n_hidden_3)
        self.dropout_3 = nn.Dropout(p=0.5)
        self.l_4 = nn.Linear(n_hidden_3, n_hidden_4)
        self.norm_4 = nn.LayerNorm(n_hidden_4)
        self.dropout_4 = nn.Dropout(p=0.5)
        self.l_5 = nn.Linear(n_hidden_4, n_eigenglaciers)

        self.V_hat = torch.nn.Parameter(V_hat, requires_grad=False)
        self.F_mean = torch.nn.Parameter(F_mean, requires_grad=False)

        self.register_buffer("area", area)

        self.train_ae = AreaAbsoluteError()
        self.test_ae = AreaAbsoluteError()

    def forward(self, x, add_mean=False):
        # Pass the input tensor through each of our operations

        a_1 = self.l_1(x)
        a_1 = self.norm_1(a_1)
        a_1 = self.dropout_1(a_1)
        z_1 = torch.relu(a_1)

        a_2 = self.l_2(z_1)
        a_2 = self.norm_2(a_2)
        a_2 = self.dropout_2(a_2)
        z_2 = torch.relu(a_2) + z_1

        a_3 = self.l_3(z_2)
        a_3 = self.norm_3(a_3)
        a_3 = self.dropout_3(a_3)
        z_3 = torch.relu(a_3) + z_2

        a_4 = self.l_4(z_3)
        a_4 = self.norm_4(a_4)
        a_4 = self.dropout_4(a_4)
        z_4 = torch.relu(a_4) + z_3

        z_5 = self.l_5(z_4)
        if add_mean:
            F_pred = z_5 @ self.V_hat.T + self.F_mean
        else:
            F_pred = z_5 @ self.V_hat.T

        return F_pred

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("NNEmulator")
        parser.add_argument("--n_hidden_1", type=int, default=128)
        parser.add_argument("--n_hidden_2", type=int, default=128)
        parser.add_argument("--n_hidden_3", type=int, default=128)
        parser.add_argument("--n_hidden_4", type=int, default=128)
        parser.add_argument("--learning_rate", type=float, default=0.01)

        return parent_parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), self.hparams.learning_rate, weight_decay=0.0
        )
        # This is an approximation to Doug's version:
        scheduler = {
            "scheduler": ExponentialLR(optimizer, 0.9975, verbose=True),
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, f, o, _ = batch
        f_pred = self.forward(x)
        loss = area_absolute_error(f_pred, f, o, self.area)

        return loss

    def validation_step(self, batch, batch_idx):
        x, f, o, o_0 = batch
        f_pred = self.forward(x)

        self.log("train_loss", self.train_ae(f_pred, f, o, self.area))
        self.log("test_loss", self.test_ae(f_pred, f, o_0, self.area))

        return {"x": x, "f": f, "f_pred": f_pred, "o": o, "o_0": o_0}

    def validation_epoch_end(self, outputs):

        self.log(
            "train_loss",
            self.train_ae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test_loss",
            self.test_ae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
