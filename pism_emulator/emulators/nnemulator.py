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

# pylint: disable=arguments-differ,too-many-instance-attributes,too-many-lines
"""
Neural Network Emulators.
"""
import math
import warnings
from argparse import ArgumentParser

import lightning as pl
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR, _LRScheduler

from pism_emulator.metrics import AreaWeightedError, area_weighted_error


def _kaiming_init(module: nn.Module) -> None:
    """
    Initialize ``nn.Linear`` layers with Kaiming-uniform weights and zero biases.

    This helper is intended to be passed to :meth:`torch.nn.Module.apply`, e.g.::

        model.apply(_kaiming_init)

    Parameters
    ----------
    module : torch.nn.Module
        Module to initialize. If ``module`` is an instance of :class:`torch.nn.Linear`,
        its ``weight`` is initialized with :func:`torch.nn.init.kaiming_uniform_` and
        its ``bias`` (when present) is set to zeros. Other module types are left
        unchanged.

    Returns
    -------
    None
        This function modifies ``module`` in place.
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class MLPBlock(nn.Module):
    """
    Feed-forward MLP block used in residual MLP emulators.

    This block applies the following sequence:

    ``Linear -> LayerNorm -> Dropout -> Activation``

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    p_dropout : float, optional
        Dropout probability applied after normalization. Default is 0.0.
    activation : {"relu", "silu", "gelu"}, optional
        Activation function name (case-insensitive). Default is ``"relu"``.

    Raises
    ------
    ValueError
        If ``activation`` is not one of ``{"relu", "silu", "gelu"}``.

    Notes
    -----
    The forward order mirrors the legacy implementation:
    ``Linear -> Norm -> Dropout -> Activation``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        p_dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)

        act = activation.lower()
        if act == "relu":
            self.act: nn.Module = nn.ReLU()
        elif act == "silu":
            self.act = nn.SiLU()
        elif act == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation '{activation}'")

        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the block to an input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(..., in_features)``.

        Returns
        -------
        torch.Tensor
            Output tensor with shape ``(..., out_features)``.
        """
        return self.act(self.drop(self.norm(self.lin(x))))


class DNNEmulator(pl.LightningModule):
    """
    Deep neural network emulator for PISM glacier fields.

    This model maps a vector of input parameters ``x`` to a set of coefficients in a
    reduced basis (the "eigenglacier" basis), and reconstructs full fields in physical
    space via a fixed matrix ``V_hat``:

    .. math::

        \\hat{F}(x) = c(x) V_{\\hat{}}^T \\, (+\\, F_{\\mathrm{mean}})

    where ``c(x)`` are learned coefficients and ``F_mean`` is an optional mean field
    added back during reconstruction.

    Parameters
    ----------
    n_parameters : int
        Number of input parameters (feature dimension of ``x``).
    V_hat : torch.Tensor
        Fixed basis matrix used to reconstruct fields from coefficients.
        Shape ``(n_nodes, n_eigenglaciers)``.
    F_mean : torch.Tensor
        Mean field in physical space. Shape ``(n_nodes,)`` (or broadcastable
        to ``(batch, n_nodes)``).
    area : torch.Tensor
        Per-node area weights used in area-weighted loss/metrics. Shape
        ``(n_nodes,)`` (or broadcastable).
    n_eigenglaciers : int or None, optional
        Number of basis vectors / coefficient dimension. If None, inferred
        from ``V_hat.shape[1]``. Default is None.
    **hparams : object
        Additional hyperparameters controlling the network and optimization.
        Common keys include:

        * ``width`` : int, hidden width (default 128)
        * ``depth`` : int, number of residual blocks (default 4)
        * ``dropout`` : float, dropout probability (default 0.5)
        * ``activation`` : {"relu","silu","gelu"}, activation function (default "relu")
        * ``learning_rate`` : float, optimizer learning rate (default 1e-2)
        * ``compile`` : bool, whether to try :func:`torch.compile` (default False)

    Notes
    -----
    * ``V_hat``, ``F_mean``, and ``area`` are registered as buffers (non-trainable)
      so they are stored in the model state but not optimized.
    * Optionally compiles the forward pass with :func:`torch.compile` when available.
    """

    def __init__(
        self,
        n_parameters: int,
        V_hat: Tensor,
        F_mean: Tensor,
        area: Tensor,
        n_eigenglaciers: int | None = None,
        **hparams: object,
    ) -> None:
        """
        Initialize the emulator.

        Parameters
        ----------
        n_parameters : int
            Number of input parameters (feature dimension of ``x``).
        V_hat : torch.Tensor
            Fixed basis matrix used to reconstruct fields from coefficients.
            Shape ``(n_nodes, n_eigenglaciers)``.
        F_mean : torch.Tensor
            Mean field in physical space. Shape ``(n_nodes,)`` (or broadcastable
            to ``(batch, n_nodes)``).
        area : torch.Tensor
            Per-node area weights used in area-weighted loss/metrics. Shape
            ``(n_nodes,)`` (or broadcastable).
        n_eigenglaciers : int or None, optional
            Number of basis vectors / coefficient dimension. If None, inferred
            from ``V_hat.shape[1]``. Default is None.
        **hparams : object
            Additional hyperparameters controlling the network and optimization.
            Common keys include:

            * ``width`` : int, hidden width (default 128)
            * ``depth`` : int, number of residual blocks (default 4)
            * ``dropout`` : float, dropout probability (default 0.5)
            * ``activation`` : {"relu","silu","gelu"}, activation function (default "relu")
            * ``learning_rate`` : float, optimizer learning rate (default 1e-2)
            * ``compile`` : bool, whether to try :func:`torch.compile` (default False)

        Raises
        ------
        ValueError
            If ``n_eigenglaciers`` is None and ``V_hat`` is None (cannot infer output size).

        Notes
        -----
        Only scalar hyperparameters are saved directly; large tensors are stored as
        buffers and also saved (CPU copies) in the hyperparameter dict for reproducibility.
        """
        super().__init__()
        flat = vars(hparams) if hasattr(hparams, "__dict__") else dict(hparams)

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

        self.inp = MLPBlock(n_parameters, width, 0.0, activation=activation)
        self.blocks = nn.ModuleList(
            [
                MLPBlock(width, width, p_drop, activation=activation)
                for _ in range(depth)
            ]
        )
        self.head = nn.Linear(width, n_eigenglaciers)

        self.register_buffer("V_hat", V_hat, persistent=True)
        self.register_buffer("F_mean", F_mean, persistent=True)
        self.register_buffer("area", area, persistent=True)

        self.train_ae = AreaWeightedError()
        self.test_ae = AreaWeightedError()

        self.apply(_kaiming_init)

        self._compiled = False

        if bool(self.hparams.get("compile", False)) and hasattr(torch, "compile"):
            try:
                self.forward = torch.compile(self.forward, dynamic=True)  # type: ignore[method-assign]
                self._compiled = True
            except (RuntimeError, TypeError, ValueError) as e:
                self._compiled = False
                warnings.warn(
                    "torch.compile was requested but failed; continuing without compilation.\n"
                    f"Reason: {type(e).__name__}: {e}\n"
                    "To silence this, set hparams['compile']=False. For debug: "
                    "TORCH_LOGS=+dynamo TORCHDYNAMO_VERBOSE=1.",
                    category=RuntimeWarning,
                    stacklevel=2,
                )

    def forward(self, x: Tensor, add_mean: bool = False) -> Tensor:
        """
        Compute reconstructed fields from input parameters.

        Parameters
        ----------
        x : torch.Tensor
            Input parameter tensor with shape ``(batch, n_parameters)``.
        add_mean : bool, optional
            If True, add the stored mean field ``F_mean`` to the reconstruction.
            Default is False.

        Returns
        -------
        torch.Tensor
            Reconstructed fields with shape ``(batch, n_nodes)``.
        """
        z = self.inp(x)
        for block in self.blocks:
            z = block(z) + z
        coeffs = self.head(z)
        F_pred = coeffs @ self.V_hat.T
        if add_mean:
            F_pred = F_pred + self.F_mean
        return F_pred

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """
        Add DNNEmulator-specific CLI arguments to an existing parser.

        Parameters
        ----------
        parent_parser : argparse.ArgumentParser
            Parser to which DNNEmulator arguments will be added.

        Returns
        -------
        argparse.ArgumentParser
            The updated parser with DNNEmulator options added.
        """
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
        """
        Configure optimizer and learning-rate scheduler.

        Returns
        -------
        list[torch.optim.Optimizer]
            List containing the configured optimizer.
        list[dict[str, torch.optim.lr_scheduler._LRScheduler]]
            List containing a scheduler configuration dictionary compatible with Lightning.
        """
        opt = torch.optim.Adam(
            self.parameters(), lr=float(self.hparams.learning_rate), weight_decay=0.0
        )
        sch = {"scheduler": ExponentialLR(opt, gamma=0.9975)}
        return [opt], [sch]

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        """
        Lightning training step.

        Parameters
        ----------
        batch : tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Batch tuple ``(x, f, o, o_0)`` where:

            * ``x`` : input parameters, shape ``(batch, n_parameters)``
            * ``f`` : target responses (mean-centered or transformed as configured),
              shape ``(batch, n_nodes)``
            * ``o`` : weights (e.g., omegas), broadcastable to ``f``
            * ``o_0`` : auxiliary weights (unused here)

        batch_idx : int
            Batch index (unused).

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        _ = batch_idx
        x, f, o, _ = batch
        f_pred = self.forward(x)
        loss = area_weighted_error(f_pred, f, o, self.area)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:
        """
        Lightning validation step.

        Parameters
        ----------
        batch : tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Batch tuple ``(x, f, o, o_0)`` (see :meth:`training_step`).
        batch_idx : int
            Batch index (unused).

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of tensors useful for epoch-end hooks/logging.
        """
        _ = batch_idx
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

    def on_validation_epoch_end(self) -> None:
        """
        Log aggregated validation metrics at the end of each validation epoch.
        """
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
    """
    Deep neural network emulator for PISM glacier fields.

    This model maps a vector of input parameters ``x`` to a set of coefficients in a
    reduced basis (the "eigenglacier" basis), and reconstructs full fields in physical
    space via a fixed matrix ``V_hat``:

    .. math::

        \\hat{F}(x) = c(x) V_{\\hat{}}^T \\, (+\\, F_{\\mathrm{mean}})

    where ``c(x)`` are learned coefficients and ``F_mean`` is an optional mean field
    added back during reconstruction.

    Parameters
    ----------
    n_parameters : int
        Number of input parameters (feature dimension of ``x``).
    V_hat : torch.Tensor
        Fixed basis matrix used to reconstruct fields from coefficients.
        Shape ``(n_nodes, n_eigenglaciers)``.
    F_mean : torch.Tensor
        Mean field in physical space. Shape ``(n_nodes,)`` (or broadcastable
        to ``(batch, n_nodes)``).
    area : torch.Tensor
        Per-node area weights used in area-weighted loss/metrics. Shape
        ``(n_nodes,)`` (or broadcastable).
    n_eigenglaciers : int or None, optional
        Number of basis vectors / coefficient dimension. If None, inferred
        from ``V_hat.shape[1]``. Default is None.
    **hparams : object
        Additional hyperparameters controlling the network and optimization.
    """

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

        self.train_ae = AreaWeightedError()
        self.test_ae = AreaWeightedError()

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
        """
        Add NNEmulator-specific CLI arguments to an existing parser.

        Parameters
        ----------
        parent_parser : argparse.ArgumentParser
            Parser to which DNNEmulator arguments will be added.

        Returns
        -------
        argparse.ArgumentParser
            The updated parser with DNNEmulator options added.
        """
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
        _ = batch_idx
        x, f, o, _ = batch
        f_pred = self.forward(x)
        area = self.area
        loss = area_weighted_error(f_pred, f, o, area)

        return loss

    def validation_step(self, batch, batch_idx):
        _ = batch_idx
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
    """
    Deep neural network emulator for PISM glacier fields.

    This model maps a vector of input parameters ``x`` to a set of coefficients in a
    reduced basis (the "eigenglacier" basis), and reconstructs full fields in physical
    space via a fixed matrix ``V_hat``:

    .. math::

        \\hat{F}(x) = c(x) V_{\\hat{}}^T \\, (+\\, F_{\\mathrm{mean}})

    where ``c(x)`` are learned coefficients and ``F_mean`` is an optional mean field
    added back during reconstruction.

    Parameters
    ----------
    n_parameters : int
        Number of input parameters (feature dimension of ``x``).
    V_hat : torch.Tensor
        Fixed basis matrix used to reconstruct fields from coefficients.
        Shape ``(n_nodes, n_eigenglaciers)``.
    F_mean : torch.Tensor
        Mean field in physical space. Shape ``(n_nodes,)`` (or broadcastable
        to ``(batch, n_nodes)``).
    area : torch.Tensor
        Per-node area weights used in area-weighted loss/metrics. Shape
        ``(n_nodes,)`` (or broadcastable).
    n_eigenglaciers : int or None, optional
        Number of basis vectors / coefficient dimension. If None, inferred
        from ``V_hat.shape[1]``. Default is None.
    **hparams : object
        Additional hyperparameters controlling the network and optimization.
    """

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

        self.train_ae = AreaWeightedError()
        self.test_ae = AreaWeightedError()

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
        """
        Add NNEmulator-specific CLI arguments to an existing parser.

        Parameters
        ----------
        parent_parser : argparse.ArgumentParser
            Parser to which NNEmulator arguments will be added.

        Returns
        -------
        argparse.ArgumentParser
            The updated parser with NNEmulator options added.
        """
        parser = parent_parser.add_argument_group("NNEmulator")
        parser.add_argument("--n_hidden", type=int, default=128)
        parser.add_argument("--learning_rate", type=float, default=0.01)

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
        _ = batch_idx
        x, f, o, _ = batch
        f_pred = self.forward(x)
        area = self.area
        loss = area_weighted_error(f_pred, f, o, area)

        return loss

    def validation_step(self, batch, batch_idx):
        _ = batch_idx
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
    """
    Deep neural network emulator for PISM glacier fields.

    This model maps a vector of input parameters ``x`` to a set of coefficients in a
    reduced basis (the "eigenglacier" basis), and reconstructs full fields in physical
    space via a fixed matrix ``V_hat``:

    .. math::

        \\hat{F}(x) = c(x) V_{\\hat{}}^T \\, (+\\, F_{\\mathrm{mean}})

    where ``c(x)`` are learned coefficients and ``F_mean`` is an optional mean field
    added back during reconstruction.

    Parameters
    ----------
    n_parameters : int
        Number of input parameters (feature dimension of ``x``).
    n_eigenglaciers : int or None, optional
        Number of basis vectors / coefficient dimension. If None, inferred
        from ``V_hat.shape[1]``. Default is None.
    V_hat : torch.Tensor
        Fixed basis matrix used to reconstruct fields from coefficients.
        Shape ``(n_nodes, n_eigenglaciers)``.
    F_mean : torch.Tensor
        Mean field in physical space. Shape ``(n_nodes,)`` (or broadcastable
        to ``(batch, n_nodes)``).
    area : torch.Tensor
        Per-node area weights used in area-weighted loss/metrics. Shape
        ``(n_nodes,)`` (or broadcastable).
    hparams : dict[str, Any] | argparse.Namespace | lightning.pytorch.utilities.parsing.AttributeDict
        Hyperparameters/configuration used to build the network and control training.
        The expected keys depend on your implementation (e.g., layer sizes, learning
        rate, dropout, etc.).
    *args : Any
        Additional positional arguments forwarded to ``pl.LightningModule`` (if any).
    **kwargs : Any
        Additional keyword arguments forwarded to ``pl.LightningModule`` (if any).
    """

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

        self.train_ae = AreaWeightedError()
        self.test_ae = AreaWeightedError()

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
        """
        Add NNEmulator-specific CLI arguments to an existing parser.

        Parameters
        ----------
        parent_parser : argparse.ArgumentParser
            Parser to which NNEmulator arguments will be added.

        Returns
        -------
        argparse.ArgumentParser
            The updated parser with NNEmulator options added.
        """
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
            "scheduler": ExponentialLR(optimizer, 0.9975),
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        _ = batch_idx
        x, f, o, _ = batch
        f_pred = self.forward(x)
        loss = area_weighted_error(f_pred, f, o, self.area)

        return loss

    def validation_step(self, batch, batch_idx):
        _ = batch_idx
        x, f, o, o_0 = batch
        f_pred = self.forward(x)

        self.log("train_loss", self.train_ae(f_pred, f, o, self.area))
        self.log("test_loss", self.test_ae(f_pred, f, o_0, self.area))

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
