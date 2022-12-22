# Copyright (C) 2022 Andy Aschwanden, Douglas C Brinkerhoff
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


class ProbModel(torch.nn.Module):

    """
    ProbModel:

    """

    def __init__(self, dataloader):
        super().__init__()
        assert isinstance(dataloader, torch.utils.data.DataLoader)
        self.dataloader = dataloader

    def log_prob(self):
        """
        If minibatches have to be sampled due to memory constraints,
        a standard PyTorch dataloader can be used.
        "Infinite minibatch sampling" can be achieved by calling:
        data, target = next(dataloader.__iter__())
        next(Iterable.__iter__()) calls a single mini-batch sampling step
        But since it's not in a loop, we can call it add infinum
        """
        raise NotImplementedError

    def sample_minibatch(self):
        """
        Idea:
        Hybrid Monte Carlo Samplers require a constant tuple (data, target) to compute trajectories
        """
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError

    def predict(self, chain):
        raise NotImplementedError

    def pretrain(self):

        pass
