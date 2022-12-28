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


import future, sys, os, datetime, argparse, copy, warnings, time
from collections.abc import MutableSequence, Iterable
from collections import OrderedDict
from itertools import compress
import numpy as np
from tqdm import tqdm
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams["figure.figsize"] = [10, 10]

import pandas as pd

import torch
from torch.nn import Module, Parameter
from torch.nn import Linear, Tanh, ReLU
import torch.nn.functional as F

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

from pismemulator.probmodel import ProbModel
from pismemulator.optimizer import (
    SGLD_Optim,
    MetropolisHastings_Optim,
    MALA_Optim,
    mMALA_Optim,
    HMC_Optim,
    SGNHT_Optim,
)
from pismemulator.acceptance import (
    SDE_Acceptance,
    MetropolisHastingsAcceptance,
)
from pismemulator.utils import RunningAverageMeter

"""
Python Container Time Complexity: https://wiki.python.org/moin/TimeComplexity
"""


class Chain(MutableSequence):

    """
    A container for storing the MCMC chain conveniently:
    samples: list of state_dicts
    log_probs: list of log_probs
    accepts: list of bools
    state_idx:
            init index of last accepted via np.where(accepts==True)[0][-1]
            can be set via len(samples) while sampling

    @property
    samples: filters the samples


    """

    def __init__(self, probmodel=None):

        super().__init__()

        if probmodel is None:
            """
            Create an empty chain
            """
            self.state_dicts = []
            self.log_probs = []
            self.accepts = []

        if probmodel is not None:
            """
            Initialize chain with given model
            """
            assert isinstance(probmodel, ProbModel)

            self.state_dicts = [copy.deepcopy(probmodel.state_dict())]
            log_prob = probmodel.log_prob(*next(probmodel.dataloader.__iter__()))
            log_prob["log_prob"].detach_()
            self.log_probs = [log_prob]
            self.accepts = [True]
            self.last_accepted_idx = 0

            self.running_avgs = {}
            for key, value in log_prob.items():
                self.running_avgs.update({key: RunningAverageMeter(0.99)})

        self.running_accepts = RunningAverageMeter(0.999)

    def __len__(self):
        return len(self.state_dicts)

    def __iter__(self):
        return zip(self.state_dicts, self.log_probs, self.accepts)

    def __delitem__(self):
        raise NotImplementedError

    def __setitem__(self):
        raise NotImplementedError

    def insert(self):
        raise NotImplementedError

    def __repr__(self):
        return f"MCMC Chain: Length:{len(self)} Accept:{self.accept_ratio:.2f}"

    def __getitem__(self, i):
        chain = copy.deepcopy(self)
        chain.state_dicts = self.samples[i]
        chain.log_probs = self.log_probs[i]
        chain.accepts = self.accepts[i]
        return chain

    def __add__(self, other):

        if type(other) in [tuple, list]:
            assert (
                len(other) == 3
            ), f"Invalid number of information pieces passed: {len(other)} vs len(Iterable(model, log_prob, accept, ratio))==4"
            self.append(*other)
        elif isinstance(other, Chain):
            self.cat(other)

        return self

    def __iadd__(self, other):

        if type(other) in [tuple, list]:
            assert (
                len(other) == 3
            ), f"Invalid number of information pieces passed: {len(other)} vs len(Iterable(model, log_prob, accept, ratio))==4"
            self.append(*other)
        elif isinstance(other, Chain):
            self.cat_chains(other)

        return self

    @property
    def state_idx(self):
        """
        Returns the index of the last accepted sample a.k.a. the state of the chain

        """
        if not hasattr(self, "state_idx"):
            """
            If the chain hasn't a state_idx, compute it from self.accepts by taking the last True of self.accepts
            """
            self.last_accepted_idx = np.where(self.accepts == True)[0][-1]
            return self.last_accepted_idx
        else:
            """
            Check that the state of the chain is actually the last True in self.accepts
            """
            last_accepted_sample_ = np.where(self.accepts == True)[0][-1]
            assert last_accepted_sample_ == self.last_accepted_idx
            assert self.accepts[self.last_accepted_idx] == True
            return self.last_accepted_idx

    @property
    def samples(self):
        """
        Filters the list of state_dicts with the list of bools from self.accepts
        :return: list of accepted state_dicts
        """
        return list(compress(self.state_dicts, self.accepts))

    @property
    def accept_ratio(self):
        """
        Sum the boolean list (=total number of Trues) and divides it by its length
        :return: float valued accept ratio
        """
        return sum(self.accepts) / len(self.accepts)

    @property
    def state(self):
        return {
            "state_dict": self.state_dicts[self.last_accepted_idx],
            "log_prob": self.log_probs[self.last_accepted_idx],
        }

    def cat_chains(self, other):

        assert isinstance(other, Chain)
        self.state_dicts += other.state_dicts
        self.log_probs += other.log_probs
        self.accepts += other.accepts

        for key, value in other.running_avgs.items():
            self.running_avgs[key].avg = (
                0.5 * self.running_avgs[key].avg + 0.5 * other.running_avgs[key].avg
            )

    def append(self, probmodel, log_prob, accept):

        if isinstance(probmodel, ProbModel):
            params_state_dict = copy.deepcopy(probmodel.state_dict())
        elif isinstance(probmodel, OrderedDict):
            params_state_dict = copy.deepcopy(probmodel)
        assert isinstance(log_prob, dict)
        assert type(log_prob["log_prob"]) == torch.Tensor
        assert log_prob["log_prob"].numel() == 1

        log_prob["log_prob"].detach_()

        self.accepts.append(accept)
        self.running_accepts.update(1 * accept)

        if accept:
            self.state_dicts.append(params_state_dict)
            self.log_probs.append(log_prob)
            self.last_accepted_idx = len(self.state_dicts) - 1
            for key, value in log_prob.items():
                self.running_avgs[key].update(value.item())

        elif not accept:
            self.state_dicts.append(False)
            self.log_probs.append(False)


class Sampler_Chain:
    def __init__(
        self,
        probmodel,
        params,
        step_size,
        num_steps,
        save_interval=None,
        save_dir=".",
        save_format="csv",
        burn_in=True,
        pretrain=True,
        tune=True,
    ):

        self.probmodel = probmodel
        self.chain = Chain(probmodel=self.probmodel)

        self.step_size = step_size
        self.num_steps = num_steps
        self.burn_in = burn_in
        self.save_interval = save_interval
        self.save_dir = save_dir
        self.save_format = save_format
        self.params = params
        self.pretrain = pretrain
        self.tune = tune

    def propose(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def tune_step_size(self):

        tune_interval_length = 100
        num_tune_intervals = int(self.burn_in // tune_interval_length)

        verbose = True

        print(f'Tuning: Init Step Size: {self.optim.param_groups[0]["step_size"]:.5f}')

        self.probmodel.reset_parameters()
        tune_chain = Chain(probmodel=self.probmodel)
        tune_chain.running_accepts.momentum = 0.5

        progress = tqdm(range(self.burn_in))
        for tune_step in progress:
            sample_log_prob, sample = self.propose()
            accept, log_ratio = self.acceptance(
                sample_log_prob["log_prob"], self.chain.state["log_prob"]["log_prob"]
            )
            tune_chain += (self.probmodel, sample_log_prob, accept)

            # if tune_step < self.burn_in and tune_step % tune_interval_length == 0 and tune_step > 0:
            if tune_step > 1:
                # self.optim.dual_average_tune(tune_chain, np.exp(log_ratio.item()))
                self.optim.dual_average_tune(
                    tune_chain.accepts[-tune_interval_length:],
                    tune_step,
                    np.exp(log_ratio.item()),
                )
                # self.optim.tune(tune_chain.accepts[-tune_interval_length:])

            if not accept:

                if torch.isnan(sample_log_prob["log_prob"]):
                    print("Chain log_prob is NaN", self.chain.state)
                    exit()
                self.probmodel.load_state_dict(self.chain.state["state_dict"])

            desc = f'Tuning: Accept: {tune_chain.running_accepts.avg:.2f}/{tune_chain.accept_ratio:.2f} StepSize: {self.optim.param_groups[0]["step_size"]:.5f}'

            progress.set_description(desc=desc)

        time.sleep(0.1)  # for cleaner printing in the console

    def sample_chain(self):

        save_interval = self.save_interval
        save_dir = self.save_dir
        save_format = self.save_format

        params = self.params
        self.probmodel.reset_parameters()
        if self.pretrain:
            self.probmodel.pretrain()

        if self.tune:
            self.tune_step_size()

        self.chain = Chain(probmodel=self.probmodel)

        progress = tqdm(range(self.num_steps))
        for step in progress:
            proposal_log_prob, sample = self.propose()
            accept, log_ratio = self.acceptance(
                proposal_log_prob["log_prob"], self.chain.state["log_prob"]["log_prob"]
            )
            self.chain += (self.probmodel, proposal_log_prob, accept)

            if not accept:

                if torch.isnan(proposal_log_prob["log_prob"]):
                    print(f"Step: {step}: log_prob is NaN\n", self.chain.state)
                    exit()
                self.probmodel.load_state_dict(self.chain.state["state_dict"])

            if step % save_interval == 0:
                X_post = (
                    torch.vstack(
                        [
                            self.chain.samples[k]["X"]
                            for k in range(len(self.chain.samples))
                        ]
                    )
                    * self.probmodel.X_std
                    + self.probmodel.X_mean
                )
                X_posterior = X_post.detach().numpy()
                posterior_dir = f"{save_dir}/posterior_samples/"
                if not os.path.isdir(posterior_dir):
                    os.makedirs(posterior_dir)

                df = pd.DataFrame(
                    data=X_posterior,
                    columns=self.probmodel.X_keys,
                )
                if save_format == "csv":
                    df.to_csv(
                        os.path.join(posterior_dir, f"X_posterior_model_0.csv.gz")
                    )
                elif save_format == "parquet":
                    df.to_parquet(
                        os.path.join(posterior_dir, f"X_posterior_model_0.parquet")
                    )
                else:
                    raise NotImplementedError(f"{out_format} not implemented")
            desc = f"{str(self)}: Accept: {self.chain.running_accepts.avg:.2f}/{self.chain.accept_ratio:.2f} \t"
            for key, running_avg in self.chain.running_avgs.items():
                desc += f" {key}: {running_avg.avg:.2f} "
            desc += f'StepSize: {self.optim.param_groups[0]["step_size"]:.3f}'
            progress.set_description(desc=desc)

        self.chain = self.chain[self.burn_in :]

        return self.chain


class SGLD_Chain(Sampler_Chain):
    def __init__(
        self,
        probmodel,
        step_size=0.0001,
        num_steps=2000,
        burn_in=100,
        pretrain=False,
        tune=False,
    ):

        Sampler_Chain.__init__(
            self, probmodel, step_size, num_steps, burn_in, pretrain, tune
        )

        self.optim = SGLD_Optim(
            self.probmodel, step_size=step_size, prior_std=1.0, addnoise=True
        )

        self.acceptance = SDE_Acceptance()

    def __repr__(self):
        return "SGLD"

    @torch.enable_grad()
    def propose(self):

        self.optim.zero_grad()
        batch = next(self.probmodel.dataloader.__iter__())
        log_prob = self.probmodel.log_prob(*batch)
        (-log_prob["log_prob"]).backward()
        self.optim.step()

        return log_prob, self.probmodel


class mMALA_Chain(Sampler_Chain):
    def __init__(
        self,
        probmodel,
        params=None,
        step_size=0.1,
        num_steps=2000,
        save_interval=1000,
        save_dir=".",
        save_format="csv",
        burn_in=100,
        pretrain=False,
        tune=False,
        num_chain=0,
    ):

        Sampler_Chain.__init__(
            self,
            probmodel,
            params,
            step_size,
            num_steps,
            save_interval,
            save_dir,
            save_format,
            burn_in,
            pretrain,
            tune,
        )

        self.num_chain = num_chain
        self.optim = mMALA_Optim(
            probmodel,
            params=params,
            step_size=step_size,
        )
        self.acceptance = MetropolisHastingsAcceptance()

    def __repr__(self):
        return "mMALA"

    @torch.enable_grad()
    def propose(self):

        self.optim.zero_grad()
        batch = next(self.probmodel.dataloader.__iter__())
        # self.probmodel.reset_parameters()
        log_prob = self.probmodel.log_prob(*batch)
        (-log_prob["log_prob"]).backward()
        self.optim.step()

        return log_prob, self.probmodel


class MALA_Chain(Sampler_Chain):
    def __init__(
        self,
        probmodel,
        step_size=0.1,
        num_steps=2000,
        burn_in=100,
        pretrain=False,
        tune=False,
        num_chain=0,
    ):

        Sampler_Chain.__init__(
            self, probmodel, step_size, num_steps, burn_in, pretrain, tune
        )

        self.num_chain = num_chain

        self.optim = MALA_Optim(
            self.probmodel, step_size=step_size, prior_std=1.0, addnoise=True
        )

        self.acceptance = MetropolisHastingsAcceptance()
        # self.acceptance = SDE_Acceptance()

    def __repr__(self):
        return "MALA"

    @torch.enable_grad()
    def propose(self):

        self.optim.zero_grad()
        batch = next(self.probmodel.dataloader.__iter__())
        log_prob = self.probmodel.log_prob(*batch)
        (-log_prob["log_prob"]).backward()
        self.optim.step()

        return log_prob, self.probmodel
