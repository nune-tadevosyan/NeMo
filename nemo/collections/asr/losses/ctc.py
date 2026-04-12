# ! /usr/bin/python
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn.functional as F
from torch import nn

from nemo.core.classes import Serialization, Typing, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType

__all__ = ['CTCLoss']


class CTCLoss(nn.CTCLoss, Serialization, Typing):
    @property
    def input_types(self):
        """Input types definitions for CTCLoss.
        """
        return {
            "log_probs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "input_lengths": NeuralType(tuple('B'), LengthsType()),
            "target_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Output types definitions for CTCLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, num_classes, zero_infinity=False, reduction='mean_batch', alpha=0.0):
        self._blank = num_classes
        # Don't forget to properly call base constructor
        if reduction not in ['none', 'mean', 'sum', 'mean_batch', 'mean_volume']:
            raise ValueError('`reduction` must be one of [mean, sum, mean_batch, mean_volume]')

        self.config_reduction = reduction
        if reduction == 'mean_batch' or reduction == 'mean_volume':
            ctc_reduction = 'none'
            self._apply_reduction = True
        elif reduction in ['sum', 'mean', 'none']:
            ctc_reduction = reduction
            self._apply_reduction = False
        super().__init__(blank=self._blank, reduction=ctc_reduction, zero_infinity=zero_infinity)

        # Label prior scaling (arXiv 2406.02560).
        # When alpha > 0, log_probs are shifted by -alpha * log(P(k)) before the CTC DP,
        # penalising over-represented tokens (blank ~80%) and boosting rare ones.
        # alpha = 0 disables the feature entirely (standard CTC behaviour).
        # Paper recommends alpha = 0.3; paper reports alpha > 0.3 causes convergence failure.
        self.alpha = alpha
        num_tokens = num_classes + 1  # vocabulary + blank
        # Initialise to uniform: log(1/V) for every token.
        # Updated at the end of each training epoch via update_priors().
        self.register_buffer('log_priors', torch.full((num_tokens,), -math.log(num_tokens)))

    def update_priors(self, counts: torch.Tensor):
        """Recompute log_priors from per-token occurrence counts.

        Args:
            counts: 1-D tensor of shape [num_classes + 1] containing raw token counts
                    (including blank at index self._blank), already all-reduced across GPUs.
        """
        counts = counts.float()
        probs = (counts + 1e-8) / (counts.sum() + 1e-8 * counts.numel())
        self.log_priors.copy_(torch.log(probs))

    def reduce(self, losses, target_lengths):
        if self.config_reduction == 'mean_batch':
            losses = losses.mean()  # global batch size average
        elif self.config_reduction == 'mean_volume':
            losses = losses.sum() / target_lengths.sum()  # same as above but longer samples weigh more

        return losses

    @typecheck()
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # override forward implementation
        # custom logic, if necessary
        input_lengths = input_lengths.long()
        target_lengths = target_lengths.long()
        targets = targets.long()
        if self.alpha != 0.0:
            # Shift log-probs by -alpha * log(P(k)).  Broadcasts [B, T, D] - [D].
            # Re-normalise with log_softmax so the result is a valid log-prob distribution
            # compatible with PyTorch's CTCLoss (the paper used k2 to avoid this requirement;
            # see arXiv 2406.02560, Section 3.3).
            log_probs = F.log_softmax(log_probs - self.alpha * self.log_priors, dim=-1)
        # here we transpose because we expect [B, T, D] while PyTorch assumes [T, B, D]
        log_probs = log_probs.transpose(1, 0)
        loss = super().forward(
            log_probs=log_probs, targets=targets, input_lengths=input_lengths, target_lengths=target_lengths
        )
        if self._apply_reduction:
            loss = self.reduce(loss, target_lengths)
        return loss
