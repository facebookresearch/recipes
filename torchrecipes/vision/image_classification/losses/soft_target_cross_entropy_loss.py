# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

import torch


def _convert_to_one_hot(targets: torch.Tensor, classes: int) -> torch.Tensor:
    """
    This function converts target class indices to one-hot vectors,
    given the number of classes.

    """
    if torch.max(targets).item() >= classes:
        raise ValueError("Class Index must be less than number of classes")
    one_hot_targets = torch.zeros(
        (targets.shape[0], classes), dtype=torch.long, device=targets.device
    )
    one_hot_targets.scatter_(1, targets.long(), 1)
    return one_hot_targets


class SoftTargetCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """This loss allows the targets for the cross entropy loss to be multi-label.

    Args:
        reduction (str): specifies reduction to apply to the output.
        normalize_targets (bool): whether the targets should be normalized to a sum of 1
            based on the total count of positive targets for a given sample.
    """

    def __init__(
        self,
        reduction: str = "mean",
        normalize_targets: bool = True,
    ) -> None:
        super().__init__(reduction=reduction)
        self.normalize_targets = normalize_targets
        self._eps: float = torch.finfo(torch.float32).eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.detach().clone()
        # Check if targets are inputted as class integers
        if target.ndim == 1:
            if input.shape[0] != target.shape[0]:
                raise ValueError(
                    "SoftTargetCrossEntropyLoss requires input and target to have same batch size!"
                )
            target = _convert_to_one_hot(target.view(-1, 1), input.shape[1])
        target = target.float()
        if self.normalize_targets:
            target /= self._eps + target.sum(dim=1, keepdim=True)

        return super().forward(input, target)
