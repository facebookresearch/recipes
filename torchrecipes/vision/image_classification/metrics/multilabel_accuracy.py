# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

from typing import Tuple

import torch
from torchmetrics.metric import Metric


class MultilabelAccuracy(Metric):
    """Computes top-k accuracy for multilabel targets. A sample is considered
    correctly classified if the top-k predictions contain any of the labels.

    Args:
        top_k: Number of highest score predictions considered to find the
            correct label.
        dist_sync_on_step: Synchronize metric state across processes at each
            forward() before returning the value at the step.
    """

    def __init__(
        self, top_k: int, compute_on_step: bool = True, dist_sync_on_step: bool = False
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step
        )

        self._top_k = top_k
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Updates the state with predictions and target.
        Args:
            preds: tensor of shape (B, C) where each value is either logit or
                class probability.
            target: tensor of shape (B, C), which is one-hot / multi-label
                encoded.
        """
        assert preds.shape == target.shape, (
            "predictions and  target must be of the same shape. "
            f"Got preds({preds.shape}) vs target({target.shape})."
        )
        num_classes = target.shape[1]
        assert (
            num_classes >= self._top_k
        ), f"top-k({self._top_k}) is greater than the number of classes({num_classes})"
        preds, target = self._format_inputs(preds, target)

        # pyre-ignore[16]: torch.Tensor has attribute topk
        _, top_idx = preds.topk(self._top_k, dim=1, largest=True, sorted=True)

        # pyre-ignore[16]: Accuracy has attribute correct
        self.correct += (
            torch.gather(target, dim=1, index=top_idx[:, : self._top_k])
            .max(dim=1)
            .values.sum()
            .item()
        )
        # pyre-ignore[16]: Accuracy has attribute total
        self.total += preds.shape[0]

    def compute(self) -> torch.Tensor:
        if torch.is_nonzero(self.total):
            return self.correct / self.total
        return torch.tensor(0.0)

    @staticmethod
    def _format_inputs(
        preds: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Since .topk() is not compatible with fp16, we promote the predictions to full precision
        return (preds.float(), target.int())
