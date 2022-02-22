#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import testslide
import torch
from torchrecipes.vision.image_classification.losses.soft_target_cross_entropy_loss import (
    SoftTargetCrossEntropyLoss,
)


class TestSoftTargetCrossEntropyLoss(testslide.TestCase):
    def _get_outputs(self) -> torch.Tensor:
        return torch.tensor([[1.0, 7.0, 0.0, 0.0, 2.0]])

    def _get_targets(self) -> torch.Tensor:
        return torch.tensor([[1, 0, 0, 0, 1]])

    def _get_loss(self) -> float:
        return 5.51097965

    def test_soft_target_cross_entropy(self) -> None:
        crit = SoftTargetCrossEntropyLoss(reduction="mean")
        outputs = self._get_outputs()
        targets = self._get_targets()
        self.assertAlmostEqual(crit(outputs, targets).item(), self._get_loss())

    def test_soft_target_cross_entropy_none_reduction(self) -> None:
        crit = SoftTargetCrossEntropyLoss(reduction="none")

        outputs = torch.tensor([[1.0, 7.0, 0.0, 0.0, 2.0], [4.0, 2.0, 1.0, 6.0, 0.5]])
        targets = torch.tensor([[1, 0, 0, 0, 1], [0, 1, 0, 1, 0]])
        loss = crit(outputs, targets)
        self.assertEqual(loss.numel(), outputs.size(0))

    def test_soft_target_cross_entropy_integer_label(self) -> None:
        crit = SoftTargetCrossEntropyLoss(reduction="mean")
        outputs = self._get_outputs()
        targets = torch.tensor([4])
        self.assertAlmostEqual(crit(outputs, targets).item(), 5.01097918)

    def test_unnormalized_soft_target_cross_entropy(self) -> None:
        crit = SoftTargetCrossEntropyLoss(reduction="none", normalize_targets=False)
        outputs = self._get_outputs()
        targets = self._get_targets()
        self.assertAlmostEqual(crit(outputs, targets).item(), 11.0219593)

    def test_deep_copy(self) -> None:
        crit = SoftTargetCrossEntropyLoss(reduction="mean")
        outputs = self._get_outputs()
        targets = self._get_targets()
        crit(outputs, targets)

        crit2 = copy.deepcopy(crit)
        self.assertAlmostEqual(crit2(outputs, targets).item(), self._get_loss())
