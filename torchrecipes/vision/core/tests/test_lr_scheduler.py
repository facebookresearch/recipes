#!/usr/bin/env python3
# pyre-strict

import unittest
from typing import List, Union

import torch
from parameterized import parameterized
from torchrecipes.vision.core.optim.lr_scheduler import CosineWithWarmup


class TestCosineWithWarmup(unittest.TestCase):
    def _get_target_schedule(self) -> List[float]:
        return [
            0.001,
            0.0055,
            0.01,
            0.009619397662556433,
            0.008535533905932736,
            0.006913417161825449,
            0.004999999999999999,
            0.003086582838174551,
            0.0014644660940672624,
            0.00038060233744356627,
        ]

    # pyre-ignore[16]: Module parameterized.parameterized has attribute expand
    @parameterized.expand([(2,), (0.2,)])
    def test_lr_schedule(self, warmup_iters: Union[int, float]) -> None:
        """Tests learning rate matches expected schedule during model training."""
        test_parameter = torch.autograd.Variable(
            torch.randn([5, 5]), requires_grad=True
        )
        optimizer = torch.optim.SGD([test_parameter], lr=0.01)
        lr_scheduler = CosineWithWarmup(
            optimizer, warmup_start_factor=0.1, max_iters=10, warmup_iters=warmup_iters
        )

        target_schedule = self._get_target_schedule()

        for epoch in range(10):
            self.assertAlmostEqual(
                lr_scheduler.get_last_lr()[0], target_schedule[epoch]
            )
            lr_scheduler.step()
