#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch


class MixupScheme(Enum):
    """Mixup scheme: Where to perform mixup within a model."""


@dataclass
class MixupParams:
    """
    alpha: float. Mixup ratio. Recommended values from (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    scheme: MixupScheme. The locations to perform mixup within a model.
    More details about the parameters and mixup can be found in FAIM WIKI:
    https://fburl.com/wiki/3f0qh0zr
    """

    alpha: float
    scheme: MixupScheme


class MixupUtil:
    @staticmethod
    def _get_lambda(alpha: float = 1.0) -> float:
        # Sample from a beta distribution
        # The result is used for linear interpolation
        if alpha > 0.0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
        return lam

    def __init__(self, batch_size: int) -> None:
        self.indices: torch.Tensor = torch.randperm(batch_size)
        self.lam: float = self._get_lambda()

    def mixup(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.lam + x[self.indices] * (1 - self.lam)

    def compute_loss(
        self,
        criterion: torch.nn.Module,
        pred: torch.Tensor,
        original_target: torch.Tensor,
        mixed_target: torch.Tensor,
    ) -> float:
        return self.lam * criterion(pred, original_target) + (1 - self.lam) * criterion(
            pred, mixed_target
        )

    def mixup_labels(self, x: torch.Tensor) -> torch.Tensor:
        return x[self.indices]
