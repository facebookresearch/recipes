#!/usr/bin/env python3
from typing import Union

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class CosineWithWarmup(SequentialLR):
    r"""Cosine Decay Learning Rate Scheduler with Linear Warmup.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_iters (int): Max number of iterations. (This should be number of epochs/steps
            based on the unit of scheduler's step size.)
        warmup_iters (int or float): number or fraction of iterations where
            linear warmup happens. Approaching the end of the linear warmup
            period the linear warmup line will intersect with the cosine decay curve.
            Default: 0
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        warmup_iters: Union[int, float] = 0,
        warmup_start_factor: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if isinstance(warmup_iters, float):
            warmup_iters = int(warmup_iters * max_iters)
        linear_lr = LinearLR(optimizer, warmup_start_factor, total_iters=warmup_iters)
        cosine_lr = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_iters)
        super().__init__(optimizer, [linear_lr, cosine_lr], [warmup_iters], last_epoch)
