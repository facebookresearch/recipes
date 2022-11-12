# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable, List

from torch.optim import lr_scheduler, Optimizer


def sequential_lr(
    optimizer: Optimizer,
    scheduler_fns: List[Callable[[Optimizer], lr_scheduler._LRScheduler]],
    milestones: List[int],
) -> lr_scheduler.SequentialLR:
    """Helper function to construct SequentialLR with scheduler callables.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        scheduler_fns (List[Callable]): List of chained scheduler callables.
        milestones (List[int]): List of integers that reflects milestone points.
    """
    schedulers = [fn(optimizer) for fn in scheduler_fns]
    # pyre-fixme[6]: For 2nd param expected `List[LRScheduler]` but got
    #  `List[_LRScheduler]`.
    return lr_scheduler.SequentialLR(optimizer, schedulers, milestones)
