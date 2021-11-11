from typing import Callable, List

from torch.optim import Optimizer, lr_scheduler


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
    return lr_scheduler.SequentialLR(optimizer, schedulers, milestones)
