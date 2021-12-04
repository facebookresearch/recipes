# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, TypeVar, Tuple, Union, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

TBatch = TypeVar("TBatch")
TTrainReturn = TypeVar("TTrainReturn")
TTestReturn = TypeVar("TTestReturn")


class TaskBase(ABC, Generic[TBatch, TTrainReturn, TTestReturn]):
    """Abstract base class for a Task.

    This allows us to be more opinionated than the base LightningModule about
    what methods must be implemented in a Standard Tasks. Furthermore, if
    these methods are not implemented, failure will occur at instantiation rather
    than dynamically at runtime.

    For example, we enforce children classes to implement `validation_step`
    and `test_step` in addition to just `training_step` as required by the base
    LightningModules. When developing new models there's no strict need for
    `validation` and `test` phases, but for a standard Task we wish to enforce
    this to allow us to collect appropriate metrics.


    Example Usage:

        from pytorch_lightning import LightningModule
        from typing import TypedDict
        if TYPE_CHECKING:
            from torch import Tensor

        class MyTaskOutput(TypedDict):
            predictions: Tensor

        class MyTask(TaskBase[MyTaskOutput], LightningModule):
            def training_step(...):
                ...

            ...
    """

    """ LightningModule Methods. See corresponding methods in pl.LightningModule for documentation. """

    @abstractmethod
    def training_step(
        self, batch: TBatch, batch_idx: int, *args: Any, **kwargs: Any
    ) -> TTrainReturn:
        raise NotImplementedError

    @abstractmethod
    def validation_step(
        self, batch: TBatch, batch_idx: int, *args: Any, **kwargs: Any
    ) -> TTestReturn:
        raise NotImplementedError

    @abstractmethod
    def test_step(
        self, batch: TBatch, batch_idx: int, *args: Any, **kwargs: Any
    ) -> TTestReturn:
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Iterable[Optimizer], Iterable[_LRScheduler]]]:
        raise NotImplementedError

    """ Optional methods. Enforce static types for batch. """

    def on_train_batch_start(
        self,
        batch: TBatch,
        batch_idx: int,
        unused: Optional[int] = None,
    ) -> None:
        ...

    def on_train_batch_end(
        self,
        outputs: Iterable[TTrainReturn],
        batch: TBatch,
        batch_idx: int,
        unused: Optional[int] = None,
    ) -> None:
        ...

    def on_validation_batch_start(
        self, batch: TBatch, batch_idx: int, dataloader_idx: int
    ) -> None:
        ...

    def on_validation_batch_end(
        self,
        outputs: Iterable[TTestReturn],
        batch: TBatch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        ...

    def on_test_batch_start(
        self, batch: TBatch, batch_idx: int, dataloader_idx: int
    ) -> None:
        ...

    def on_test_batch_end(
        self,
        outputs: Iterable[TTestReturn],
        batch: TBatch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        ...
