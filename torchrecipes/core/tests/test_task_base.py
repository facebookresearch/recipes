# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

from typing import Any, TypedDict, Optional

import testslide
from pytorch_lightning import LightningModule
from torch.optim import Optimizer
from torchrecipes.core.task_base import TaskBase


class Output(TypedDict):
    metric: int


class TestTaskBase(testslide.TestCase):
    def test_enforce_abstract_method(self) -> None:
        class IncompleteTask(TaskBase[int, float, float], LightningModule):
            def training_step(
                self, batch: int, batch_index: int, *args: Any, **kwargs: Any
            ) -> float:
                return 10

        with self.assertRaises(TypeError):
            # pyre-ignore[45]: Pyre catches this statically as well, but we ignore
            # for the purpose of verifying that an error is raised dynamically
            # as well.
            _ = IncompleteTask()

    def test_enforce_type_checks(self) -> None:
        class CompleteClassWrongTypes(
            TaskBase[Output, Output, Output], LightningModule
        ):
            def training_step(
                self, batch: Output, batch_index: int, *args: Any, **kwargs: Any
            ) -> str:
                return "10"

            def validation_step(
                self, batch: Output, batch_index: int, *args: Any, **kwargs: Any
            ) -> Output:
                return Output(metric=10)

            def test_step(
                self, batch: Output, batch_index: int, *args: Any, **kwargs: Any
            ) -> Output:
                return Output(metric=12)

            def configure_optimizers(self) -> Optional[Optimizer]:
                return None

        # This "works" but pyre will catch the fact the user is returning "10"
        # instead of Output.
        _ = CompleteClassWrongTypes()
