# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

import pickle
from abc import ABC, abstractmethod

import testslide
from pytorch_lightning import LightningModule


class TaskTestCaseBase(ABC, testslide.TestCase):
    """All Standard Task unit tests should inherit from this class."""

    @abstractmethod
    def get_standard_task(self) -> LightningModule:
        """Subclasses should implement a standard method of retrieving and instance of the Task to test."""
        raise NotImplementedError

    def test_standard_task_is_torchscriptable(self) -> None:
        task = self.get_standard_task()
        _ = task.to_torchscript()

    def test_standard_task_is_pickleable(self) -> None:
        task = self.get_standard_task()
        pickle.dumps(task)
