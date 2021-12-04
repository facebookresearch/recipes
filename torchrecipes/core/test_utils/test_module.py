# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

import random
from typing import Any, Callable, Optional, Tuple

import hydra
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


class TestModule(LightningModule):
    """
    A sample classification LightningModule for testing purpose.
    """

    def __init__(
        self,
        model: Any,  # pyre-ignore[2]
        optim: Any,  # pyre-ignore[2]
        loss: Any,  # pyre-ignore[2]
        metric: Optional[Any],  # pyre-ignore[2]
    ) -> None:
        super().__init__()

        self.model: torch.nn.Module = hydra.utils.instantiate(model)
        self.optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            optim, self.model.parameters()
        )
        self.loss: Callable[
            [torch.Tensor, torch.Tensor], STEP_OUTPUT
        ] = hydra.utils.instantiate(loss)

    # pyre-ignore[14]: torchscript doesn't support *args and **kwargs
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)

    def _step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        input, target = batch
        output = self.forward(input)
        return self.loss(output, target)

    def training_step(
        self, batch: torch.Tensor, batch_idx: int, *args: Any, **kwargs: Any
    ) -> STEP_OUTPUT:
        loss = self._step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int, *args: Any, **kwargs: Any
    ) -> None:
        loss = self._step(batch, batch_idx)
        self.log("val_loss", loss, on_step=True, on_epoch=True)

    def test_step(
        self, batch: torch.Tensor, batch_idx: int, *args: Any, **kwargs: Any
    ) -> None:
        loss = self._step(batch, batch_idx)
        self.log("test_loss", loss, on_step=True, on_epoch=True)

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> None:
        self._step(batch, batch_idx)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer


class RandomDataset(Dataset[torch.Tensor]):
    def __init__(self, size: int, length: int) -> None:
        self.len: int = length
        self.data: torch.Tensor = torch.randn(length, size)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return (self.data[index], random.randint(0, 1))

    def __len__(self) -> int:
        return self.len


class TestDataModule(LightningDataModule):
    def __init__(self, size: int, length: int) -> None:
        super().__init__()
        self.size = size
        self.length = length

    def train_dataloader(self) -> DataLoader[torch.Tensor]:
        return torch.utils.data.DataLoader(RandomDataset(self.size, self.length))

    def val_dataloader(self) -> DataLoader[torch.Tensor]:
        return torch.utils.data.DataLoader(RandomDataset(self.size, self.length))

    def test_dataloader(self) -> DataLoader[torch.Tensor]:
        return torch.utils.data.DataLoader(RandomDataset(self.size, self.length))

    def predict_dataloader(self) -> DataLoader[torch.Tensor]:
        return torch.utils.data.DataLoader(RandomDataset(self.size, self.length))
