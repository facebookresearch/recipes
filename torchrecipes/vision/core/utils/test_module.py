# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from pytorch_lightning import LightningModule
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


TTrainOutput = Dict[str, torch.Tensor]
TTestOutput = Dict[str, torch.Tensor]


class RandomDataset(Dataset[torch.Tensor]):
    def __init__(self, size: int, length: int) -> None:
        self.len: int = length
        self.data: torch.Tensor = torch.randn(length, size)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return self.len


class TestModule(LightningModule):
    def __init__(self, epoch_min_loss_override: Optional[int] = None) -> None:
        """LightningModule for testing purposes

        Args:
            epoch_min_loss_override (int, optional): Pass in an epoch that will be set to the minimum
                validation loss for testing purposes (zero based). If None this is ignored. Defaults to None.
        """
        super().__init__()
        self.layer = torch.nn.Linear(in_features=32, out_features=2)
        self.another_layer = torch.nn.Linear(in_features=2, out_features=2)
        self.epoch_min_loss_override = epoch_min_loss_override

    # pyre-ignore: Correct signature is forward(self, x, *args, **kwargs) but
    # torchscript does not support *args, **kwargs, so we ignore.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        return self.another_layer(x)

    def loss(self, batch: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def training_step(
        self, batch: torch.Tensor, batch_idx: int, *args: Any, **kwargs: Any
    ) -> TTrainOutput:
        output = self.forward(batch)
        loss = self.loss(batch, output)
        return {"output": output.detach(), "loss": loss, "checkpoint_on": loss.detach()}

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int, *args: Any, **kwargs: Any
    ) -> TTrainOutput:
        output = self.forward(batch)
        loss = self.loss(batch, output)
        return {"output": output.detach(), "loss": loss, "checkpoint_on": loss.detach()}

    def test_step(
        self, batch: torch.Tensor, batch_idx: int, *args: Any, **kwargs: Any
    ) -> TTestOutput:
        output = self.forward(batch)
        loss = self.loss(batch, output)
        return {"output": output.detach(), "loss": loss}

    # pyre-fixme[14]: `training_epoch_end` overrides method defined in
    #  `LightningModule` inconsistently.
    def training_epoch_end(self, outputs: Iterable[TTrainOutput]) -> None:
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("avg_loss", avg_loss)

    # pyre-fixme[14]: `validation_epoch_end` overrides method defined in
    #  `LightningModule` inconsistently.
    def validation_epoch_end(self, outputs: Iterable[TTrainOutput]) -> None:
        avg_val_loss = torch.stack(
            [torch.randn(1, requires_grad=True) for _ in outputs]
        ).mean()
        # For testing purposes allow a nominated epoch to have a low loss
        if self.current_epoch == self.epoch_min_loss_override:
            avg_val_loss -= 1e10
        self.log("val_loss", avg_val_loss)
        self.log("checkpoint_on", avg_val_loss)

    # pyre-fixme[14]: `test_epoch_end` overrides method defined in `LightningModule`
    #  inconsistently.
    def test_epoch_end(self, outputs: Iterable[TTestOutput]) -> None:
        avg_loss = torch.stack(
            [torch.randn(1, requires_grad=True) for _ in outputs]
        ).mean()
        self.log("val_loss", avg_loss)

    def configure_optimizers(
        self,
    ) -> Tuple[
        Iterable[torch.optim.Optimizer], Iterable[torch.optim.lr_scheduler.StepLR]
    ]:
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self) -> DataLoader[torch.Tensor]:
        return DataLoader(RandomDataset(32, 64))

    def val_dataloader(self) -> DataLoader[torch.Tensor]:
        return DataLoader(RandomDataset(32, 64))

    def test_dataloader(self) -> DataLoader[torch.Tensor]:
        return DataLoader(RandomDataset(32, 64))


class TrackedTestModule(TestModule):
    def __init__(self) -> None:
        super().__init__()
        self.num_epochs_seen = 0
        self.num_batches_seen = 0
        self.num_on_load_checkpoint_called = 0

    def on_train_epoch_end(self) -> None:
        self.num_epochs_seen += 1

    def on_train_batch_start(
        self, batch: torch.Tensor, batch_idx: int, unused: Optional[int] = None
    ) -> None:
        self.num_batches_seen += 1

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.num_on_load_checkpoint_called += 1
