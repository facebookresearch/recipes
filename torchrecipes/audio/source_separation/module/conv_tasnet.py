# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3
# pyre-strict

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

Batch = Union[List[torch.Tensor], Mapping[str, torch.Tensor]]


class ConvTasNetModule(LightningModule):
    """
    The Lightning Module for speech separation.

    Args:
        loss (Any): The loss function to use.
        optim (Any): The optimizer function to use.
        metrics (List of methods): The metrics to track, which will be used for both train and validation.
        lr_scheduler (Any or None, optional): The LR Scheduler.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Callable,
        optim_fn: Callable[Iterable[Parameter], Optimizer],
        metrics: Mapping[str, Callable],
        lr_scheduler: Optional[_LRScheduler] = None,
    ) -> None:
        super().__init__()
        self.model: nn.Module = model
        self.loss: Callable = loss
        self.optim: Optimizer = optim_fn(self.model.parameters())
        self.lr_scheduler: Optional[_LRScheduler] = (
            lr_scheduler(self.optim) if lr_scheduler else None
        )
        self.metrics: Mapping[str, Callable] = metrics

        self.train_metrics: Dict = {}
        self.val_metrics: Dict = {}
        self.test_metrics: Dict = {}
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            self.train_metrics.update(self.metrics)
            self.val_metrics.update(self.metrics)
        else:
            self.test_metrics.update(self.metrics)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward defines the prediction/inference actions.
        """
        return self.model(x)

    def training_step(self, batch: Batch, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self._step(batch, subset="train")

    def validation_step(
        self, batch: Batch, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Operates on a single batch of data from the validation set.
        """
        return self._step(batch, subset="val")

    def test_step(
        self, batch: Batch, *args: Any, **kwargs: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Operates on a single batch of data from the test set.
        """
        return self._step(batch, subset="test")

    def _step(self, batch: Batch, subset: str) -> Dict[str, Any]:
        """
        Common step for training, validation, and testing.
        """
        mix, src, mask = batch
        pred = self.model(mix)
        loss = self.loss(pred, src, mask)
        self.log(f"losses/{subset}_loss", loss.item(), on_step=True, on_epoch=True)

        metrics_result = self._compute_metrics(pred, src, mix, mask, subset)
        self.log_dict(metrics_result, on_epoch=True)

        return loss

    def configure_optimizers(self) -> Tuple[Any]:
        epoch_schedulers = {
            "scheduler": self.lr_scheduler,
            "monitor": "losses/val_loss",
            "interval": "epoch",
        }
        return [self.optim], [epoch_schedulers]

    def _compute_metrics(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        subset: str,
    ) -> Dict[str, torch.Tensor]:
        metrics_dict = getattr(self, f"{subset}_metrics")
        metrics_result = {}
        for name, metric in metrics_dict.items():
            metrics_result[f"metrics/{subset}/{name}"] = metric(
                pred, label, inputs, mask
            )
        return metrics_result
