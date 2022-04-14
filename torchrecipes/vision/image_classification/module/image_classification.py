# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyre_extensions import none_throws
from torch.nn import Parameter
from torch.optim import Optimizer
from torchvision.ops._utils import split_normalization_params


OptimCallable = Callable[[Union[Iterable[Parameter], List[Dict[str, Any]]]], Optimizer]
LRSchedulerCallable = Callable[[Optimizer], torch.optim.lr_scheduler._LRScheduler]
Batch = Union[List[torch.Tensor], Mapping[str, torch.Tensor]]


class ImageClassificationModule(pl.LightningModule):
    """
    Generic module for image classification.

    Args:
        model: model instance.
        loss: loss function used for computing the loss.
        optim_fn: callable that returns an optimizer.
        metrics: meters to calculate metrics during.
        lr_scheduler_fn: callable that returns a LR scheduler.
        apply_softmax: whether to apply softmax on prediction.
        process_weighted_labels: whether to process weighted labels.
        norm_weight_decay: weight decay for batch norm layers.
        lr_scheduler_interval: interval to update learning rate. It should
            be either "epoch" or "step".
    """

    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optim: OptimCallable,
        metrics: Mapping[str, nn.Module],
        lr_scheduler: Optional[LRSchedulerCallable] = None,
        apply_softmax: bool = False,
        process_weighted_labels: bool = False,
        norm_weight_decay: float = 0.0,
        lr_scheduler_interval: str = "epoch",
    ) -> None:
        super().__init__()
        self.model: nn.Module = model
        self.loss: nn.Module = loss
        self.optim_fn: OptimCallable = optim
        self.lr_scheduler_fn: Optional[LRSchedulerCallable] = lr_scheduler
        self.metrics: nn.ModuleDict = nn.ModuleDict(metrics)
        self.apply_softmax: bool = apply_softmax
        self.process_weighted_labels: bool = process_weighted_labels
        self.norm_weight_decay: float = norm_weight_decay
        self.lr_scheduler_interval: str = lr_scheduler_interval

    def _postprocess_preds(self, preds: torch.Tensor) -> torch.Tensor:
        """
        This function postprocesses extracted predictions by running softmax
        if user requests it.

        For additional flexibility, this function can be overridden
        after subclassing.
        """
        # Shape of prediction tensor should be (batch_size, ...)
        # Softmax will sum to 1 along last dimension
        return F.softmax(preds, dim=-1) if self.apply_softmax else preds

    def _postprocess_labels(self, labels: torch.Tensor, phase: str) -> torch.Tensor:
        """
        This function postprocesses weighted label vector by converting it
        to index label by applying argmax if user requests it.
        For example, it converts [[0.1, 0.9, 0],[0.7, 0, 0.3]] to [1, 0].

        For additional flexibility, this function can be overridden
        after subclassing.
        """
        # TODO: Remove `process_weighted_labels` when torchmetrics support weighted labels
        # https://torchmetrics.readthedocs.io/en/stable/references/modules.html#input-types
        if self.process_weighted_labels and phase == "train":
            return torch.argmax(labels, dim=-1)
        return labels

    def compute_metrics(
        self, pred: torch.Tensor, label: torch.Tensor, phase: str
    ) -> Dict[str, torch.Tensor]:
        return {
            f"metrics/{phase}_{name}": metric(pred, label)
            for name, metric in self.metrics.items()
        }

    def forward(self, input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.model(input)

    def _step(self, batch: Batch, phase: str) -> Dict[str, Any]:
        if isinstance(batch, dict) and "input" in batch and "target" in batch:
            input, target = batch["input"], batch["target"]
        else:
            input, target = batch[0], batch[1]
        model_output = self.forward(input)
        loss = self.loss(model_output, target)
        loss_key = f"losses/{phase}_loss"
        self.log(loss_key, loss, on_step=True, on_epoch=True)

        preds = self._postprocess_preds(model_output)
        labels = self._postprocess_labels(target, phase)
        metrics_result = self.compute_metrics(preds, labels, phase)
        self.log_dict(metrics_result, on_epoch=True)

        if phase == "test":
            return {"loss": loss, "output": model_output}
        else:
            return {"loss": loss}

    def training_step(self, batch: Batch, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self._step(batch, "train")

    def validation_step(
        self, batch: Batch, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        return self._step(batch, "val")

    def test_step(self, batch: Batch, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self._step(batch, "test")

    def get_optimizer_param_groups(self) -> List[Dict[str, Any]]:
        norm_params, other_params = split_normalization_params(self.model)
        param_groups: List[Dict[str, Any]] = [{"params": other_params}]
        if len(norm_params) > 0:
            param_groups.append(
                {"params": norm_params, "weight_decay": self.norm_weight_decay}
            )
        return param_groups

    def configure_optimizers(self) -> Union[Optimizer, Dict[str, Any]]:
        param_groups = self.get_optimizer_param_groups()
        optim = self.optim_fn(param_groups)
        if not self.lr_scheduler_fn:
            return optim
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": none_throws(self.lr_scheduler_fn)(optim),
                "interval": self.lr_scheduler_interval,
            },
        }
