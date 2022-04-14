# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, Mapping, Optional

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as metrics
from omegaconf import DictConfig
from torch.nn.modules import CrossEntropyLoss
from torch.optim import Optimizer
from torchrecipes.utils.config_utils import config_entry

logger: logging.Logger = logging.getLogger(__name__)


class DocClassificationModule(pl.LightningModule):
    """
    Generic module for doc classification
    The components(model, optim etc.) can be configured and instantiated by hydra

    Note: this is a simple demo Module. Please use torchrecipes.fb.text.module.doc_classification
    for training, which supports advanced features like FSDP
    """

    # See P340190896. torchscripting fails if annotated at class-level.

    def __init__(
        self,
        transform: nn.Module,
        model: nn.Module,
        optim: Optimizer,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.model = model
        self.optim = optim

        self.loss = CrossEntropyLoss()
        self.accuracy = metrics.Accuracy()
        self.fbeta = metrics.FBetaScore(
            num_classes=num_classes,
            average="macro",
        )

    @config_entry
    @staticmethod
    def from_config(
        transform: DictConfig,
        model: DictConfig,
        optim: DictConfig,
        num_classes: int,
    ) -> "DocClassificationModule":
        transform = hydra.utils.instantiate(transform)
        model = hydra.utils.instantiate(
            model,
        )
        optim = hydra.utils.instantiate(
            optim,
            model.parameters(),
        )
        return DocClassificationModule(transform, model, optim, num_classes)

    def setup(self, stage: Optional[str]) -> None:
        """
        Called at the beginning of fit and test.
        This is a good hook when you need to build models dynamically or adjust something about them.
        This hook is called on every process when using DDP.

        Args:
            stage: either 'fit' or 'test'
        """
        pass

    # pyre-ignore[14]: This is for torchscript compatibility.
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        token_ids = self.transform(batch)["token_ids"]
        assert torch.jit.isinstance(token_ids, torch.Tensor)
        return self.model(token_ids)

    def configure_optimizers(self) -> Optimizer:
        return self.optim

    def training_step(
        self,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        logits = self.model(batch["token_ids"])
        loss = self.loss(logits, batch["label_ids"])
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        logits = self.model(batch["token_ids"])
        loss = self.loss(logits, batch["label_ids"])
        scores = F.softmax(logits)

        self.accuracy(scores, batch["label_ids"])
        self.fbeta(scores, batch["label_ids"])
        self.log("val_loss", loss)
        self.log("val_acc", self.accuracy)
        self.log("val_f1", self.fbeta)

    def test_step(
        self,
        batch: Mapping[str, torch.Tensor],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        logits = self.model(batch["token_ids"])
        loss = self.loss(logits, batch["label_ids"])
        scores = F.softmax(logits)

        self.accuracy(scores, batch["label_ids"])
        self.fbeta(scores, batch["label_ids"])
        self.log("test_loss", loss)
        self.log("test_acc", self.accuracy)
        self.log("test_f1", self.fbeta)
