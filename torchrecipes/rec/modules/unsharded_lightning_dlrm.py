# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

import logging
import sys
from dataclasses import dataclass
from typing import Any, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics as metrics
from hydra.core.config_store import ConfigStore
from torchrec import EmbeddingBagCollection, KeyedJaggedTensor
from torchrec.datasets.utils import Batch
from torchrec.models.dlrm import DLRM
from torchrecipes.core.conf import ModuleConf
from torchrecipes.utils.config_utils import get_class_name_str

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger: logging.Logger = logging.getLogger()


class UnshardedLightningDLRM(pl.LightningModule):
    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
    ) -> None:
        super().__init__()
        self.model: DLRM = DLRM(
            embedding_bag_collection=embedding_bag_collection,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=dense_arch_layer_sizes,
            over_arch_layer_sizes=over_arch_layer_sizes,
        )
        self.loss_fn: nn.Module = nn.BCEWithLogitsLoss()
        self.accuracy: metrics.Metric = metrics.Accuracy()
        self.auroc: metrics.Metric = metrics.AUROC()

    # pyre-ignore[14] - `forward` overrides method defined in `pl.core.lightning.LightningModule`
    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        output = self.model(dense_features, sparse_features)
        return output.squeeze()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters())

    def _step(
        self,
        batch: Batch,
        batch_idx: int,
        step_phase: str,
    ) -> torch.Tensor:
        logits = self.forward(
            dense_features=batch.dense_features,
            sparse_features=batch.sparse_features,
        )
        loss = self.loss_fn(logits, batch.labels.float())
        preds = torch.sigmoid(logits)
        accuracy = self.accuracy(preds, batch.labels)

        self.log(f"{step_phase}_accuracy", accuracy)
        self.log(f"{step_phase}_loss", loss)

        if 1 in batch.labels and 0 in batch.labels:
            auroc = self.auroc(preds, batch.labels)
            self.log(f"{step_phase}_auroc", auroc)
        else:
            logger.warning(
                f"{step_phase}:Could not compute AUROC. The labels were missing either a positive or negative sample."
            )

        return loss

    # pyre-fixme[14]: `training_step` overrides method defined in `LightningModule`
    #  inconsistently.
    def training_step(
        self,
        batch: Batch,
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._step(batch, batch_idx, "train")

    # pyre-fixme[14]: `validation_step` overrides method defined in
    #  `LightningModule` inconsistently.
    def validation_step(
        self,
        batch: Batch,
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._step(batch, batch_idx, "val")

    # pyre-fixme[14]: `test_step` overrides method defined in `LightningModule`
    #  inconsistently.
    def test_step(
        self,
        batch: Batch,
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._step(batch, batch_idx, "test")


@dataclass
class UnshardedLightningDLRMModuleConf(ModuleConf):
    _target_: str = get_class_name_str(UnshardedLightningDLRM)


cs: ConfigStore = ConfigStore.instance()

cs.store(
    group="schema/module",
    name="unsharded_lightning_dlrm",
    node=UnshardedLightningDLRMModuleConf,
    package="module",
)
