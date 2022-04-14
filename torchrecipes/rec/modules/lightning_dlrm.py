# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3
import logging
import os
import sys
from typing import Iterator, Any, List, TypeVar

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
import torchmetrics as metrics
from ai_codesign.benchmarks.dlrm.torchrec_dlrm.modules.dlrm_train import DLRMTrain
from torchrec import EmbeddingBagCollection
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.train_pipeline import In
from torchrec.optim.keyed import KeyedOptimizerWrapper


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger: logging.Logger = logging.getLogger()

M = TypeVar("M", bound=nn.Module)


class LightningDLRM(pl.LightningModule):
    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        batch_size: int,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
    ) -> None:
        super().__init__()

        rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            backend = "nccl"
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
            backend = "gloo"

        if not torch.distributed.is_initialized():
            dist.init_process_group(backend=backend)
        self.to(device=device)

        model = DLRMTrain(
            embedding_bag_collection=embedding_bag_collection,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=dense_arch_layer_sizes,
            over_arch_layer_sizes=over_arch_layer_sizes,
            dense_device=device,
        )

        self.model = DistributedModelParallel(
            module=model,
            device=device,
        )

        self.train_pipeline: TrainPipelineSparseDist = TrainPipelineSparseDist(
            self.model,
            self.configure_optimizers(),
            device,
        )
        self.automatic_optimization = False

        self.accuracy: metrics.Metric = metrics.Accuracy().to(device=device)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return KeyedOptimizerWrapper(
            dict(self.model.named_parameters()),
            lambda params: torch.optim.SGD(params, lr=0.01),
        )

    def _step(
        self,
        dataloader_iter: Iterator[In],
        batch_idx: int,
        step_phase: str,
    ) -> torch.Tensor:
        loss, logits, labels = self.train_pipeline.progress(dataloader_iter)

        preds = torch.sigmoid(logits)
        accuracy = self.accuracy(preds, labels)
        batch_size = len(labels)

        self.log(f"{step_phase}_accuracy", accuracy, batch_size=batch_size)
        self.log(f"{step_phase}_loss", loss, batch_size=batch_size)

        return loss

    def training_step(
        self,
        dataloader_iter: Iterator[In],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._step(dataloader_iter, batch_idx, "train")

    def validation_step(
        self,
        dataloader_iter: Iterator[In],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._step(dataloader_iter, batch_idx, "val")

    def test_step(
        self,
        dataloader_iter: Iterator[In],
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._step(dataloader_iter, batch_idx, "test")
