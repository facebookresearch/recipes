# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import List, Tuple

import hydra
import pytorch_lightning as pl
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torch.utils.data import DataLoader
from torch.utils.data import IterDataPipe
from torch.utils.data.backward_compatibility import worker_init_fn
from torchrecipes.core.conf import DataModuleConf
from torchrecipes.text.doc_classification.conf.common import (
    DatasetConf,
    DocClassificationTransformConf,
)
from torchrecipes.utils.config_utils import (
    config_entry,
    get_class_config_method,
)
from torchtext.functional import to_tensor


class DocClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: IterDataPipe[Tuple[str, str]],
        val_dataset: IterDataPipe[Tuple[str, str]],
        test_dataset: IterDataPipe[Tuple[str, str]],
        transform: nn.Module,
        label_transform: nn.Module,
        columns: List[str],
        label_column: str,
        batch_size: int,
        num_workers: int = 0,
        drop_last: bool = False,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.transform = transform
        self.label_transform = label_transform

        self.columns = columns
        self.label_column = label_column

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.pin_memory = pin_memory

    @config_entry
    @staticmethod
    def from_config(
        transform: DocClassificationTransformConf,
        dataset: DatasetConf,
        columns: List[str],
        label_column: str,
        batch_size: int,
        num_workers: int = 0,
        drop_last: bool = False,
        pin_memory: bool = False,
    ) -> "DocClassificationDataModule":
        train_dataset, val_dataset, test_dataset = hydra.utils.call(dataset)
        text_transform = hydra.utils.instantiate(transform.transform, _recursive_=False)
        label_transform = hydra.utils.instantiate(
            transform.label_transform,
            _recursive_=False,
        )
        return DocClassificationDataModule(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            # TODO: Note that the following line should be replaced by
            # `test_dataset` once we update the lightning module to support
            # test data with and without labels
            test_dataset=val_dataset,
            transform=text_transform,
            label_transform=label_transform,
            columns=columns,
            label_column=label_column,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )

    def _get_data_loader(self, dataset: IterDataPipe[Tuple[str, str]]) -> DataLoader:
        dataset = dataset.batch(self.batch_size).rows2columnar(self.columns)
        dataset = dataset.map(self.transform)
        dataset = dataset.map(
            lambda x: {
                **x,
                "label_ids": to_tensor(self.label_transform(x[self.label_column])),
            }
        )
        dataset = dataset.add_index()

        return DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_data_loader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        return self._get_data_loader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._get_data_loader(self.test_dataset)


@dataclass
class DocClassificationDataModuleConf(DataModuleConf):
    _target_: str = get_class_config_method(DocClassificationDataModule)
    transform: DocClassificationTransformConf = MISSING
    dataset: DatasetConf = MISSING
    columns: List[str] = field(default_factory=lambda: ["text", "label"])
    label_column: str = "label"
    batch_size: int = 16
    num_workers: int = 0
    drop_last: bool = False
    pin_memory: bool = False


cs: ConfigStore = ConfigStore.instance()

cs.store(
    group="schema/datamodule",
    name="doc_classification",
    node=DocClassificationDataModuleConf,
    package="datamodule",
)
