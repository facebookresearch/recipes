# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import List, Optional

import pytorch_lightning as pl
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader
from torchrec.datasets.random import RandomRecDataset
from torchrecipes.core.conf import DataModuleConf
from torchrecipes.utils.config_utils import get_class_name_str


class RandomRecDataModule(pl.LightningDataModule):
    """
    DataModule that wraps RandomRecDataset. This dataset generates _RandomRecBatch, or random
    batches of sparse_features in the form of KeyedJaggedTensor, dense_features and labels

    {
        "dense_features": torch.Tensor,
        "sparse_features": KeyedJaggedTensor,
        "labels": torch.Tensor,
    }
    """

    def __init__(
        self,
        batch_size: int = 3,
        hash_size: Optional[int] = 100,
        hash_sizes: Optional[List[int]] = None,
        manual_seed: Optional[int] = None,
        pin_memory: bool = False,
        keys: Optional[List[str]] = None,
        ids_per_feature: int = 2,
        num_dense: int = 50,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.keys: List[str] = keys if keys else ["f1", "f3", "f2"]
        self.batch_size = batch_size
        self.manual_seed = manual_seed
        self.pin_memory = pin_memory
        self.hash_size = hash_size
        self.hash_sizes = hash_sizes
        self.ids_per_feature = ids_per_feature
        self.num_dense = num_dense
        self.num_workers = num_workers
        self.init_loader: DataLoader = DataLoader(
            RandomRecDataset(
                keys=self.keys,
                batch_size=self.batch_size,
                hash_size=self.hash_size,
                hash_sizes=self.hash_sizes,
                manual_seed=self.manual_seed,
                ids_per_feature=self.ids_per_feature,
                num_dense=self.num_dense,
            ),
            batch_size=None,
            batch_sampler=None,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self.init_loader

    def val_dataloader(self) -> DataLoader:
        return self.init_loader

    def test_dataloader(self) -> DataLoader:
        return self.init_loader


@dataclass
class RandomRecDataModuleConf(DataModuleConf):
    _target_: str = get_class_name_str(RandomRecDataModule)


cs: ConfigStore = ConfigStore.instance()

cs.store(
    group="schema/datamodule",
    name="random_rec_datamodule",
    node=RandomRecDataModuleConf,
    package="datamodule",
)
