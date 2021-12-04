# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from hydra.core.config_store import ConfigStore
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader, Dataset, random_split
from torchrecipes.utils.config_utils import get_class_name_str
from torchvision import transforms as transform_lib
from torchvision.datasets import MNIST


class MNISTDataModule(LightningDataModule):
    """The data module for MNIST dataset (http://yann.lecun.com/exdb/mnist/).

    Args:
        data_dir: Where to save/load the data
        val_split: Percent (float) or number (int) of samples to use for the validation split
        num_workers: How many workers to use for loading data
        normalize: If true applies image normalize
        batch_size: How many samples per batch to load
        seed: Random seed to be used for train/val/test splits
        shuffle: If true shuffles the train data every epoch
        pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                    returning them
        drop_last: If true drops the last incomplete batch
        tran_transforms: transforms for train dataset
        val_transforms: transforms for validation dataset
        test_transforms: transforms for test dataset
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 16,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        train_transforms: Optional[Callable] = None,  # pyre-ignore[24]
        val_transforms: Optional[Callable] = None,  # pyre-ignore[24]
        test_transforms: Optional[Callable] = None,  # pyre-ignore[24]
    ) -> None:
        super().__init__(
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
        )

        self.data_dir: str = data_dir if data_dir else os.getcwd()
        self.val_split = val_split
        self.num_workers = num_workers
        self.normalize = normalize
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        # pyre-ignore[24]
        self.train_transforms: Callable = (
            train_transforms if train_transforms else self.default_transforms()
        )
        # pyre-ignore[24]
        self.val_transforms: Callable = (
            val_transforms if val_transforms else self.default_transforms()
        )
        # pyre-ignore[24]
        self.test_transforms: Callable = (
            test_transforms if test_transforms else self.default_transforms()
        )

        self.datasets: Dict[str, Dataset] = {}

        self.__validate_init_configuration()

    def default_transforms(self) -> Callable:  # pyre-ignore[24]
        transforms = [transform_lib.ToTensor()]
        if self.normalize:
            # pyre-ignore[6]: callable
            transforms.append(transform_lib.Normalize(mean=(0.5,), std=(0.5,)))
        return transform_lib.Compose(transforms)  # pyre-ignore[6]: callable

    def prepare_data(self) -> None:
        """Downloads files to data_dir."""
        MNIST(root=self.data_dir, train=True, download=True)
        MNIST(root=self.data_dir, train=False, download=True)

    def _get_splits(self, dataset_len: int) -> List[int]:
        """Computes split lengths for train and validation set."""
        val_split = self.val_split
        if isinstance(val_split, int):
            val_len = val_split
        elif isinstance(val_split, float):
            val_len = int(val_split * dataset_len)
        else:
            raise ValueError(f"Unsupported type {type(val_split)}")
        return [dataset_len - val_len, val_len]

    def _split_dataset(self, dataset: Dataset, train: bool = True) -> Dataset:
        """Splits the full dataset into train and validation set."""
        splits = self._get_splits(len(dataset))  # pyre-ignore[6]: dataset length
        dataset_train, dataset_val = random_split(
            dataset,
            splits,
            # pyre-ignore[16]: Generator has manual_seed
            generator=torch.Generator().manual_seed(self.seed),
        )

        if train:
            return dataset_train
        return dataset_val

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val and test dataset."""
        if stage == "fit" or stage is None:
            dataset_train = MNIST(
                self.data_dir, train=True, transform=self.train_transforms
            )
            dataset_val = MNIST(
                self.data_dir, train=True, transform=self.val_transforms
            )

            # Split
            self.datasets["train"] = self._split_dataset(dataset_train)
            self.datasets["val"] = self._split_dataset(dataset_val, train=False)

        if stage == "test" or stage is None:
            self.datasets["test"] = MNIST(
                self.data_dir, train=False, transform=self.test_transforms
            )

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader"""
        return self._data_loader(self.datasets["train"], shuffle=self.shuffle)

    def val_dataloader(
        self, *args: Any, **kwargs: Any
    ) -> Union[DataLoader, List[DataLoader]]:
        """The val dataloader"""
        return self._data_loader(self.datasets["val"])

    def test_dataloader(
        self, *args: Any, **kwargs: Any
    ) -> Union[DataLoader, List[DataLoader]]:
        """The test dataloader"""
        return self._data_loader(self.datasets["test"])

    def __validate_init_configuration(self) -> None:
        val_split = self.val_split
        wrong_type = not isinstance(val_split, (int, float))
        wrong_int = isinstance(val_split, int) and val_split < 0
        wrong_float = isinstance(val_split, float) and not (0.0 <= val_split <= 1.0)
        if wrong_type or wrong_int or wrong_float:
            raise MisconfigurationException(
                f"Invalid value for val_split={val_split}, Must be integer >= 0 or float between [0, 1]"
            )


@dataclass
class MNISTDataModuleConf:
    _target_: str = get_class_name_str(MNISTDataModule)
    data_dir: Optional[str] = None
    val_split: Any = 0.2  # pyre-ignore[4]: Union[int, float]
    num_workers: int = 16
    normalize: bool = False
    batch_size: int = 32
    seed: int = 42
    shuffle: bool = False
    pin_memory: bool = False
    drop_last: bool = False


cs = ConfigStore()
cs.store(
    group="datamodule/datamodule",
    name="mnist_data_module",
    node=MNISTDataModuleConf,
)
