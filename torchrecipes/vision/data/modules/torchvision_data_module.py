#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Union

import hydra
import torch
from hydra.core.config_store import ConfigStore

# @manual "//github/third-party/omry/omegaconf:omegaconf"
from omegaconf import MISSING
from pyre_extensions import none_throws
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
from torchrecipes.core.conf import DataModuleConf
from torchrecipes.utils.config_utils import get_class_config_method, config_entry
from torchrecipes.vision.data.transforms.builder import (
    build_transforms_from_dataset_config,
)
from torchvision.datasets.vision import VisionDataset


class TorchVisionDataModule(LightningDataModule):
    """The data module wraps around a PyTorch Vision dataset and
    generates a dataloader for each phase.

    Args:
        datasets: The torchvision datasets, should be a mapping from each phase
            ["train", "val", "test"] to either a VisionDataset or None.
        batch_size: How many samples per batch to load.
        drop_last: If true drops the last incomplete batch.
        normalize: If true applies image normalize.
        num_workers: How many workers to use for loading data.
        pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                    returning them.
        seed: Random seed to be used for train/val/test splits.
        val_split: Percent (float) or number (int) of samples to use for the validation split.
    """

    def __init__(
        self,
        datasets: Mapping[str, Optional[VisionDataset]],
        batch_size: int = 32,
        drop_last: bool = False,
        normalize: bool = False,
        num_workers: int = 16,
        pin_memory: bool = False,
        seed: int = 42,
        val_split: Optional[Union[int, float]] = None,
    ) -> None:
        super().__init__()
        self.datasets = datasets

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.normalize = normalize
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.val_split = val_split

    @config_entry
    @staticmethod
    def from_config(
        datasets: Dict[str, Any],
        batch_size: int = 32,
        drop_last: bool = False,
        normalize: bool = False,
        num_workers: int = 16,
        pin_memory: bool = False,
        seed: int = 42,
        val_split: Optional[Union[int, float]] = None,
    ) -> "TorchVisionDataModule":
        datasets_module: Dict[str, Optional[VisionDataset]] = {}
        for split in datasets:
            assert split in ["train", "val", "test"]
            dataset_conf = datasets[split]
            if dataset_conf:
                dataset_conf = dict(dataset_conf)
                dataset_conf = build_transforms_from_dataset_config(dataset_conf)
                datasets_module[split] = hydra.utils.instantiate(
                    dataset_conf, _recursive_=False
                )
            else:
                datasets_module[split] = None

        return TorchVisionDataModule(
            datasets_module,
            batch_size,
            drop_last,
            normalize,
            num_workers,
            pin_memory,
            seed,
            val_split,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val and test dataset."""
        if stage == "fit" or stage is None:
            assert self.datasets[
                "train"
            ], "In fit stage, the train dataset shouldn't be None!"
            if not self.datasets["val"] and self.val_split:
                dataset_train, dataset_val = self._split_dataset(
                    none_throws(self.datasets["train"])
                )
                # pyre-fixme[16]: `Mapping` has no attribute `__setitem__`.
                self.datasets["train"] = dataset_train
                self.datasets["val"] = dataset_val
                logging.info("We have split part of the train set into val set!")

    def _get_splits(self, dataset_len: int) -> List[int]:
        """Computes split lengths for train and validation set."""
        if isinstance(self.val_split, int):
            val_len = self.val_split
        elif isinstance(self.val_split, float):
            val_len = int(self.val_split * dataset_len)
        else:
            raise ValueError(f"Unsupported type {type(self.val_split)}")
        return [dataset_len - val_len, val_len]

    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def _split_dataset(self, dataset: VisionDataset) -> Any:
        """Splits the full dataset into train and validation set."""
        splits = self._get_splits(len(dataset))
        dataset_train, dataset_val = random_split(
            dataset,
            splits,
            # pyre-ignore[16]: Generator has manual_seed
            generator=torch.Generator().manual_seed(self.seed),
        )
        return dataset_train, dataset_val

    def _get_data_loader(self, dataset: VisionDataset, phase: str) -> DataLoader:
        if phase == "train":
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader"""
        assert self.datasets["train"] is not None, "Train dataset should be specified!"
        return self._get_data_loader(
            dataset=none_throws(self.datasets["train"]),
            phase="train",
        )

    def val_dataloader(
        self, *args: Any, **kwargs: Any
    ) -> Union[DataLoader, List[DataLoader]]:
        """The val dataloader"""
        assert (
            self.datasets["val"] is not None
        ), "Validation dataset should be specified!"

        return self._get_data_loader(
            dataset=none_throws(self.datasets["val"]), phase="val"
        )

    def test_dataloader(
        self, *args: Any, **kwargs: Any
    ) -> Union[DataLoader, List[DataLoader]]:
        """The test dataloader"""
        assert self.datasets["test"] is not None, "Test dataset should be specified!"
        return self._get_data_loader(
            dataset=none_throws(self.datasets["test"]), phase="test"
        )


@dataclass
class TorchVisionDataModuleConf(DataModuleConf):
    _target_: str = get_class_config_method(TorchVisionDataModule)
    datasets: Dict[str, Any] = MISSING
    batch_size: int = 32
    drop_last: bool = False
    normalize: bool = False
    num_workers: int = 16
    pin_memory: bool = False
    seed: int = 42
    val_split: Any = None  # pyre-ignore[4]: Union[int, float]


cs = ConfigStore()
cs.store(
    group="schema/datamodule/datamodule",
    name="torchvision_datamodule_conf",
    node=TorchVisionDataModuleConf,
    package="datamodule.datamodule",
)
