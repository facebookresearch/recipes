# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3
# pyre-strict

from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchaudio.datasets import LibriMix

from .utils import CollateFn


class LibriMixDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int = 6,
        tr_split: str = "train-360",
        num_speakers: int = 2,
        sample_rate: int = 8000,
        task: str = "sep_clean",
        num_workers: int = 4,
    ) -> None:
        """The LightningDataModule for LibriMix Dataset.
        Args:
            root_dir (str): the root directory of the dataset.
            batch_size (int, optional): the batch size of the dataset. (Default: 6)
            tr_split (str, optional): the training split in LibriMix dataset.
                Options: [``train-360`, ``train-100``] (Default: ``train-360``)
            num_speakers (int, optional): The number of speakers, which determines the directories
                to traverse. The datamodule will traverse ``s1`` to ``sN`` directories to collect
                N source audios. (Default: 2)
            sample_rate (int, optional): the sample rate of the audio. (Default: 8000)
            task (str, optional): the task of LibriMix.
                Options: [``enh_single``, ``enh_both``, ``sep_clean``, ``sep_noisy``]
                (Default: ``sep_clean``)
            num_workers (int, optional): the number of workers for each dataloader. (Default: 4)
            testing (bool, optional): To test the training recipe. If set to ``True``, the dataset will
                output random Tensors without need of the real dataset. (Default: ``False``)
        """
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.tr_split = tr_split
        self.num_speakers = num_speakers
        self.sample_rate = sample_rate
        self.task = task
        self.num_workers = num_workers
        self.datasets = {}

    def _get_dataset(self, subset):
        return LibriMix(
            root=self.root_dir,
            subset=subset,
            num_speakers=self.num_speakers,
            sample_rate=self.sample_rate,
            task=self.task,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.datasets["train"] = self._get_dataset(subset=self.tr_split)
            self.datasets["val"] = self._get_dataset(subset="dev")
        if stage == "test" or stage is None:
            self.datasets["test"] = self._get_dataset(subset="test")

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            collate_fn=CollateFn(sample_rate=self.sample_rate, duration=3),
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            collate_fn=CollateFn(sample_rate=self.sample_rate, duration=-1),
            num_workers=self.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            collate_fn=CollateFn(sample_rate=self.sample_rate, duration=-1),
            num_workers=self.num_workers,
        )
