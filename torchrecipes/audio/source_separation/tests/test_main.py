# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3
# pyre-strict

import unittest
from typing import List, Tuple

import hydra
import torch
import torchrecipes.audio.source_separation.conf  # noqa
from torch.utils.data import Dataset
from torchrecipes.audio.source_separation.main import main
from torchrecipes.utils.test import tempdir


class TestDataset(Dataset):
    def __len__(self) -> int:
        return 10

    def __getitem__(self, key: int) -> Tuple[int, torch.Tensor, List[torch.Tensor]]:
        """Load the n-th sample from the dataset.
        Args:
            key (int): The index of the sample to be loaded
        Returns:
            (int, Tensor, List[Tensor]): ``(sample_rate, mix_waveform, list_of_source_waveforms)``
        """
        return 8000, torch.rand(1, 24000), [torch.rand(1, 24000), torch.rand(1, 24000)]


class TestMain(unittest.TestCase):
    @tempdir
    def test_train_model(self, root_dir: str) -> None:
        with hydra.initialize_config_module(
            config_module="torchrecipes.audio.source_separation.conf"
        ):
            config = hydra.compose(
                config_name="default_config",
                overrides=[
                    f"datamodule.root_dir={root_dir}",
                    "trainer.accelerator=cpu",
                    "trainer.devices=null",
                    "trainer.strategy=null",
                    "trainer.max_epochs=2",
                    "+trainer.fast_dev_run=true",
                ],
            )
        with unittest.mock.patch(
            "torchrecipes.audio.source_separation.datamodule.librimix.LibriMixDataModule._get_dataset",
            return_value=TestDataset(),
        ):
            main(config)
