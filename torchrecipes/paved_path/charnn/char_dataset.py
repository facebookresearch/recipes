#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import fsspec
import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    """Character dataset"""

    def __init__(self, data_path: str, block_size: int) -> None:
        self._init_data(data_path, block_size)

    def __len__(self) -> int:
        return self.data_size - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._getitem(idx)

    def _init_data(self, data_path: str, block_size) -> None:
        fs, path = fsspec.core.url_to_fs(data_path)
        with fs.open(path, "r") as f:
            self.data = f.read()
        self.data_path = data_path
        self.block_size = block_size
        chars = sorted(set(self.data))
        self.data_size = len(self.data)
        self.vocab_size = len(chars)
        print(f"Data has {self.data_size} characters, {self.vocab_size} unique.")
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def _getitem(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx : idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return (x, y)


def get_dataset(data_path: str, block_size: int) -> Dataset:
    """
    Get a dataset by name and config. Will be extended to support more datasets
    """
    return CharDataset(data_path, block_size)
