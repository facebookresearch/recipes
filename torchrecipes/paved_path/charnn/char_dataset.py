#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch

from char_transform import CharTransform
from torch.utils.data import Dataset


class CharDataset(Dataset):
    """Character dataset"""

    def __init__(self, data_path: str, block_size: int) -> None:
        # self._init_data(data_path, block_size)
        self.block_size = block_size
        self.transform = CharTransform(data_path)
        self.data = self.transform.data
        self.vocab_size = self.transform.vocab_size

    def __len__(self) -> int:
        return self.transform.data_size - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx : idx + self.block_size + 1]
        # encode every character to an integer
        # dix = [self.stoi[s] for s in chunk]
        ids = self.transform(chunk)
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return (x, y)
