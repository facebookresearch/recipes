#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import fsspec
import torch
import torch.nn as nn


class CharTransform(nn.Module):
    def __init__(self, data_path: str):
        super().__init__()
        fs, path = fsspec.core.url_to_fs(data_path)
        with fs.open(path, "r") as f:
            self.data = f.read()
        self.data_path = data_path
        chars = sorted(set(self.data))
        self.data_size = len(self.data)
        self.vocab_size = len(chars)
        print(f"Data has {self.data_size} characters, {self.vocab_size} unique.")
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def forward(self, text: str):
        return self.encode(text)

    def encode(self, text: str):
        ids = [self.stoi[s] for s in text]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: List[int]):
        return "".join([self.itos[i] for i in ids])
