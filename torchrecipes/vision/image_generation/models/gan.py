# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3
# based on https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/gans/basic/basic_gan_module.py
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Generator(nn.Module):
    """Generator model from the
    `"Generative Adversarial Networks" <https://arxiv.org/abs/1406.2661>`_ paper.

    Args:
        latent_dim (int): dimension of latent
        img_shape (tuple): shape of image tensor
        hidden_dim (int): dimension of hidden layer
    """

    def __init__(
        self, latent_dim: int, img_shape: Tuple[int, int, int], hidden_dim: int = 256
    ) -> None:
        super().__init__()
        feats = int(np.prod(img_shape))
        self.img_shape = img_shape
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, feats)

    # forward method
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.leaky_relu(self.fc1(z), 0.2)
        z = F.leaky_relu(self.fc2(z), 0.2)
        z = F.leaky_relu(self.fc3(z), 0.2)
        img = torch.tanh(self.fc4(z))
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    """Discriminator model from the
    `"Generative Adversarial Networks" <https://arxiv.org/abs/1406.2661>`_ paper.

    Args:
        img_shape (tuple): shape of image tensor
        hidden_dim (int): dimension of hidden layer
    """

    def __init__(self, img_shape: Tuple[int, int, int], hidden_dim: int = 1024) -> None:
        super().__init__()
        in_dim = int(np.prod(img_shape))
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = img.view(img.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))
