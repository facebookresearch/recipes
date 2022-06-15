# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3
# Base on https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/infogan/infogan.py
from typing import List, Tuple

import torch
from torch import nn


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_classes: int,
        code_dim: int,
        img_size: int,
        channels: int,
    ) -> None:
        """
        Args:
            latent_dim: Dimension of the latent space
            n_classes: number of classes for dataset
            code_dim: latent code
            img_size: size of each image dimension
            channels: number of image channels
        """
        super().__init__()
        input_dim = latent_dim + n_classes + code_dim

        self.init_size: int = img_size // 4  # Initial size before upsampling
        self.l1: nn.modules.Sequential = nn.Sequential(
            nn.Linear(input_dim, 128 * self.init_size**2)
        )

        self.conv_blocks: nn.modules.Sequential = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(
        self, noise: torch.Tensor, labels: torch.Tensor, code: torch.Tensor
    ) -> torch.Tensor:
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(
        self,
        n_classes: int,
        code_dim: int,
        img_size: int,
        channels: int,
    ) -> None:
        """
        Args:
            n_classes: number of classes for dataset
            code_dim: latent code
            img_size: size of each image dimension
            channels: number of image channels
        """
        super().__init__()

        def discriminator_block(
            in_filters: int, out_filters: int, bn: bool = True
        ) -> List[nn.Module]:
            """Returns layers of each discriminator block"""
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                # pyre-fixme[6]: For 1st param expected `Union[LeakyReLU, Conv2d,
                #  Dropout2d]` but got `BatchNorm2d`.
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            # pyre-fixme[7]: Expected `List[Module]` but got `List[Union[LeakyReLU,
            #  Conv2d, Dropout2d]]`.
            return block

        self.conv_blocks: nn.modules.Sequential = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2**4

        # Output layers
        self.adv_layer: nn.modules.Sequential = nn.Sequential(
            nn.Linear(128 * ds_size**2, 1)
        )
        self.aux_layer: nn.modules.Sequential = nn.Sequential(
            nn.Linear(128 * ds_size**2, n_classes), nn.Softmax()
        )
        self.latent_layer: nn.modules.Sequential = nn.Sequential(
            nn.Linear(128 * ds_size**2, code_dim)
        )

    def forward(
        self, img: torch.Tensor
    ) -> Tuple[nn.modules.Sequential, nn.modules.Sequential, nn.modules.Sequential]:
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code
