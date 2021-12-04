# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# Based on https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/gans/dcgan/components.py
import torch
from torch import nn
from torch.nn.modules import Sequential


class Generator(nn.Module):
    def __init__(self, latent_dim: int, feature_maps: int, image_channels: int) -> None:
        """
        Args:
            latent_dim: Dimension of the latent space
            feature_maps: Number of feature maps to use
            image_channels: Number of channels of the images from the dataset
        """
        super().__init__()
        self.gen = Sequential(
            self._make_gen_block(
                latent_dim, feature_maps * 8, kernel_size=4, stride=1, padding=0
            ),
            self._make_gen_block(feature_maps * 8, feature_maps * 4),
            self._make_gen_block(feature_maps * 4, feature_maps * 2),
            self._make_gen_block(feature_maps * 2, feature_maps),
            self._make_gen_block(feature_maps, image_channels, last_block=True),
        )

    @staticmethod
    def _make_gen_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        bias: bool = False,
        last_block: bool = False,
    ) -> Sequential:
        if not last_block:
            gen_block = Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=bias
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            gen_block = Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=bias
                ),
                nn.Tanh(),
            )

        return gen_block

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        noise = noise.view(*noise.shape, 1, 1)
        return self.gen(noise)


class Discriminator(nn.Module):
    def __init__(self, feature_maps: int, image_channels: int) -> None:
        """
        Args:
            feature_maps: Number of feature maps to use
            image_channels: Number of channels of the images from the dataset
        """
        super().__init__()
        self.disc = Sequential(
            self._make_disc_block(image_channels, feature_maps, batch_norm=False),
            self._make_disc_block(feature_maps, feature_maps * 2),
            self._make_disc_block(feature_maps * 2, feature_maps * 4),
            self._make_disc_block(feature_maps * 4, feature_maps * 8),
            self._make_disc_block(
                feature_maps * 8, 1, kernel_size=4, stride=1, padding=0, last_block=True
            ),
        )

    @staticmethod
    def _make_disc_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        bias: bool = False,
        batch_norm: bool = True,
        last_block: bool = False,
    ) -> Sequential:
        if not last_block:
            disc_block = Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=bias
                ),
                nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        else:
            disc_block = Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=bias
                ),
                nn.Sigmoid(),
            )

        return disc_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.disc(x).view(-1, 1).squeeze(1)
