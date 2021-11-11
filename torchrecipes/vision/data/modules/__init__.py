#!/usr/bin/env python3

from torchrecipes.vision.data.modules.mnist_data_module import MNISTDataModule
from torchrecipes.vision.data.modules.torchvision_data_module import (
    TorchVisionDataModule,
)

__all__ = [
    "MNISTDataModule",
    "TorchVisionDataModule",
]
