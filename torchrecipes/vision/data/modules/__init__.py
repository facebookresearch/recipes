# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

from torchrecipes.vision.data.modules.mnist_data_module import MNISTDataModule
from torchrecipes.vision.data.modules.torchvision_data_module import (
    TorchVisionDataModule,
)

__all__ = [
    "MNISTDataModule",
    "TorchVisionDataModule",
]
