# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3
# flake8: noqa
import torch
import torchvision.models
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchmetrics import AveragePrecision
from torchrecipes.vision.core.datamodule.torchvision_data_module import (
    TorchVisionDataModule,
)
from torchrecipes.vision.image_classification.module.image_classification import (
    ImageClassificationModule,
)
