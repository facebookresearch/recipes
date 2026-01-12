# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

from typing import List, Optional

import hydra
# @manual "fbsource//third-party/pypi/omegaconf:omegaconf"

from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import Callback
from torchrecipes.core.base_train_app import BaseTrainApp
from torchrecipes.core.conf import DataModuleConf, TrainerConf
from torchrecipes.vision.image_generation.callbacks import (
    TensorboardGenerativeModelImageSampler,
)
from torchrecipes.vision.image_generation.module.gan import GANModuleConf


class GANTrainApp(BaseTrainApp):
    def __init__(
        self,
        module: GANModuleConf,
        trainer: TrainerConf,
        datamodule: DataModuleConf,
    ) -> None:
        super().__init__(module, trainer, datamodule)

    def get_data_module(self) -> Optional[LightningDataModule]:
        """
        Instantiate a LightningDataModule.
        """
        return hydra.utils.instantiate(self.datamodule_conf)

    def get_callbacks(self) -> List[Callback]:
        # TODO(kaizh): make callback configurable
        return [TensorboardGenerativeModelImageSampler()]
