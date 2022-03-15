# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3
from dataclasses import dataclass
from typing import List, Optional

import hydra

# @manual "//github/third-party/omry/omegaconf:omegaconf"
from omegaconf import MISSING
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import Callback
from torchrecipes.core.base_train_app import BaseTrainApp
from torchrecipes.core.conf import DataModuleConf, TrainerConf, TrainAppConf
from torchrecipes.utils.config_utils import get_class_name_str
from torchrecipes.vision.image_generation.callbacks import (
    TensorboardGenerativeModelImageSampler,
)
from torchrecipes.vision.image_generation.module.gan import GANModuleConf
from torchrecipes.vision.image_generation.module.infogan import InfoGANModuleConf


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


@dataclass
class GANTrainAppConf(TrainAppConf):
    _target_: str = get_class_name_str(GANTrainApp)
    datamodule: DataModuleConf = MISSING
    module: GANModuleConf = MISSING
    trainer: TrainerConf = MISSING


@dataclass
class InfoGANTrainAppConf(TrainAppConf):
    _target_: str = get_class_name_str(GANTrainApp)
    datamodule: DataModuleConf = MISSING
    module: InfoGANModuleConf = MISSING
    trainer: TrainerConf = MISSING
