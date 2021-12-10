# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

# pyre-strict

from dataclasses import dataclass
from typing import Any, Optional

import hydra
from omegaconf import MISSING
from pytorch_lightning import LightningModule, LightningDataModule
from torchrecipes.core.base_train_app import BaseTrainApp
from torchrecipes.core.conf import TrainAppConf, TrainerConf
from torchrecipes.text.doc_classification.conf.common import (
    TransformConf,
)
from torchrecipes.text.doc_classification.datamodule.doc_classification import (
    DocClassificationDataModule,
    DocClassificationDataModuleConf,
)
from torchrecipes.text.doc_classification.module.doc_classification import (
    DocClassificationModuleConf,
)
from torchrecipes.utils.config_utils import get_class_name_str


class DocClassificationTrainApp(BaseTrainApp):
    """
    This app is used to launch the doc classification training / testing.
    """

    module_conf: DocClassificationModuleConf
    datamodule_conf: DocClassificationDataModuleConf

    def __init__(
        self,
        module: DocClassificationModuleConf,
        trainer: TrainerConf,
        datamodule: DocClassificationDataModuleConf,
        transform: TransformConf,
    ) -> None:
        self.transform_conf = transform
        super().__init__(module, trainer, datamodule)

    def get_lightning_module(self) -> LightningModule:
        # check whether this is the OSS or internal transform
        # the OSS TransformConf has a `label_transform` field whereas the
        # internal transforms don't
        if hasattr(self.transform_conf, "label_transform"):
            num_classes = len(
                # pyre-ignore[16]: Subclass of `TransformConf` has relevant attribute
                self.transform_conf.label_transform.label_names
            )
            transform_conf = (
                # pyre-ignore[16]: Subclass of `TransformConf` has relevant attribute
                self.transform_conf.transform
            )
        else:
            # pyre-ignore[16]: Subclass of `TransformConf` has relevant attribute
            num_classes = len(self.transform_conf.label_names)
            transform_conf = self.transform_conf

        return hydra.utils.instantiate(
            self.module_conf,
            transform=transform_conf,
            num_classes=num_classes,
            _recursive_=False,
        )

    def get_data_module(self) -> Optional[LightningDataModule]:
        # check whether this is the OSS or internal DataLoader
        # note that the internal dataloader expects a transform `nn.Module`
        # object whereas the OSS DataLoader expects a transform config
        if (
            get_class_name_str(DocClassificationDataModule)
            in self.datamodule_conf._target_
        ):
            datamodule = hydra.utils.instantiate(
                self.datamodule_conf, transform=self.transform_conf, _recursive_=False
            )
        else:
            transform = hydra.utils.instantiate(self.transform_conf, _recursive_=False)
            datamodule = hydra.utils.instantiate(
                self.datamodule_conf, transform=transform, _recursive_=False
            )

        return datamodule


@dataclass
class DocClassificationTrainAppConf(TrainAppConf):
    _target_: str = get_class_name_str(DocClassificationTrainApp)
    # pyre-ignore[4]: Cannot use complex types with hydra.
    transform: Any = MISSING
