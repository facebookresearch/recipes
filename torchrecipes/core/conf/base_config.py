# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from torchrecipes.core.conf import TrainAppConf


@dataclass
class BaseTrainAppConf(TrainAppConf):
    _target_: str = "torchrecipes.core.base_train_app.BaseTrainApp"


cs: ConfigStore = ConfigStore.instance()
cs.store(name="base_config", node=BaseTrainAppConf)
