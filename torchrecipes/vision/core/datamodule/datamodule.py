# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore

# @manual "//github/third-party/omry/omegaconf:omegaconf"
from omegaconf import MISSING
from torchrecipes.core.conf import DataModuleConf

cs = ConfigStore()


@dataclass
class VisionDataModuleConf(DataModuleConf):
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    datamodule: Any = MISSING
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    transforms: Any = None


cs.store(
    group="schema/datamodule",
    name="vision_module_conf",
    node=VisionDataModuleConf,
    package="datamodule",
)
