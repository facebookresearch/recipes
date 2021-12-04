# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Union

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torchrecipes.core.conf import (
    ModuleConf,
    TrainAppConf,
    DataModuleConf,
)

cs: ConfigStore = ConfigStore.instance()


_APP_DEFAULTS: List[Union[str, Dict[str, str]]] = [
    "_self_",
    # Module
    {"schema/module": "test_module"},
    {"module/model": "linear_dummy"},
    {"module/optim": "sgd"},
    {"module/loss": "cross_entropy"},
    # Trainer
    {"schema/trainer": "trainer"},
    {"trainer": "cpu"},
    # DataModule
    {"schema/datamodule": "test_datamodule"},
    {"datamodule": "random_data"},
]


@dataclass
class TestModuleConf(ModuleConf):
    _target_: str = "torchrecipes.core.test_utils.test_module.TestModule"
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    model: Any = MISSING
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    optim: Any = MISSING
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    loss: Any = MISSING
    # pyre-fixme[4]: Attribute annotation cannot be `Any`.
    metric: Optional[Any] = None


cs.store(
    group="schema/module", name="test_module", node=TestModuleConf, package="module"
)


@dataclass
class TestDataModuleConf(DataModuleConf):
    _target_: str = "torchrecipes.core.test_utils.test_module.TestDataModule"
    size: int = MISSING
    length: int = MISSING


cs.store(
    group="schema/datamodule",
    name="test_datamodule",
    node=TestDataModuleConf,
    package="datamodule",
)


@dataclass
class BaseTrainAppConf(TrainAppConf):
    _target_: str = "torchrecipes.core.base_train_app.BaseTrainApp"
    # pyre-fixme[4]: Attribute annotation cannot contain `Any`.
    defaults: List[Any] = field(default_factory=lambda: _APP_DEFAULTS)


cs.store(name="base_config", node=BaseTrainAppConf)
