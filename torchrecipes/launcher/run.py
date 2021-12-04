# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

import importlib
import logging
import os
from enum import auto, unique, Enum
from typing import Union, Optional

import hydra
from omegaconf import OmegaConf
from pytorch_lightning.utilities.types import _EVALUATE_OUTPUT, _PREDICT_OUTPUT
from torch.distributed.elastic.multiprocessing import errors
from torchrecipes.core.base_train_app import BaseTrainApp
from torchrecipes.core.base_train_app import TrainOutput
from torchrecipes.core.conf.base_config import BaseTrainAppConf

logger: logging.Logger = logging.getLogger(__name__)

# Set default value of these environment variables in
# fbcode/torchx/components/fb/stl_apps.py

# Your TrainApp's hydra conf module. We need to import this module before calling hydra
CONFIG_MODULE = "CONFIG_MODULE"
# Which mode your App will run in.
#   - prod: (Default) train + test, return test result
#   - train: train only
#   - test: test only
#   - predict: train + predict
MODE = "MODE"


@unique
class Mode(Enum):
    PROD = auto()
    TRAIN = auto()
    TEST = auto()
    PREDICT = auto()


def _get_mode() -> Mode:
    """Fetch operating environment."""
    mode_key = os.getenv(MODE, "").upper()
    try:
        return Mode[mode_key]
    except KeyError:
        logger.warning("Unknown MODE, run train and test by default")
        return Mode.PROD


def run_in_certain_mode(
    app: BaseTrainApp,
) -> Union[TrainOutput, _EVALUATE_OUTPUT, Optional[_PREDICT_OUTPUT]]:
    mode = _get_mode()
    if mode == Mode.TRAIN:
        logger.info("MODE set to train, run train only.")
        return app.train()
    elif mode == Mode.TEST:
        logger.info("MODE set to test, run test only.")
        return app.test()
    elif mode == Mode.PREDICT:
        logger.info("MODE set to predict, run train and precit.")
        app.train()
        return app.predict()
    else:
        # By default, run train and test
        app.train()
        return app.test()


@hydra.main()
def run_with_hydra(
    cfg: BaseTrainAppConf,
) -> Union[TrainOutput, _EVALUATE_OUTPUT, Optional[_PREDICT_OUTPUT]]:
    logger.info(OmegaConf.to_yaml(cfg))
    app = hydra.utils.instantiate(cfg, _recursive_=False)
    return run_in_certain_mode(app)


# pyre-ignore[56]: Decorator is not defined in typeshed_internal.
@errors.record
def main() -> None:
    config_module = os.getenv(CONFIG_MODULE)
    logger.info(f"CONFIG_MODULE: {config_module}")
    # only needed for apps that use hydra config
    if config_module:
        importlib.import_module(config_module)
        run_with_hydra()
    else:
        # TODO: T93277666 add entry point for non-hydra apps
        raise NotImplementedError


if __name__ == "__main__":
    main()
