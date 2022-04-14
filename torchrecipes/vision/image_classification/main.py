# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3
import logging
from dataclasses import dataclass
from typing import Optional

import hydra
import torchrecipes.vision.image_classification.conf  # noqa
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

log: logging.Logger = logging.getLogger(__name__)


@dataclass
class TrainOutput:
    log_dir: Optional[str] = None
    best_model_path: Optional[str] = None


@hydra.main(config_path="conf", config_name="default_config")
def main(config: DictConfig) -> TrainOutput:
    seed = config.get("seed", 0)
    seed_everything(seed, workers=True)
    log.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    log.info("Instantiating a datamodule, a module, and a trainer")
    datamodule = hydra.utils.instantiate(config.datamodule)
    trainer = hydra.utils.instantiate(config.trainer)
    module = hydra.utils.instantiate(config.module)

    if getattr(config, "pretrained_checkpoint_path", None):
        log.info(f"Loading module from checkpoint {config.pretrained_checkpoint_path}")
        module = module.load_from_checkpoint(
            checkpoint_path=config.pretrained_checkpoint_path
        )

    log.info("Training started")
    trainer.fit(module, datamodule=datamodule)
    logging.info("Testing started")
    trainer.test(module, datamodule=datamodule)

    train_output = TrainOutput(
        best_model_path=getattr(trainer.checkpoint_callback, "best_model_path", None),
        log_dir=getattr(trainer.logger, "save_dir", None),
    )
    log.info(f"Training output: {train_output}")
    return train_output


if __name__ == "__main__":
    main()
