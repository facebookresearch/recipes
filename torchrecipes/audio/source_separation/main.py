# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3
# pyre-strict

import logging

import hydra
import torchrecipes.audio.source_separation.conf  # noqa
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import seed_everything

log: logging.Logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="default_config")
def main(config: DictConfig) -> None:
    seed = config.get("seed", 0)
    seed_everything(seed, workers=True)
    log.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    datamodule = hydra.utils.instantiate(config.datamodule)
    trainer = hydra.utils.instantiate(config.trainer)
    module = hydra.utils.instantiate(config.module)

    if getattr(config, "pretrained_checkpoint_path", None):
        module = module.load_from_checkpoint(
            checkpoint_path=config.pretrained_checkpoint_path
        )
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
