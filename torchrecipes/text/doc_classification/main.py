# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

# pyre-strict

from dataclasses import dataclass
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from torchrecipes.text.doc_classification.datamodule.doc_classification import (
    DocClassificationDataModule,
)
from torchrecipes.text.doc_classification.module.doc_classification import (
    DocClassificationModule,
)


@dataclass
class TrainOutput:
    tensorboard_log_dir: Optional[str] = None


def train_and_test(cfg: DictConfig) -> TrainOutput:
    if cfg.get("random_seed") is not None:
        seed_everything(cfg.random_seed)

    module = DocClassificationModule.from_config(
        model=cfg.module.model,
        optim=cfg.module.optim,
        transform=cfg.transform,
        num_classes=cfg.transform.num_labels,
    )
    datamodule = DocClassificationDataModule.from_config(
        transform=cfg.transform,
        dataset=cfg.datamodule.dataset,
        columns=cfg.datamodule.columns,
        label_column=cfg.datamodule.label_column,
        batch_size=cfg.datamodule.batch_size,
        num_workers=cfg.datamodule.num_workers,
        drop_last=cfg.datamodule.drop_last,
        pin_memory=cfg.datamodule.pin_memory,
    )

    trainer_conf = dict(cfg.trainer)
    trainer_conf["logger"] = TensorBoardLogger(save_dir=cfg.trainer.default_root_dir)
    trainer = Trainer(**trainer_conf)
    trainer.fit(module, datamodule=datamodule)
    trainer.test(module, datamodule=datamodule)
    return TrainOutput(
        tensorboard_log_dir=trainer_conf["logger"].log_dir,
    )


@hydra.main(config_path="conf", config_name="default")
def main(cfg: DictConfig) -> TrainOutput:
    print(f"config:\n{OmegaConf.to_yaml(cfg)}")
    return train_and_test(cfg)


if __name__ == "__main__":
    main()
