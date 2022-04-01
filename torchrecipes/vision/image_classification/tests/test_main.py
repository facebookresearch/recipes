# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

# pyre-strict
from copy import deepcopy

import hydra
import testslide
import torch
import torchrecipes.vision.image_classification.conf  # noqa
from omegaconf import DictConfig
from torch import nn
from torchrecipes.utils.test import tempdir
from torchrecipes.vision.core.ops.fine_tuning_wrapper import FineTuningWrapper
from torchrecipes.vision.image_classification.main import main
from torchvision.models.resnet import resnet18
from torchvision.ops.misc import FrozenBatchNorm2d


class TestMain(testslide.TestCase):
    def _get_config(self, tb_save_dir: str) -> DictConfig:
        with hydra.initialize_config_module(
            config_module="torchrecipes.vision.image_classification.conf"
        ):
            config = hydra.compose(
                config_name="default_config",
                overrides=[
                    "datamodule/datasets=fake_data",
                    "+module.model.num_classes=10",
                    "trainer.enable_checkpointing=false",
                    "trainer.fast_dev_run=true",
                    f"trainer.logger.save_dir={tb_save_dir}",
                ],
            )
        return config

    @tempdir
    def test_train_model(self, root_dir: str) -> None:
        config = self._get_config(tb_save_dir=root_dir)
        output = main(config)
        self.assertIsNotNone(output)

    @tempdir
    def test_fine_tuning(self, root_dir: str) -> None:
        trunk = resnet18(norm_layer=FrozenBatchNorm2d)
        head = nn.Linear(in_features=512, out_features=10)
        fine_tune_model = FineTuningWrapper(trunk, "flatten", head)
        origin_trunk = deepcopy(fine_tune_model.trunk)

        config = self._get_config(tb_save_dir=root_dir)
        datamodule = hydra.utils.instantiate(config.datamodule)
        trainer = hydra.utils.instantiate(config.trainer)
        module = hydra.utils.instantiate(config.module, model=fine_tune_model)
        trainer.fit(module, datamodule=datamodule)

        with torch.no_grad():
            inp = torch.randn(1, 3, 28, 28)
            origin_out = origin_trunk(inp)
            tuned_out = module.model.trunk(inp)
            self.assertTrue(torch.equal(origin_out["flatten"], tuned_out["flatten"]))
