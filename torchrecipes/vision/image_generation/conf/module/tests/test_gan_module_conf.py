# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

import unittest

import hydra
from hydra.experimental import compose, initialize_config_module
from torchrecipes.vision.image_generation.module.gan import GAN


class TestGANModuleConf(unittest.TestCase):
    def test_init_with_hydra(self) -> None:
        with initialize_config_module(
            config_module="torchrecipes.vision.image_generation.conf"
        ):
            test_conf = compose(
                config_name="gan_train_app",
            )
            test_module = hydra.utils.instantiate(test_conf.module, _recursive_=False)
            self.assertIsInstance(test_module, GAN)
            self.assertIsNotNone(test_module.generator)
            self.assertIsNotNone(test_module.discriminator)
            self.assertIsNotNone(test_module.criterion)
