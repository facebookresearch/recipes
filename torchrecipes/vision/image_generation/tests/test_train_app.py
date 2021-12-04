# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

from typing import cast

from torchrecipes.core.test_utils.test_base import BaseTrainAppTestCase
from torchrecipes.vision.image_generation.train_app import GANTrainApp


class TestGANTrainApp(BaseTrainAppTestCase):
    def get_train_app(self, config_name: str) -> GANTrainApp:
        app = self.create_app_from_hydra(
            config_module="torchrecipes.vision.image_generation.conf",
            config_name=config_name,
            overrides=[
                "+schema/datamodule/datamodule=torchvision_datamodule_conf",
                "datamodule/datamodule=fake_data",
            ],
        )
        self.mock_trainer_params(app)
        return cast(GANTrainApp, app)

    def test_gan_train_app(self) -> None:
        train_app = self.get_train_app("gan_train_app")
        output = train_app.train()
        self.assertIsNotNone(output)

    def test_infogan_train_app(self) -> None:
        train_app = self.get_train_app("infogan_train_app")
        output = train_app.train()
        self.assertIsNotNone(output)
