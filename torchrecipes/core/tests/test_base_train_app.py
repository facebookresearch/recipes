# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

# pyre-strict
from pytorch_lightning.callbacks import ModelCheckpoint
from torchrecipes.core.base_train_app import BaseTrainApp
from torchrecipes.core.conf import TrainerConf
from torchrecipes.core.test_utils.test_base import BaseTrainAppTestCase


class TestTrainApp(BaseTrainAppTestCase):
    def test_ckpt_callback_fallback_to_default(self) -> None:
        app = BaseTrainApp(None, TrainerConf(), None)
        app._set_trainer_params(trainer_params={})
        self.assertIsNotNone(app._checkpoint_callback)
        self.assertIsNone(app._checkpoint_callback.monitor)

    def test_ckpt_callback_user_provided(self) -> None:
        app = BaseTrainApp(None, TrainerConf(), None)
        self.mock_callable(app, "get_callbacks").to_return_value(
            [ModelCheckpoint(monitor="some_metrics")]
        )
        app._set_trainer_params(trainer_params={})
        self.assertIsNotNone(app._checkpoint_callback)
        self.assertEqual(app._checkpoint_callback.monitor, "some_metrics")
