# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

# pyre-strict
import os

import torchrecipes.core.conf.base_config  # noqa
from torchrecipes._internal_patches import ModelCheckpoint
from torchrecipes.core.test_utils.test_base import BaseTrainAppTestCase
from torchrecipes.utils.test import tempdir


class TestBaseTrainApp(BaseTrainAppTestCase):
    def test_task_train(self) -> None:
        app = self.create_app_from_hydra(
            config_module="torchrecipes.core.conf",
            config_name="base_config",
        )
        self.mock_trainer_params(app)
        output = app.train()
        self.assert_train_output(output)

    def test_task_test(self) -> None:
        app = self.create_app_from_hydra(
            config_module="torchrecipes.core.conf",
            config_name="base_config",
        )
        self.mock_trainer_params(app)
        output = app.test()
        self.assertIsNotNone(output)

    def test_task_predict(self) -> None:
        app = self.create_app_from_hydra(
            config_module="torchrecipes.core.conf",
            config_name="base_config",
        )
        self.mock_trainer_params(app)
        app.predict()

    @tempdir
    def test_auto_resume_from_checkpoint(self, root_dir: str) -> None:
        app = self.create_app_from_hydra(
            config_module="torchrecipes.core.conf",
            config_name="base_config",
        )

        mock_ckpt = ModelCheckpoint(
            save_last=True,
            dirpath=root_dir,
            save_top_k=-1,
            has_user_data=False,
            ttl_days=1,
            monitor=None,
        )

        # By default checkpoints are disabled by base test
        self.mock_trainer_params(
            app,
            overrides={
                "checkpoint_callback": True,
                # checkpoint is disabled in fast dev run
                "fast_dev_run": False,
            },
        )
        # Call it after mock_trainer_params to override the mock_checkpoint in it
        self.mock_callable(
            app, "get_default_model_checkpoint", allow_private=True
        ).to_return_value(mock_ckpt)

        expected_checkpoint_path = f"{root_dir}/last.ckpt"
        self.assertFalse(os.path.exists(expected_checkpoint_path))

        # First run. Check that checkpoint is successfully saved.
        output = app.train()
        self.assert_train_output(output)
        self.assertTrue(os.path.exists(expected_checkpoint_path))

        # Second run. When creating trainer again, automatically find ckpt
        trainer, _ = app._get_trainer()
        self.assertIsNotNone(trainer.resume_from_checkpoint)
        self.assertEqual(trainer.resume_from_checkpoint, expected_checkpoint_path)
