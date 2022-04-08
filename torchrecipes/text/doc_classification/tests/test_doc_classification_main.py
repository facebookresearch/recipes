# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

# pyre-strict
from unittest.mock import patch

import torchrecipes.text.doc_classification.conf  # noqa
from hydra import compose, initialize_config_module
from omegaconf import DictConfig
from torchrecipes.core.test_utils.test_base import BaseTrainAppTestCase
from torchrecipes.text.doc_classification.main import train_and_test


class TestDocClassificationMain(BaseTrainAppTestCase):
    def setUp(self) -> None:
        super().setUp()
        # patch the _hash_check() fn output to make it work with the dummy dataset
        self.patcher = patch(
            "torchdata.datapipes.iter.util.cacheholder._hash_check", return_value=True
        )
        self.patcher.start()

    def tearDown(self) -> None:
        self.patcher.stop()
        super().tearDown()

    def _get_config(self, config_name: str) -> DictConfig:
        with initialize_config_module("torchrecipes.text.doc_classification.conf"):
            cfg = compose(
                config_name=config_name,
                overrides=[
                    # train with 1 batch of data and skip checkpointing
                    "trainer.fast_dev_run=true",
                ],
            )
        return cfg

    def test_default_config(self) -> None:
        cfg = self._get_config("default_config")
        output = train_and_test(cfg)
        self.assertIsNotNone(output)

    def test_tiny_model_full_config(self) -> None:
        cfg = self._get_config("tiny_model_full_config")
        output = train_and_test(cfg)
        self.assertIsNotNone(output)

    def test_tiny_model_mixed_config(self) -> None:
        cfg = self._get_config("tiny_model_mixed_config")
        output = train_and_test(cfg)
        self.assertIsNotNone(output)
