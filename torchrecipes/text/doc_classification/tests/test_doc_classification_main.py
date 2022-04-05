# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

# pyre-strict
import os.path
from unittest.mock import patch

import torchrecipes.text.doc_classification.conf  # noqa
from hydra import compose, initialize_config_module
from omegaconf import DictConfig
from torchrecipes.core.test_utils.test_base import BaseTrainAppTestCase
from torchrecipes.text.doc_classification.main import train_and_test
from torchrecipes.text.doc_classification.module.doc_classification import (
    DocClassificationModule,
)
from torchrecipes.text.doc_classification.tests.common.assets import (
    copy_partial_sst2_dataset,
    get_asset_path,
    copy_asset,
)
from torchrecipes.utils.config_utils import get_class_config_method
from torchrecipes.utils.test import tempdir


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

    def get_config(self, root_dir: str) -> DictConfig:
        # copy the asset files into their expected download locations
        # note we need to do this anywhere we use hydra overrides
        # otherwise we get a `LexerNoViableAltException`
        vocab_path = os.path.join(root_dir, "vocab_example.pt")
        spm_model_path = os.path.join(root_dir, "spm_example.model")
        copy_asset(get_asset_path("vocab_example.pt"), vocab_path)
        copy_asset(get_asset_path("spm_example.model"), spm_model_path)
        copy_partial_sst2_dataset(root_dir)

        with initialize_config_module("torchrecipes.text.doc_classification.conf"):
            cfg = compose(
                config_name="default",
                overrides=[
                    f"+module._target_={get_class_config_method(DocClassificationModule)}",
                    "module/model=xlmrbase_classifier_tiny",
                    f"datamodule.dataset.root={root_dir}",
                    f"trainer.default_root_dir={root_dir}",
                    f"transform.transform.vocab_path={vocab_path}",
                    f"transform.transform.spm_model_path={spm_model_path}",
                    "transform.num_labels=2",
                    "trainer.logger=false",
                    "trainer.enable_checkpointing=false",
                    "trainer.fast_dev_run=true",
                ],
            )
        return cfg

    @tempdir
    def test_train_and_test(self, root_dir: str) -> None:
        cfg = self.get_config(root_dir=root_dir)
        output = train_and_test(cfg)
        self.assertIsNotNone(output)
