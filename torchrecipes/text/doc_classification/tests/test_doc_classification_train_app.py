# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

# pyre-strict
import os.path
from unittest.mock import patch

import torchrecipes.text.doc_classification.conf  # noqa
from torchrecipes.core.base_train_app import BaseTrainApp
from torchrecipes.core.test_utils.test_base import BaseTrainAppTestCase
from torchrecipes.text.doc_classification.tests.common.assets import (
    copy_partial_sst2_dataset,
    get_asset_path,
    copy_asset,
)
from torchrecipes.utils.test import tempdir


class TestDocClassificationTrainApp(BaseTrainAppTestCase):
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

    def get_train_app(self, root_dir: str) -> BaseTrainApp:
        # copy the asset files into their expected download locations
        # note we need to do this anywhere we use hydra overrides
        # otherwise we get a `LexerNoViableAltException`
        vocab_path = os.path.join(root_dir, "vocab_example.pt")
        spm_model_path = os.path.join(root_dir, "spm_example.model")
        copy_asset(get_asset_path("vocab_example.pt"), vocab_path)
        copy_asset(get_asset_path("spm_example.model"), spm_model_path)
        copy_partial_sst2_dataset(root_dir)

        app = self.create_app_from_hydra(
            config_module="torchrecipes.text.doc_classification.conf",
            config_name="train_app",
            overrides=[
                "module.model.checkpoint=null",
                "module.model.freeze_encoder=True",
                f"datamodule.dataset.root={root_dir}",
                f"trainer.default_root_dir={root_dir}",
                "trainer.logger=False",
                "trainer.enable_checkpointing=False",
                f"transform.transform.vocab_path={vocab_path}",
                f"transform.transform.spm_model_path={spm_model_path}",
                "transform.num_labels=2",
            ],
        )
        self.mock_trainer_params(app)
        return app

    @tempdir
    def test_doc_classification_task(self, root_dir: str) -> None:
        train_app = self.get_train_app(root_dir=root_dir)
        train_app.train()
        output = train_app.test()
        self.assertIsNotNone(output)
