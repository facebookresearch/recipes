# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

# pyre-strict
import os.path

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
    def get_train_app(self, root_dir: str) -> BaseTrainApp:
        # copy the asset files into their expected download locations
        # note we need to do this anywhere we use hydra overrides
        # otherwise we get a `LexerNoViableAltException`
        vocab_path = os.path.join(root_dir, "xlmr.vocab.pt")
        spm_model_path = os.path.join(root_dir, "xlmr.sentencepiece.bpe.model")
        copy_asset(get_asset_path("xlmr.vocab.pt"), vocab_path)
        copy_asset(get_asset_path("xlmr.sentencepiece.bpe.model"), spm_model_path)
        copy_partial_sst2_dataset(root_dir)

        app = self.create_app_from_hydra(
            config_module="torchrecipes.text.doc_classification.conf",
            config_name="train_app",
            overrides=[
                "module.model.checkpoint=null",
                "module.model.freeze_encoder=True",
                f"datamodule.dataset.root={root_dir}",
                "datamodule.dataset.validate_hash=False",
                f"trainer.default_root_dir={root_dir}",
                "trainer.logger=False",
                "trainer.checkpoint_callback=False",
                f"transform.transform.vocab_path={vocab_path}",
                f"transform.transform.spm_model_path={spm_model_path}",
            ],
        )
        self.mock_trainer_params(app)
        return app

    @tempdir
    def test_doc_classification_task_train(self, root_dir: str) -> None:
        train_app = self.get_train_app(root_dir=root_dir)
        output = train_app.train()
        self.assert_train_output(output)

    @tempdir
    def test_doc_classification_task_test(self, root_dir: str) -> None:
        train_app = self.get_train_app(root_dir=root_dir)
        train_app.train()
        output = train_app.test()
        self.assertIsNotNone(output)
