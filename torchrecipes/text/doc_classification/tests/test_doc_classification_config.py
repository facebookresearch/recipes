# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict
import os.path
from typing import Tuple
from unittest.mock import patch

import hydra
import testslide

# Register our config class to use it in Hydra's helpers.
import torchrecipes.text.doc_classification.conf  # noqa
from hydra.experimental import compose, initialize_config_module
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.trainer import Trainer
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


class TestDocClassificationConfig(testslide.TestCase):
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

    @tempdir
    def test_doc_classification_task(self, root_dir: str) -> None:
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
                config_name="default_config",
                overrides=[
                    f"+module._target_={get_class_config_method(DocClassificationModule)}",
                    "module/model=xlmrbase_classifier_tiny",
                    f"datamodule.dataset.root={root_dir}",
                    f"trainer.default_root_dir={root_dir}",
                    "trainer.logger=False",
                    "trainer.enable_checkpointing=False",
                    f"transform.transform.vocab_path={vocab_path}",
                    f"transform.transform.spm_model_path={spm_model_path}",
                    "transform.num_labels=2",
                ],
            )
        task, datamodule = self._instantiate_config(cfg)
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer)
        trainer.fit(task, datamodule=datamodule)

        result = trainer.test(task, datamodule=datamodule)
        self.assertGreaterEqual(result[0]["test_f1"], 0)

        pred1 = task.forward({"text": ["hello world", "how are you?"]})
        pred2 = task.forward(
            {"text": ["hello world", "how are you?"], "label": ["1", "0"]}
        )
        self.assertIsNotNone(pred1)
        self.assertIsNotNone(pred2)

    @tempdir
    def test_doc_classification_task_torchscript(self, root_dir: str) -> None:
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
                config_name="default_config",
                overrides=[
                    f"+module._target_={get_class_config_method(DocClassificationModule)}",
                    "module/model=xlmrbase_classifier_tiny",
                    f"datamodule.dataset.root={root_dir}",
                    f"trainer.default_root_dir={root_dir}",
                    "trainer.logger=False",
                    "trainer.enable_checkpointing=False",
                    f"transform.transform.vocab_path={vocab_path}",
                    f"transform.transform.spm_model_path={spm_model_path}",
                    "transform.num_labels=2",
                ],
            )
        task, datamodule = self._instantiate_config(cfg)
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer)
        trainer.fit(task, datamodule=datamodule)

        result = trainer.test(task, datamodule=datamodule)
        self.assertGreaterEqual(result[0]["test_f1"], 0)

        ts_model = task.to_torchscript()
        pred1 = ts_model({"text": ["hello world", "how are you?"]})
        pred2 = ts_model({"text": ["hello world", "how are you?"], "label": ["1", "0"]})
        self.assertIsNotNone(pred1)
        self.assertIsNotNone(pred2)

    def _instantiate_config(
        self, cfg: DictConfig
    ) -> Tuple[LightningModule, LightningDataModule]:
        num_classes = cfg.transform.num_labels
        datamodule = hydra.utils.instantiate(
            cfg.datamodule,
            transform=cfg.transform,
            _recursive_=False,
        )
        task = hydra.utils.instantiate(
            cfg.module,
            transform=cfg.transform.transform,
            num_classes=num_classes,
            _recursive_=False,
        )
        return task, datamodule
