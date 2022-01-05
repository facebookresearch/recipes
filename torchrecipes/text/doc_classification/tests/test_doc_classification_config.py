# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict
import os.path
from typing import Tuple

import hydra
import testslide

# Register our config class to use it in Hydra's helpers.
import torchrecipes.text.doc_classification.conf  # noqa
from hydra.experimental import compose, initialize_config_module
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.trainer import Trainer
from torchrecipes.text.doc_classification.tests.common.assets import (
    copy_partial_sst2_dataset,
    get_asset_path,
    copy_asset,
)
from torchrecipes.utils.test import tempdir


class TestDocClassificationConfig(testslide.TestCase):
    @tempdir
    def test_doc_classification_task(self, root_dir: str) -> None:
        # copy the asset files into their expected download locations
        # note we need to do this anywhere we use hydra overrides
        # otherwise we get a `LexerNoViableAltException`
        vocab_path = os.path.join(root_dir, "xlmr.vocab.pt")
        spm_model_path = os.path.join(root_dir, "xlmr.sentencepiece.bpe.model")
        copy_asset(get_asset_path("xlmr.vocab.pt"), vocab_path)
        copy_asset(get_asset_path("xlmr.sentencepiece.bpe.model"), spm_model_path)
        copy_partial_sst2_dataset(root_dir)

        with initialize_config_module("torchrecipes.text.doc_classification.conf"):
            cfg = compose(
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
        task, datamodule = self._instantiate_config(cfg)
        trainer = Trainer(**cfg.trainer)
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
        vocab_path = os.path.join(root_dir, "xlmr.vocab.pt")
        spm_model_path = os.path.join(root_dir, "xlmr.sentencepiece.bpe.model")
        copy_asset(get_asset_path("xlmr.vocab.pt"), vocab_path)
        copy_asset(get_asset_path("xlmr.sentencepiece.bpe.model"), spm_model_path)
        copy_partial_sst2_dataset(root_dir)

        with initialize_config_module("torchrecipes.text.doc_classification.conf"):
            cfg = compose(
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
        task, datamodule = self._instantiate_config(cfg)
        trainer = Trainer(**cfg.trainer)
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
        num_classes = len(cfg.transform.label_transform.label_names)
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
