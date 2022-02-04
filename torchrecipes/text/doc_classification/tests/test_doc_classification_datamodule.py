# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

# pyre-strict

from unittest.mock import patch

import hydra
import testslide
import torch
from torchrecipes.text.doc_classification.conf.common import (
    SST2DatasetConf,
    LabelTransformConf,
    DocClassificationTransformConf,
)
from torchrecipes.text.doc_classification.datamodule.doc_classification import (
    DocClassificationDataModuleConf,
    DocClassificationDataModule,
)
from torchrecipes.text.doc_classification.tests.common.assets import _DATA_DIR_PATH
from torchrecipes.text.doc_classification.tests.common.assets import get_asset_path
from torchrecipes.text.doc_classification.transform.doc_classification_text_transform import (
    DocClassificationTextTransformConf,
)


class TestDocClassificationDataModule(testslide.TestCase):
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

    def get_datamodule(self) -> DocClassificationDataModule:
        doc_transform_conf = DocClassificationTextTransformConf(
            vocab_path=get_asset_path("vocab_example.pt"),
            spm_model_path=get_asset_path("spm_example.model"),
        )
        transform_conf = DocClassificationTransformConf(
            transform=doc_transform_conf,
            num_labels=2,
        )

        dataset_conf = SST2DatasetConf(root=_DATA_DIR_PATH)
        datamodule_conf = DocClassificationDataModuleConf(
            transform=transform_conf,
            dataset=dataset_conf,
            columns=["text", "label"],
            label_column="label",
            batch_size=8,
        )
        return hydra.utils.instantiate(
            datamodule_conf,
            _recursive_=False,
        )

    def test_doc_classification_datamodule(self) -> None:
        datamodule = self.get_datamodule()
        self.assertIsInstance(datamodule, DocClassificationDataModule)

        dataloader = datamodule.train_dataloader()
        batch = next(iter(dataloader))

        self.assertTrue(torch.is_tensor(batch["label_ids"]))
        self.assertTrue(torch.is_tensor(batch["token_ids"]))

        self.assertEqual(batch["label_ids"].size(), torch.Size([8]))
        self.assertEqual(batch["token_ids"].size(), torch.Size([8, 35]))
