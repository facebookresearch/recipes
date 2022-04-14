# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

# pyre-strict

from unittest.mock import patch

import testslide
import torch
from torchrecipes.text.doc_classification.datamodule.doc_classification import (
    DocClassificationDataModule,
)
from torchrecipes.text.doc_classification.tests.common.assets import (
    _DATA_DIR_PATH,
    get_asset_path,
)
from torchrecipes.text.doc_classification.transform.doc_classification_text_transform import (
    DocClassificationTextTransform,
)
from torchtext.datasets.sst2 import SST2
from torchtext.transforms import LabelToIndex


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
        train_dataset, val_dataset, test_dataset = SST2(root=_DATA_DIR_PATH)
        text_transform = DocClassificationTextTransform(
            vocab_path=get_asset_path("vocab_example.pt"),
            spm_model_path=get_asset_path("spm_example.model"),
        )
        label_transform = LabelToIndex(label_names=["0", "1"])
        return DocClassificationDataModule(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            # TODO: Note that the following line should be replaced by
            # `test_dataset` once we update the lightning module to support
            # test data with and without labels
            test_dataset=val_dataset,
            transform=text_transform,
            label_transform=label_transform,
            columns=["text", "label"],
            label_column="label",
            batch_size=8,
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
