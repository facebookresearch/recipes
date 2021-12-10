# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import hashlib
import json

import hydra
import testslide
from torchrecipes.text.doc_classification.conf.common import SST2DatasetConf
from torchrecipes.text.doc_classification.tests.common.assets import _DATA_DIR_PATH
from torchtext.experimental.datasets import sst2


class TestDocClassificationDataset(testslide.TestCase):
    def test_doc_classification_sst2_dataset(self) -> None:
        dataset_conf = SST2DatasetConf(root=_DATA_DIR_PATH, validate_hash=False)
        train_dataset, dev_dataset, test_dataset = hydra.utils.call(
            dataset_conf, _recursive_=False
        )

        # verify datasets objects are instances of SST2Dataset
        for dataset in (train_dataset, dev_dataset, test_dataset):
            self.assertTrue(isinstance(dataset, sst2.SST2Dataset))

        # verify hashes of first line in dataset
        self.assertEqual(
            hashlib.md5(
                json.dumps(next(iter(train_dataset)), sort_keys=True).encode("utf-8")
            ).hexdigest(),
            sst2._FIRST_LINE_MD5["train"],
        )
        self.assertEqual(
            hashlib.md5(
                json.dumps(next(iter(dev_dataset)), sort_keys=True).encode("utf-8")
            ).hexdigest(),
            sst2._FIRST_LINE_MD5["dev"],
        )
        self.assertEqual(
            hashlib.md5(
                json.dumps(next(iter(test_dataset)), sort_keys=True).encode("utf-8")
            ).hexdigest(),
            sst2._FIRST_LINE_MD5["test"],
        )
