# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import hydra
import testslide
import torch
from torchrecipes.text.doc_classification.transform.doc_classification_text_transform import (
    DocClassificationTextTransform,
    DocClassificationTextTransformConf,
)


class TestDocClassificationTransform(testslide.TestCase):
    def test_doc_classification_transform(self) -> None:
        transform_conf = DocClassificationTextTransformConf(
            vocab_path="https://download.pytorch.org/models/text/xlmr.vocab.pt",
            spm_model_path="https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model",
        )
        transform = hydra.utils.instantiate(transform_conf, _recursive_=False)

        # check whether correct class is being instantiated by hydra
        self.assertIsInstance(transform, DocClassificationTextTransform)

        test_input = {"text": ["XLMR base Model Comparison"]}
        actual = transform(test_input)
        expected_token_ids = torch.tensor(
            [[0, 43523, 52005, 3647, 13293, 113307, 40514, 2]], dtype=torch.long
        )
        self.assertTrue(torch.all(actual["token_ids"].eq(expected_token_ids)))
