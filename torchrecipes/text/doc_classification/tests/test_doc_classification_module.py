# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict
import os
from unittest.mock import patch

import torchtext
from pytorch_lightning.trainer import Trainer
from torch.optim import AdamW
from torchrecipes.text.doc_classification.datamodule.doc_classification import (
    DocClassificationDataModule,
)
from torchrecipes.text.doc_classification.module.doc_classification import (
    DocClassificationModule,
)
from torchrecipes.text.doc_classification.tests.common.assets import (
    _DATA_DIR_PATH,
    get_asset_path,
)
from torchrecipes.text.doc_classification.transform.doc_classification_text_transform import (
    DocClassificationTextTransform,
)
from torchrecipes.utils.task_test_base import TaskTestCaseBase
from torchtext.datasets.sst2 import SST2
from torchtext.transforms import LabelToIndex


class TestDocClassificationModule(TaskTestCaseBase):
    def setUp(self) -> None:
        self.base_dir = os.path.join(os.path.dirname(__file__), "data")
        # patch the _hash_check() fn output to make it work with the dummy dataset
        self.patcher = patch(
            "torchdata.datapipes.iter.util.cacheholder._hash_check", return_value=True
        )
        self.patcher.start()

    def tearDown(self) -> None:
        self.patcher.stop()
        super().tearDown()

    def get_transform(self) -> DocClassificationTextTransform:
        return DocClassificationTextTransform(
            vocab_path=get_asset_path("vocab_example.pt"),
            spm_model_path=get_asset_path("spm_example.model"),
        )

    def get_standard_task(self) -> DocClassificationModule:
        model = torchtext.models.RobertaBundle.build_model(
            encoder_conf=torchtext.models.roberta.model.RobertaEncoderConf(
                vocab_size=102,
                embedding_dim=8,
                ffn_dimension=8,
                padding_idx=1,
                max_seq_len=64,
                num_attention_heads=1,
                num_encoder_layers=1,
            ),
            head=torchtext.models.roberta.model.RobertaClassificationHead(
                num_classes=2,
                input_dim=8,
                inner_dim=8,
            ),
        )
        optim = AdamW(model.parameters())
        return DocClassificationModule(
            transform=self.get_transform(),
            model=model,
            optim=optim,
            num_classes=2,
        )

    def get_datamodule(self) -> DocClassificationDataModule:
        train_dataset, val_dataset, test_dataset = SST2(root=_DATA_DIR_PATH)
        label_transform = LabelToIndex(label_names=["0", "1"])
        return DocClassificationDataModule(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            # TODO: Note that the following line should be replaced by
            # `test_dataset` once we update the lightning module to support
            # test data with and without labels
            test_dataset=val_dataset,
            transform=self.get_transform(),
            label_transform=label_transform,
            columns=["text", "label"],
            label_column="label",
            batch_size=8,
        )

    def test_train(self) -> None:
        # pyre-fixme[16]: `TestDocClassificationModule` has no attribute `datamodule`.
        self.datamodule = self.get_datamodule()
        task = self.get_standard_task()
        trainer = Trainer(fast_dev_run=True)
        trainer.fit(task, datamodule=self.datamodule)

        pred1 = task.forward({"text": ["hello world", "how are you?"]})
        pred2 = task.forward(
            {"text": ["hello world", "how are you?"], "label": ["1", "0"]}
        )
        self.assertTrue(pred1 is not None)
        self.assertTrue(pred2 is not None)
