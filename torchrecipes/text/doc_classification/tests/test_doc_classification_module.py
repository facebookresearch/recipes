# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict
import os
from unittest.mock import patch

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.trainer import Trainer
from torchrecipes.text.doc_classification.datamodule.doc_classification import (
    DocClassificationDataModule,
)
from torchrecipes.text.doc_classification.module.doc_classification import (
    DocClassificationModule,
    DocClassificationModuleConf,
)
from torchrecipes.text.doc_classification.tests.common.assets import _DATA_DIR_PATH
from torchrecipes.text.doc_classification.tests.common.assets import get_asset_path
from torchrecipes.text.doc_classification.transform.doc_classification_text_transform import (
    DocClassificationTextTransformConf,
)
from torchrecipes.utils.config_utils import get_class_name_str
from torchrecipes.utils.task_test_base import TaskTestCaseBase
from torchtext.datasets.sst2 import SST2


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

    def get_transform_conf(self) -> DictConfig:
        doc_transform_conf = DocClassificationTextTransformConf(
            vocab_path=get_asset_path("vocab_example.pt"),
            spm_model_path=get_asset_path("spm_example.model"),
        )
        return OmegaConf.create(
            {"transform": doc_transform_conf, "num_labels": 2, "label_transform": None}
        )

    def get_standard_task(self) -> DocClassificationModule:
        module_conf = DocClassificationModuleConf(
            model=OmegaConf.load(
                "torchrecipes/text/doc_classification/conf/module/model/xlmrbase_classifier_tiny.yaml"
            ),
            optim=OmegaConf.load(
                "torchrecipes/text/doc_classification/conf/module/optim/adamw.yaml"
            ),
        )
        transform_conf = self.get_transform_conf()
        num_classes = transform_conf.num_labels
        return hydra.utils.instantiate(
            module_conf,
            transform=transform_conf.transform,
            num_classes=num_classes,
            _recursive_=False,
        )

    def get_datamodule(self) -> DocClassificationDataModule:
        transform_conf = self.get_transform_conf()
        dataset_conf = OmegaConf.create(
            {"root": _DATA_DIR_PATH, "_target_": get_class_name_str(SST2)}
        )
        datamodule_conf = OmegaConf.create(
            {
                "_target_": "torchrecipes.text.doc_classification.datamodule.doc_classification.DocClassificationDataModule.from_config",
                "transform": transform_conf,
                "dataset": dataset_conf,
                "columns": ["text", "label"],
                "label_column": "label",
                "batch_size": 8,
            }
        )
        return hydra.utils.instantiate(
            datamodule_conf,
            _recursive_=False,
        )

    def test_python_conf(self) -> None:
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
