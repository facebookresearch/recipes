#!/usr/bin/env python3

# pyre-strict
import os
from copy import deepcopy
from typing import List, Optional

import torch
from torch import nn
from torchrecipes.core.test_utils.test_base import BaseTrainAppTestCase
from torchrecipes.utils.test import tempdir
from torchrecipes.vision.core.ops.fine_tuning_wrapper import FineTuningWrapper
from torchrecipes.vision.core.utils.model_weights import load_model_weights
from torchrecipes.vision.image_classification.train_app import (
    ImageClassificationTrainApp,
)
from torchvision.models.resnet import resnet18
from torchvision.ops.misc import FrozenBatchNorm2d


class TestTrainApp(BaseTrainAppTestCase):
    def _get_train_app(
        self, tb_save_dir: str, test_overrides: Optional[List[str]] = None
    ) -> ImageClassificationTrainApp:
        overrides: List[str] = [
            "datamodule/datamodule=fake_data",
            "+module.model.num_classes=10",
            f"+tb_save_dir={tb_save_dir}",
        ]
        app = self.create_app_from_hydra(
            config_module="torchrecipes.vision.image_classification.conf",
            config_name="train_app",
            overrides=test_overrides if test_overrides else overrides,
        )
        self.mock_trainer_params(app, {"logger": True})
        # pyre-fixme[7]: Expected `ImageClassificationTrainApp` but got `BaseTrainApp`.
        return app

    @tempdir
    def test_train_model(self, root_dir: str) -> None:
        train_app = self._get_train_app(tb_save_dir=root_dir)
        # Train the model with the config
        train_app.train()

    @tempdir
    def test_fine_tuning(self, root_dir: str) -> None:
        pretrained = resnet18()

        weights_path = os.path.join(root_dir, "weights.pth")
        torch.save(pretrained.state_dict(), weights_path)

        # prepare model for fine-tuning
        trunk = resnet18(norm_layer=FrozenBatchNorm2d)
        load_model_weights(trunk, weights_path)
        head = nn.Linear(in_features=512, out_features=10)
        fine_tune_model = FineTuningWrapper(trunk, "flatten", head)
        origin_trunk = deepcopy(fine_tune_model.trunk)

        # start fine-tuning
        classification_train_app = self._get_train_app(tb_save_dir=root_dir)
        # pyre-ignore[16]: ImageClassificationModule has model
        classification_train_app.module.model = fine_tune_model
        classification_train_app.train()

        with torch.no_grad():
            inp = torch.randn(1, 3, 28, 28)
            origin_out = origin_trunk(inp)
            # pyre-ignore[16]: ImageClassificationModule has model
            tuned_out = classification_train_app.module.model.trunk(inp)
            self.assertTrue(torch.equal(origin_out["flatten"], tuned_out["flatten"]))
