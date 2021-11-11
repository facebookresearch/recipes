#!/usr/bin/env python3

# pyre-strict
from typing import List, Optional

from torchrecipes.core.test_utils.test_base import BaseTrainAppTestCase
from torchrecipes.utils.test import tempdir
from torchrecipes.vision.image_classification.train_app import (
    ImageClassificationTrainApp,
)


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
