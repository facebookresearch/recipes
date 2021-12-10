#!/usr/bin/env python3

# pyre-strict
from typing import List, Optional

from torchrecipes.audio.source_separation.train_app import SourceSeparationTrainApp
from torchrecipes.core.test_utils.test_base import BaseTrainAppTestCase
from torchrecipes.utils.test import tempdir


class TestTrainApp(BaseTrainAppTestCase):
    def _get_train_app(
        self, tb_save_dir: str, test_overrides: Optional[List[str]] = None
    ) -> SourceSeparationTrainApp:
        overrides: List[str] = [
            "datamodule=test_data",
            "trainer.max_epochs=2",
            f"trainer.default_root_dir={tb_save_dir}",
            f"+tb_save_dir={tb_save_dir}",
        ]
        app = self.create_app_from_hydra(
            config_module="torchrecipes.audio.source_separation.conf",
            config_name="train_app",
            overrides=test_overrides if test_overrides else overrides,
        )
        self.mock_trainer_params(app, {"logger": True})
        return app

    @tempdir
    def test_train_model(self, root_dir: str) -> None:
        train_app = self._get_train_app(tb_save_dir=root_dir)
        # Train the model with the config
        train_app.train()
