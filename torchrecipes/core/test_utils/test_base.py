#!/usr/bin/env python3

# pyre-strict

from typing import Any, Callable, Dict, List, Optional

import testslide
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torchrecipes.core.base_train_app import BaseTrainApp, TrainOutput


def get_mock_init_trainer_params(
    overrides: Optional[Dict[str, Any]] = None,
) -> Callable[..., Dict[str, Any]]:
    """
    Order of trainer_params setting in unit test:
      - First call original function, which sets params from config
      - Then override some params to disable logger and checkpoint
      - Apply any test-specific overrides.
    """

    def mock_init_trainer_params(
        original: Callable[..., Dict[str, Any]],
    ) -> Dict[str, Any]:
        trainer_params = original()

        trainer_params["logger"] = False
        trainer_params["checkpoint_callback"] = False
        trainer_params["fast_dev_run"] = True

        if overrides:
            trainer_params.update(overrides)

        return trainer_params

    return mock_init_trainer_params


class BaseTrainAppTestCase(testslide.TestCase):
    """All Standard TrainApp unit tests should inherit from this class."""

    def mock_trainer_params(
        self, app: BaseTrainApp, overrides: Optional[Dict[str, Any]] = None
    ) -> None:
        self.mock_callable(
            app, "_init_trainer_params", allow_private=True
        ).with_wrapper(get_mock_init_trainer_params(overrides))

    def create_app_from_hydra(
        self,
        config_module: str,
        config_name: str,
        overrides: Optional[List[str]] = None,
    ) -> BaseTrainApp:
        with initialize_config_module(config_module=config_module):
            cfg = compose(config_name=config_name, overrides=overrides or [])
        print(OmegaConf.to_yaml(cfg))
        return instantiate(cfg, _recursive_=False)

    def assert_train_output(self, output: TrainOutput) -> None:
        self.assertIsNotNone(output)
        # Ensure logger is set to False in test to avoid dependency on Manifold
        self.assertIsNone(output.tensorboard_log_dir)
