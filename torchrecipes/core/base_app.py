# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseApp(ABC):
    def __init__(self) -> None:
        # log API usage. It's a no-op for OSS
        torch._C._log_api_usage_once(
            f"torchrecipes.{self.__module__}.{self.__class__.__name__}"
        )

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> None:
        """
        The main method to run a recipe. It could be any script like
        training, testing, predicting or any combination of them
        """
