# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from abc import ABC, abstractmethod
from enum import auto, unique, Enum
from typing import Any

import torch


class BaseApp(ABC):
    def __init__(self) -> None:
        # log API usage. It's a no-op for OSS
        torch._C._log_api_usage_once(
            f"torchrecipes.{self.__module__}.{self.__class__.__name__}"
        )

    @abstractmethod
    def get_data(self) -> Any:
        """
        Return data
        """

    @abstractmethod
    def get_model(self) -> Any:
        """
        Return model
        """

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> Any:
        """
        Train a model
        """

    @abstractmethod
    def test(self, *args: Any, **kwargs: Any) -> Any:
        """
        Train a model
        """

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """
        Predict with a trained model
        """


@unique
class Mode(Enum):
    TRAIN = auto()
    TEST = auto()
    PREDICT = auto()


def get_mode(mode_key) -> Mode:
    mode_key = mode_key.upper()
    try:
        return Mode[mode_key]
    except KeyError:
        return Mode.TRAIN