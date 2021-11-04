#!/usr/bin/env python3

import importlib
import os
from typing import Any

import mock
import testslide
from testslide import StrictMock
from torchrecipes.core.base_train_app import BaseTrainApp, TrainOutput
from torchrecipes.launcher import run


class TestLauncherMain(testslide.TestCase):
    @mock.patch.dict(os.environ, {}, clear=True)
    def test_no_env(self) -> None:
        self.assertRaises(NotImplementedError, run.main)

    @mock.patch.dict(os.environ, {"CONFIG_MODULE": "test_module"})
    def test_import_module(self) -> None:
        self.mock_callable(importlib, "import_module").for_call(
            "test_module"
        ).to_return_value(None)
        self.mock_callable(run, "run_with_hydra").to_return_value(
            StrictMock(TrainOutput)
        )
        run.main()


class TestRunWithCertainEnv(testslide.TestCase):
    def assert_train(self, app: BaseTrainApp) -> None:
        mock_output = StrictMock(TrainOutput)
        self.mock_callable(app, "train").to_return_value(
            mock_output
        ).and_assert_called_once()

    def assert_test(self, app: BaseTrainApp) -> None:
        mock_output = []
        self.mock_callable(app, "test").to_return_value(
            mock_output
        ).and_assert_called_once()

    @mock.patch.dict(os.environ, {"MODE": "prod"})
    def test_prod_mode(self) -> None:
        app = StrictMock(template=BaseTrainApp)
        self.assert_train(app)
        self.assert_test(app)
        run.run_in_certain_mode(app)

    @mock.patch.dict(os.environ, {"MODE": "train"})
    def test_train_only(self) -> None:
        app = StrictMock(template=BaseTrainApp)
        self.assert_train(app)
        run.run_in_certain_mode(app)

    @mock.patch.dict(os.environ, {"MODE": "test"})
    def test_test_only(self) -> None:
        app = StrictMock(template=BaseTrainApp)
        self.assert_test(app)
        run.run_in_certain_mode(app)

    @mock.patch.dict(os.environ, {"MODE": "random"})
    def test_wrong_input(self) -> None:
        app = StrictMock(template=BaseTrainApp)
        self.assert_train(app)
        self.assert_test(app)
        run.run_in_certain_mode(app)

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_no_input(self) -> None:
        app = StrictMock(template=BaseTrainApp)
        self.assert_train(app)
        self.assert_test(app)
        run.run_in_certain_mode(app)
