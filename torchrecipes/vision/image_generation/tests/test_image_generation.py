# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3
import unittest
from unittest.mock import patch

import torch
import torchvision
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torchrecipes.utils.test import tempdir
from torchrecipes.vision.core.utils.test_module import TestModule
from torchrecipes.vision.image_generation.callbacks import (
    TensorboardGenerativeModelImageSampler,
)


class TestGANModule(unittest.TestCase):
    @tempdir
    def test_module_without_dimension(self, tmp_dir: str) -> None:
        """tests using the callback with a module that doesn't define image and
        latent dimension.
        """
        module = TestModule()
        trainer = Trainer(
            default_root_dir=tmp_dir,
            fast_dev_run=True,
            callbacks=[TensorboardGenerativeModelImageSampler()],
        )

        with self.assertRaises(AssertionError):
            trainer.fit(module)

    @tempdir
    def test_logger_without_add_image(self, tmp_dir: str) -> None:
        """tests using the callback with an unsupported logger."""
        module = TestModule()
        trainer = Trainer(
            default_root_dir=tmp_dir,
            fast_dev_run=True,
            logger=CSVLogger(tmp_dir),
            callbacks=[TensorboardGenerativeModelImageSampler()],
        )

        with self.assertRaises(AssertionError):
            trainer.fit(module)

    @tempdir
    def test_callback_triggered(self, tmp_dir: str) -> None:
        """tests image generation is triggered by end of an epoch."""

        class MyModule(TestModule):
            def __init__(self) -> None:
                super().__init__()
                self.latent_dim = 32
                self.img_dim = (1, 1, 2)

            def forward(self, x) -> torch.Tensor:
                assert x.size() == torch.Size([1, 32])
                return super().forward(x)

        module = MyModule()
        trainer = Trainer(
            default_root_dir=tmp_dir,
            fast_dev_run=True,
            callbacks=[TensorboardGenerativeModelImageSampler(num_samples=1)],
        )

        with patch.object(
            torchvision.utils, "make_grid", return_value=torch.randn(1, 1, 1)
        ) as mock_call:
            trainer.fit(module)
            # called twice: once for training, once for validation
            self.assertEqual(mock_call.call_count, 2)
