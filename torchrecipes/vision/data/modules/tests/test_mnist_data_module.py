# Copyright (c) Meta Platforms, Inc. and its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

import unittest
from tempfile import TemporaryDirectory

import torch
from hydra.core.config_store import ConfigStore
from hydra.experimental import compose, initialize
from hydra.utils import instantiate
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torchrecipes.core.test_utils.conf_utils import conf_asdict
from torchrecipes.vision.data.modules.mnist_data_module import (
    MNISTDataModule,
    MNISTDataModuleConf,
)
from torchrecipes.vision.data.transforms import build_transforms
from torchvision.datasets import MNIST


class TestMNISTDataModule(unittest.TestCase):
    data_path: str

    @classmethod
    def setUpClass(cls) -> None:
        data_path_ctx = TemporaryDirectory()
        cls.addClassCleanup(data_path_ctx.cleanup)
        cls.data_path = data_path_ctx.name

        # download the dataset
        MNIST(cls.data_path, train=True, download=True)
        MNIST(cls.data_path, train=False, download=True)

    def test_misconfiguration(self) -> None:
        """Tests init configuration validation."""
        with self.assertRaises(MisconfigurationException):
            MNISTDataModule(val_split=-1)

    def test_dataloading(self) -> None:
        """Tests loading batches from the dataset."""
        module = MNISTDataModule(data_dir=self.data_path, batch_size=1)
        module.prepare_data()
        module.setup()
        dataloder = module.train_dataloader()
        batch = next(iter(dataloder))
        # batch contains images and labels
        self.assertEqual(len(batch), 2)
        self.assertEqual(len(batch[0]), 1)

    def test_split_dataset(self) -> None:
        """Tests splitting the full dataset into train and validation set."""
        module = MNISTDataModule(data_dir=self.data_path, val_split=100)
        module.prepare_data()
        module.setup()
        # pyre-ignore[6]: dataset has length
        self.assertEqual(len(module.datasets["train"]), 59900)
        # pyre-ignore[6]: dataset has length
        self.assertEqual(len(module.datasets["val"]), 100)

    def test_transforms(self) -> None:
        """Tests images being transformed correctly."""
        transform_config = [
            {
                "_target_": "torchvision.transforms.Resize",
                "size": 64,
            },
            {
                "_target_": "torchvision.transforms.ToTensor",
            },
        ]
        transforms = build_transforms(transform_config)
        module = MNISTDataModule(
            data_dir=self.data_path, batch_size=1, train_transforms=transforms
        )
        module.prepare_data()
        module.setup()
        dataloder = module.train_dataloader()
        image, _ = next(iter(dataloder))
        self.assertEqual(image.size(), torch.Size([1, 1, 64, 64]))

    def test_module_conf_dataclass(self) -> None:
        """Tests creating module with dataclass."""
        module = MNISTDataModule(**conf_asdict(MNISTDataModuleConf()))
        self.assertIsInstance(module, MNISTDataModule)

    def test_init_with_hydra(self) -> None:
        """Tests creating module with Hydra."""
        # Set up Hydra configs
        cs = ConfigStore.instance()
        cs.store(name="mnist_data_module", node=MNISTDataModuleConf)
        with initialize():
            test_conf = compose(config_name="mnist_data_module")
            mnist_data_module = instantiate(test_conf)
            self.assertIsInstance(mnist_data_module, MNISTDataModule)
