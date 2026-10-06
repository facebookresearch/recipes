# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3

import os
import unittest
from tempfile import TemporaryDirectory

import hydra
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torchrecipes.vision.core.datamodule.mnist_data_module import MNISTDataModule
from torchrecipes.vision.core.datamodule.transforms import build_transforms
from torchvision.datasets import MNIST

IMAGE_SIZE = (28, 28)
NUM_TRAIN_IMAGES = 60000
NUM_TEST_IMAGES = 10000
UINT8_TYPE_CODE = 8


def write_idx_file(path: str, data: torch.Tensor) -> None:
    """Writes a tensor in the big-endian "Pascal Vincent" format MNIST uses."""
    with open(path, "wb") as f:
        f.write((UINT8_TYPE_CODE * 256 + data.dim()).to_bytes(4, "big"))
        for dim in data.shape:
            f.write(dim.to_bytes(4, "big"))
        f.write(data.numpy().tobytes())


def create_mnist_dataset(root: str) -> None:
    """Creates a synthetic MNIST dataset on disk, avoiding a network download."""
    raw_folder = os.path.join(root, MNIST.__name__, "raw")
    os.makedirs(raw_folder, exist_ok=True)
    generator = torch.Generator().manual_seed(42)
    for prefix, num_images in (
        ("train", NUM_TRAIN_IMAGES),
        ("t10k", NUM_TEST_IMAGES),
    ):
        images = torch.randint(
            256, (num_images, *IMAGE_SIZE), dtype=torch.uint8, generator=generator
        )
        labels = torch.randint(
            10, (num_images,), dtype=torch.uint8, generator=generator
        )
        write_idx_file(os.path.join(raw_folder, f"{prefix}-images-idx3-ubyte"), images)
        write_idx_file(os.path.join(raw_folder, f"{prefix}-labels-idx1-ubyte"), labels)


class TestMNISTDataModule(unittest.TestCase):
    data_path: str

    @classmethod
    def setUpClass(cls) -> None:
        data_path_ctx = TemporaryDirectory()
        cls.addClassCleanup(data_path_ctx.cleanup)
        cls.data_path = data_path_ctx.name

        create_mnist_dataset(cls.data_path)

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
        # pyre-fixme[6]: For 1st param expected `Sized` but got `Dataset[typing.Any]`.
        self.assertEqual(len(module.datasets["train"]), 59900)
        # pyre-fixme[6]: For 1st param expected `Sized` but got `Dataset[typing.Any]`.
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

    def test_init_with_hydra(self) -> None:
        """Tests creating module with Hydra."""
        test_conf = {
            "_target_": "torchrecipes.vision.core.datamodule.mnist_data_module.MNISTDataModule",
            "data_dir": None,
            "val_split": 0.2,
            "num_workers": 16,
            "normalize": False,
            "batch_size": 32,
            "seed": 42,
            "shuffle": False,
            "pin_memory": False,
            "drop_last": False,
        }
        mnist_data_module = hydra.utils.instantiate(test_conf)
        self.assertIsInstance(mnist_data_module, MNISTDataModule)
