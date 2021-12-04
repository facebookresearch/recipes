# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3
# Base on https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/infogan/infogan.py
import itertools
from dataclasses import dataclass
from typing import Any, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

# @manual "//github/third-party/omry/omegaconf:omegaconf"
from omegaconf import MISSING
from torch import nn, Tensor
from torchrecipes.core.conf import ModuleConf
from torchrecipes.core.task_base import TaskBase
from torchrecipes.utils.config_utils import get_class_name_str
from torchvision.utils import make_grid

Batch = List[Tensor]
TrainOutput = Tensor
TestOutput = Tuple[Tensor, Tensor, Tensor]

# Loss functions
adversarial_loss: torch.nn.modules.loss.MSELoss = torch.nn.MSELoss()
categorical_loss: torch.nn.modules.loss.CrossEntropyLoss = torch.nn.CrossEntropyLoss()
continuous_loss: torch.nn.modules.loss.MSELoss = torch.nn.MSELoss()

# Loss weights
LAMDBA_CAT = 1
LAMBDA_CON = 0.1


def weights_init_normal(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # pyre-fixme[6]: Expected `Tensor` for 1st positional only
        # parameter to call `nn.init.normal_` but got
        # `typing.Union[Tensor, nn.Module]`
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        # pyre-fixme[6]: Expected `Tensor` for 1st positional only
        # parameter to call `nn.init.normal_` but got
        # `typing.Union[Tensor, nn.Module]`
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        # pyre-fixme[6]: Expected `Tensor` for 1st positional only
        # parameter to call `nn.init.constant_` but got
        # `typing.Union[Tensor, nn.Module]`
        torch.nn.init.constant_(m.bias.data, 0.0)


def to_categorical(
    y: np.ndarray, num_columns: int, device: Union[str, torch.device]
) -> Tensor:
    """Returns one-hot encoded Tensor"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return torch.tensor(y_cat, dtype=torch.float, device=device)


class InfoGAN(TaskBase[Batch, TrainOutput, TestOutput], pl.LightningModule):
    """
    Implements a Lighting module for training vision InfoGAN.

    Args:
        generator: generator model.
        discriminator: discriminator model.
        optim: config for optimizers.
        img_dim: dimension of generated image.
    """

    def __init__(
        self,
        generator: Any,  # pyre-ignore[2]
        discriminator: Any,  # pyre-ignore[2]
        optim: Any,  # pyre-ignore[2]
        img_dim: Tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.generator: nn.Module = instantiate(generator)
        self.discriminator: nn.Module = instantiate(discriminator)
        self.save_hyperparameters()
        self.img_dim = img_dim

        self.n_classes: int = generator.n_classes
        self.latent_dim: int = generator.latent_dim
        self.code_dim: int = generator.code_dim

        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

        # init state variables
        self.cur_batch_size: int = 0
        self.gen_imgs: Tensor = Tensor([0])

        # Adversarial ground truths
        self.valid: Tensor = Tensor([0])
        self.fake: Tensor = Tensor([0])

        # Configure input
        self.real_imgs: Tensor = Tensor([0])

        # Static generator inputs for sampling
        self.register_buffer(
            "static_z", torch.zeros([self.n_classes ** 2, self.latent_dim])
        )
        self.register_buffer(
            "static_label",
            to_categorical(
                np.array(
                    [
                        num
                        for _ in range(self.n_classes)
                        for num in range(self.n_classes)
                    ]
                ),
                num_columns=self.n_classes,
                device=self.device,
            ),
        )
        self.register_buffer(
            "static_code", torch.zeros([self.n_classes ** 2, self.code_dim])
        )

    def training_step(
        self,
        batch: Batch,
        batch_idx: int,
        optimizer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> TrainOutput:
        imgs, _ = batch

        # train generator
        if optimizer_idx == 0:
            self._update(imgs)
            return self.generator_step()

        # train discriminator
        if optimizer_idx == 1:
            return self.discriminator_step()

        # information loss
        if optimizer_idx == 2:
            return self.info_loss_step()

        raise AssertionError(
            "There should be three optimizers. "
            f"Invalid optimizer_idx({optimizer_idx})."
        )

    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Any]]:
        lr = self.hparams["optim"].lr
        b1 = self.hparams["optim"].b1
        b2 = self.hparams["optim"].b2

        optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(b1, b2)
        )
        optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(b1, b2)
        )
        optimizer_info = torch.optim.Adam(
            itertools.chain(
                self.generator.parameters(), self.discriminator.parameters()
            ),
            lr=lr,
            betas=(b1, b2),
        )
        return [optimizer_G, optimizer_D, optimizer_info], []

    def _update(self, imgs: Tensor) -> None:
        self.cur_batch_size = imgs.shape[0]

        self.valid = torch.ones(
            [self.cur_batch_size, 1], device=self.device, requires_grad=False
        )
        self.fake = torch.zeros(
            [self.cur_batch_size, 1], device=self.device, requires_grad=False
        )
        self.real_imgs = imgs.type(torch.float32)

    # pyre-ignore[14]: *args, **kwargs are not torchscriptable.
    def forward(self, z: Tensor) -> Tensor:
        return self.generate_sample_image()

    def generator_step(self) -> TrainOutput:
        # Sample noise and labels as generator input
        z = torch.tensor(
            np.random.normal(0, 1, (self.cur_batch_size, self.latent_dim)),
            dtype=torch.float,
            device=self.device,
        )

        label_input = to_categorical(
            np.random.randint(0, self.n_classes, self.cur_batch_size),
            num_columns=self.n_classes,
            device=self.device,
        )
        code_input = torch.tensor(
            np.random.uniform(-1, 1, (self.cur_batch_size, self.code_dim)),
            dtype=torch.float,
            device=self.device,
        )

        # Generate a batch of images
        self.gen_imgs = self.generator(z, label_input, code_input)

        # Loss measures generator's ability to fool the discriminator
        validity, _, _ = self.discriminator(self.gen_imgs)
        g_loss = adversarial_loss(validity, self.valid)
        self.log("g_loss", g_loss, prog_bar=True)
        return g_loss

    def discriminator_step(self) -> TrainOutput:
        # Loss for real images
        real_pred, _, _ = self.discriminator(self.real_imgs)
        d_real_loss = adversarial_loss(real_pred, self.valid)

        # Loss for fake images
        fake_pred, _, _ = self.discriminator(self.gen_imgs.detach())
        d_fake_loss = adversarial_loss(fake_pred, self.fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        return d_loss

    def info_loss_step(self) -> TrainOutput:
        # Sample labels
        sampled_labels = np.random.randint(0, self.n_classes, self.cur_batch_size)

        # Ground truth labels
        gt_labels = torch.tensor(
            sampled_labels, dtype=torch.long, device=self.device, requires_grad=False
        )

        # Sample noise, labels and code as generator input
        z = torch.tensor(
            np.random.normal(0, 1, (self.cur_batch_size, self.latent_dim)),
            dtype=torch.float,
            device=self.device,
        )

        label_input = to_categorical(
            sampled_labels, num_columns=self.n_classes, device=self.device
        )
        code_input = torch.tensor(
            np.random.uniform(-1, 1, (self.cur_batch_size, self.code_dim)),
            dtype=torch.float,
            device=self.device,
        )

        gen_imgs = self.generator(z, label_input, code_input)
        _, pred_label, pred_code = self.discriminator(gen_imgs)

        info_loss = LAMDBA_CAT * categorical_loss(
            pred_label, gt_labels
        ) + LAMBDA_CON * continuous_loss(pred_code, code_input)

        self.log("info_loss", info_loss, prog_bar=True)
        return info_loss

    def _evaluation_step(self, batch: Batch) -> TestOutput:
        imgs, labels = batch
        self._update(imgs)
        return (self.generator_step(), self.discriminator_step(), self.info_loss_step())

    def test_step(
        self, batch: Batch, batch_idx: int, *args: Any, **kwargs: Any
    ) -> TestOutput:
        return self._evaluation_step(batch)

    def validation_step(
        self, batch: Batch, batch_idx: int, *args: Any, **kwargs: Any
    ) -> TestOutput:
        return self._evaluation_step(batch)

    def generate_sample_image(self, n_row: int = 10) -> Tensor:
        # Static sample
        z = torch.tensor(
            np.random.normal(0, 1, (n_row ** 2, self.latent_dim)),
            dtype=torch.float,
            device=self.device,
        )

        static_sample = self.generator(z, self.static_label, self.static_code)
        static_img = make_grid(static_sample, nrow=n_row, normalize=True, padding=0)

        # Get varied c1 and c2
        zeros = np.zeros((n_row ** 2, 1))
        c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
        c1 = torch.tensor(
            np.concatenate((c_varied, zeros), -1),
            dtype=torch.float,
            device=self.device,
        )
        c2 = torch.tensor(
            np.concatenate((zeros, c_varied), -1),
            dtype=torch.float,
            device=self.device,
        )
        sample1 = self.generator(self.static_z, self.static_label, c1)
        sample1_img = make_grid(sample1, nrow=n_row, normalize=True, padding=0)
        sample2 = self.generator(self.static_z, self.static_label, c2)
        sample2_img = make_grid(sample2, nrow=n_row, normalize=True, padding=0)

        return torch.cat((static_img, sample1_img, sample2_img), 0)


@dataclass
class InfoGANModuleConf(ModuleConf):
    _target_: str = get_class_name_str(InfoGAN)
    generator: Any = MISSING  # pyre-ignore[4]
    discriminator: Any = MISSING  # pyre-ignore[4]
    optim: Any = MISSING  # pyre-ignore[4]
    img_dim: Tuple[int, int, int] = (3, 320, 320)


cs = ConfigStore()
cs.store(
    group="schema/module",
    name="infogan_module_conf",
    node=InfoGANModuleConf,
    package="module",
)
