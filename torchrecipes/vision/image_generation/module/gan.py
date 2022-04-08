# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3
# based on https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/gans/basic/basic_gan_module.py

from dataclasses import dataclass
from typing import Any, List, Tuple, Union

import pytorch_lightning as pl
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

# @manual "//github/third-party/omry/omegaconf:omegaconf"
from omegaconf import MISSING
from torch import nn, Tensor
from torchrecipes.core.conf import ModuleConf
from torchrecipes.utils.config_utils import get_class_name_str

Batch = List[Tensor]
TrainOutput = Tensor
TestOutput = Tuple[Tensor, Tensor]


def _weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        # pyre-fixme[6]: Expected `Tensor` but got `Union[Tensor, nn.Module]`.
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        # pyre-fixme[6]: Expected `Tensor` but got `Union[Tensor, nn.Module]`.
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        # pyre-fixme[6]: Expected `Tensor` but got `Union[Tensor, nn.Module]`.
        torch.nn.init.zeros_(m.bias)


class GAN(pl.LightningModule):
    """Implements a Lighting module for training vision generative adversarial
    networks.

    Args:
        generator: generator model.
        discriminator: discriminator model.
        criterion: criterion to calculate adversarial loss.
        img_dim: dimension of generated image.
        latent_dim: dimension of latent.
        init_weights: whether to initialize Conv and BatchNorm weights.
    """

    def __init__(
        self,
        generator: Any,  # pyre-ignore[2]
        discriminator: Any,  # pyre-ignore[2]
        criterion: Any,  # pyre-ignore[2]
        optim: Any,  # pyre-ignore[2]
        img_dim: Tuple[int, int, int],
        latent_dim: int = 32,
        init_weights: bool = False,
    ) -> None:
        super().__init__()
        self.generator: nn.Module = instantiate(generator)
        self.discriminator: nn.Module = instantiate(discriminator)
        self.criterion: nn.Module = instantiate(criterion)
        self.img_dim = img_dim
        self.latent_dim = latent_dim

        if init_weights:
            self.generator.apply(_weights_init)
            self.discriminator.apply(_weights_init)

        self.save_hyperparameters()

    def training_step(
        self,
        batch: Batch,
        batch_idx: int,
        optimizer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> TrainOutput:
        x, _ = batch

        # train generator
        if optimizer_idx == 0:
            return self.generator_step(x)

        # train discriminator
        if optimizer_idx == 1:
            return self.discriminator_step(x)

        raise AssertionError(
            "There should be two optimizers. "
            f"Invalid optimizer_idx({optimizer_idx})."
        )

    def generator_step(self, x: Tensor) -> Tensor:
        g_loss = self.generator_loss(x)

        self.log("g_loss", g_loss, on_epoch=True, prog_bar=True)
        return g_loss

    def discriminator_step(self, x: Tensor) -> Tensor:
        d_loss = self.discriminator_loss(x)

        self.log("d_loss", d_loss, on_epoch=True, prog_bar=True)
        return d_loss

    def _evaluation_step(self, batch: Batch) -> TestOutput:
        # TODO(kaizh): support quantitative GAN generator evaluation,
        # for example Frechet Inception Distance, Inception Score etc.
        x, _ = batch
        return (self.generator_step(x), self.discriminator_step(x))

    # pyre-fixme[15]: `test_step` overrides method defined in `LightningModule`
    #  inconsistently.
    def test_step(
        self, batch: Batch, batch_idx: int, *args: Any, **kwargs: Any
    ) -> TestOutput:
        return self._evaluation_step(batch)

    # pyre-fixme[15]: `validation_step` overrides method defined in
    #  `LightningModule` inconsistently.
    def validation_step(
        self, batch: Batch, batch_idx: int, *args: Any, **kwargs: Any
    ) -> TestOutput:
        return self._evaluation_step(batch)

    @staticmethod
    def get_latent(
        n_samples: int, latent_dim: int, device: Union[str, torch.device]
    ) -> Tensor:
        return torch.randn(n_samples, latent_dim, device=device)

    # pyre-ignore[14]: *args, **kwargs are not torchscriptable.
    def forward(self, z: Tensor) -> Tensor:
        return self.generator(z)

    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Any]]:
        lr = self.hparams["optim"].lr
        betas = (self.hparams["optim"].beta1, 0.999)

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        return [opt_g, opt_d], []

    def generator_loss(self, x: Tensor) -> Tensor:
        z = self.get_latent(x.shape[0], self.hparams["latent_dim"], self.device)

        generated_imgs = self(z)

        D_output = self.discriminator(generated_imgs)
        y = torch.ones_like(D_output)

        g_loss = self.criterion(D_output, y)

        return g_loss

    def discriminator_loss(self, x: Tensor) -> Tensor:
        # calculate real score
        D_output = self.discriminator(x)
        y_real = torch.ones_like(D_output)
        D_real_loss = self.criterion(D_output, y_real)

        # train discriminator on fake
        z = self.get_latent(x.shape[0], self.hparams["latent_dim"], self.device)
        x_fake = self(z)

        # calculate fake score
        D_output = self.discriminator(x_fake)
        y_fake = torch.zeros_like(D_output)
        D_fake_loss = self.criterion(D_output, y_fake)

        # gradient backprop & optimize ONLY D's parameters
        D_loss = D_real_loss + D_fake_loss

        return D_loss


@dataclass
class GANModuleConf(ModuleConf):
    _target_: str = get_class_name_str(GAN)
    generator: Any = MISSING  # pyre-ignore[4]
    discriminator: Any = MISSING  # pyre-ignore[4]
    criterion: Any = MISSING  # pyre-ignore[4]
    optim: Any = MISSING  # pyre-ignore[4]
    img_dim: Tuple[int, int, int] = (1, 28, 28)
    latent_dim: int = 32
    init_weights: bool = False


cs = ConfigStore()
cs.store(
    group="schema/module",
    name="gan_module_conf",
    node=GANModuleConf,
    package="module",
)
