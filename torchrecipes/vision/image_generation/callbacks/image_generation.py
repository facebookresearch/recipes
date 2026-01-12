# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


#!/usr/bin/env python3
# Based on https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/callbacks/vision/image_generation.py
from contextlib import contextmanager
from typing import Iterator, Optional, Protocol, runtime_checkable, Tuple

import torch
import torchvision
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn


@contextmanager
def mode(net: torch.nn.Module, training: bool) -> Iterator[torch.nn.Module]:
    """Temporarily switch to training/evaluation mode."""
    istrain = net.training
    try:
        net.train(training)
        yield net
    finally:
        net.train(istrain)


@runtime_checkable
class HasInputOutputDimension(Protocol):
    """
    Variables:
        latent_dim: dimension of the latent
        img_dim: dimension of the generated image
    """

    latent_dim: int
    img_dim: Tuple[int, int, int]


class TensorboardGenerativeModelImageSampler(Callback):
    """
    Generates images and logs to tensorboard once per epoch.
    Your model must implement the ``forward`` function for generation
    Requirements::
        # model must have img_dim and latent_dim arg
        model.img_dim = (1, 28, 28)
        # model forward must work for sampling
        z = torch.rand(batch_size, model.latent_dim)
        img_samples = your_model(z)
    Example:
        from torchrecipes.vision.GAN.callbacks import TensorboardGenerativeModelImageSampler
        trainer = Trainer(callbacks=[TensorboardGenerativeModelImageSampler()])
    """

    def __init__(
        self,
        num_samples: int = 3,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``False``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """

        super().__init__()
        self.num_samples = num_samples
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str]
    ) -> None:
        assert isinstance(pl_module, HasInputOutputDimension), (
            "Lightning module should define the dimension of latent and generated image."
        )

    @rank_zero_only
    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not (logger := trainer.logger):
            rank_zero_warn("Trainer must have a logger configured.")
            return
        if not hasattr(logger, "experiment"):
            rank_zero_warn("Trainer must have a logger configured that can log images.")
            return
        # pyre-ignore[16]: `pytorch_lightning.loggers.base.LightningLoggerBase` has no attribute `experiment`.
        experiment = logger.experiment
        dim = (self.num_samples, pl_module.latent_dim)
        # pyre-fixme[6]: For 3rd param expected `Union[List[int], Size,
        #  typing.Tuple[int, ...]]` but got `Tuple[int, Union[Tensor, Module]]`.
        z = torch.normal(mean=0.0, std=1.0, size=dim, device=pl_module.device)

        # generate images
        with torch.no_grad(), mode(pl_module, training=False) as eval_module:
            images = eval_module(z)

        img_dim = pl_module.img_dim
        images = images.view(self.num_samples, *img_dim)

        grid = torchvision.utils.make_grid(
            tensor=images,
            nrow=self.nrow,
            padding=self.padding,
            normalize=self.normalize,
            value_range=self.norm_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )
        str_title = f"{pl_module.__class__.__name__}_images"
        experiment.add_image(str_title, grid, global_step=trainer.global_step)
